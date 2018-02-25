//
// Created by yanhang on 3/5/17.
//

#include "imu_localization.h"

#include <chrono>

#include "algorithm/geometry.h"
#include "speed_regression/feature_target.h"

namespace ridi {

using Functor600 = LocalSpeedFunctor<FunctorSize::kVar_600, FunctorSize::kCon_600>;
using Functor800 = LocalSpeedFunctor<FunctorSize::kVar_800, FunctorSize::kCon_800>;
using Functor1000 = LocalSpeedFunctor<FunctorSize::kVar_1000, FunctorSize::kCon_1000>;
using Functor5000 = LocalSpeedFunctor<FunctorSize::kVar_5000, FunctorSize::kCon_5000>;
using FunctorLarge = LocalSpeedFunctor<FunctorSize::kVar_large, FunctorSize::kCon_large>;

const Eigen::Vector3d IMUTrajectory::local_gravity_dir_ = Eigen::Vector3d(0, 1, 0);

IMUTrajectory::IMUTrajectory(const IMULocalizationOption option,
                             const TrainingDataOption td_option,
                             const Eigen::Vector3d &init_speed,
                             const Eigen::Vector3d &init_position,
                             const ModelWrapper* model):
    option_(option), td_option_(td_option),
    init_speed_(init_speed), init_position_(init_position), num_frames_(0),
    last_constraint_ind_(td_option.window_size - option.reg_interval),
    model_(model){
  ts_.reserve(kInitCapacity_);
  gyro_.reserve(kInitCapacity_);
  linacce_.reserve(kInitCapacity_);
  orientation_.reserve(kInitCapacity_);
  speed_.reserve(kInitCapacity_);
  terminate_flag_.store(false);
  can_add_.store(true);

  opt_thread_ = std::move(std::thread(&IMUTrajectory::StartOptmizationThread, this));
}

void IMUTrajectory::AddRecord(const double t, const Eigen::Vector3d &gyro, const Eigen::Vector3d &linacce,
                              const Eigen::Vector3d &gravity, const Eigen::Quaterniond &orientation) {
  num_frames_ += 1;
  ts_.push_back(t);
  linacce_.push_back(linacce);
  orientation_.push_back(orientation);
  gyro_.push_back(gyro);
  gravity_.push_back(gravity);

  Eigen::Quaterniond rotor_g = Eigen::Quaterniond::FromTwoVectors(gravity, local_gravity_dir_);
  R_GW_.push_back(rotor_g * orientation.conjugate());

  std::lock_guard<std::mutex> guard(mt_);
  if (num_frames_ > 1) {
    const double dt = ts_[num_frames_ - 1] - ts_[num_frames_ - 2];
    speed_.emplace_back(orientation * linacce * dt);
    position_.emplace_back(speed_[num_frames_ - 1] * dt);
  } else {
    speed_.push_back(init_speed_);
    position_.push_back(init_position_);
  }
}

void IMUTrajectory::CommitOptimizationResult(const SparseGrid *grid, const int start_id,
                                             const double *bx, const double *by, const double *bz) {
  // Correct acceleration and re-do double integration
  CHECK_NOTNULL(grid)->correct_linacce_bias<double>(&linacce_[start_id], bx, by, bz);
  std::lock_guard<std::mutex> guard(mt_);
  for (int i = start_id + 1; i < num_frames_; ++i) {
    const double dt = ts_[i] - ts_[i - 1];
    speed_[i] = speed_[i - 1] + orientation_[i - 1] * linacce_[i - 1] * dt;
    position_[i] = position_[i - 1] + speed_[i - 1] * dt;
  }
}

  
int IMUTrajectory::RegressSpeed(const int end_ind) {
  const int old_label_counts = static_cast<int>(labels_.size());
  for (int i = last_constraint_ind_ + option_.reg_interval; i < end_ind; i += option_.reg_interval) {
    auto clock = cv::getTickCount();
    std::vector<Eigen::Vector3d> gyro_slice(gyro_.begin() + i - td_option_.window_size, gyro_.begin() + i);
    std::vector<Eigen::Vector3d> linacce_slice(linacce_.begin() + i - td_option_.window_size,
                                               linacce_.begin() + i);
    std::vector<Eigen::Vector3d> gravity_slice(gravity_.begin() + i - td_option_.window_size,
                                               gravity_.begin() + i);
    cv::Mat feature = ComputeDirectFeatureGravity(gyro_slice.data(), linacce_slice.data(), gravity_slice.data(),
                                                  (int) gyro_slice.size(), td_option_.feature_smooth_sigma);

    Eigen::VectorXd regressed(2);
    int predicted_label;
    model_->Predict(feature, &regressed, &predicted_label);
    labels_.emplace_back(predicted_label);
    transition_counts_.emplace_back(0);

    const double ls_x = regressed[0];
    const double ls_z = regressed[1];

    constraint_ind_.push_back(i);
    // The forward direction is defined in the stabilized IMU frame, therefore no gravity compensation is needed.
    Eigen::Vector3d forward_dir(0, 0, -1);

    if (option_.reg_option == FULL) {
      local_speed_.emplace_back(ls_x, 0, ls_z);
    } else if (option_.reg_option == CONST) {
      local_speed_.emplace_back(forward_dir * option_.const_speed);
    } else if (option_.reg_option == MAG) {
      local_speed_.emplace_back(forward_dir * std::sqrt(ls_x * ls_x + ls_z * ls_z));
    } else if (option_.reg_option == ORI) {
      double ang = std::atan2(ls_z, ls_x);
      local_speed_.emplace_back(option_.const_speed * std::cos(ang), 0, option_.const_speed * std::sin(ang));
    } else { // Z_ONLY
      local_speed_.emplace_back(ls_x, 0, 0);
    }

    time_regression_.push_back((cv::getTickCount() - clock) / static_cast<float>(cv::getTickFrequency()));
  }

  // Inspect the predicted labels and update transition counts;
  // We also need to update transition counts as before as (last_constraint_id - label_filter_radius).
  const int update_start_id = std::max(old_label_counts - option_.label_filter_radius, 0);
  for (int i=update_start_id; i < labels_.size(); ++i){
    const int window_start_id = std::max(i - option_.label_filter_radius, 0);
    const int window_end_id = std::min(i + option_.label_filter_radius,  static_cast<int>(labels_.size()));
    transition_counts_[i] = 0;
    for (int j=window_start_id + 1; j < window_end_id; ++j){
      if (labels_[j] != labels_[j-1]){
        ++transition_counts_[i];
      }
    }
     if (transition_counts_[i] > option_.max_allowed_transition){
       // Transition state identified, overwrite the regressed speed to 0.
       local_speed_[i] = Eigen::Vector3d(0, 0, 0);
     }
  }

  // Update index of last regressed velocity.
  last_constraint_ind_ = constraint_ind_.back();
  return 0;
}

  
void IMUTrajectory::RunOptimization(const int start_id, const int N) {
  CHECK_GE(start_id, 0);
  CHECK_GE(N, 600);
  // Complete the speed regression up to this point
  RegressSpeed(start_id + N);

  ceres::Problem problem;
  const SparseGrid *grid = nullptr;
  std::vector<double> bx, by, bz;

  std::vector<int> cur_constraint_id;
  std::vector<int> cur_transition_counts;
  std::vector<Eigen::Vector3d> cur_local_speed;

  Eigen::Vector3d cur_init_speed;

  auto ConstructConstraint = [&](const int kCon) {
    CHECK_GE(constraint_ind_.size(), kCon);
    if (start_id > 0) {
      for (auto i = constraint_ind_.size() - kCon; i < constraint_ind_.size(); ++i) {
        cur_constraint_id.push_back(constraint_ind_[i] - start_id);
        cur_local_speed.push_back(local_speed_[i]);
        cur_transition_counts.push_back(transition_counts_[i]);
      }
      cur_init_speed = speed_[start_id];
    } else {
      // Be sure to include the first and last constraint
      cur_init_speed = init_speed_;
      const float inc = (float) constraint_ind_.size() / ((float) kCon - 1);
      for (auto i = 0; i < kCon - 1; ++i) {
        cur_constraint_id.push_back(constraint_ind_[(int) (i * inc)]);
        cur_local_speed.push_back(local_speed_[(int) (i * inc)]);
        cur_transition_counts.push_back(transition_counts_[i * inc]);
      }
      cur_constraint_id.push_back(constraint_ind_.back());
      cur_local_speed.push_back(local_speed_.back());
      cur_transition_counts.push_back(transition_counts_.back());
    }
  };

  if (N >= 600 && N < 800) {
    ConstructConstraint(FunctorSize::kCon_600);
    CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_600);
    grid = ConstructProblem<Functor600, FunctorSize::kVar_600, FunctorSize::kCon_600>
        (start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_transition_counts.data(),
         cur_init_speed, bx, by, bz);
  } else if (N >= 800 && N < 1000) {
    ConstructConstraint(FunctorSize::kCon_800);
    CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_800);
    grid = ConstructProblem<Functor800, FunctorSize::kVar_800, FunctorSize::kCon_800>
        (start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_transition_counts.data(),
         cur_init_speed, bx, by, bz);
  } else if (N >= 1000 && N < 5000) {
    ConstructConstraint(FunctorSize::kCon_1000);
    CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_1000);
    grid = ConstructProblem<Functor1000, FunctorSize::kVar_1000, FunctorSize::kCon_1000>
        (start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_transition_counts.data(),
         cur_init_speed, bx, by, bz);
  } else if (N >= 5000 && N < 10100) {
    ConstructConstraint(FunctorSize::kCon_5000);
    CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_5000);
    grid = ConstructProblem<Functor5000, FunctorSize::kVar_5000, FunctorSize::kCon_5000>
        (start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_transition_counts.data(),
         cur_init_speed, bx, by, bz);
  } else {
    ConstructConstraint(FunctorSize::kCon_large);
    CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_large);
    grid = ConstructProblem<FunctorLarge, FunctorSize::kVar_large, FunctorSize::kCon_large>
        (start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_transition_counts.data(),
         cur_init_speed, bx, by, bz);
  }

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 3;
  solver_options.linear_solver_type = ceres::DENSE_QR;

  auto clock = cv::getTickCount();

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();

  time_optimization_.push_back((cv::getTickCount() - clock) / static_cast<float>(cv::getTickFrequency()));

  CommitOptimizationResult(grid, start_id, bx.data(), by.data(), bz.data());
}

void IMUTrajectory::StartOptmizationThread() {
  auto check_interval = std::chrono::microseconds(10);

  LOG(INFO) << "Background thread started";
  while (true) {
    std::unique_lock<std::mutex> guard(queue_lock_);
    cv_.wait_for(guard, check_interval, [this] { return !task_queue_.empty(); });
    if (terminate_flag_.load()) {
      // Make sure to finish the remaining optimization task
      std::vector<std::pair<int, int> > remaining_tasks;
      for (const auto &t: task_queue_) {
        remaining_tasks.push_back(t);
      }
      guard.unlock();

      for (const auto &t: remaining_tasks) {
        RunOptimization(t.first, t.second);
      }
      break;
    }
    if (task_queue_.empty()) {
      guard.unlock();
      continue;
    }

    std::pair<int, int> cur_task = task_queue_.front();
    task_queue_.pop_front();
    if (task_queue_.size() < max_queue_task_) {
      can_add_.store(true);
    }
    guard.unlock();

    RunOptimization(cur_task.first, cur_task.second);

  }
  LOG(INFO) << "Background thread terminated";
}
}//namespace ridi
