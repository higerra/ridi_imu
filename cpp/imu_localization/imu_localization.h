//
// Created by yanhang on 3/5/17.
//

#ifndef PROJECT_IMU_LOCALIZATION_H
#define PROJECT_IMU_LOCALIZATION_H

#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

#include <Eigen/Eigen>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imu_optimization/imu_optimization.h"
#include "speed_regression/model_wrapper.h"
#include "speed_regression/feature_target.h"

namespace ridi {

  enum RegressionOption {
    // Use the full local speed (with gravity).
    FULL,
    // Use only the magnitude of the speed.
    MAG,
    // Use only the orientation of the speed.
    ORI,
    // Assume constant speed.
    CONST
  };

  struct IMULocalizationOption {
    // Number of frames between two consecutive optimizations.
    int local_opt_interval = 200;
    // Temporal window size (in frames) for each optimization.
    int local_opt_window = 1000;

    // The weight parameter for local speed constraint.
    double weight_ls = 1.0;
    // The weight parameter for zero vertical speed constraint.
    double weight_vs = 1.0;

    RegressionOption reg_option = FULL;

    // When reg_option is set to CONST, the following const speed will be used.
    double const_speed = 1.0;

    static constexpr int reg_interval = 30;

    // The following two terms are used to identify unreliable classification/regression results. It works as follows:
    // For i'th predicted label, we consider the window: [i - $label_filter_radius, i + $label_filter_radius].
    // If the predicted label changes more than $max_allowed_transition, we say that the classification/regression
    // is unreliable.
    const int max_allowed_transition = 2;
    const int label_filter_radius = 7;
  };

  // We pre-define functor to ease the problem formulation with Ceres solver.
  struct FunctorSize {
    static constexpr int kVar_600 = 12;
    static constexpr int kCon_600 = 8;

    static constexpr int kVar_800 = 16;
    static constexpr int kCon_800 = 12;

    static constexpr int kVar_1000 = 20;
    static constexpr int kCon_1000 = 16;

    static constexpr int kVar_5000 = 100;
    static constexpr int kCon_5000 = 96;

    static constexpr int kVar_large = 100;
    static constexpr int kCon_large = 200;
  };

  
  // This class encapsulates the data and processes for RIDI. It receives the input IMU data, performs regression
  // at subsampled frames and estimates bias in linear accelerations. Regressions and optimizations are executed
  // in a background thread.
  class IMUTrajectory {
  public:
    IMUTrajectory(const IMULocalizationOption option,
		  const TrainingDataOption td_option,
		  const Eigen::Vector3d &init_speed,
		  const Eigen::Vector3d &init_position,
		  const ridi::ModelWrapper* model);

    ~IMUTrajectory() {
      terminate_flag_.store(true);
      if (opt_thread_.joinable()) {
	opt_thread_.join();
      }
    }

    void AddRecord(const double t, const Eigen::Vector3d &gyro, const Eigen::Vector3d &linacce,
		   const Eigen::Vector3d &gravity, const Eigen::Quaterniond &orientation);

    /// Run optimization
    /// \param start_id The start index of optimization window. Pass -1 to run global optimization
    /// \param N
    void RunOptimization(const int start_id, const int N);

    void StartOptmizationThread();

    inline void ScheduleOptimization(const int start_id, const int N) {
      std::lock_guard<std::mutex> guard(queue_lock_);
      task_queue_.push_back(std::make_pair(start_id, N));
      if (task_queue_.size() >= max_queue_task_) {
	can_add_.store(false);
      }
      cv_.notify_all();
    }

    // Perform regression at subsampled frames up to end_ind.
    int RegressSpeed(const int end_ind);

    // Construct a ceres problem with the information in a local temporal window.
    template<class FunctorType, int kVar, int kCon>
      const SparseGrid *ConstructProblem(const int start_id, const int N, ceres::Problem &problem,
					 const int* constraint_ind, const Eigen::Vector3d *local_speed,
					 const int* transition_counts,
					 const Eigen::Vector3d init_speed,
					 std::vector<double> &bx, std::vector<double> &by, std::vector<double> &bz) {
      CHECK_GE(constraint_ind_.size(), kCon);

      std::vector<double> weight_ls(kCon, option_.weight_ls);
      std::vector<double> weight_vs(kCon, option_.weight_vs);
      for (int i=0; i<kCon; ++i){

	// Handle the transition period (e.g. move the phone from the bag to the hand). There are two options:
	// 1. Set the weighting for local speed functor to 0, effectively ignoring the regressed speed and
	//    performing the naive double integration.
	// 2. Overwrite the regressed speed to 0. 
	// Here we adopts the second strategy for its robustness.
	if (transition_counts[i] > option_.max_allowed_transition){
	  // weight_ls[i] = std::numeric_limits<double>::epsilon();
	  // weight_vs[i] = std::numeric_limits<double>::epsilon();
	}
      }

      FunctorType *functor = new FunctorType(&ts_[start_id], N, &linacce_[start_id],
					     &orientation_[start_id], &R_GW_[start_id],
					     constraint_ind, local_speed, init_speed,
					     weight_ls.data(), weight_vs.data());
      bx.resize((size_t) kVar, 0.0);
      by.resize((size_t) kVar, 0.0);
      bz.resize((size_t) kVar, 0.0);

      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FunctorType, 3 * kCon, kVar, kVar, kVar>(functor),
			       nullptr, bx.data(), by.data(), bz.data());

      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)),
			       nullptr,
			       bx.data());
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)),
			       nullptr,
			       by.data());
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)),
			       nullptr,
			       bz.data());

      return functor->GetLinacceGrid();
    }

    inline const std::vector<float>& GetRegressionTimes() const{
      return time_regression_;
    };

    inline const std::vector<float>& GetOptimizationTimes() const{
      return time_optimization_;
    };

    inline const Eigen::Vector3d &GetCurrentSpeed() const {
      std::lock_guard<std::mutex> guard(mt_);
      return position_.back();
    }

    inline const Eigen::Quaterniond &GetCurrentOrientation() const {
      return orientation_.back();
    }

    inline const Eigen::Vector3d &GetCurrentPosition() const {
      std::lock_guard<std::mutex> guard(mt_);
      return position_.back();
    }

    inline const std::vector<Eigen::Vector3d> &GetPositions() const {
      std::lock_guard<std::mutex> guard(mt_);
      return position_;
    }

    inline const std::vector<Eigen::Vector3d> &GetSpeed() const {
      std::lock_guard<std::mutex> guard(mt_);
      return speed_;
    }

    inline const std::vector<Eigen::Quaterniond> &GetOrientations() const {
      std::lock_guard<std::mutex> guard(mt_);
      return orientation_;
    }

    inline const std::vector<Eigen::Vector3d> &GetLinearAcceleration() const {
      std::lock_guard<std::mutex> guard(mt_);
      return linacce_;
    }

    inline const std::vector<int> &GetConstraintInd() const {
      std::lock_guard<std::mutex> guard(mt_);
      return constraint_ind_;
    }

    inline const std::vector<Eigen::Vector3d> &GetLocalSpeed() const {
      std::lock_guard<std::mutex> guard(mt_);
      return local_speed_;
    }

    inline const std::vector<int>& GetLabels() const{
      return labels_;
    }

    inline const std::vector<int>& GetTransitionCounts() const{
      return transition_counts_;
    }

    inline const int GetNumFrames() const {
      std::lock_guard<std::mutex> guard(mt_);
      return num_frames_;
    }

    inline bool CanAdd() const {
      return can_add_.load();
    }

    inline void EndTrajectory() {
      terminate_flag_.store(true);
      cv_.notify_all();
      if (opt_thread_.joinable()) {
	opt_thread_.join();
      }
    }

    void CommitOptimizationResult(const SparseGrid *grid, const int start_id,
				  const double *bx, const double *by, const double *bz);

    static constexpr int kInitCapacity_ = 10000;

    // If the the regressed speed exceeds this value, set the "weight_ls" for that constraint to 0.
    static constexpr double kMaxSpeed = 100;

    static const Eigen::Vector3d local_gravity_dir_;

  private:
    std::vector<double> ts_;
    std::vector<Eigen::Vector3d> linacce_;
    std::vector<Eigen::Vector3d> gyro_;
    std::vector<Eigen::Vector3d> gravity_;
    std::vector<Eigen::Quaterniond> orientation_;
    std::vector<Eigen::Quaterniond> R_GW_;

    std::vector<Eigen::Vector3d> speed_;
    std::vector<Eigen::Vector3d> position_;

    std::vector<float> time_regression_;
    std::vector<float> time_optimization_;

    std::vector<int> constraint_ind_;

    std::vector<int> labels_;
    std::vector<int> transition_counts_;
    std::vector<Eigen::Vector3d> local_speed_;
    int last_constraint_ind_;

    const ridi::ModelWrapper* model_;

    Eigen::Vector3d init_speed_;
    Eigen::Vector3d init_position_;

    std::deque<std::pair<int, int> > task_queue_;

    int num_frames_;

    // Options for localization
    const IMULocalizationOption option_;
    // Options for extracting feature
    const TrainingDataOption td_option_;

    mutable std::mutex mt_;
    mutable std::mutex queue_lock_;
    static constexpr int max_queue_task_ = 3;

    std::condition_variable cv_;
    std::atomic<bool> terminate_flag_;
    std::atomic<bool> can_add_;
    std::thread opt_thread_;
  };

} //namespace ridi

#endif //PROJECT_IMU_LOCALIZATION_H
