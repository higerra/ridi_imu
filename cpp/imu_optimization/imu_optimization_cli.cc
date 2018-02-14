//
// Created by yanhang on 2/6/17.
//

/*
  This code performs testing and optimization in the offline fashion. That means only
  one optimization will be executed for the entire trajectory.
  The training is done by the python code. Please refer to
  code/python/regression_cascade.py
*/
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <random>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include "algorithm/geometry.h"
#include "utility/data_io.h"
#include "speed_regression/feature_target.h"
#include "speed_regression/model_wrapper.h"

#include "imu_optimization.h"

using namespace std;

DEFINE_int32(max_iter, 500, "maximum iteration");
DEFINE_int32(window, 200, "Window size");
DEFINE_string(model_path, "", "Path to models");
DEFINE_bool(gt, false, "Use ground truth");
DEFINE_bool(rv, false, "Use rotation vector");
DEFINE_double(feature_smooth_sigma, 2.0, "Sigma for feature smoothing.");
DEFINE_double(target_smooth_sigma, 30.0, "Sigma for target smoothing.");
DEFINE_double(weight_ls, 1.0, "The weight of local speed residual");
DEFINE_double(weight_vs, 1.0, "The weight of vertical speed residual");

// Pre-defined the grid parameter.
struct Config {
  // Number of regressions.
  static constexpr int kConstriantPoints = 200;
  // Number of frames where variables are defined..
  static constexpr int kSparsePoints = 200;
};

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./IMUOptimization <path-to-datasets>" << endl;
    return 1;
  }

  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);

  char buffer[256] = {};

  printf("Loading...\n");
  ridi::IMUDataset dataset(argv[1]);

  const int kTotalCount = (int) dataset.GetTimeStamp().size();

  const std::vector<double>& ts = dataset.GetTimeStamp();
  const std::vector<Eigen::Vector3d>& gyro = dataset.GetGyro();
  const std::vector<Eigen::Vector3d>& linacce = dataset.GetLinearAcceleration();
  const std::vector<Eigen::Vector3d>& gravity = dataset.GetGravity();

  std::vector<Eigen::Quaterniond> orientation((size_t) kTotalCount);

  if (FLAGS_rv) {
    const Eigen::Quaterniond &tango_init_ori = dataset.GetOrientation()[0];
    const Eigen::Quaterniond &imu_init_ori = dataset.GetRotationVector()[0];
    const Eigen::Quaterniond align_ori = tango_init_ori * imu_init_ori.conjugate();
    for (auto i = 0; i < orientation.size(); ++i) {
      orientation[i] = align_ori * dataset.GetRotationVector()[i];
    }
  } else {
    LOG(WARNING) << "Using ground truth orientation";
    orientation = std::vector<Eigen::Quaterniond>(dataset.GetOrientation().begin(),
                                                  dataset.GetOrientation().begin() + kTotalCount);
  }

  // The rotation from stabilized-IMU frame to the world frame.
  std::vector<Eigen::Quaterniond> R_GW((size_t) kTotalCount);
  const Eigen::Vector3d local_g_dir(0, 1, 0);
  for (auto i = 0; i < kTotalCount; ++i) {
    Eigen::Quaterniond rotor = Eigen::Quaterniond::FromTwoVectors(gravity[i], local_g_dir);
    R_GW[i] = rotor * orientation[i].conjugate();
  }

  // Indices of frames where regression will be performed.
  std::vector<int> constraint_ind;
  // TODO(yanhang): Change "local_speed" to "local_speed_gravity" or "target_value" to avoid confusion.
  std::vector<Eigen::Vector3d> local_speed;

  // regress local speed
  constraint_ind.resize(Config::kConstriantPoints);
  local_speed.resize(constraint_ind.size(), Eigen::Vector3d(0, 0, 0));
  constraint_ind[0] = FLAGS_window;
  const int constraint_interval = (kTotalCount - FLAGS_window) /
      (Config::kConstriantPoints - 1);

  for (int i = 1; i < constraint_ind.size(); ++i) {
    constraint_ind[i] = constraint_ind[i - 1] + constraint_interval;
  }

  if (FLAGS_gt) {
    LOG(WARNING) << "Using ground truth as constraint";
    const std::vector<double> ts_slice(ts.begin(), ts.begin() + kTotalCount);
    const std::vector<Eigen::Vector3d> positions_slice(
        dataset.GetPosition().begin(),
        dataset.GetPosition().begin() + kTotalCount);
    const std::vector<Eigen::Quaterniond> orientations_slice(
        dataset.GetOrientation().begin(),
        dataset.GetOrientation().begin() + kTotalCount
    );
    const std::vector<Eigen::Vector3d> gravity_slice(gravity.begin(), gravity.begin() + kTotalCount);
    cv::Mat local_speed_mat =
        ridi::ComputeLocalSpeedTargetGravityAligned(ts_slice, positions_slice, orientations_slice, gravity_slice,
                                                          constraint_ind, FLAGS_target_smooth_sigma);
    CHECK_EQ(local_speed_mat.rows, local_speed.size());
    for (auto i = 0; i < local_speed.size(); ++i) {
      local_speed[i][0] = (double) local_speed_mat.at<float>(i, 0);
      local_speed[i][1] = (double) local_speed_mat.at<float>(i, 1);
      local_speed[i][2] = (double) local_speed_mat.at<float>(i, 2);
    }
  } else {
    printf("Regressing local speed...\n");
    std::unique_ptr<ridi::ModelWrapper> model(new ridi::SVRCascade(FLAGS_model_path));
#pragma omp parallel for
    for (int i = 0; i < constraint_ind.size(); ++i) {
      const int sid = constraint_ind[i] - FLAGS_window;
      const int eid = constraint_ind[i];
      cv::Mat feature = ridi::ComputeDirectFeatureGravity(&gyro[sid], &linacce[sid], &gravity[sid],
                                                                FLAGS_window, FLAGS_feature_smooth_sigma);
      Eigen::VectorXd response(2);
      model->Predict(feature, &response);
      local_speed[i][0] = response[0];
      local_speed[i][2] = response[1];
    }
  }

  constexpr int kResiduals = Config::kConstriantPoints;
  constexpr int kSparsePoint = Config::kSparsePoints;

  std::vector<Eigen::Vector3d> corrected_linacce = linacce;
  std::vector<Eigen::Quaterniond> corrected_orientation = orientation;
  {
    printf("Optimizing linear acceleration bias...\n");
    ceres::Problem problem_linacce;
    // Initialize bias with gaussian distribution
    std::vector<double> bx((size_t) kSparsePoint, 0.0), by((size_t) kSparsePoint, 0.0), bz((size_t) kSparsePoint,
                                                                                           0.0);
    using FunctorTypeLinacce = ridi::LocalSpeedFunctor<kSparsePoint, kResiduals>;

    FunctorTypeLinacce
        *functor = new FunctorTypeLinacce(ts.data(),
                                          (int) ts.size(),
                                          linacce.data(),
                                          orientation.data(),
                                          R_GW.data(),
                                          constraint_ind.data(),
                                          local_speed.data(),
                                          Eigen::Vector3d(0, 0, 0),
                                          FLAGS_weight_ls,
                                          FLAGS_weight_vs);
    problem_linacce.AddResidualBlock(
        new ceres::AutoDiffCostFunction<FunctorTypeLinacce, 3 * kResiduals, kSparsePoint, kSparsePoint, kSparsePoint>(
            functor), nullptr, bx.data(), by.data(), bz.data());

    problem_linacce.AddResidualBlock(
        new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay<kSparsePoint>(1.0)
        ), nullptr, bx.data());

    problem_linacce.AddResidualBlock(
        new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay<kSparsePoint>(1.0)
        ), nullptr, by.data());

    problem_linacce.AddResidualBlock(
        new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay<kSparsePoint>(1.0)
        ), nullptr, bz.data());

    float start_t = cv::getTickCount();
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 6;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = FLAGS_max_iter;
    ceres::Solver::Summary summary;

    printf("Solving...\n");
    ceres::Solve(options, &problem_linacce, &summary);

    std::cout << summary.BriefReport() << endl;
    printf("Time usage: %.3fs\n", ((float) cv::getTickCount() - start_t) / cv::getTickFrequency());

    functor->GetLinacceGrid()->correct_linacce_bias<double>(corrected_linacce.data(), bx.data(), by.data(),
                                                            bz.data());

    // Write result to a text file for plotting
    sprintf(buffer, "%s/sparse_grid_bias.txt", argv[1]);
    ofstream bias_out(buffer);
    CHECK(bias_out.is_open());
    for (auto i = 0; i < bx.size(); ++i) {
      sprintf(buffer, "%d %.6f %.6f %.6f\n", functor->GetLinacceGrid()->GetVariableIndAt(i),
              bx[i], by[i], bz[i]);
      bias_out << buffer;
    }

    sprintf(buffer, "%s/sparse_grid_constraint.txt", argv[1]);
    ofstream ls_out(buffer);
    CHECK(ls_out.is_open());
    for (auto i = 0; i < constraint_ind.size(); ++i) {
      sprintf(buffer,
              "%d %.6f %.6f %.6f\n",
              constraint_ind[i],
              local_speed[i][0],
              local_speed[i][1],
              local_speed[i][2]);
      ls_out << buffer;
    }

    sprintf(buffer, "%s/corrected_linacce.txt", argv[1]);
    ofstream cs_out(buffer);
    CHECK(cs_out.is_open());
    for (auto i = 0; i < corrected_linacce.size(); ++i) {
      sprintf(buffer,
              "%d %.6f %.6f %.6f\n",
              i,
              corrected_linacce[i][0],
              corrected_linacce[i][1],
              corrected_linacce[i][2]);
      cs_out << buffer;
    }
  }

  std::vector<Eigen::Vector3d>
      directed_corrected_linacce = ridi::Rotate3DVector(corrected_linacce, corrected_orientation);
  std::vector<Eigen::Vector3d> corrected_speed = ridi::Integration(ts, directed_corrected_linacce);
  std::vector<Eigen::Vector3d>
      corrected_position = ridi::Integration(ts, corrected_speed, dataset.GetPosition()[0]);

  sprintf(buffer, "%s/optimized_cpp_gaussian.ply", argv[1]);
  ridi::WriteToPly(std::string(buffer), ts.data(), corrected_position.data(), orientation.data(), kTotalCount);

  std::vector<Eigen::Vector3d> directed_linacce = ridi::Rotate3DVector(linacce, orientation);
  std::vector<Eigen::Vector3d> speed = ridi::Integration(ts, directed_linacce);
  std::vector<Eigen::Vector3d> raw_position = ridi::Integration(ts, speed, dataset.GetPosition()[0]);

  sprintf(buffer, "%s/raw.ply", argv[1]);
  ridi::WriteToPly(std::string(buffer), ts.data(), raw_position.data(), orientation.data(), kTotalCount);

  return 0;
}
