//
// Created by Yan Hang on 3/4/17.
//

#ifndef PROJECT_SPEED_REGRESSION_H
#define PROJECT_SPEED_REGRESSION_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include "utility/data_io.h"

namespace ridi {

enum TargetType {
  LOCAL_SPEED,
  LOCAL_SPEED_GRAVITY_ALIGNED,
};

enum FeatureType {
  DIRECT,
  DIRECT_GRAVITY_ALIGNED,
};

struct TrainingDataOption {
  explicit TrainingDataOption(const int step = 10, const int window = 200,
                              const FeatureType feature = DIRECT_GRAVITY_ALIGNED,
                              const TargetType target = LOCAL_SPEED_GRAVITY_ALIGNED) :
      step_size(step), window_size(window), feature_type(feature), target_type(target) {}
  // A feature vector/regression target will be computed every "step_size" frames.
  int step_size;
  // Frames inside the local window (i-window_size, i] will be used to construct the feature vector for frame i.
  int window_size;

  // The IMU readings will be gaussian-smoothed with the "feature_smooth_sigma" before feature construction.
  double feature_smooth_sigma = 2.0;
  // The computed regression target (i.e. velocity) will be gaussian-smoothed with the "target_smooth_sigma".
  double target_smooth_sigma = 30.0;

  FeatureType feature_type;
  TargetType target_type;

  const int kThreads = 1;
};

// Compute different type of features. Please refer code/python/training_data.py for the definition of each feature.
cv::Mat ComputeLocalSpeedTarget(const std::vector<double> &time_stamp,
                                const std::vector<Eigen::Vector3d> &position,
                                const std::vector<Eigen::Quaterniond> &orientation,
                                const std::vector<int> &sample_points,
                                const double sigma = -1);

cv::Mat ComputeLocalSpeedTargetGravityAligned(const std::vector<double>& time_stamp,
                                              const std::vector<Eigen::Vector3d>& position,
                                              const std::vector<Eigen::Quaterniond>& orientation,
                                              const std::vector<Eigen::Vector3d>& gravity,
                                              const std::vector<int>& sample_points,
                                              const double sigma = -1,
                                              const Eigen::Vector3d local_gravity = Eigen::Vector3d(0, 1, 0));

cv::Mat ComputeDirectFeature(const Eigen::Vector3d* gyro,
                             const Eigen::Vector3d* linacce,
                             const int N, const double sigma = -1);


cv::Mat ComputeDirectFeatureGravity(const Eigen::Vector3d* gyro,
                                    const Eigen::Vector3d* linacce,
                                    const Eigen::Vector3d* gravity,
                                    const int N, const double sigma = -1,
                                    const Eigen::Vector3d local_gravity = Eigen::Vector3d(0, 1, 0));

// Compute all feature vectors given the full dataset. The number of feature vectors is controlled by the option.
void CreateFeatureMat(const TrainingDataOption& option, const IMUDataset& data, cv::Mat* feature);

} // namespace ridi

#endif //PROJECT_SPEED_REGRESSION_H
