//
// Created by Yan Hang on 3/4/17.
//

#include <thread>

#include "feature_target.h"

using namespace std;
using namespace cv;

namespace ridi {

cv::Mat ComputeLocalSpeedTarget(const std::vector<double> &time_stamp,
                                const std::vector<Eigen::Vector3d> &position,
                                const std::vector<Eigen::Quaterniond> &orientation,
                                const std::vector<int> &sample_points,
                                const double sigma) {
  const auto N = (int) time_stamp.size();
  CHECK_EQ(position.size(), N);
  CHECK_EQ(orientation.size(), N);

  std::vector<Eigen::Vector3d> global_speed(position.size(), Eigen::Vector3d(0, 0, 0));
  for (auto i = 0; i < N - 1; ++i) {
    global_speed[i] = (position[i + 1] - position[i]) / (time_stamp[i + 1] - time_stamp[i]);
  }
  global_speed[global_speed.size() - 2] = global_speed[global_speed.size() - 1];

  Mat local_speed_all(N, 3, CV_32FC1, cv::Scalar::all(0));
  auto *ls_ptr = (float *) local_speed_all.data;
  for (auto i = 0; i < N; ++i) {
    Eigen::Vector3d ls = orientation[i].conjugate() * global_speed[i];
    ls_ptr[i * 3] = (float) ls[0];
    ls_ptr[i * 3 + 1] = (float) ls[1];
    ls_ptr[i * 3 + 2] = (float) ls[2];
  }

  Mat local_speed_filtered = local_speed_all.clone();
  if (sigma > 0){
    cv::GaussianBlur(local_speed_all, local_speed_filtered, cv::Size(0, 0), 0, sigma);
  }
  Mat local_speed((int) sample_points.size(), 3, CV_32FC1, cv::Scalar::all(0));
  for (auto i = 0; i < sample_points.size(); ++i) {
    const int ind = sample_points[i];
    local_speed.at<float>(i, 0) = local_speed_filtered.at<float>(ind, 0);
    local_speed.at<float>(i, 1) = local_speed_filtered.at<float>(ind, 1);
    local_speed.at<float>(i, 2) = local_speed_filtered.at<float>(ind, 2);
  }

  return local_speed;
}


cv::Mat ComputeLocalSpeedTargetGravityAligned(const std::vector<double>& time_stamp,
                                              const std::vector<Eigen::Vector3d>& position,
                                              const std::vector<Eigen::Quaterniond>& orientation,
                                              const std::vector<Eigen::Vector3d>& gravity,
                                              const std::vector<int>& sample_points,
                                              const double sigma, const Eigen::Vector3d local_gravity){
  LOG(ERROR) << "The C++ code is only for testing. This function shouldn't be called.";
  const auto N = (int) time_stamp.size();
  CHECK_EQ(position.size(), N);
  CHECK_EQ(orientation.size(), N);

  std::vector<Eigen::Vector3d> global_speed(position.size(), Eigen::Vector3d(0, 0, 0));
  for (auto i = 0; i < N - 1; ++i) {
    global_speed[i] = (position[i + 1] - position[i]) / (time_stamp[i + 1] - time_stamp[i]);
  }
  global_speed[global_speed.size() - 2] = global_speed[global_speed.size() - 1];

  Mat local_speed_all(N, 3, CV_32FC1, cv::Scalar::all(0));
  auto *ls_ptr = (float *) local_speed_all.data;
  for (auto i = 0; i < N; ++i) {
    Eigen::Quaterniond rotor = Eigen::Quaterniond::FromTwoVectors(gravity[i], local_gravity);
    Eigen::Vector3d ls = rotor * orientation[i].conjugate() * global_speed[i];
    ls_ptr[i * 3] = (float) ls[0];
    ls_ptr[i * 3 + 1] = (float) ls[1];
    ls_ptr[i * 3 + 2] = (float) ls[2];
  }

  Mat local_speed_filtered = local_speed_all.clone();
  if (sigma > 0){
    cv::GaussianBlur(local_speed_all, local_speed_filtered, cv::Size(1, 0), 0, sigma);
  }

  Mat local_speed_gravity((int) sample_points.size(), 3, CV_32FC1, cv::Scalar::all(0));
  for (auto i = 0; i < sample_points.size(); ++i) {
    const int ind = sample_points[i];
    local_speed_gravity.at<float>(i, 0) = local_speed_filtered.at<float>(ind, 0);
    local_speed_gravity.at<float>(i, 1) = local_speed_filtered.at<float>(ind, 1);
    local_speed_gravity.at<float>(i, 2) = local_speed_filtered.at<float>(ind, 2);
  }

  return local_speed_gravity;
}


cv::Mat ComputeDirectFeature(const Eigen::Vector3d* gyro,
                             const Eigen::Vector3d* linacce,
                             const int N, const double sigma) {
  Mat feature(N, 6, CV_32FC1, cv::Scalar::all(0));
  for (int i = 0; i < N; ++i) {
    feature.at<float>(i, 0) = (float) gyro[i][0];
    feature.at<float>(i, 1) = (float) gyro[i][1];
    feature.at<float>(i, 2) = (float) gyro[i][2];
    feature.at<float>(i, 3) = (float) linacce[i][0];
    feature.at<float>(i, 4) = (float) linacce[i][1];
    feature.at<float>(i, 5) = (float) linacce[i][2];
  }
  cv::Mat feature_filtered = feature.clone();
  if (sigma > 0){
    cv::GaussianBlur(feature, feature_filtered, cv::Size(1, 0), 0, sigma, cv::BORDER_REFLECT);
  }
  // Flatten the feature matrix to a row vector
  return feature_filtered.reshape(1, 1);
}

cv::Mat ComputeDirectFeatureGravity(const Eigen::Vector3d* gyro,
                                    const Eigen::Vector3d* linacce,
                                    const Eigen::Vector3d* gravity,
                                    const int N, const double sigma,
                                    const Eigen::Vector3d local_gravity) {
  Mat feature(N, 6, CV_32FC1, cv::Scalar::all(0));
  const double epsilon = 1e-03;
  for (auto i = 0; i < N; ++i) {

    Eigen::Vector3d aligned_linacce = linacce[i];
    Eigen::Vector3d aligned_gyro = gyro[i];

    // There are two singular points when computing the rotation between the measured gravity direction
    // and the target gravity direction: in parallel or in oppsite direction.
    const double dot_pro = local_gravity.dot(gravity[i]) / (gravity[i].norm() * local_gravity.norm());
    if(dot_pro < -1.0 + epsilon){
      // In opposite direction, reverse the Y component.
      aligned_linacce[1] *= -1;
      aligned_gyro[1] *= -1;
    } else if (std::fabs(dot_pro) < 1.0 - epsilon){
      // Within non-singular range.
      Eigen::Quaterniond rotor = Eigen::Quaterniond::FromTwoVectors(gravity[i], local_gravity);
      aligned_linacce = rotor * linacce[i];
      aligned_gyro = rotor * gyro[i];
    }

    feature.at<float>(i, 0) = (float) aligned_gyro[0];
    feature.at<float>(i, 1) = (float) aligned_gyro[1];
    feature.at<float>(i, 2) = (float) aligned_gyro[2];
    feature.at<float>(i, 3) = (float) aligned_linacce[0];
    feature.at<float>(i, 4) = (float) aligned_linacce[1];
    feature.at<float>(i, 5) = (float) aligned_linacce[2];
  }
  cv::Mat feature_filtered = feature.clone();
  if (sigma > 0){
    cv::GaussianBlur(feature, feature_filtered, cv::Size(1, 0), 0, sigma, cv::BORDER_REFLECT);
  }
  // Flatten the matrix to a row vector
  return feature_filtered.reshape(1, 1);
}

void CreateFeatureMat(const TrainingDataOption& option, const IMUDataset& data, cv::Mat* feature){
  const std::vector<double>& ts = data.GetTimeStamp();
  const std::vector<Eigen::Vector3d>& gyro = data.GetGyro();
  const std::vector<Eigen::Vector3d>& linacce = data.GetLinearAcceleration();
  const std::vector<Eigen::Vector3d>& gravity = data.GetGravity();
  const int kSamples = ts.size();
  CHECK_NOTNULL(feature)->create((kSamples - option.window_size) / option.step_size,
                                 6 * option.window_size, CV_32FC1);
  auto thread_func = [&](const int tid) {
    for (int i = tid; i < feature->rows; i += option.kThreads){
      cv::Mat feat;
      const int sid = i * option.step_size;
      if (option.feature_type == DIRECT) {
        feat = ComputeDirectFeature(&gyro[sid], &linacce[sid], option.window_size, option.feature_smooth_sigma);
      } else {
        feat = ComputeDirectFeatureGravity(&gyro[sid], &linacce[sid], &gravity[sid], option.window_size,
                                           option.feature_smooth_sigma);
      }
      feat.copyTo(feature->row(i));
    }
  };

  // Parallel execution
  if (option.kThreads == 1){
    thread_func(0);
  } else {
    std::vector<std::thread> threads;
    for (int i=0; i<option.kThreads; ++i){
      std::thread t(thread_func, i);
      threads.push_back(std::move(t));
    }

    for (auto& t: threads){
      if (t.joinable()){
        t.join();
      }
    }
  }
}

}  // namespace ridi
