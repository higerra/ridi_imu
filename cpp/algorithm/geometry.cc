//
// Created by yanhang on 10/9/17.
//

#include "geometry.h"

namespace ridi{

std::vector<Eigen::Vector3d> Rotate3DVector(const std::vector<Eigen::Vector3d> &input,
                                            const std::vector<Eigen::Quaterniond> &orientation) {
  std::vector<Eigen::Vector3d> output(input.size());
  for (int i = 0; i < input.size(); ++i) {
    output[i] = orientation[i].toRotationMatrix() * input[i];
  }

  return output;
}


std::vector<Eigen::Vector3d> Integration(const std::vector<double> &ts,
                                         const std::vector<Eigen::Vector3d> &input,
                                         const Eigen::Vector3d &initial) {
  CHECK_EQ(ts.size(), input.size());
  std::vector<Eigen::Vector3d> output(ts.size());
  output[0] = initial;
  for (int i = 1; i < ts.size(); ++i) {
    output[i] = output[i - 1] + (input[i - 1] + input[i]) / 2.0 * (ts[i] - ts[i - 1]);
  }

  return output;
}


}  // namespace ridi