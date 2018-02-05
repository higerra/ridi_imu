//
// Created by yanhang on 10/9/17.
//

#ifndef ALGORITHM_GEOMETRY_H_
#define ALGORITHM_GEOMETRY_H_

#include <Eigen/Eigen>
#include <glog/logging.h>

namespace ridi {

// Rotate an array of 3D vectors by an array of quaternions.
std::vector<Eigen::Vector3d> Rotate3DVector(const std::vector<Eigen::Vector3d> &input,
                                            const std::vector<Eigen::Quaterniond> &orientation);

// Perform single integration.
std::vector<Eigen::Vector3d> Integration(const std::vector<double> &ts,
                                         const std::vector<Eigen::Vector3d> &input,
                                         const Eigen::Vector3d &initial = Eigen::Vector3d(0, 0, 0));

// We need to eliminate device tilting before computing heading from the magnet vector.
inline Eigen::Quaterniond OrientationFromMagnet(const Eigen::Vector3d& magnet,
                                                const Eigen::Vector3d& gravity,
                                                const Eigen::Vector3d global_gravity = Eigen::Vector3d(0, 0, 1),
                                                const Eigen::Vector3d global_north = Eigen::Vector3d(0, 1, 0)){
  Eigen::Quaterniond rot_grav = Eigen::Quaterniond::FromTwoVectors(gravity, global_gravity);
  Eigen::Vector3d magnet_grav = rot_grav * magnet;
  magnet_grav[2] = 0.0;
  Eigen::Quaterniond rot_magnet = Eigen::Quaterniond::FromTwoVectors(magnet_grav, global_north);
  return rot_magnet;
}

// This function estimates the rigid transformation from source to target. The source and target array should contain
// the same number of points. A homogenous representation of the transformation will be returned. In addition,
// the rotation and/or translation part of the transformation can be required separately.
template<int DIM>
void EstimateTransformation(const std::vector<Eigen::Matrix<double, DIM, 1>> &source,
                            const std::vector<Eigen::Matrix<double, DIM, 1>> &target,
                            Eigen::Matrix<double, DIM + 1, DIM + 1> *transformation_homo,
                            Eigen::Matrix<double, DIM, DIM> *rotation = nullptr,
                            Eigen::Matrix<double, DIM, 1> *translation = nullptr);


//////////////////////////////////////
// Implementation

template<int DIM>
void EstimateTransformation(const std::vector<Eigen::Matrix<double, DIM, 1>> &source,
                            const std::vector<Eigen::Matrix<double, DIM, 1>> &target,
                            Eigen::Matrix<double, DIM + 1, DIM + 1> *transformation_homo,
                            Eigen::Matrix<double, DIM, DIM> *rotation,
                            Eigen::Matrix<double, DIM, 1> *translation) {
  CHECK_EQ(source.size(), target.size());
  const auto kPoints = source.size();
  Eigen::Matrix<double, DIM, 1> source_center = Eigen::Matrix<double, DIM, 1>::Zero();
  Eigen::Matrix<double, DIM, 1> target_center = Eigen::Matrix<double, DIM, 1>::Zero();
  for (int i=0; i<kPoints; ++i){
    source_center += source[i];
    target_center += target[i];
  }
  source_center /= static_cast<double>(kPoints);
  target_center /= static_cast<double>(kPoints);

  Eigen::MatrixXd source_zeromean(kPoints, DIM);
  Eigen::MatrixXd target_zeromean(kPoints, DIM);
  for (int i=0; i<kPoints; ++i){
    source_zeromean.block<1, DIM>(i, 0) = source[i] - source_center;
    target_zeromean.block<1, DIM>(i, 0) = target[i] - target_center;
  }

  Eigen::Matrix<double, DIM, DIM> covariance = source_zeromean.transpose() * target_zeromean;
  Eigen::JacobiSVD<Eigen::Matrix<double, DIM, DIM>> svd(covariance, Eigen::ComputeFullU|Eigen::ComputeFullV);
  Eigen::Matrix<double, DIM, DIM> V = svd.matrixV();
  Eigen::Matrix<double, DIM, DIM> U = svd.matrixU();
  Eigen::Matrix<double, DIM, DIM> R = V * U.transpose();
  if (R.determinant() < 0){
    for (int i=0; i<DIM; ++i){
      V(i, DIM-1) *= -1;
    }
    R = V * U.transpose();
  }

  Eigen::Matrix<double, DIM, 1> t = target_center - R * source_center;
  CHECK_NOTNULL(transformation_homo)->setIdentity();
  transformation_homo->block(0, 0, DIM, DIM) = R;
  transformation_homo->block(0, DIM, DIM, 1) = t;
  if (rotation != nullptr){
    *rotation = R;
  }
  if (translation != nullptr){
    *translation = t;
  }
}

}  // namespace ridi

#endif //ALGORITHM_GEOMETRY_H_
