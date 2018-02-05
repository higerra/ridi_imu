//
// Created by Yan Hang on 3/1/17.
//

#ifndef PROJECT_UTILITY_H
#define PROJECT_UTILITY_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace ridi {


void TrajectoryOverlay(const double pixel_length, const Eigen::Vector2d &sp, const Eigen::Vector3d &map_ori,
                       const std::vector<Eigen::Vector3d> &positions, const Eigen::Vector3d &color, cv::Mat &map);



}//namespace ridi

#endif //PROJECT_UTILITY_H
