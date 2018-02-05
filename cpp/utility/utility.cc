//
// Created by Yan Hang on 3/1/17.
//

#include "utility.h"

namespace ridi {

void TrajectoryOverlay(const double pixel_length, const Eigen::Vector2d &sp, const Eigen::Vector3d &map_ori,
                       const std::vector<Eigen::Vector3d> &positions, const Eigen::Vector3d &color, cv::Mat &map) {
  CHECK(map.data);

  // Assume
  constexpr int forward_start = 600;
  constexpr int forward_end = 1200;

  Eigen::Vector3d ori_traj = positions[forward_end] - positions[forward_start];
  ori_traj[1] *= -1;
  ori_traj[2] = 0.0;

  Eigen::Quaterniond rotor;
  rotor.setFromTwoVectors(ori_traj, map_ori);

  for (const auto &pos: positions) {
    Eigen::Vector3d new_pos = rotor * Eigen::Vector3d(pos[0], -pos[1], pos[2]);
    Eigen::Vector2d pix_loc = (new_pos / pixel_length).block<2, 1>(0, 0) + sp;
    cv::circle(map, cv::Point(pix_loc[0], pix_loc[1]), 1, cv::Scalar(color[0], color[1], color[2]));
  }
}

}//namespace ridi
