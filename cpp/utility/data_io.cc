//
// Created by yanhang on 2/6/17.
//

#include "data_io.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <fstream>

using namespace std;

namespace ridi {

IMUDataset::IMUDataset(const std::string &directory, unsigned char load_control) : file_io_(directory) {

  fstream fin(file_io_.GetPlainTextData().c_str());
  CHECK(fin.is_open()) << "Can not open file " << file_io_.GetPlainTextData();

  int kSamples, kColumns;
  fin >> kSamples >> kColumns;
  CHECK_EQ(kColumns, layout_.rotation_vector + 4);
  timestamp_.resize((size_t) kSamples, 0);

  for (int i = 0; i < kSamples; ++i) {
    Eigen::Vector3d gyro, acce, linacce, gravity, magnet, pos;
    Eigen::Quaterniond ori, rv;
    fin >> timestamp_[i] >> gyro[0] >> gyro[1] >> gyro[2];
    fin >> acce[0] >> acce[1] >> acce[2];
    fin >> linacce[0] >> linacce[1] >> linacce[2];
    fin >> gravity[0] >> gravity[1] >> gravity[2];
    fin >> magnet[0] >> magnet[1] >> magnet[2];
    fin >> pos[0] >> pos[1] >> pos[2];
    fin >> ori.w() >> ori.x() >> ori.y() >> ori.z();
    fin >> rv.w() >> rv.x() >> rv.y() >> rv.z();
    // Perform bit-wise add operation to decide what domains to load.
    if (load_control & IMU_ORIENTATION) {
      orientation_.push_back(ori);
    }
    if (load_control & IMU_POSITION) {
      position_.push_back(pos);
    }
    if (load_control & IMU_GYRO) {
      gyrocope_.push_back(gyro);
    }
    if (load_control & IMU_ACCELEROMETER) {
      accelerometer_.push_back(acce);
    }
    if (load_control & IMU_LINEAR_ACCELERATION) {
      linear_acceleration_.push_back(linacce);
    }
    if (load_control & IMU_GRAVITY) {
      gravity_.push_back(gravity);
    }
    if (load_control & IMU_MAGNETOMETER){
      magnet_.push_back(magnet);
    }
    if (load_control & IMU_ROTATION_VECTOR) {
      rotation_vector_.push_back(rv);
    }
  }
  // Convert the unit of time from nano-second to second to simplity velocity computation.
  for (int i = 0; i < kSamples; ++i) {
    timestamp_[i] = timestamp_[i] / kNanoToSec;
  }
}

void WriteToPly(const std::string &path, const double *ts, const Eigen::Vector3d *position,
                const Eigen::Quaterniond *orientation, const int N, const bool only_xy,
                const Eigen::Vector3d traj_color, const double axis_length, const int kpoints, const int interval) {
  using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
  TriMesh mesh;
  mesh.request_vertex_colors();

  constexpr int axis_color[3][3] = {{0, 200, 0}, {0, 255, 0}, {0, 0, 255}};

  // First add trajectory points
  for (int i = 0; i < N; ++i) {
    Eigen::Vector3d pt = position[i];
    if (only_xy) {
      pt[2] = 0.0;
    }
    TriMesh::VertexHandle vertex = mesh.add_vertex(TriMesh::Point(pt[0], pt[1], pt[2]));
    Eigen::Vector3d cur_color = traj_color;
    mesh.set_color(vertex, TriMesh::Color((int) cur_color[0], (int) cur_color[1], (int) cur_color[2]));
  }

  // Then add axis points
  if (kpoints > 0 && interval > 0 && axis_length > 0) {
    Eigen::Matrix3d local_axis = Eigen::Matrix3d::Identity();
    for (int i = 0; i < N; i += interval) {
      Eigen::Matrix3d axis_dir = orientation[i].toRotationMatrix() * local_axis;
      for (int j = 0; j < kpoints; ++j) {
        for (int k = 0; k < 1; ++k) {
          Eigen::Vector3d pos = position[i];
          if (only_xy) {
            pos[2] = 0.0;
          }
          Eigen::Vector3d pt = pos + axis_length / kpoints * j * axis_dir.block<3, 1>(0, k);
          TriMesh::VertexHandle vertex = mesh.add_vertex(TriMesh::Point(pt[0], pt[1], pt[2]));
          mesh.set_color(vertex, TriMesh::Color(axis_color[k][0], axis_color[k][1], axis_color[k][2]));
        }
      }
    }
  }
  // Write file
  OpenMesh::IO::Options wopt;
  wopt += OpenMesh::IO::Options::VertexColor;

  try {
    OpenMesh::IO::write_mesh(mesh, path, wopt);
  } catch (const std::runtime_error &e) {
    CHECK(true) << e.what();
  }
}

}//namespace ridi
