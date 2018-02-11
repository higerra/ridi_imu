//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_DATASET_H
#define PROJECT_IMU_DATASET_H

#include <vector>
#include <string>

#include <Eigen/Eigen>
#include <glog/logging.h>

namespace ridi {

// This class manages the path of data files w.r.t to the data folder.
class FileIO {
 public:
  explicit FileIO(const std::string& directory) : directory_(directory) {}

  const std::string &GetDirectory() const {
    return directory_;
  }

  inline const std::string GetProcessedData() const {
    char buffer[128] = {};
    sprintf(buffer, "%s/processed/data.csv", directory_.c_str());
    return std::string(buffer);
  }

  inline const std::string GetPlainTextData() const {
    char buffer[128] = {};
    sprintf(buffer, "%s/processed/data_plain.txt", directory_.c_str());
    return std::string(buffer);
  }
 private:
  const std::string directory_;
};

// Structure to specify the column layout. The numbers indicate the starting column index for each domain.
struct DataLayout {
  const int time_stamp = 0;
  const int gyro = 1;
  const int accelerometer = 4;
  const int linear_acceleration = 7;
  const int gravity = 10;
  const int magnet = 13;
  const int position = 16;
  // "orientation" is the device orientation from the ground truth, while "rotation_vector" is the device orientation
  // from the Android system.
  const int orientation = 19;
  const int rotation_vector = 23;
};

class IMUDataset {
 public:
  // Constructor for IMUDataset. When loading the dataset, a bit-wise and operation will be performed with various
  // pre-defined values to decide what domains to load. See static constexpr variables defined below.
  explicit IMUDataset(const std::string &directory, unsigned char load_control = 255);

  inline const std::vector<Eigen::Vector3d> &GetGyro() const {
    return gyrocope_;
  }
  inline std::vector<Eigen::Vector3d> &GetGyro() {
    return gyrocope_;
  }

  inline const std::vector<Eigen::Vector3d> &GetLinearAcceleration() const {
    return linear_acceleration_;
  }
  inline std::vector<Eigen::Vector3d> &GetLinearAcceleration() {
    return linear_acceleration_;
  }

  inline const std::vector<Eigen::Vector3d> &GetAccelerometer() const {
    return accelerometer_;
  }

  inline std::vector<Eigen::Vector3d> &GetAccelerometer() {
    return accelerometer_;
  }

  inline const std::vector<Eigen::Vector3d> &GetGravity() const {
    return gravity_;
  }

  inline std::vector<Eigen::Vector3d> &GetGravity() {
    return gravity_;
  }

  inline const std::vector<Eigen::Vector3d> &GetMagnet() const{
    return magnet_;
  }

  inline std::vector<Eigen::Vector3d> &GetMagnet(){
    return magnet_;
  }

  inline std::vector<Eigen::Quaterniond> &GetRotationVector() {
    return rotation_vector_;
  }

  inline const std::vector<Eigen::Quaterniond> &GetRotationVector() const {
    return rotation_vector_;
  }

  inline const std::vector<Eigen::Quaterniond> &GetOrientation() const {
    return orientation_;
  }

  inline std::vector<Eigen::Quaterniond> &GetOrientation() {
    return orientation_;
  }

  inline const std::vector<Eigen::Vector3d> &GetPosition() const {
    return position_;
  }

  inline std::vector<Eigen::Vector3d> &GetPosition() {
    return position_;
  }

  inline const std::vector<double> &GetTimeStamp() const {
    return timestamp_;
  }

  // The binary codes for each domain. For example, if "load_control & IMU_GYRO == 1", the gyro data will be loaded.
  static constexpr unsigned char IMU_GYRO = 1;
  static constexpr unsigned char IMU_ACCELEROMETER = 2;
  static constexpr unsigned char IMU_LINEAR_ACCELERATION = 4;
  static constexpr unsigned char IMU_GRAVITY = 8;
  static constexpr unsigned char IMU_MAGNETOMETER = 16;
  static constexpr unsigned char IMU_ROTATION_VECTOR = 32;
  static constexpr unsigned char IMU_POSITION = 64;
  static constexpr unsigned char IMU_ORIENTATION = 128;

  static constexpr double kNanoToSec = 1000000000.0;

 private:
  const FileIO file_io_;
  const DataLayout layout_;

  //data from IMU
  std::vector<double> timestamp_;
  std::vector<Eigen::Vector3d> gyrocope_;
  std::vector<Eigen::Vector3d> accelerometer_;
  std::vector<Eigen::Vector3d> linear_acceleration_;
  std::vector<Eigen::Vector3d> gravity_;
  std::vector<Eigen::Vector3d> magnet_;
  std::vector<Eigen::Quaterniond> rotation_vector_;

  // data from tango
  std::vector<Eigen::Quaterniond> orientation_;
  std::vector<Eigen::Vector3d> position_;
};

// Write a trajecotry to PLY file. If "only_xy" is set to true, the z axis of the trajectory is set to 0.
// In addition to trajectory, local axes can be drawn. Namely, axes with length "axis_length" represented
// by "kpoints" points will be drawn every "interval" frames on the trajectory.
void WriteToPly(const std::string &path, const double *ts, const Eigen::Vector3d *position,
                const Eigen::Quaterniond *orientation, const int N, const bool only_xy = false,
                const Eigen::Vector3d traj_color = Eigen::Vector3d(0, 255, 255),
                const double axis_length = 0.5, const int kpoints = 100, const int interval = 200);

} //namespace ridi
#endif //PROJECT_IMU_DATASET_H
