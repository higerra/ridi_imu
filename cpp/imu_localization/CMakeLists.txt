cmake_minimum_required(VERSION 3.0)

project(IMULocalization)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(IMULocalization_cli imu_localization.h imu_localization.cc imu_localization_cli.cc)
target_link_libraries(IMULocalization_cli ridi_utility ridi_regression IMUOptimization stlplus3)
target_link_libraries(IMULocalization_cli ${OpenCV_LIBS} ${GLOG_LIBRARY} ${GFLAGS_LIBRARY})
