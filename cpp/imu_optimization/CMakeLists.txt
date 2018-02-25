cmake_minimum_required(VERSION 3.0)

project(IMUOptimization)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} {OpenCV_INCLUDE_DIRS})

add_library(IMUOptimization imu_optimization.h imu_optimization.cc)
target_link_libraries(IMUOptimization ${CERES_LIBRARIES} ${GLOG_LIBRARY})
