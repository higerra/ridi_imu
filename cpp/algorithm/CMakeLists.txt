cmake_minimum_required(VERSION 3.0)

project(IMUAlgorithm)

find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)

include_directories(${EIGEN_INCLUDE_DIR})
add_library(algorithm geometry.cc geometry.h)

target_link_libraries(algorithm ${GLOG_LIBRARIES})
