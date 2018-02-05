cmake_minimum_required(VERSION 3.0)

project(IMUUtility)

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(stlplus3)
add_library(IMUUtility utility.h utility.cc data_io.h data_io.cc)
target_link_libraries(IMUUtility ${OpenCV_LIBS} ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${OPENMESH_LIBRARIES})