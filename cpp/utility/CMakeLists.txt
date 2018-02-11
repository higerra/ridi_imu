cmake_minimum_required(VERSION 3.0)

project(ridi_utility)

find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(stlplus3)
add_library(ridi_utility data_io.h data_io.cc)
target_link_libraries(ridi_utility ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${OPENMESH_LIBRARIES})
