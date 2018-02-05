cmake_minimum_required(VERSION 3.0)

if(WIN32)
list(APPEND CMAKE_PREFIX_PATH "D:/library")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP QUIET)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

if(OPENMP_FOUND)
    message("-- Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(algorithm)
add_subdirectory(utility)
add_subdirectory(imu_optimization)
add_subdirectory(speed_regression)
add_subdirectory(imu_localization)

# add_subdirectory(viewer)

enable_testing()
