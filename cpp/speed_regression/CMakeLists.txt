cmake_minimum_required(VERSION 3.0)

project (SpeedRegression)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS} ${GLOG_INCLUDE_DIR})

add_library(ridi_regression feature_target.h feature_target.cc
        model_wrapper.h model_wrapper.cc speed_regression_cli.cc)
target_link_libraries(ridi_regression ${OpenCV_LIBS} ${GLOG_LIBRARY})

add_executable(SpeedRegression_cli speed_regression_cli.cc)
target_link_libraries(SpeedRegression_cli ridi_regression ${GFLAGS_LIBRARY} ridi_utility)
