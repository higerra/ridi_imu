//
// Created by Hang Yan on 10/1/17.
//

// This file is for debugging purpose.

#include <fstream>
#include <vector>
#include <memory>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "speed_regression/feature_target.h"
#include "speed_regression/model_wrapper.h"
#include "utility/data_io.h"

DEFINE_string(model_path, "", "Path to the model");
DEFINE_string(data_path, "", "Path to the csv file containing the data");
DEFINE_string(output_path, "", "Path to the output file. If empty, the output file will be named regressed.txt"
    " and saved to the data folder");

using ridi::ModelWrapper;
using ridi::SVRCascade;
using ridi::TrainingDataOption;
using ridi::IMUDataset;
using cv::Mat;

int main(int argc, char** argv){
  if (argc < 3){
    std::cerr << "Usage: ./SpeedRegression_cli <path-to-data> <path-to-model> [<output-path>]" << std::endl;
    return 1;
  }
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  TrainingDataOption td_option;
  td_option.feature_smooth_sigma = 2.0;
  td_option.target_smooth_sigma = 30.0;
  td_option.feature_type = ridi::DIRECT_GRAVITY_ALIGNED;
  td_option.target_type = ridi::LOCAL_SPEED_GRAVITY_ALIGNED;
  
  IMUDataset data(FLAGS_data_path);
  const std::vector<double>& ts = data.GetTimeStamp();
  printf("Number of records: %d\n", (int)ts.size());

  Mat feature;
  printf("Compute features...\n");
  ridi::CreateFeatureMat(td_option, data, &feature);
  printf("Num samples: %d\n", feature.rows);
  std::unique_ptr<ModelWrapper> model(new SVRCascade());
  CHECK(model->LoadFromFile(FLAGS_model_path)) << "Load model failed: " << FLAGS_model_path;
  auto model_cast = dynamic_cast<SVRCascade*>(model.get());
  auto classifier = model_cast->GetClassifier();
  printf("Number of SV in the classifier: %d\n", classifier->getSupportVectors().rows);

  for (int i=0; i<10; ++i){
    for (int j=0; j<10; ++j){
      printf("%.6f\t", feature.at<float>(i, j));
    }
  }
  printf("\n");
  
  for (int i=0; i<10; ++i){
    int label;
    Eigen::VectorXd response(2);
    model_cast->Predict(feature.row(i), &response, &label);
    printf("%d: %d\t%f\t%f\n", i, label, response[0], response[1]);
  }
  return 0;
}
