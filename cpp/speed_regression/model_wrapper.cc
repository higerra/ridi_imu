//
// Created by yanhang on 9/30/17.
//

#include "speed_regression/model_wrapper.h"

#include <fstream>

namespace ridi {

const std::string SVRCascadeOption::kVersionTag = "v1.0";

std::istream &operator>>(std::istream &stream, SVRCascadeOption &option) {
  std::string version_tag;
  stream >> version_tag;
  CHECK_EQ(version_tag, option.kVersionTag) << "The version of the file " << version_tag
                                            << " doesn't match the current version " << option.kVersionTag;
  stream >> option.num_classes >> option.num_channels;
  return stream;
}

bool SVRCascade::LoadFromFile(const std::string &path) {
  std::ifstream option_in(path + "/option.txt");
  if (!option_in.is_open()) {
    LOG(ERROR) << "Can not open option file: " << path + "/option.txt";
    return false;
  }

  // Load the option
  option_in >> option_;

  // Load the class map
  class_names_.resize(GetNumClasses());
  std::ifstream classmap_in(path + "/class_map.txt");
  if (!classmap_in.is_open()){
    LOG(ERROR) << "Can not open class map file " << path + "/class_map.txt";
    return false;
  }
  std::string name;
  int number;
  classmap_in >> number;
  if (number != GetNumClasses()){
    LOG(ERROR) << "The number of classes in the class map file doesn't match the one in the option file: "
        << number << " vs " << GetNumClasses();
    return false;
  }
  for (int i=0; i<GetNumClasses(); ++i){
    classmap_in >> name >> number;
    if (number < 0 || number > GetNumClasses()){
      LOG(ERROR) << "Invalid class number encountered: " << number;
      return false;
    }
    class_names_[number] = name;
  }

  // Load the classifier
  if (GetNumClasses() > 1) {
    classifier_ = cv::ml::SVM::load(path + "/classifier.yaml");
    if (!classifier_.get()) {
      LOG(ERROR) << "Can not read the classifier from " << path + "/classifier.yaml";
      return false;
    }
    LOG(INFO) << "Classifier " << path + "/classifier.yaml loaded";
  }

  regressors_.resize(GetNumChannels() * GetNumClasses());
  char buffer[128] = {};
  for (int cls = 0; cls < GetNumClasses(); ++cls) {
    for (int chn = 0; chn < GetNumChannels(); ++chn) {
      int rid = cls * GetNumChannels() + chn;
      sprintf(buffer, "%s/regressor_%d_%d.yaml", path.c_str(), cls, chn);
      regressors_[rid] = cv::ml::SVM::load(buffer);
      if (!regressors_[rid].get()){
        LOG(ERROR) << "Can not load regressor " << buffer;
        return false;
      }
      LOG(INFO) << "Regressor " << rid << ':' << buffer << " loaded";
    }
  }
  return true;
}

void SVRCascade::Predict(const cv::Mat &feature, Eigen::VectorXd* response) const {
  int label;
  return Predict(feature, response, &label);
}

void SVRCascade::Predict(const cv::Mat &feature, Eigen::VectorXd* response, int *label) const {
  // Predict the label
  CHECK(response) << "The provided output response is empty";
  CHECK_EQ(response->rows(), GetNumChannels());
  CHECK(label) << "The output label is empty";
  if (GetNumClasses() > 1) {
    *label = static_cast<int>(CHECK_NOTNULL(classifier_.get())->predict(feature));
  } else {
    *label = 0;
  }
  CHECK_LT(*label, GetNumClasses()) << "The predicted label is unknown: " << *label;

  // Then pass the sample to corresponding regressors.
  for (int chn = 0; chn < GetNumChannels(); ++chn) {
    (*response)[chn] = regressors_[(*label) * GetNumChannels() + chn]->predict(feature);
  }
}


}  // namespace ridi
