//
// Created by Hang Yan.
//

#ifndef CODE_SPEED_REGRESSION_MODEL_WRAPPER_H
#define CODE_SPEED_REGRESSION_MODEL_WRAPPER_H

#include <string>
#include <glog/logging.h>
#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

namespace ridi{

// This class provide unified predicting interface for regression model. The training of these models are done through
// python codes. See code/python/speed_regression for details.
class ModelWrapper{
 public:
  virtual bool LoadFromFile(const std::string& path) = 0;
  virtual void Predict(const cv::Mat& feature, Eigen::VectorXd* response) const = 0;
  virtual void Predict(const cv::Mat& feature, Eigen::VectorXd* response, int* label) const = 0;
};

struct SVRCascadeOption{
  int num_classes = 0;
  int num_channels = 0;
  const static std::string kVersionTag;
};


// Read SVRCascadeOption from a stream object.
std::istream& operator >> (std::istream& stream, SVRCascadeOption& option);

// This class defines the cascading model. The model consists of a classifier and multiple regressors. A sample is
// firstly passed to the classifier to obtain a placement type. It then passed to corresponding two regressors (one
// for each of the horizontal channel) based on the label.
class SVRCascade: public ModelWrapper{
 public:
  SVRCascade() = default;
  explicit SVRCascade(const std::string& path){
    CHECK(LoadFromFile(path)) << "Can not load SVRCascade model from " << path;
  }
  bool LoadFromFile(const std::string& path) override;
  void Predict(const cv::Mat& feature, Eigen::VectorXd* response) const override;
  void Predict(const cv::Mat& feature, Eigen::VectorXd* response, int* label) const override ;

  inline int GetNumClasses() const{
    return option_.num_classes;
  }

  inline int GetNumChannels() const{
    return option_.num_channels;
  }

  inline const cv::ml::SVM* GetClassifier() const{
    return classifier_.get();
  }

  inline const cv::ml::SVM* GetRegressor(int id) const {
    if (id > regressors_.size()){
      LOG(ERROR) << "Regressor index out of bound.";
      return nullptr;
    }
    return regressors_[id].get();
  }

  inline const std::vector<cv::Ptr<cv::ml::SVM>>& GetRegressors() const{
    return regressors_;
  }
  SVRCascade(const SVRCascade& model) = delete;
  bool operator = (const SVRCascade& model) = delete;
 private:
  SVRCascadeOption option_;

  std::vector<cv::Ptr<cv::ml::SVM>> regressors_;
  cv::Ptr<cv::ml::SVM> classifier_;
  std::vector<std::string> class_names_;
};


}  // namespace ridi

#endif // CODE_SPEED_REGRESSION_MODEL_WRAPPER_H
