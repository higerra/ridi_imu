//
// Created by yanhang on 3/5/17.
//

#include "imu_localization.h"

#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <gflags/gflags.h>

#include "algorithm/geometry.h"
#include "utility/data_io.h"
#include "utility/utility.h"
#include "utility/stlplus3/file_system.hpp"

DEFINE_string(model_path, "../../../../models/svr_cascade1116_2", "Path to model");
DEFINE_string(mapinfo_path, "default", "path to map info");
DEFINE_int32(log_interval, 1000, "logging interval");
DEFINE_string(color, "blue", "color");
DEFINE_double(weight_vs, 1.0, "The weight parameter for vertical speed. Larger weight_vs imposes more penalty for"
    " the vertical speed not being 0.");
DEFINE_double(weight_ls, 10.0, "The weight parameter for the local speed. Larger weight_ls imposes more penalty for"
    " the integrated local speed being different than the regressed speed.");
DEFINE_string(suffix, "full", "suffix");
DEFINE_string(preset, "none", "preset mode");

DEFINE_bool(register_to_reference_global, true, "If the ground truth trajectory is provided and this term is set"
    " to true, a global transformation will be estimated that aligns the start portion of the estimated trajectory "
    "to the ground truth.");
DEFINE_bool(register_start_portion_2d, true,
            "If set to true, estimate a 2D global transformation that aligns the start portion of the estimated "
                "trajector with the ground truth. Only useful with FLAGS_estimate_global_transformation is true.");
DEFINE_int32(start_portion_length, 2500, "The length (in frames) of the start portion. These frames will be used to "
    "estimate the global transformation from the estimated trajectory to the ground truth. Set to -1 to use the "
    "entire trajectory");

DEFINE_bool(run_global, true, "Run global optimization at the end");
DEFINE_bool(tango_ori, false, "Use ground truth orientation");
using namespace std;

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./IMULocalization_cli <path-to-data>" << endl;
    return 1;
  }

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  char buffer[256] = {};

  LOG(INFO) << "Initializing...";
  // load data
  IMUProject::IMUDataset dataset(argv[1]);


  // Run the system
  const int N = (int) dataset.GetTimeStamp().size();
  const std::vector<double> &ts = dataset.GetTimeStamp();
  const std::vector<Eigen::Vector3d> &gyro = dataset.GetGyro();
  const std::vector<Eigen::Vector3d> &linacce = dataset.GetLinearAcceleration();
  const std::vector<Eigen::Vector3d> &gravity = dataset.GetGravity();
  const std::vector<Eigen::Vector3d> &magnet = dataset.GetMagnet();

  std::vector<Eigen::Quaterniond> orientation;
  if (FLAGS_tango_ori) {
    LOG(WARNING) << "Use Tango orientation";
    orientation = dataset.GetOrientation();
  } else {
    LOG(INFO) << "Use rotation vector";
    orientation = dataset.GetRotationVector();
    Eigen::Quaterniond rotor = IMUProject::OrientationFromMagnet(gravity[0], magnet[0]);
    for (int i=0; i < orientation.size(); ++i){
      orientation[i] = rotor * orientation[i];
    }
  }

  Eigen::Vector3d traj_color(0, 0, 255);
  if (FLAGS_color == "yellow") {
    traj_color = Eigen::Vector3d(128, 128, 0);
  } else if (FLAGS_color == "green") {
    traj_color = Eigen::Vector3d(0, 180, 0);
  } else if (FLAGS_color == "brown") {
    traj_color = Eigen::Vector3d(0, 128, 128);
  }

  // crete trajectory instance
  IMUProject::IMULocalizationOption loc_option;
  loc_option.weight_ls = FLAGS_weight_ls;
  loc_option.weight_vs = FLAGS_weight_vs;
  if (FLAGS_preset == "full") {
    loc_option.reg_option = IMUProject::FULL;
    FLAGS_suffix = "full";
    traj_color = Eigen::Vector3d(0, 0, 255);
  } else if (FLAGS_preset == "ori_only") {
    loc_option.reg_option = IMUProject::ORI;
    FLAGS_suffix = "ori_only";
    traj_color = Eigen::Vector3d(0, 200, 0);
  } else if (FLAGS_preset == "mag_only") {
    loc_option.reg_option = IMUProject::MAG;
    FLAGS_suffix = "mag_only";
    traj_color = Eigen::Vector3d(139, 0, 139);
  } else if (FLAGS_preset == "const") {
    loc_option.reg_option = IMUProject::CONST;
    FLAGS_suffix = "const";
    traj_color = Eigen::Vector3d(150, 150, 0);
  } else if (FLAGS_preset == "raw"){
    FLAGS_suffix = "raw";
    traj_color = Eigen::Vector3d(0, 128, 128);
  }

  // Create output directory
  char result_dir_path[128];
  sprintf(result_dir_path, "%s/result_%s/", argv[1], FLAGS_suffix.c_str());
  if (stlplus::file_exists(result_dir_path)) {
    LOG(ERROR) << "Path " << result_dir_path << " is the name of an existing file.";
    return 1;
  }

  if (!stlplus::folder_exists(result_dir_path)) {
    LOG(INFO) << "Creating folder: " << result_dir_path;
    CHECK(stlplus::folder_create(result_dir_path)) << "Can not create folder " << result_dir_path << " for output.";
  }

  std::vector<Eigen::Vector3d> output_positions;
  std::vector<Eigen::Vector3d> output_speed;
  std::vector<Eigen::Vector3d> output_linacce;
  std::vector<Eigen::Vector3d> output_bias(dataset.GetTimeStamp().size(), Eigen::Vector3d(0, 0, 0));

  if (FLAGS_preset == "raw"){
    // Write trajectory with double integration
    output_positions.resize(dataset.GetTimeStamp().size(), dataset.GetPosition()[0]);
    output_speed.resize(dataset.GetTimeStamp().size(), Eigen::Vector3d(0, 0, 0));
    output_linacce = dataset.GetLinearAcceleration();

    for (auto i = 1; i < output_positions.size(); ++i) {
      Eigen::Vector3d acce = orientation[i - 1] * dataset.GetLinearAcceleration()[i - 1];
      output_speed[i] = output_speed[i - 1] + acce * (ts[i] - ts[i - 1]);
      output_positions[i] = output_positions[i - 1] + output_speed[i - 1] * (ts[i] - ts[i - 1]);
      output_positions[i][2] = 0;
    }
  } else {
    // load regressor
    std::unique_ptr<IMUProject::ModelWrapper> model(new IMUProject::SVRCascade(FLAGS_model_path));
    LOG(INFO) << "Model " << FLAGS_model_path << " loaded";

    IMUProject::TrainingDataOption td_option;
    IMUProject::IMUTrajectory trajectory(loc_option, td_option, Eigen::Vector3d(0, 0, 0),
                                         dataset.GetPosition()[0], model.get());

    float start_t = (float) cv::getTickCount();

    constexpr int init_capacity = 20000;

    printf("Start adding records...\n");
    for (int i = 0; i < N; ++i) {
      trajectory.AddRecord(ts[i], gyro[i], linacce[i], gravity[i], orientation[i]);
      if (i > loc_option.local_opt_window) {
        if (i % loc_option.local_opt_interval == 0) {
          LOG(INFO) << "Running local optimzation at frame " << i;
          while (true) {
            if (trajectory.CanAdd()) {
              break;
            }
          }
          trajectory.ScheduleOptimization(i - loc_option.local_opt_window, loc_option.local_opt_window);
        }
      }
      if (FLAGS_log_interval > 0 && i > 0 && i % FLAGS_log_interval == 0) {
        const float time_passage = std::max(((float) cv::getTickCount() - start_t) / (float) cv::getTickFrequency(),
                                            std::numeric_limits<float>::epsilon());
        sprintf(buffer, "%d records added in %.5fs, fps=%.2fHz\n", i, time_passage, (float) i / time_passage);
        LOG(INFO) << buffer;
      }

    }
    trajectory.EndTrajectory();
    if (FLAGS_run_global) {
      printf("Running global optimization on the whole sequence...\n");
      trajectory.RunOptimization(0, trajectory.GetNumFrames());
    }
    printf("All done. Number of points on trajectory: %d\n", trajectory.GetNumFrames());
    const auto total_time = ((float) cv::getTickCount() - start_t) / (float) cv::getTickFrequency();
    const float fps_all = (float) trajectory.GetNumFrames() / total_time;
    printf("Time usage: %.3fs. Overall framerate: %.3f\n", total_time, fps_all);
    const std::vector<float> &time_regression = trajectory.GetRegressionTimes();
    const std::vector<float> &time_optimization = trajectory.GetOptimizationTimes();
    printf("%d regressions executed; %d optimization executed.\n", static_cast<int>(time_regression.size()),
           static_cast<int>(time_optimization.size()));
    printf("Average time for regression: %.3f, average time for optimization: %.3f\n",
           std::accumulate(time_regression.begin(), time_regression.end(), 0.0f) / time_regression.size(),
           std::accumulate(time_optimization.begin(), time_optimization.end(), 0.0f) / time_optimization.size());

    output_positions = trajectory.GetPositions();
    output_speed = trajectory.GetSpeed();
    output_linacce = trajectory.GetLinearAcceleration();

    sprintf(buffer, "%s/regression_%s.txt", result_dir_path, FLAGS_suffix.c_str());
    ofstream reg_out(buffer);
    const std::vector<int> &cids = trajectory.GetConstraintInd();
    const std::vector<Eigen::Vector3d> &lss = trajectory.GetLocalSpeed();
    const std::vector<int>& labels = trajectory.GetLabels();
    const std::vector<int>& trans_counts = trajectory.GetTransitionCounts();
    for (auto i = 0; i < cids.size(); ++i) {
      reg_out << cids[i] << ' ' << lss[i][0] << ' ' << lss[i][1] << ' ' << lss[i][2] << ' ' << labels[i] << ' '
              << trans_counts[i] << endl;
    }
  }

  // Register with the ground truth (if applicable).
  std::vector<Eigen::Quaterniond> output_orientation = orientation;
  const std::vector<Eigen::Vector3d> &gt_positions = dataset.GetPosition();
  Eigen::Vector3d sum_gt_position = std::accumulate(gt_positions.begin(), gt_positions.end(),
						    Eigen::Vector3d(0, 0, 0));
  bool is_gt_valid = sum_gt_position.norm() > std::numeric_limits<double>::epsilon();
  if (FLAGS_register_to_reference_global) {
    printf("Estimating global transformation\n");
    Eigen::Matrix4d global_transform;
    Eigen::Matrix3d global_rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d global_translation;
    if(is_gt_valid) {
      IMUProject::EstimateTransformation(output_positions, gt_positions, &global_transform, &global_rotation,
                                         &global_translation);
    }
    for (int i = 0; i < output_positions.size(); ++i) {
      output_positions[i] = global_rotation * (output_positions[i] - gt_positions[0]) + gt_positions[0];
      output_orientation[i] = global_rotation * output_orientation[i];
    }

    if (FLAGS_register_start_portion_2d && is_gt_valid) {
      if (FLAGS_start_portion_length < 0) {
        FLAGS_start_portion_length = static_cast<int>(output_positions.size());
      }
      printf("Registring start portion. Length: %d\n", FLAGS_start_portion_length);
      CHECK_GT(FLAGS_start_portion_length, 3) << "The start portion length must be larger than 3";
      std::vector<Eigen::Vector2d> source;
      std::vector<Eigen::Vector2d> target;
      for (int i = 0; i < FLAGS_start_portion_length; ++i) {
        source.push_back(output_positions[i].block<2, 1>(0, 0));
        target.push_back(gt_positions[i].block<2, 1>(0, 0));
      }
      Eigen::Matrix3d transform_2d;
      Eigen::Matrix2d rotation_2d;
      Eigen::Vector2d translation_2d;
      IMUProject::EstimateTransformation<2>(source, target, &transform_2d, &rotation_2d, &translation_2d);
      Eigen::Matrix3d rotation_2d_as_3d = Eigen::Matrix3d::Identity();
      rotation_2d_as_3d.block<2, 2>(0, 0) = rotation_2d;
      for (int i = 0; i < output_positions.size(); ++i) {
//        Eigen::Vector2d transformed = rotation_2d * (output_positions[i].block<2, 1>(0, 0) - translation_2d)
//            + translation_2d;
        Eigen::Vector2d pt_centered = (output_positions[i] - gt_positions[0]).block<2, 1>(0, 0);
        Eigen::Vector2d transformed = rotation_2d * pt_centered + gt_positions[0].block<2, 1>(0, 0);

        output_positions[i][0] = transformed[0];
        output_positions[i][1] = transformed[1];
        output_orientation[i] = rotation_2d_as_3d * output_orientation[i];
      }
    }


  }

  const int kFrames = output_positions.size();
  sprintf(buffer, "%s/result_trajectory_%s.ply", result_dir_path, FLAGS_suffix.c_str());
  IMUProject::WriteToPly(std::string(buffer), dataset.GetTimeStamp().data(), output_positions.data(),
                         output_orientation.data(), kFrames, false, traj_color, 0);

  sprintf(buffer, "%s/tango_trajectory.ply", result_dir_path);
  IMUProject::WriteToPly(std::string(buffer), dataset.GetTimeStamp().data(), dataset.GetPosition().data(),
                         dataset.GetOrientation().data(), (int) dataset.GetPosition().size(),
                         false, Eigen::Vector3d(255, 0, 0), 0.8, 100, 300);

  {
    // // Write the trajectory and bias as txt
    // sprintf(buffer, "%s/result_%s.csv", result_dir_path, FLAGS_suffix.c_str());
    // ofstream traj_out(buffer);
    // CHECK(traj_out.is_open());
    // traj_out << ",time,pos_x,pos_y,pos_z,speed_x,speed_y,speed_z,bias_x,bias_y,bias_z" << endl;
    // for (auto i = 0; i < kFrames; ++i) {
    //   const Eigen::Vector3d &pos = output_positions[i];
    //   const Eigen::Vector3d &acce = output_linacce[i];
    //   const Eigen::Vector3d &spd = output_speed[i];
    //   sprintf(buffer, "%d,%.9f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
    //           i, dataset.GetTimeStamp()[i], pos[0], pos[1], pos[2], spd[0], spd[1], spd[2],
    //           acce[0] - linacce[i][0], acce[1] - linacce[i][1], acce[2] - linacce[i][2]);
    //   traj_out << buffer;
    // }
  }

  if (FLAGS_mapinfo_path == "default") {
    sprintf(buffer, "%s/map.txt", argv[1]);
  } else {
    sprintf(buffer, "%s", FLAGS_mapinfo_path.c_str());
  }
  ifstream map_in(buffer);
  if (map_in.is_open()) {
    LOG(INFO) << "Found map info file, creating overlay";
    string line;
    map_in >> line;
    sprintf(buffer, "%s/%s", argv[1], line.c_str());

    cv::Mat map_img = cv::imread(buffer);
    CHECK(map_img.data) << "Can not open image file: " << buffer;

    Eigen::Vector2d sp1, sp2;
    Eigen::Vector3d op1(0, 0, 0), op2(0, 0, 0);
    double scale_length;

    map_in >> sp1[0] >> sp1[1] >> sp2[0] >> sp2[1];
    map_in >> scale_length;
    map_in >> op1[0] >> op1[1] >> op2[0] >> op2[1];

    Eigen::Vector2d start_pix(op1[0], op1[1]);

    const double pixel_length = scale_length / (sp2 - sp1).norm();

   IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2 - op1, output_positions,
                                 Eigen::Vector3d(255, 0, 0), map_img);

    IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2 - op1, dataset.GetPosition(),
                                  Eigen::Vector3d(0, 0, 255), map_img);
    sprintf(buffer, "%s/overlay_%s.png", result_dir_path, FLAGS_suffix.c_str());
    cv::imwrite(buffer, map_img);
    printf("Overlay image written to %s\n", buffer);
  }

  return 0;
}
