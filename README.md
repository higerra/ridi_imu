# RIDI: Robust IMU Double Integration
## Prerequisite
  * Python: numpy, scipy, opencv-python (>3.0), numpy-quaternion, plyfile
  * C++: Glog, Gflags, OpenCV (>3.0), Eigen, Ceres-Solver, OpenMesh
## Data format
The dataset used by this project is collected by a specific Tango App:
[https://github.com/higerra/TangoIMURecorder](https://github.com/higerra/TangoIMURecorder)
Please read the README.md file of this repository for detailed file format.
If you do not have a Tango phone, you can use the alternative app:
[https://github.com/higerra/AndroidIMURecorder](https://github.com/higerra/AndroidIMURecorder)
Note that without a Tango phone, the ground truth trajectory will not be available.

## Usage:
  1. Clone the repository.
  2. (Optional) Download the dataset from [HERE](https://wustl.box.com/s/ysj1m8qcda92r00etz894vz3sfq7mxtg) and the pre-trained model from [HERE](https://wustl.box.com/s/fsjta6399idcb9lmd6maf4e215wxbp6i). Note that the pre-trained model is trained from a small group of people. For the best result please train your own model.
  3. For newly captured dataset, run ```python/gen_dataset.py```(with Tango phone) or ```python/gen_dataset_nopose.py```(without Tango phone) to preprocess the dataset. Please refer to the source code for command line arguments.
  4. To train a model, run ```python/regression_cascade.py```. Please refer to the source code for command line arguments. One possible call is:
    * ```python/regression_cascade.py --list <path-to-dataset-list> --model_output <path-to-output-folder>```.
  5. Compile the C++ source code:
    * ```cd <project-root>```
    * ```mkdir cpp/build & cd cpp/build```
    * ```cmake ..```
    * ```make -j4```
  6. After the compilation finishes, run the following executable to perform RIDI:
    * ```cd <project-root>/cpp/build/imu_localization```
    * ```./IMULocalization_cli <path-to-dataset> --model_path <path-to-model>```
  7. The result will be written to ```<path-to-dataset>/result_full```
  
  ## Citation
  Please cite the following paper is you use the code:
  
  [Yan, Hang, Qi Shan, and Yasutaka Furukawa. "RIDI: Robust IMU Double Integration." arXiv preprint arXiv:1712.09004 (2017).](https://arxiv.org/abs/1712.09004)
