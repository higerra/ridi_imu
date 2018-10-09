# RIDI: Robust IMU Double Integration
## Prerequisite
  * Python3: numpy, scipy, opencv-python (>3.0), numpy-quaternion, plyfile
  * C++: Glog, Gflags, OpenCV (>3.0), Eigen, Ceres-Solver, OpenMesh

## Installing the dependencies
### C++ Dependencies

1. *Installing Glog, Gflags, CMake, BLAS & LAPACK*

Referring to the instructions stated on [Ceres-Solver Documentation](http://ceres-solver.org/installation.html) the following commands need to be executed to install the dependencies for Ceres: 
```
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# SuiteSparse and CXSparse (optional)
# - If you want to build Ceres as a *static* library (the default)
#   you can use the SuiteSparse package in the main Ubuntu package
#   repository:
sudo apt-get install libsuitesparse-dev
# - However, if you want to build Ceres as a *shared* library, you must
#   add the following PPA:
sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
sudo apt-get update
sudo apt-get install libsuitesparse-dev
```
*NOTE*: We cannot use `sudo apt-get install libeigen3-dev` to download eigen3 as it installs Eigen 3.2.92 As we need a version > 3.3.0, we need to manually install and compile Eigen3!

2. **Installing Eigen3:** You need to check which version of eigen3 is installed in your system. To check the version you can use `pkg-config --modversion eigen3`

If the returned value is more than or equal to 3.3.0 then you are through. Else, you need to find where eigen is located and delete it, if present. It isn't necessary, however, to avoid version conflicts we might want it.  

For systems involving to many workspaces and virtaul environments we can use `sudo apt-get remove libeigen3-dev`

After removing the current versions of eigen in your system, you need to download the latest stable release of Eigen3 from [here](http://eigen.tuxfamily.org/index.php?title=Main_Page) and follow the following commands: 

```
tar xvf eigen-eigen-5a0156e40feb.tar.gz
cd eigen-eigen-5a0156e40feb
mkdir build
cd build 
cmake ..
make all
make install
```
Once through check your eigen version using `pkg-config --modversion eigen3`
Congratulations! You have successfully installed eigen3 in your system!

3. **Installing Ceres-Solver**: Run the following commands to build Ceres-Solver: 

```
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver 
mkdir ceres-bin
cd ceres-bin
cmake ..
make all
sudo make install #using sudo as you might need permission to write files!
```
That installs ceres solver in our system! 

4.**Installing OpenCV(> 3)**
Download a stable release of OpenCV from [here](https://opencv.org/releases.html). Once through, extract the files and then:
```
cd opencv-3.3
mkdir build
cd build
cmake ..
make all
make install
```
It's better to use `make install`, as to remove the it you could directly use `make uninstall`

5. **Installing OpenMesh**
Download the latest stable release of OpenMesh from [here](https://www.openmesh.org/download/). After extracting the files use the following commands: 

```
cd OpenMesh-7.1
mkdir build
cd build
cmake ..
make all
make install
```
Most likely, OpenMesh will install with the above commands. However, there is a possibility that you might have to use the flag variables that have been described in the [documentation](https://www.openmesh.org/media/Documentations/OpenMesh-Doc-Latest/a03923.html) judiciously to make it work.

One standard problem is to link c++11 explicitly, in that case you might need to specify c++ version in the cmake 
```
cmake .. -DCMAKE_CXX_FLAGS=-std=c++11
```

*Please do refer to the flags that can be passed to cmake very judiciously in case you are not able to build OpenMesh!* 

### Python3
1. **Install Anaconda3**: We would recommend that you use Anaconda, and hence the guide goes along assuming you have Anaconda installed in your system. You can get the latest version of Anaconda supporting python3 [here](https://www.anaconda.com/download/#linux).

2. As you install the above you already have the latest version of Numpy and Scipy, there within it. So next we need to install **plyfile** and **quaternion**.
```
pip install plyfile 
conda install -c conda-forge quaternion 
```

3. **Installing opencv3**
```
conda install -c menpo opencv3
```

Now as your system is ready with all the dependencies to run the code refer to the **Data Format** and **Usage** to finally make it work!

## Data format
The dataset used by this project is collected by a specific Tango App:
[https://github.com/higerra/TangoIMURecorder](https://github.com/higerra/TangoIMURecorder)
Please read the README.md file of this repository for detailed file format.
If you do not have a Tango phone, you can use the alternative app:
[https://github.com/higerra/AndroidIMURecorder](https://github.com/higerra/AndroidIMURecorder)
Note that without a Tango phone, the ground truth trajectory will not be available.

## Usage:

  1. Clone the repository.
  2. (Optional) Download the dataset from [HERE](https://wustl.box.com/s/6lzfkaw00w76f8dmu0axax7441xcrzd9) and the pre-trained model from [HERE](https://wustl.box.com/s/fsjta6399idcb9lmd6maf4e215wxbp6i). Note that the pre-trained model is trained from a small group of people. For the best result please train your own model.
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

