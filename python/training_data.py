import numpy as np
import quaternion
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter1d

import geometry


class TrainingDataOption:
    def __init__(self, sample_step=10, window_size=200, feature='direct_gravity', target='local_speed_gravity'):
        # Feature vectors and targets will be computed once $self.sample_step_ frames.
        self.sample_step_ = sample_step
        # The window size used for constructing the feature vector.
        self.window_size_ = window_size
        # Feature type, choices are 'direct', 'direct_gravity' and 'fourier'
        self.feature_ = feature
        # Target type, choices are 'local_speed', 'local_speed_gravity'
        self.target_ = target
        self.nanoToSec = 1000000000.0


def compute_fourier_features(data, samples, window_size, threshold, discard_direct=False):
    """
    Compute fourier coefficients as feature vector. Not used.
    
    :param data: NxM array for N samples with M dimensions
    :return: Nxk array
    """
    skip = 0
    if discard_direct:
        skip = 1
    features = np.empty([samples.shape[0], data.shape[1] * (threshold - skip)], dtype=np.float)
    for i in range(samples.shape[0]):
        features[i, :] = np.abs(fft(data[samples[i]-window_size:samples[i]], axis=0)[skip:threshold]).flatten()
    return features


def compute_direct_features(data, samples_points, window_size, sigma=-1):
    """
    Construct feature vectors by concatenating source channels.
    
    :param data: NxM array. Each row contains all information of a frame.
    :param samples_points: Indices of frames where feature vectors are constructed.
    :param window_size: When constructing the feature vector at frame i, information between (i-window_size, i]
                        is used.
    :param sigma: When set to positive value, the data matrix will be filtered along the first dimension before
                  concatenation.
    :return: A matrix containing feature vectors.
    """
    features = np.empty([samples_points.shape[0], data.shape[1] * window_size], dtype=np.float)
    for i in range(samples_points.shape[0]):
        data_slice = data[samples_points[i] - window_size:samples_points[i]]
        if sigma > 0:
            data_slice = gaussian_filter1d(data_slice, sigma, axis=0)
        features[i, :] = data_slice.flatten()
    return features


def compute_direct_feature_gravity(gyro, linacce, gravity, samples, window_size, sigma=-1):
    """
    Construct feature vectors by concatenating angular rates and linear accelerations in stabilized IMU frame.
    
    :param gyro: Nx3 array. Angular rates.
    :param linacce: Nx3 array. Linear accelerations.
    :param gravity: Nx3 array. Gravity vectors in local device frame.
    :param samples: Indices where feature vectors are constructed.
    :param window_size: Number of frames used when constructing feature vectors.
    :param sigma: Sigma used for pre-filtering the signal before concatenating.
    :return: A matrix containing feature vectors.
    """
    gyro_gravity = geometry.align_3dvector_with_gravity(gyro, gravity)
    linacce_gravity = geometry.align_3dvector_with_gravity(linacce, gravity)
    return compute_direct_features(np.concatenate([gyro_gravity, linacce_gravity], axis=1), samples_points=samples,
                                   window_size=window_size, sigma=sigma)


def compute_speed(time_stamp, position, sample_points=None):
    """
    Compute speed vectors in the global frame giving position and time_stamp.
    
    :param time_stamp: N array. Time stamps of each frame.
    :param position: Nx3 array. Positions of each frame.
    :param sample_points: M array. Indices where feature vectors are constructed.
    :return: Mx3 array, each contraining a speed vector at sampled frames.
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    speed = (position[sample_points+1] - position[sample_points]) / (time_stamp[sample_points+1] -
                                                                     time_stamp[sample_points])[:, None]
    return speed


def compute_local_speed(time_stamp, position, orientation, sample_points=None):
    """
    Compute the speed in local (IMU) frame.
    
    :param time_stamp: Nx1 array containing time stamps for each frame.
    :param position: Nx3 array of positions
    :param orientation: Nx4 array of orientations as quaternion
    :param sample_points: Mx1 integer array indicating where feature vectors should be computed.
    :return: Mx3 array containin speed vectors in the IMU frame.
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    speed = compute_speed(time_stamp, position, sample_points)
    for i in range(speed.shape[0]):
        q = quaternion.quaternion(*orientation[sample_points[i]])
        speed[i] = (q.conj() * quaternion.quaternion(1.0, *speed[i]) * q).vec
    return speed


def compute_local_speed_with_gravity(time_stamp, position, orientation, gravity,
                                     sample_points=None, local_gravity=np.array([0., 1., 0.])):
    """
    Compute the speed vector in the stabilized IMU frame. That is, remove the pitch and roll by the gravity vector.
    :param time_stamp: Nx1 array containing time stamps for each frame.
    :param position: Nx3 array containing positions.
    :param orientation:  Nx4 array containing orientations as quaternions.
    :param gravity:  Nx3 vector containing gravity vectors.
    :param sample_points: Mx1 integer array indicating where feature vectors should be computed.
    :param local_gravity: The vector in the IMU frame to which the gravity vector should be aligned.
    :return: Mx3 array containing speed vectors in stabilized-IMU frame.
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    local_speed = compute_local_speed(time_stamp, position, orientation, sample_points)
    return geometry.align_3dvector_with_gravity(local_speed, gravity[sample_points], local_gravity)


def compute_delta_angle(time_stamp, position, orientation, sample_points=None,
                        local_axis=quaternion.quaternion(1.0, 0., 0., -1.)):
    """
    Compute the cosine between the moving direction and viewing direction. Not used.
    
    :param time_stamp: Time stamp
    :param position: Position. When passing Nx2 array, compute ignore z direction
    :param orientation: Orientation as quaternion
    :param local_axis: the viewing direction in the device frame. Default is set w.r.t. to android coord frame
    :return:
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    epsilon = 1e-10
    speed_dir = compute_speed(time_stamp, position)
    speed_dir = np.concatenate([np.zeros([1, position.shape[1]]), speed_dir], axis=0)
    speed_mag = np.linalg.norm(speed_dir, axis=1)
    cos_array = np.zeros(sample_points.shape[0], dtype=float)
    valid_array = np.empty(sample_points.shape[0], dtype=bool)
    for i in range(sample_points.shape[0]):
        if speed_mag[sample_points[i]] <= epsilon:
            valid_array[i] = False
        else:
            q = quaternion.quaternion(*orientation[sample_points[i]])
            camera_axis = (q * local_axis * q.conj()).vec[:position.shape[1]]
            cos_array[i] = min(np.dot(speed_dir[sample_points[i]], camera_axis) / speed_mag[sample_points[i]], 1.0)
            valid_array[i] = True
    return cos_array, valid_array


def get_training_data(data_all, option, sample_points=None, extra_args=None):
    """
    Create training data.
    
    :param data_all: The whole dataset. Must include 'time' column and all columns inside imu_columns
    :param imu_columns: Columns used for constructing feature vectors. Fields must exist in the dataset
    :param option: An instance of TrainingDataOption
    :param sample_points: an array of locations where the data sample is computed. If not provided, samples are
           uniformly distributed based on option.sample_step_
    :param extra_args: Extra arguments.
    :return: [Nx(d+1)] array. Target value is appended at back
    """
    if sample_points is None:
        sample_points = np.arange(option.window_size_,
                                  data_all.shape[0] - 1,
                                  option.sample_step_,
                                  dtype=int)
    assert sample_points[-1] < data_all.shape[0]
    pose_data = data_all[['pos_x', 'pos_y', 'pos_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    data_used = data_all[['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']].values
    time_stamp = data_all['time'].values / 1e09

    targets = None

    if option.target_ == 'speed_magnitude':
        targets = np.linalg.norm(compute_speed(time_stamp, pose_data), axis=1)
    elif option.target_ == 'angle':
        targets, valid_array = compute_delta_angle(time_stamp, pose_data, orientation)
    elif option.target_ == 'local_speed':
        targets = compute_local_speed(time_stamp, pose_data, orientation)
    elif option.target_ == 'local_speed_gravity':
        gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
        targets = compute_local_speed_with_gravity(time_stamp, pose_data, orientation, gravity)

    if extra_args is not None:
        if 'target_smooth_sigma' in extra_args:
            targets = gaussian_filter1d(targets, sigma=extra_args['target_smooth_sigma'], axis=0)

    targets = targets[sample_points]

    gaussian_sigma = -1
    if extra_args is not None:
        if 'feature_smooth_sigma' in extra_args:
            gaussian_sigma = extra_args['feature_smooth_sigma']

    if option.feature_ == 'direct':
        features = compute_direct_features(data_used, sample_points, option.window_size_, gaussian_sigma)
    elif option.feature_ == 'fourier':
        print('Additional parameters: ', extra_args)
        features = compute_fourier_features(data_used, sample_points, option.window_size_, extra_args['frq_threshold'],
                                            extra_args['discard_direct'])
    elif option.feature_ == 'direct_gravity':
        gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
        features = compute_direct_feature_gravity(data_used[:, :3], data_used[:, -3:],
                                                  gravity, sample_points, option.window_size_, gaussian_sigma)
    else:
        print('Feature type not supported: ' + option.feature_)
        raise ValueError

    return features, targets
