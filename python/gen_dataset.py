#pylint: disable=C0103,C0111,C0301

import sys
import os
import subprocess
import argparse
import numpy as np
import scipy.interpolate
import quaternion
import quaternion.quaternion_time_series
import pandas

from write_trajectory_to_ply import write_ply_to_file


def interpolate_quaternion_linear(quat_data, input_timestamp, output_timestamp):
    """
    This function interpolate the input quaternion array into another time stemp.
    
    Args:
        quat_data: Nx4 array containing N quaternions.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mx4 array containing M quaternions.
    """
    n_input = quat_data.shape[0]
    assert input_timestamp.shape[0] == n_input
    assert quat_data.shape[1] == 4
    n_output = output_timestamp.shape[0]

    quat_inter = np.zeros([n_output, 4])
    ptr1 = 0
    ptr2 = 0
    for i in range(n_output):
        if ptr1 >= n_input - 1 or ptr2 >= n_input:
            raise ValueError("")
        # Forward to the correct interval
        while input_timestamp[ptr1 + 1] < output_timestamp[i]:
            ptr1 += 1
            if ptr1 == n_input - 1:
                break
        while input_timestamp[ptr2] < output_timestamp[i]:
            ptr2 += 1
            if ptr2 == n_input:
                break
        q1 = quaternion.quaternion(*quat_data[ptr1])
        q2 = quaternion.quaternion(*quat_data[ptr2])
        quat_inter[i] = quaternion.as_float_array(quaternion.quaternion_time_series.slerp(q1, q2, input_timestamp[ptr1],
                                                                                          input_timestamp[ptr2],
                                                                                          output_timestamp[i]))
    return quat_inter


def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.
    
    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default=None, help='Path to a list file.')
    parser.add_argument('--path', type=str, default=None, help='Path to a dataset folder.')
    parser.add_argument('--skip_front', type=int, default=800, help='Number of discarded records at beginning.')
    parser.add_argument('--skip_end', type=int, default=800, help='Numbef of discarded records at end')
    parser.add_argument('--recompute', action='store_true',
                        help='When set, the previously computed results will be over-written.')
    parser.add_argument('--no_trajectory', action='store_true',
                        help='When set, no ply files will be written.')
    parser.add_argument('--no_magnet', action='store_true',
                        help='If set to true, Magnetometer data will not be processed. This is to deal with'
                             'occasion magnet data corruption.')
    parser.add_argument('--no_remove_duplicate', action='store_true')
    parser.add_argument('--clear_result', action='store_true')

    args = parser.parse_args()

    dataset_list = []
    root_dir = ''
    if args.path:
        dataset_list.append(args.path)
    elif args.list:
        root_dir = os.path.dirname(args.list) + '/'
        with open(args.list) as f:
            for s in f.readlines():
                if s[0] is not '#':
                    dataset_list.append(s.strip('\n'))
    else:
        raise ValueError('No data specified')

    print(dataset_list)

    nano_to_sec = 1000000000.0
    total_length = 0.0
    length_dict = {}
    for dataset in dataset_list:
        if len(dataset.strip()) == 0:
            continue
        if dataset[0] == '#':
            continue
        info = dataset.split(',')
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]
        data_root = root_dir + info[0]
        length = 0
        if os.path.exists(data_root + '/processed/data.csv') and not args.recompute:
            data_pandas = pandas.read_csv(data_root + '/processed/data.csv')
        else:
            print('------------------\nProcessing ' + data_root, ', type: ' + motion_type)
            # drop the head and tail
            pose_data = np.genfromtxt(data_root+'/pose.txt')[args.skip_front:-args.skip_end, :]

            # swap tango's orientation from [x,y,z,w] to [w,x,y,z]
            pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]
            # For some reason there might be a few duplicated records...
            if not args.no_remove_duplicate:
                unique_ts, unique_inds = np.unique(pose_data[:, 0], return_index=True)
                print('Portion of unique records: ', unique_inds.shape[0] / pose_data.shape[0])
                pose_data = pose_data[unique_inds, :]
            output_timestamp = pose_data[:, 0]
            output_samplerate = output_timestamp.shape[0] * nano_to_sec / (output_timestamp[-1] - output_timestamp[0])
            assert 195 < output_samplerate < 205, 'Wrong output sample rate: %f' % output_samplerate

            print('Pose sample rate: {:2f}Hz'.format(output_samplerate))

            acce_data = np.genfromtxt(data_root+'/acce.txt')
            print('Acceleration found. Sample rate:{:2f} Hz'
                  .format((acce_data.shape[0] - 1.0) * nano_to_sec / (acce_data[-1, 0] - acce_data[0, 0])))
            gyro_data = np.genfromtxt(data_root+'/gyro.txt')
            print('Gyroscope found. Sample rate:{:2f} Hz'
                  .format((gyro_data.shape[0] - 1.0) * nano_to_sec / (gyro_data[-1, 0] - gyro_data[0, 0])))
            linacce_data = np.genfromtxt(data_root+'/linacce.txt')
            print('Linear acceleration found. Sample rate:{:2f} Hz'
                  .format((linacce_data.shape[0] - 1.0) * nano_to_sec / (linacce_data[-1, 0] - linacce_data[0, 0])))
            gravity_data = np.genfromtxt(data_root+'/gravity.txt')
            print('Gravity found. Sample rate:{:2f} Hz'
                  .format((gravity_data.shape[0] - 1.0) * nano_to_sec / (gravity_data[-1, 0] - gravity_data[0, 0])))

            magnet_data = np.genfromtxt(data_root + '/magnet.txt')
            print('Magnetometer: {:.2f}Hz'.
                  format((magnet_data.shape[0] - 1.0) * nano_to_sec / (magnet_data[-1, 0] - magnet_data[0, 0])))

            orientation_data = np.genfromtxt(data_root + '/orientation.txt')
            print('Orientation found. Sample rate:{:2f}'
                  .format((orientation_data.shape[0] - 1.0) * nano_to_sec /
                          (orientation_data[-1, 0] - orientation_data[0, 0])))

            output_folder = data_root + '/processed'
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            # Generate dataset

            # output_gyro_linear = interpolateAngularRateLinear(gyro_data, output_timestamp)
            output_gyro_linear = interpolate_3dvector_linear(gyro_data[:, 1:], gyro_data[:, 0], output_timestamp)
            output_accelerometer_linear = interpolate_3dvector_linear(acce_data[:, 1:], acce_data[:, 0],
                                                                      output_timestamp)
            output_linacce_linear = interpolate_3dvector_linear(linacce_data[:, 1:], linacce_data[:, 0],
                                                                output_timestamp)
            output_gravity_linear = interpolate_3dvector_linear(gravity_data[:, 1:], gravity_data[:, 0],
                                                                output_timestamp)
            output_magnet_linear = np.zeros([output_timestamp.shape[0], 3])
            if not args.no_magnet:
                output_magnet_linear = interpolate_3dvector_linear(magnet_data[:, 1:], magnet_data[:, 0],
                                                                   output_timestamp)
            # swap from x,y,z,w to w,x,y,z
            orientation_data[:, [1, 2, 3, 4]] = orientation_data[:, [4, 1, 2, 3]]
            # Convert rotation vector to quaternion
            output_orientation = interpolate_quaternion_linear(orientation_data[:, 1:], orientation_data[:, 0],
                                                               output_timestamp)

            # construct a Pandas DataFrame
            column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                          'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') + \
                          'magnet_x,magnet_y,magnet_z'.split(',') + \
                          'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z'.split(',')
            data_mat = np.concatenate([output_timestamp[:, None], output_gyro_linear,
                                       output_accelerometer_linear,
                                       output_linacce_linear,
                                       output_gravity_linear,
                                       output_magnet_linear,
                                       pose_data[:, 1:4],
                                       pose_data[:, -4:],
                                       output_orientation], axis=1)

            data_pandas = pandas.DataFrame(data_mat, columns=column_list)
            data_pandas.to_csv(output_folder + '/data.csv')
            print('Dataset written to ' + output_folder + '/data.txt')

            # write data in plain text file for C++
            with open(output_folder + '/data_plain.txt', 'w') as f:
                f.write('{} {}\n'.format(data_mat.shape[0], data_mat.shape[1]))
                for i in range(data_mat.shape[0]):
                    for j in range(data_mat.shape[1]):
                        f.write('{}\t'.format(data_mat[i][j]))
                    f.write('\n')

            if not args.no_trajectory:
                print("Writing trajectory to ply file")
                viewing_dir = np.zeros([data_mat.shape[0], 3], dtype=float)
                viewing_dir[:, 2] = -1.0
                write_ply_to_file(path=output_folder + '/trajectory.ply', position=pose_data[:, 1:4],
                                  orientation=pose_data[:, -4:])

        length = (data_pandas['time'].values[-1] - data_pandas['time'].values[0]) / nano_to_sec
        hertz = data_pandas.shape[0] / length
        print(info[0] + ', length: {:.2f}s, sample rate: {:.2f}Hz'.format(length, hertz))
        if motion_type not in length_dict:
            length_dict[motion_type] = length
        else:
            length_dict[motion_type] += length
        total_length += length
        if args.clear_result:
            command = 'rm -r %s/result*' % (data_root)
            subprocess.call(command, shell=True)

    print('All done. Total length: {:.2f}s ({:.2f}min)'.format(total_length, total_length / 60.0))
    for k, v in length_dict.items():
        print(k + ': {:.2f}s ({:.2f}min)'.format(v, v / 60.0))
