import math

import numpy as np
import quaternion


def rotate_vector(input, orientation):
    """
    Rotate 3D vectors with quaternions.
    
    :param input: Nx3 array containing N 3D vectors.
    :param orientation: Nx4 array containing N quaternions.
    :return: Nx3 array containing rotated vectors.
    """
    output = np.empty(input.shape, dtype=float)
    for i in range(input.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        output[i] = (q * quaternion.quaternion(1.0, *input[i]) * q.conj()).vec
    return output


def rotation_matrix_from_two_vectors(v1, v2):
    """
    Compute 3x3 rotation matrix between two vectors using Rodrigues rotation formula. Two vectors need not be
    normalized.
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    
    :param v1: starting vector
    :param v2: ending vector
    :return 3x3 rotation matrix
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    theta = np.dot(v1, v2)
    if theta == 1:
        return np.identity(3)
    if theta == -1:
        raise ValueError
    k = np.cross(v1, v2)
    k /= np.linalg.norm(k)
    K = np.matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.identity(3) + math.sqrt(1 - theta * theta) * K + np.dot((1 - theta) * K * K, v1)


def quaternion_from_two_vectors(v1, v2):
    """
    Compute quaternion from two vectors. v1 and v2 need not be normalized.
    
    :param v1: starting vector
    :param v2: ending vector
    :return Quaternion representation of rotation that rotate v1 to v2.
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    w = np.cross(v1n, v2n)
    q = np.array([1.0 + np.dot(v1n, v2n), *w])
    q /= np.linalg.norm(q)
    return quaternion.quaternion(*q)


def align_3dvector_with_gravity(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Eliminate pitch and roll from a 3D vector by aligning gravity vector to local_g_direction.
    
    @:param data: N x 3 array
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return
    """
    assert data.ndim == 2, 'Expect 2 dimensional array input'
    assert data.shape[1] == 3, 'Expect Nx3 array input'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])

    epsilon = 1e-03
    gravity_normalized = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    output = np.copy(data)
    for i in range(data.shape[0]):
        # Be careful about two singular conditions where gravity[i] and local_g_direction are parallel.
        gd = np.dot(gravity_normalized[i], local_g_direction)
        if gd > 1. - epsilon:
            continue
        if gd < -1. + epsilon:
            # Invert the Y and Z axis
            output[i, 1:] *= -1
            continue
        q = quaternion_from_two_vectors(gravity[i], local_g_direction)
        output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec
    return output


def orientation_from_gravity_and_magnet(grav, magnet,
                                        global_gravity=np.array([0, 0, 1]),
                                        global_north=np.array([0, 1, 0])):
    """
    Give the magnet vector and gravity vector in the local IMU frame, the device
    orientation can be uniquely determined.
    """
    rot_grav = quaternion_from_two_vectors(grav, global_gravity)
    # remove tilting
    magnet_grav = (rot_grav * quaternion.quaternion(1.0, *magnet) * rot_grav.conj()).vec
    magnet_grav[2] = 0.0
    rot_magnet = quaternion_from_two_vectors(magnet_grav, global_north)
    return rot_magnet * rot_grav


# complmentary filter
def correct_gyro_drifting(rv, magnet, gravity, alpha=0.98,
                          min_cos=0.0, global_orientation=None):
    """
    This function implements the complementary filter that reduces the drifting error
    of angular rates from the gyroscope with magnetometer data. In indoor environment
    the magnetic field is highly unstable, thus this function is unused.
    """
    assert rv.shape[0] == magnet.shape[0]
    assert rv.shape[0] == gravity.shape[0]

    rv_quats = []
    for r in rv:
        rv_quats.append(quaternion.quaternion(*r))
    if global_orientation is None:
        global_orientation = rv_quats[0]

    # fake the angular velocity by differentiating the rotation vector
    rv_dif = [quaternion.quaternion(1.0, 0.0, 0.0, 0.0) for _ in range(rv.shape[0])]
    for i in range(1, rv.shape[0]):
        rv_dif[i] = rv_quats[i] * rv_quats[i - 1].inverse()

    # complementary filter
    rv_filtered = [global_orientation for _ in range(rv.shape[0])]
    rv_mag_init_trans = global_orientation * orientation_from_gravity_and_magnet(magnet[0], gravity[0]).inverse()
    fused = [False for _ in range(rv.shape[0])]
    for i in range(1, rv.shape[0]):
        # from gyroscop
        rv_filtered[i] = rv_dif[i] * rv_filtered[i - 1]
        # from magnetometer
        rv_mag = rv_mag_init_trans * orientation_from_gravity_and_magnet(magnet[i], gravity[i])
        diff_angle = rv_filtered[i].inverse() * rv_mag
        # only fuse when the magnetometer is "reasonable"
        if diff_angle.w >= min_cos:
            fused[i] = True
            rv_filtered[i] = quaternion.slerp(rv_filtered[i], rv_mag, 0.0, 1.0, 1.0 - alpha)
    return quaternion.as_float_array(rv_filtered), fused
