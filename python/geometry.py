import math

import numpy as np
import quaternion


def rotate_vector(input, orientation):
    output = np.empty(input.shape, dtype=float)
    for i in range(input.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        output[i] = (q * quaternion.quaternion(1.0, *input[i]) * q.conj()).vec
    return output


def rotation_matrix_from_two_vectors(v1, v2):
    """
    Using Rodrigues rotation formula
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
    Compute quaternion from two vectors
    :param v1:
    :param v2:
    :return Quaternion representation of rotation between v1 and v2
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    w = np.cross(v1n, v2n)
    q = np.array([1.0 + np.dot(v1n, v2n), *w])
    q /= np.linalg.norm(q)
    return quaternion.quaternion(*q)


def align_3dvector_with_gravity(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Adjust pose such that the gravity is at $target$ direction
    @:param data: N x 3 array
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return
    """
    assert data.ndim == 2, 'Expect 2 dimensional array input'
    assert data.shape[1] == 3, 'Expect Nx3 array input'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])

    # output = np.empty(data.shape, dtype=float)
    output = np.copy(data)
    for i in range(data.shape[0]):
        q = quaternion_from_two_vectors(gravity[i], local_g_direction)
        # TODO(yanhang): Is setting a a slack (0.99) reasonable here?
        if q.w < 0.99:
            output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec
    return output


def adjust_eular_angle(source):
    # convert the euler angle s.t. pitch is in (-pi/2, pi/2), roll and yaw are in (-pi, pi)
    output = np.copy(source)
    if output[0] < -math.pi / 2:
        output[0] += math.pi
        output[1] *= -1
        output[2] += math.pi
    elif output[0] > math.pi / 2:
        output[0] -= math.pi
        output[1] *= -1
        output[2] -= math.pi

    for j in [1, 2]:
        if output[j] < -math.pi:
            output[j] += 2 * math.pi
        if output[j] > math.pi:
            output[j] -= 2 * math.pi
    return output


def align_eular_rotation_with_gravity(data, gravity, local_g_direction=np.array([0, 1, 0])):
    """
    Transform the coordinate frame of orientations such that the gravity is aligned with $local_g_direction
    :param data: input orientation in Eular
    :param gravity:
    :param local_g_direction:
    :return:
    """
    assert data.shape[1] == 3, 'Expect Nx3 array'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])

    # output = np.empty(data.shape, dtype=float)
    output = np.copy(data)

    # be careful of the ambiguity of eular angle representation
    for i in range(data.shape[0]):
        rotor = quaternion_from_two_vectors(gravity[i], local_g_direction)
        if np.linalg.norm(rotor.vec) > 1e-3 and rotor.w < 0.999:
            q = rotor * quaternion.from_euler_angles(*data[i]) * rotor.conj()
            output[i] = adjust_eular_angle(quaternion.as_euler_angles(q))
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
