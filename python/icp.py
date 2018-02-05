import numpy as np


def fit_transformation(source, target):
    """
    This function computes the best rigid transformation between two point sets. It assumes that "source" and
    "target" are with the same length and "source[i]" corresponds to "target[i]".
    
    :param source: Nxd array. 
    :param target: Nxd array.
    :return: A transformation as (d+1)x(d+1) matrix; the rotation part as a dxd matrix and the translation
    part as a dx1 vector.
    """
    assert source.shape == target.shape
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)
    m = source.shape[1]
    source_zeromean = source - center_source
    target_zeromean = target - center_target
    W = np.dot(source_zeromean.T, target_zeromean)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = center_target.T - np.dot(R, center_source.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t