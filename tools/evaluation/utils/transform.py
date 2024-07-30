import math

from pyproj import Proj
import numpy as np


def rotation_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll


def rot_translate_combine(rotation, translation):
    assert rotation.shape == (3, 3)
    assert translation.shape == (3, 1)
    return np.vstack([np.hstack([rotation, translation]), [0, 0, 0, 1]])


def transform_matrix(rotation, translation):
    yaw, _, _ = quaternion_to_euler(rotation["qw"], rotation["qx"], rotation["qy"], rotation["qz"])
    rotation = rotation_matrix(yaw)
    translation = np.array([translation["x"], translation["y"], translation["z"]], dtype=np.float64).reshape((3, 1))
    return rot_translate_combine(rotation, translation)

def transform_matrix_(rotation, translation):
    """
    Args:
        rotation: quaternion in order of (w, x, y, z)
        translation: translation in order of (x, y z)
    Returns:
        (4, 4) transformation matrix
    """
    yaw, _, _ = quaternion_to_euler(*rotation)
    rotation = rotation_matrix(yaw)
    translation = np.array([*translation], dtype=np.float64).reshape((3, 1))
    return rot_translate_combine(rotation, translation)



def scalar_transform(point, transform):
    assert transform.shape == (4, 4)
    if len(point) == 2:
        point = [*point, 0, 1]
        dst_point = transform.dot(point)
        return dst_point[:2]
    elif len(point) == 3:
        point = [*point, 1]
        dst_point = transform.dot(point)
        return dst_point[:3]
    else:
        raise AssertionError("dimension of point should be 2 or 3, got {} instead".format(len(point)))


def vector_transform(velocity, transform):
    # assert len(velocity) == 2
    if len(velocity) == 2:
        velocity = [*velocity, 0]
    if transform.shape == (4, 4):
        transform = transform[:3, :3]
    assert transform.shape == (3, 3)
    dst_velocity = transform.dot(velocity)
    return dst_velocity[:2]


ll2utm_converter = Proj(proj='utm', zone=51, ellps='WGS84')


def ll2utm(lat, lon):
    return ll2utm_converter(lon, lat)
