from functools import reduce
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ['transform_matrix', 'transform_offset', 'convert_points', 'convert_boxes', 'convert_velos']


def quaternion_to_rot_matrix(quaternion: List[float]) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rot_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrix).as_quat()


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.array([0, 0, 0, 1]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix

    @param translation: <np.float32: 3>. Translation in x, y, z.
    @param rotation: Rotation in quaternions (ri rj rk, w).
    @param inverse: Whether to compute inverse transform matrix.
    return: <np.float32: 4, 4>. Transformation matrix.
    """
    vector_T = np.array(translation)
    matrix_R = quaternion_to_rot_matrix(rotation) if rotation.shape == (4,) else rotation

    tm = np.eye(4)
    if inverse:
        rot_inv = matrix_R.T
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(-vector_T)
    else:
        tm[:3, :3] = matrix_R
        tm[:3, 3] = vector_T
    return tm


def transform_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix_inv = np.eye(4)
    matrix_inv[:3, :3] = matrix[:3, :3].T
    matrix_inv[:3, 3] = matrix_inv[:3, :3].dot(-matrix[:3, 3].T)
    return matrix_inv


def transform_offset(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Calculate transform from source pos to reference pos
    """
    return reduce(np.dot, [transform_matrix_inverse(ref), src])


def convert_points(points: np.ndarray, transform: np.ndarray, new_array=True) -> np.ndarray:
    """
    Transform points from src(current) to ref(target)
    @param points: (N, D)
    @param [src_pose, ref_pose]: [(4, 4), (4, 4)] transformation matrix
        or
    @param tm: (4, 4) transformation matrix
    """
    points_converted = points.copy() if new_array else points
    if isinstance(transform, List):
        transform = transform_offset(*transform)

    R, T = transform[:3, :3], transform[:3, 3]
    points_converted[:, :3] = np.einsum('ij,jk->ik', points[:, :3], R.T) + T
    if new_array:
        return points_converted


def convert_boxes(boxes: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform boxes from src(current) to ref(target)
    @param points: (N, D)
    @param src_pose: (4, 4) transformation matrix
    @param ref_pose: (4, 4) transformation matrix
    """
    if len(boxes) == 0:
        return boxes
    boxes_converted = np.atleast_2d(boxes).copy()

    if isinstance(transform, List):
        transform = transform_offset(*transform)

    R, T = transform[:3, :3], transform[:3, 3]
    boxes_converted[:, :3] = np.einsum('ij,jk->ik', boxes_converted[:, :3], R.T) + T

    angular_transform = np.arctan2(R[1, 0], R[0, 0])
    boxes_converted[:, -1] = (boxes_converted[:, -1] + angular_transform + np.pi) % (2 * np.pi) - np.pi

    if boxes_converted.shape[0] == 1 and len(boxes.shape) == 1:
        boxes_converted = boxes_converted.squeeze(0)
    return boxes_converted


def convert_velos(velos: np.ndarray, transform: np.ndarray) -> np.ndarray:
    converted_velos = np.atleast_2d(velos)

    if isinstance(transform, List):
        transform = transform_offset(*transform)

    converted_velos = np.einsum('ij,jk->ik', converted_velos, transform[:3, :3])

    if converted_velos.shape[0] == 1 and len(velos.shape) == 1:
        converted_velos = converted_velos.squeeze(0)
    return converted_velos