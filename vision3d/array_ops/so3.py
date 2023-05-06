from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation


def get_rotation_along_axis(scale: float, axis: int = "z") -> ndarray:
    assert axis in ["x", "y", "z"]
    theta = 2.0 * np.pi * scale
    if axis == "x":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]]
        )
    elif axis == "y":
        rotation = np.asarray(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 1.0], [-np.sin(theta), 0.0, np.cos(theta)]]
        )
    else:
        rotation = np.asarray(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
    return rotation


def apply_rotation(
    points: ndarray, rotation: ndarray, normals: Optional[ndarray] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    points = np.matmul(points, rotation.T)
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    return points


def skew_symmetric_matrix(vector: ndarray) -> ndarray:
    """Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Args:
        vector (ndarray): input vectors (3)

    Returns:
        skews (ndarray): output skew-symmetric matrix (3, 3)
    """
    skews = np.zeros(shape=(3, 3))
    skews[0, 1] = -vector[2]
    skews[0, 2] = vector[1]
    skews[1, 0] = vector[2]
    skews[1, 2] = -vector[0]
    skews[2, 0] = -vector[1]
    skews[2, 1] = vector[0]
    return skews


def rodrigues_rotation_formula(axis: ndarray, angle: float) -> ndarray:
    """Compute Rodrigues Rotation Matrix.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Args:
        axis (array<float>): normalized axis vectors (3)
        angle (float): rotation angles in right-hand direction in rad.

    Returns:
        rotation (array<float>): Rodrigues rotation matrix (3, 3)
    """
    skews = skew_symmetric_matrix(axis)  # (3, 3)
    rotation = np.eye(3) + np.sin(angle) * skews + (1.0 - np.cos(angle)) * np.matmul(skews, skews)
    return rotation


def axis_angle_to_rotation_matrix(phi: ndarray) -> ndarray:
    rotation = Rotation.from_rotvec(phi).as_matrix()
    return rotation


def axis_angle_to_quaternion(phi: ndarray) -> ndarray:
    q = Rotation.from_rotvec(phi).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (xyzw) -> (wxyz)
    return q


def quaternion_to_axis_angle(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (xyzw) -> (wxyz)
    phi = Rotation.from_quat(q).as_rotvec()
    return phi


def quaternion_to_rotation_matrix(q: ndarray) -> ndarray:
    q = q[..., [1, 2, 3, 0]]  # (xyzw) -> (wxyz)
    rotation = Rotation.from_quat(q).as_matrix()
    return rotation


def rotation_matrix_to_axis_angle(rotation: ndarray) -> ndarray:
    phi = Rotation.from_matrix(rotation).as_rotvec()
    return phi


def rotation_matrix_to_quaternion(rotation: ndarray) -> ndarray:
    q = Rotation.from_matrix(rotation).as_quat()
    q = q[..., [3, 0, 1, 2]]  # (xyzw) -> (wxyz)
    return q


def rotation_matrix_to_euler(rotation: ndarray, order: str, use_degree: bool = False) -> ndarray:
    euler = Rotation.from_matrix(rotation).as_euler(order, degrees=use_degree)
    return euler


def euler_to_rotation_matrix(euler: ndarray, order: str) -> ndarray:
    rotation = Rotation.from_euler(order, euler).as_matrix()
    return rotation
