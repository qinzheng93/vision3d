from typing import Optional

import numpy as np
from numpy import ndarray

from .se3 import get_transform_from_rotation_translation


def weighted_procrustes(
    src_points: ndarray,
    tgt_points: ndarray,
    weights: Optional[ndarray] = None,
    weight_threshold: float = 0.0,
    eps: float = 1e-6,
) -> ndarray:
    """Estimate the alignment transformation with weighted SVD.

    Args:
        src_points (array<float>): The correspondence points in the source point cloud in the shape of (N, 3).
        tgt_points (array<float>): The correspondence points in the target point cloud in the shape of (N, 3).
        weights (array<float>, optional): The weights of the correspondences in the shape of (N,).
        weight_threshold (float): The threshold value for the weights. Default: 0.
        eps (float): The safe number for division. Default: 1e-6.

    Returns:
        A ndarray of the estimated transformation in the shape of (4, 4).
    """
    if weights is None:
        weights = np.ones(shape=(src_points.shape[0],))  # (N,)
    weights[weights < weight_threshold] = 0.0
    weights = weights / (weights.sum() + eps)
    weights = weights[:, None]  # (N, 1)

    src_centroid = np.sum(src_points * weights, axis=0)  # (3)
    tgt_centroid = np.sum(tgt_points * weights, axis=0)  # (3)

    src_points_centered = src_points - src_centroid[None, :]  # (N, 3)
    tgt_points_centered = tgt_points - tgt_centroid[None, :]  # (N, 3)

    H = src_points_centered.T @ (weights * tgt_points_centered)
    U, _, Vt = np.linalg.svd(H)  # H = USV^T
    Ut = U.T
    V = Vt.T
    eye = np.eye(3)
    eye[-1, -1] = np.sign(np.linalg.det(V @ Ut))
    R = V @ eye @ Ut

    t = tgt_centroid - R @ src_centroid

    transform = get_transform_from_rotation_translation(R, t)

    return transform
