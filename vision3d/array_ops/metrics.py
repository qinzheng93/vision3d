from typing import Optional, Union

import numpy as np
from numpy import ndarray

from .knn import knn
from .procrustes import weighted_procrustes
from .se3 import apply_transform, get_rotation_translation_from_transform
from .so3 import rotation_matrix_to_euler


def psnr(
    inputs: ndarray, targets: ndarray, masks: Optional[ndarray] = None, reduction: str = "mean"
) -> Union[float, ndarray]:
    assert reduction in ["mean", "none"]
    errors = (inputs - targets) ** 2
    if masks is not None:
        errors = errors[masks]
    if reduction == "mean":
        errors = errors.mean()
    psnr = -10.0 * np.log10(errors)
    return psnr


def relative_rotation_error(gt_rotation: ndarray, est_rotation: ndarray):
    """Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translation: ndarray, est_translation: ndarray):
    """Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)


def isotropic_registration_error(gt_transform: ndarray, est_transform: ndarray):
    """Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = relative_rotation_error(gt_rotation, est_rotation)
    rte = relative_translation_error(gt_translation, est_translation)
    return rre, rte


def rotation_mse_and_mae(gt_rotation: ndarray, est_rotation: ndarray):
    """Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = rotation_matrix_to_euler(gt_rotation, "xyz", use_degree=True)  # (3,)
    est_euler_angles = rotation_matrix_to_euler(est_rotation, "xyz", use_degree=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def translation_mse_and_mae(gt_translation: ndarray, est_translation: ndarray):
    """Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def anisotropic_registration_error(gt_transform: ndarray, est_transform: ndarray):
    """Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def registration_rmse(src_points: ndarray, gt_transform: ndarray, est_transform: ndarray):
    """Compute re-alignment error (approximated RMSE in 3DMatch).

    RMSE = \sqrt{\sum \lVert R p_i + t - q_i \rVert^2}

    Used in Rotated 3DMatch.

    Args:
        src_points (array): source point cloud. (N, 3)
        gt_transform (array): ground-truth transformation. (4, 4)
        est_transform (array): estimated transformation. (4, 4)

    Returns:
        error (float): root mean square error.
    """
    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    rmse = np.sqrt(np.sum((gt_points - est_points) ** 2, axis=1).mean())
    # rmse = np.linalg.norm(gt_points - est_points, axis=1).mean()
    return rmse


def registration_chamfer_distance(
    raw_points: ndarray,
    src_points: ndarray,
    tgt_points: ndarray,
    gt_transform: ndarray,
    est_transform: ndarray,
):
    """Compute the modified chamfer distance (RPMNet)."""
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = np.power(knn(aligned_src_points, raw_points, k=1, return_distance=True)[0], 2).mean()
    # Q -> P_raw
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = np.power(knn(tgt_points, aligned_raw_points, k=1, return_distance=True)[0], 2).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def registration_corr_distance(src_corr_points, tgt_corr_points, transform):
    """Computing the mean distance between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    distances = np.sqrt(((tgt_corr_points - src_corr_points) ** 2).sum(1))
    mean_distance = np.mean(distances)
    return mean_distance


def registration_inlier_ratio(src_corr_points, tgt_corr_points, transform, positive_radius=0.1):
    """Computing the inlier ratio between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    corr_distances = np.sqrt(((tgt_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(corr_distances < positive_radius)
    return inlier_ratio


def point_cloud_overlap(src_points, tgt_points, transform=None, positive_radius=0.1):
    """Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances, _ = knn(tgt_points, src_points, k=1, return_distance=True)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


def absolute_trajectory_error(gt_trajectory, est_trajectory):
    """Compute absolute trajectory error."""
    transform = weighted_procrustes(gt_trajectory, est_trajectory)
    gt_trajectory = apply_transform(gt_trajectory, transform)
    error = np.linalg.norm(gt_trajectory - est_trajectory, axis=1)
    rmse = np.sqrt(np.mean(error**2))
    return rmse
