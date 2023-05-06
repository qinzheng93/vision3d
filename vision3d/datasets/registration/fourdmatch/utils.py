import numpy as np
from numpy import ndarray

from vision3d.array_ops import apply_transform, knn_interpolate


def compute_nonrigid_inlier_ratio(
    src_corr_points: ndarray,
    tgt_corr_points: ndarray,
    corr_flows: ndarray,
    transform: ndarray,
    acceptance_radius: float = 0.04,
):
    """Non-rigid Inlier Ratio (from 4DMatch).

    NIR = \frac{1}{|C|} \sum_{(p, q) in C} [ || R x (p + f) + t - q || < \tau ]

    Args:
        tgt_corr_points (array): (N, 3)
        src_corr_points (array): (N, 3)
        corr_flows (array): (N, 3)
        transform (array): (4, 4)
        acceptance_radius (float): default: 0.04 (4DMatch)

    Returns:
        inlier_ratio (float): non-rigid inlier ratio
    """
    src_corr_points = apply_transform(src_corr_points + corr_flows, transform)
    residuals = np.linalg.norm(tgt_corr_points - src_corr_points, axis=1)
    inlier_ratio = (residuals < acceptance_radius).mean()
    return inlier_ratio


def compute_nonrigid_feature_matching_recall(
    src_corr_points: ndarray,
    tgt_corr_points: ndarray,
    src_points: ndarray,
    flows: ndarray,
    transform: ndarray,
    test_indices: ndarray,
    acceptance_radius: float = 0.04,
    distance_limit: float = 0.1,
):
    corr_flows = tgt_corr_points - src_corr_points
    q_points = src_points[test_indices]
    pred_flows = knn_interpolate(q_points, src_corr_points, corr_flows, k=3, distance_limit=distance_limit)
    pred_tgt_points = q_points + pred_flows
    gt_flows = flows[test_indices]
    gt_tgt_points = apply_transform(q_points + gt_flows, transform)
    residuals = np.linalg.norm(pred_tgt_points - gt_tgt_points, axis=1)
    recall = (residuals < acceptance_radius).mean()
    return recall


def compute_end_point_error(
    warped_src_points: ndarray,
    src_points: ndarray,
    scene_flows: ndarray,
    transform: ndarray,
):
    aligned_src_points = apply_transform(src_points + scene_flows, transform)
    error = np.linalg.norm(warped_src_points - aligned_src_points, axis=1).mean()
    return error


def compute_3d_accuracy(
    warped_src_points: ndarray,
    src_points: ndarray,
    scene_flows: ndarray,
    transform: ndarray,
    acceptance_relative_error: float,
    acceptance_absolute_error: float,
):
    aligned_src_points = apply_transform(src_points + scene_flows, transform)
    warped_motions = warped_src_points - src_points
    aligned_motions = aligned_src_points - src_points
    absolute_errors = np.linalg.norm(warped_motions - aligned_motions, axis=1)
    absolute_results = absolute_errors < acceptance_absolute_error
    relative_errors = absolute_errors / np.linalg.norm(aligned_motions, axis=1)
    relative_results = relative_errors < acceptance_relative_error
    results = absolute_results & relative_results
    accuracy = results.astype(np.float64).mean()
    return accuracy


def compute_outlier_ratio(
    warped_src_points: ndarray,
    src_points: ndarray,
    scene_flows: ndarray,
    transform: ndarray,
    acceptance_error: float = 0.3,
):
    aligned_src_points = apply_transform(src_points + scene_flows, transform)
    warped_motions = warped_src_points - src_points
    aligned_motions = aligned_src_points - src_points
    absolute_errors = np.linalg.norm(warped_motions - aligned_motions, axis=1)
    relative_errors = absolute_errors / np.linalg.norm(aligned_motions, axis=1)
    outlier_ratio = (relative_errors > acceptance_error).astype(np.float64).mean()
    return outlier_ratio
