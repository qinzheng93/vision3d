from typing import Optional, Tuple

import torch
from torch import Tensor

from vision3d.array_ops import anisotropic_registration_error

from .knn import knn
from .knn_interpolate import knn_interpolate
from .pairwise_distance import pairwise_distance
from .se3 import apply_transform, get_rotation_translation_from_transform


def evaluate_binary_classification(
    inputs: Tensor, targets: Tensor, positive_threshold: float = 0.5, use_logits: bool = False, eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """Binary classification precision and recall metric.

    Args:
        inputs (Tensor): inputs (*)
        targets (Tensor): targets, 0 or 1 (*)
        positive_threshold (float=0.5): considered as positive if larger than this value.
        use_logits (bool=False): If True, the inputs are logits, a sigmoid is needed.
        eps (float=1e-6): safe number.

    Return:
        precision (Tensor): precision
        recall (Tensor): recall
    """
    if use_logits:
        inputs = torch.sigmoid(inputs)

    if targets.dtype != torch.float32:
        targets = targets.float()

    results = torch.gt(inputs, positive_threshold).float()
    correct_results = results * targets

    precision = correct_results.sum() / (results.sum() + eps)
    recall = correct_results.sum() / (targets.sum() + eps)

    return precision, recall


def evaluate_multiclass_classification(
    inputs: Tensor, targets: Tensor, dim: int, eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor]:
    """Multi-class classification precision and recall metric.

    This method compute overall accuracy, macro-precision and macro-recall.

    Args:
        inputs (Tensor): inputs (*, C, *)
        targets (LongTensor): targets, [0, C-1] (*)
        dim (int): the category dim.
        eps (float=1e-6): safe number

    Return:
        accuracy (Tensor): overall accuracy
        mean_precision (Tensor): mean precision over all categories.
        mean_recall (Tensor): mean recall over all categories.
    """
    num_classes = inputs.shape[dim]

    # 1. accuracy
    results = inputs.argmax(dim=dim)
    accuracy = torch.eq(results, targets).float().mean()

    # 2. precision and recall
    results = results.flatten()  # (N,)
    targets = targets.flatten()  # (N,)
    num_rows = results.shape[0]
    row_indices = torch.arange(num_rows).cuda()

    result_mat = torch.zeros(size=(num_rows, num_classes)).cuda()  # (N, C)
    result_mat[row_indices, results] = 1.0
    result_sum = result_mat.sum(dim=0)  # (C,)

    target_mat = torch.zeros(size=(num_rows, num_classes)).cuda()  # (N, C)
    target_mat[row_indices, targets] = 1.0
    target_sum = target_mat.sum(dim=0)  # (C,)

    positive_mat = result_mat * target_mat
    positive_sum = positive_mat.sum(dim=0)  # (C,)

    per_class_precision = positive_sum / (result_sum + eps)  # (C,)
    per_class_recall = positive_sum / (target_sum + eps)  # (C,)

    # 3. mask out unseen categories
    class_masks = torch.gt(target_sum, 0).float()  # (C,)
    precision = (per_class_precision * class_masks).sum() / (class_masks.sum() + eps)
    recall = (per_class_recall * class_masks).sum() / (class_masks.sum() + eps)

    return accuracy, precision, recall


def psnr(inputs: Tensor, targets: Tensor, masks: Optional[Tensor] = None, reduction: str = "mean") -> Tensor:
    assert reduction in ["mean", "none"]
    errors = (inputs - targets) ** 2
    if masks is not None:
        errors = errors[masks]
    if reduction == "mean":
        errors = errors.mean()
    psnr = -10.0 * torch.log10(errors)
    return psnr


def compute_relative_rotation_error(inputs: Tensor, targets: Tensor) -> Tensor:
    """Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        inputs (Tensor): estimated rotation matrix (*, 3, 3)
        targets (Tensor): ground-truth rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(inputs.transpose(-1, -2), targets)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / torch.pi
    return rre


def compute_relative_translation_error(inputs: Tensor, targets: Tensor) -> Tensor:
    """Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        inputs (Tensor): estimated translation vector (*, 3)
        targets (Tensor): ground-truth translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(inputs - targets, dim=-1)
    return rte


def compute_isotropic_transform_error(
    inputs: Tensor, targets: Tensor, reduction: str = "mean"
) -> Tuple[Tensor, Tensor]:
    """Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        targets (Tensor): ground-truth transformation matrix (*, 4, 4)
        inputs (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ["mean", "sum", "none"]

    input_rotations, input_translations = get_rotation_translation_from_transform(inputs)
    target_rotations, target_translations = get_rotation_translation_from_transform(targets)

    rre = compute_relative_rotation_error(input_rotations, target_rotations)  # (*)
    rte = compute_relative_translation_error(input_translations, target_translations)  # (*)

    if reduction == "mean":
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == "sum":
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte


def compute_anisotropic_transform_error(
    inputs: Tensor, targets: Tensor, reduction: str = "mean"
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the anisotropic Relative Rotation Error and Relative Translation Error.

    This function calls numpy-based implementation to achieve batch-wise computation and thus is non-differentiable.

    Args:
        inputs (Tensor): estimated transformation matrix (B, 4, 4)
        targets (Tensor): ground-truth transformation matrix (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        r_mse (Tensor): rotation mse.
        r_mae (Tensor): rotation mae.
        t_mse (Tensor): translation mse.
        t_mae (Tensor): translation mae.
    """
    assert reduction in ["mean", "sum", "none"]

    batch_size = targets.shape[0]
    inputs_array = inputs.detach().cpu().numpy()
    targets_array = targets.detach().cpu().numpy()

    all_r_mse = []
    all_r_mae = []
    all_t_mse = []
    all_t_mae = []
    for i in range(batch_size):
        r_mse, r_mae, t_mse, t_mae = anisotropic_registration_error(targets_array[i], inputs_array[i])
        all_r_mse.append(r_mse)
        all_r_mae.append(r_mae)
        all_t_mse.append(t_mse)
        all_t_mae.append(t_mae)
    r_mse = torch.as_tensor(all_r_mse).to(targets)
    r_mae = torch.as_tensor(all_r_mae).to(targets)
    t_mse = torch.as_tensor(all_t_mse).to(targets)
    t_mae = torch.as_tensor(all_t_mae).to(targets)

    if reduction == "mean":
        r_mse = r_mse.mean()
        r_mae = r_mae.mean()
        t_mse = t_mse.mean()
        t_mae = t_mae.mean()
    elif reduction == "sum":
        r_mse = r_mse.sum()
        r_mae = r_mae.sum()
        t_mse = t_mse.sum()
        t_mae = t_mae.sum()

    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(points: Tensor, inputs: Tensor, targets: Tensor, reduction: str = "mean"):
    """Compute registration RMSE.

    RMSE = sqrt(sum(|| TP - T*P ||^2_2) / | P |)

    Args:
        points (Tensor): the point cloud in the shape of (B, N, 3).
        inputs (Tensor): the estimated transform in the shape of (B, 4, 4).
        targets (Tensor): the ground-truth transform in the shape of (B, 4, 4).
        reduction (str): the reduction method: "mean", "sum", "none". Default: "mean".

    Returns:
        A Tensor of the rmse in the shape of () ("mean" or "sum") or (B) ("none").
    """
    assert reduction in ["mean", "sum", "none"]

    input_points = apply_transform(points, inputs)
    target_points = apply_transform(points, targets)
    rmse = torch.sqrt((input_points - target_points).pow(2).sum(-1).mean())

    if reduction == "mean":
        rmse = rmse.mean()
    elif reduction == "sum":
        rmse = rmse.sum()

    return rmse


def compute_nonrigid_inlier_ratio(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    corr_scene_flows: Tensor,
    transform: Optional[Tensor] = None,
    acceptance_radius: float = 0.04,
) -> Tensor:
    """Non-rigid Inlier Ratio for 4DMatch.

    NIR = \frac{1}{|C|} \sum_{(p, q) in C [ || R x (p + f) + t - q || ]

    Args:
        src_corr_points (Tensor): (N, 3)
        tgt_corr_points (Tensor): (N, 3)
        corr_scene_flows (Tensor): (N, 3)
        transform (Tensor, optional): (4, 4)
        acceptance_radius (float): default: 0.04 (4DMatch)

    Returns:
        inlier_ratio (Tensor): non-rigid inlier ratio
    """
    src_corr_points = src_corr_points + corr_scene_flows
    if transform is not None:
        src_corr_points = apply_transform(src_corr_points, transform)
    residuals = torch.linalg.norm(tgt_corr_points - src_corr_points, dim=1)
    inlier_ratio = torch.lt(residuals, acceptance_radius).float().mean().nan_to_num_()
    return inlier_ratio


def compute_nonrigid_feature_matching_recall(
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    src_points: Tensor,
    scene_flows: Tensor,
    test_indices: Tensor,
    transform: Optional[Tensor] = None,
    acceptance_radius: float = 0.04,
    distance_limit: float = 0.1,
) -> Tensor:
    """Non-rigid Feature Matching Recall for 4DMatch.

    Args:
        src_corr_points (Tensor): (N, 3)
        tgt_corr_points (Tensor): (N, 3)
        src_points (Tensor): (M, 3)
        scene_flows (Tensor): (M, 3)
        test_indices (LongTensor): (K)
        transform (Tensor, optional): (4, 4)
        acceptance_radius (float=0.04): acceptance radius
        distance_limit (float=0.1): max distance for scene flow interpolation

    Returns:
        recall (Tensor): non-rigid feature matching recall
    """
    corr_motions = tgt_corr_points - src_corr_points  # (N, 3)
    src_test_points = src_points[test_indices]  # (K, 3)
    pred_motions = knn_interpolate(src_test_points, src_corr_points, corr_motions, k=3, distance_limit=distance_limit)
    pred_tgt_test_points = src_test_points + pred_motions  # (K, 3)
    gt_scene_flows = scene_flows[test_indices]  # (K, 3)
    gt_tgt_test_points = apply_transform(src_test_points + gt_scene_flows, transform)
    residuals = torch.linalg.norm(pred_tgt_test_points - gt_tgt_test_points, dim=1)
    recall = torch.lt(residuals, acceptance_radius).float().mean().nan_to_num_()
    return recall


def compute_scene_flow_accuracy(
    inputs: Tensor,
    targets: Tensor,
    acceptance_absolute_error: float,
    acceptance_relative_error: float,
    eps: float = 1e-20,
) -> Tensor:
    absolute_errors = torch.linalg.norm(inputs - targets, dim=1)
    target_lengths = torch.linalg.norm(targets, dim=1)
    relative_errors = absolute_errors / (target_lengths + eps)
    absolute_results = torch.lt(absolute_errors, acceptance_absolute_error)
    relative_results = torch.lt(relative_errors, acceptance_relative_error)
    results = torch.logical_or(absolute_results, relative_results)
    accuracy = results.float().mean()
    return accuracy


def compute_scene_flow_outlier_ratio(
    inputs: Tensor,
    targets: Tensor,
    acceptance_absolute_error: Optional[float],
    acceptance_relative_error: Optional[float],
    eps: float = 1e-20,
) -> Tensor:
    absolute_errors = torch.linalg.norm(inputs - targets, dim=1)
    target_lengths = torch.linalg.norm(targets, dim=1)
    relative_errors = absolute_errors / (target_lengths + eps)
    results = inputs.new_zeros(size=(inputs.shape[0],), dtype=torch.bool)
    if acceptance_absolute_error is not None:
        results = torch.logical_or(results, torch.gt(absolute_errors, acceptance_absolute_error))
    if acceptance_relative_error is not None:
        results = torch.logical_or(results, torch.gt(relative_errors, acceptance_relative_error))
    outlier_ratio = results.float().mean()
    return outlier_ratio


def compute_corr_coverage(q_points: Tensor, s_points: Tensor, distance_limit: float) -> Tensor:
    nn_distances, nn_indices = knn(q_points, s_points, k=1, return_distance=True)
    coverage = torch.lt(nn_distances, distance_limit).float().mean().nan_to_num_()
    return coverage


def compute_chamfer_distance(q_points: Tensor, s_points: Tensor) -> Tensor:
    q_nn_distances, q_nn_indices = knn(q_points, s_points, k=1, return_distance=True)
    s_nn_distances, s_nn_indices = knn(s_points, q_points, k=1, return_distance=True)
    chamfer_distance = q_nn_distances.mean() + s_nn_distances.mean()
    return chamfer_distance


def compute_modified_chamfer_distance(
    raw_points: Tensor,
    src_points: Tensor,
    tgt_points: Tensor,
    gt_transform: Tensor,
    transform: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute the modified chamfer distance.

    Note:
        1. The chamfer distance is squared.

    Args:
        raw_points (Tensor): (B, N_raw, 3)
        src_points (Tensor): (B, N_src, 3)
        tgt_points (Tensor): (B, N_tgt, 3)
        gt_transform (Tensor): (B, 4, 4)
        transform (Tensor): (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        chamfer_distance
    """
    assert reduction in ["mean", "sum", "none"]

    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, transform)  # (B, N_src, 3)
    sq_dist_mat_p_q = pairwise_distance(aligned_src_points, raw_points)  # (B, N_src, N_raw)
    nn_sq_distances_p_q = sq_dist_mat_p_q.min(dim=-1)[0]  # (B, N_src)
    chamfer_distance_p_q = nn_sq_distances_p_q.mean(dim=-1)  # (B)

    # Q -> P_raw
    composed_transform = torch.matmul(transform, torch.linalg.inv(gt_transform))  # (B, 4, 4)
    aligned_raw_points = apply_transform(raw_points, composed_transform)  # (B, N_raw, 3)
    sq_dist_mat_q_p = pairwise_distance(tgt_points, aligned_raw_points)  # (B, N_tgt, N_raw)
    nn_sq_distances_q_p = sq_dist_mat_q_p.min(dim=-1)[0]  # (B, N_tgt)
    chamfer_distance_q_p = nn_sq_distances_q_p.mean(dim=-1)  # (B)

    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p  # (B)

    if reduction == "mean":
        chamfer_distance = chamfer_distance.mean()
    elif reduction == "sum":
        chamfer_distance = chamfer_distance.sum()
    return chamfer_distance
