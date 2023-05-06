from typing import Optional, Tuple

import torch
from torch import Tensor

from .index_select import index_select
from .pairwise_distance import pairwise_distance
from .point_cloud_partition import point_to_node_partition
from .se3 import apply_transform

# Patch correspondences


@torch.no_grad()
def dense_correspondences_to_patch_correspondences(
    src_points: Tensor,
    tgt_points: Tensor,
    src_nodes: Tensor,
    tgt_nodes: Tensor,
    src_corr_indices: Tensor,
    tgt_corr_indices: Tensor,
    return_score: bool = False,
) -> Tuple[Tensor, ...]:
    """Generate patch correspondences from point correspondences and the number of point correspondences within each
    patch correspondences.

    For each point correspondence, convert it to patch correspondence by replacing the point indices to the
    corresponding patch indices.

    We also define the proxy score for each patch correspondence as a estimation of the overlap ratio:
    s = (#point_corr / #point_in_ref_patch + #point_corr / #point_in_src_patch) / 2

    Args:
        src_points (Tensor): source point cloud
        tgt_points (Tensor): target point cloud
        src_nodes (Tensor): source patch points
        tgt_nodes (Tensor): target patch points
        src_corr_indices (LongTensor): correspondence indices in source point cloud
        tgt_corr_indices (LongTensor): correspondence indices in target point cloud
        return_score (bool=True): If True, return the proxy score for each patch correspondences

    Returns:
        src_node_corr_indices (LongTensor): (C), [src, tgt]
        tgt_node_corr_indices (LongTensor): (C), [src, tgt]
        node_corr_counts (LongTensor): (C)
        node_corr_scores (Tensor): (C)
    """
    src_point_to_node, src_node_sizes = point_to_node_partition(
        src_points, src_nodes, return_counts=True, gather_points=False
    )
    tgt_point_to_node, tgt_node_sizes = point_to_node_partition(
        tgt_points, tgt_nodes, return_counts=True, gather_points=False
    )

    src_node_corr_indices = src_point_to_node[src_corr_indices]
    tgt_node_corr_indices = tgt_point_to_node[tgt_corr_indices]

    node_corr_indices = src_node_corr_indices * tgt_nodes.shape[0] + tgt_node_corr_indices
    node_corr_indices, node_corr_counts = torch.unique(node_corr_indices, return_counts=True)
    src_node_corr_indices = node_corr_indices // tgt_nodes.shape[0]
    tgt_node_corr_indices = node_corr_indices % tgt_nodes.shape[0]

    if return_score:
        src_node_corr_scores = node_corr_counts / src_node_sizes[src_node_corr_indices]
        tgt_node_corr_scores = node_corr_counts / tgt_node_sizes[tgt_node_corr_indices]
        node_corr_scores = (src_node_corr_scores + tgt_node_corr_scores) / 2
        return src_node_corr_indices, tgt_node_corr_indices, node_corr_counts, node_corr_scores

    return src_node_corr_indices, tgt_node_corr_indices, node_corr_counts


@torch.no_grad()
def get_patch_correspondences(
    src_nodes: Tensor,
    tgt_nodes: Tensor,
    src_knn_points: Tensor,
    tgt_knn_points: Tensor,
    transform: Tensor,
    pos_radius: float,
    src_masks: Optional[Tensor] = None,
    tgt_masks: Optional[Tensor] = None,
    src_knn_masks: Optional[Tensor] = None,
    tgt_knn_masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k-nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        tgt_nodes (Tensor): (M, 3)
        src_nodes (Tensor): (N, 3)
        tgt_knn_points (Tensor): (M, K, 3)
        src_knn_points (Tensor): (N, K, 3)
        transform (Tensor): (4, 4)
        pos_radius (float)
        tgt_masks (BoolTensor=None): (M,)
        src_masks (BoolTensor=None): (N,)
        tgt_knn_masks (BoolTensor=None): (M, K)
        src_knn_masks (BoolTensor=None): (N, K)

    Returns:
        src_corr_indices (LongTensor): (C,)
        tgt_corr_indices (LongTensor): (C,)
        corr_overlaps (Tensor): (C,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    # filter out non-overlapping patches using enclosing sphere
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (M, K)
    if src_knn_masks is not None:
        src_knn_dists[~src_knn_masks] = 0.0
    src_max_dists = src_knn_dists.max(1)[0]  # (M,)
    tgt_knn_dists = torch.linalg.norm(tgt_knn_points - tgt_nodes.unsqueeze(1), dim=-1)  # (N, K)
    if tgt_knn_masks is not None:
        tgt_knn_dists[~tgt_knn_masks] = 0.0
    tgt_max_dists = tgt_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(src_nodes, tgt_nodes))  # (M, N)
    intersect_mat = torch.gt(src_max_dists.unsqueeze(1) + tgt_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    if src_masks is not None:
        intersect_mat = torch.logical_and(intersect_mat, src_masks.unsqueeze(1))
    if tgt_masks is not None:
        intersect_mat = torch.logical_and(intersect_mat, tgt_masks.unsqueeze(0))
    sel_src_indices, sel_tgt_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs, compute correspondence matrix
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)
    tgt_knn_points = tgt_knn_points[sel_tgt_indices]  # (B, K, 3)
    dist_mat = pairwise_distance(src_knn_points, tgt_knn_points)  # (B, K, K)
    point_corr_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    if src_knn_masks is not None:
        src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
        point_corr_mat = torch.logical_and(point_corr_mat, src_knn_masks.unsqueeze(2))  # (B, K, K)
    if tgt_knn_masks is not None:
        tgt_knn_masks = tgt_knn_masks[sel_tgt_indices]  # (B, K)
        point_corr_mat = torch.logical_and(point_corr_mat, tgt_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    src_overlap_counts = torch.any(point_corr_mat, dim=-1).sum(-1)  # (B,)
    tgt_overlap_counts = torch.any(point_corr_mat, dim=-2).sum(-1)  # (B,)
    src_total_counts = src_knn_masks.sum(-1) if src_knn_masks is not None else src_knn_points.shape[1]  # 1 or (B,)
    tgt_total_counts = tgt_knn_masks.sum(-1) if tgt_knn_masks is not None else tgt_knn_points.shape[1]  # 1 or (B,)
    src_overlap_ratios = src_overlap_counts.float() / src_total_counts.float()  # (B,)
    tgt_overlap_ratios = tgt_overlap_counts.float() / tgt_total_counts.float()  # (B,)
    overlap_ratios = (src_overlap_ratios + tgt_overlap_ratios) / 2  # (B,)

    overlap_masks = torch.gt(overlap_ratios, 0)
    src_corr_indices = sel_src_indices[overlap_masks]
    tgt_corr_indices = sel_tgt_indices[overlap_masks]
    corr_overlaps = overlap_ratios[overlap_masks]

    return src_corr_indices, tgt_corr_indices, corr_overlaps


@torch.no_grad()
def patch_correspondences_to_dense_correspondences(
    src_knn_points: Tensor,
    tgt_knn_points: Tensor,
    src_knn_indices: Tensor,
    tgt_knn_indices: Tensor,
    src_node_corr_indices: Tensor,
    tgt_node_corr_indices: Tensor,
    transform: Tensor,
    matching_radius: float,
    src_knn_masks: Optional[Tensor] = None,
    tgt_knn_masks: Optional[Tensor] = None,
    return_distance: bool = False,
) -> Tuple[Tensor, ...]:
    src_knn_points = apply_transform(src_knn_points, transform)
    src_node_corr_knn_indices = src_knn_indices[src_node_corr_indices]  # (P, K)
    tgt_node_corr_knn_indices = tgt_knn_indices[tgt_node_corr_indices]  # (P, K)
    src_node_corr_knn_points = src_knn_points[src_node_corr_indices]  # (P, K, 3)
    tgt_node_corr_knn_points = tgt_knn_points[tgt_node_corr_indices]  # (P, K, 3)
    dist_mat = pairwise_distance(src_node_corr_knn_points, tgt_node_corr_knn_points, squared=False)  # (P, K, K)
    corr_mat = torch.lt(dist_mat, matching_radius)
    if src_knn_masks is not None:
        src_node_corr_knn_masks = src_knn_masks[src_node_corr_indices]  # (P, K)
        corr_mat = torch.logical_and(corr_mat, src_node_corr_knn_masks.unsqueeze(2))  # (P, K, K)
    if tgt_knn_masks is not None:
        tgt_node_corr_knn_masks = tgt_knn_masks[tgt_node_corr_indices]  # (P, K)
        corr_mat = torch.logical_and(corr_mat, tgt_node_corr_knn_masks.unsqueeze(1))  # (P, K, K)
    batch_indices, row_indices, col_indices = torch.nonzero(corr_mat, as_tuple=True)  # (C,) (C,) (C,)
    src_corr_indices = src_node_corr_knn_indices[batch_indices, row_indices]
    tgt_corr_indices = tgt_node_corr_knn_indices[batch_indices, col_indices]

    if return_distance:
        corr_distances = dist_mat[batch_indices, row_indices, col_indices]
        return src_corr_indices, tgt_corr_indices, corr_distances

    return src_corr_indices, tgt_corr_indices


@torch.no_grad()
def get_patch_overlap_ratios(
    src_points: Tensor,
    tgt_points: Tensor,
    src_knn_points: Tensor,
    tgt_knn_points: Tensor,
    src_knn_indices: Tensor,
    tgt_knn_indices: Tensor,
    src_node_corr_indices: Tensor,
    tgt_node_corr_indices: Tensor,
    transform: Tensor,
    matching_radius: float,
    src_knn_masks: Optional[Tensor] = None,
    tgt_knn_masks: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    src_corr_indices, tgt_corr_indices = patch_correspondences_to_dense_correspondences(
        src_knn_points,
        tgt_knn_points,
        src_knn_indices,
        tgt_knn_indices,
        src_node_corr_indices,
        tgt_node_corr_indices,
        transform,
        matching_radius,
        src_knn_masks=src_knn_masks,
        tgt_knn_masks=tgt_knn_masks,
    )
    unique_src_corr_indices = torch.unique(src_corr_indices)
    unique_tgt_corr_indices = torch.unique(tgt_corr_indices)
    src_overlap_masks = torch.zeros(src_points.shape[0] + 1).cuda()  # pad for following indexing
    tgt_overlap_masks = torch.zeros(tgt_points.shape[0] + 1).cuda()  # pad for following indexing
    src_overlap_masks.index_fill_(0, unique_src_corr_indices, 1.0)
    tgt_overlap_masks.index_fill_(0, unique_tgt_corr_indices, 1.0)
    src_knn_overlap_masks = index_select(src_overlap_masks, src_knn_indices, dim=0)  # (N', K)
    tgt_knn_overlap_masks = index_select(tgt_overlap_masks, tgt_knn_indices, dim=0)  # (M', K)
    if src_knn_masks is not None:
        src_knn_overlap_ratios = (src_knn_overlap_masks * src_knn_masks).sum(1) / (src_knn_masks.sum(1) + eps)  # (N')
    else:
        src_knn_overlap_ratios = src_knn_overlap_masks.mean(dim=1)  # (N')
    if tgt_knn_masks is not None:
        tgt_knn_overlap_ratios = (tgt_knn_overlap_masks * tgt_knn_masks).sum(1) / (tgt_knn_masks.sum(1) + eps)  # (M')
    else:
        tgt_knn_overlap_ratios = tgt_knn_overlap_masks.mean(dim=1)  # (M')
    return src_knn_overlap_ratios, tgt_knn_overlap_ratios


@torch.no_grad()
def get_patch_occlusion_ratios(
    src_points: Tensor,
    tgt_points: Tensor,
    src_knn_points: Tensor,
    tgt_knn_points: Tensor,
    src_knn_indices: Tensor,
    tgt_knn_indices: Tensor,
    src_node_corr_indices: Tensor,
    tgt_node_corr_indices: Tensor,
    transform: Tensor,
    matching_radius: float,
    src_knn_masks: Optional[Tensor] = None,
    tgt_knn_masks: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    src_knn_overlap_ratios, tgt_knn_overlap_ratios = get_patch_overlap_ratios(
        src_points,
        tgt_points,
        src_knn_points,
        tgt_knn_points,
        src_knn_indices,
        tgt_knn_indices,
        src_node_corr_indices,
        tgt_node_corr_indices,
        transform,
        matching_radius,
        src_knn_masks=src_knn_masks,
        tgt_knn_masks=tgt_knn_masks,
        eps=eps,
    )
    src_knn_occlusion_ratios = 1.0 - src_knn_overlap_ratios
    tgt_knn_occlusion_ratios = 1.0 - tgt_knn_overlap_ratios
    return src_knn_occlusion_ratios, tgt_knn_occlusion_ratios
