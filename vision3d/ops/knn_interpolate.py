from typing import Optional

import torch
from torch import Tensor

from .index_select import index_select
from .knn import keops_knn


def knn_interpolate(
    q_points: Tensor,
    s_points: Tensor,
    s_feats: Tensor,
    k: int = 3,
    eps: float = 1e-10,
    distance_limit: Optional[float] = None,
):
    """kNN interpolate.

    Args:
        q_points (Tensor): a Tensor of the query points in the shape of (M, 3).
        s_points (Tensor): a Tensor of the support points in the shape of (N, 3).
        s_feats (Tensor): a Tensor of the support features in the shape of (N, C).
        k (int): the number of the neighbors. Default: 3.
        eps (float): the safe number for division. Default: 1e-10.
        distance_limit (float, optional): the distance limit for the neighbors. If not None, the neighbors further than
            this distance are ignored.

    Returns:
        A Tensor of the features of the query points in the shape of (M, C).
    """
    knn_distances, knn_indices = keops_knn(q_points, s_points, k)  # (M, K)
    if distance_limit is not None:
        masks = torch.gt(knn_distances, distance_limit)  # (M, K)
        knn_distances.masked_fill_(masks, 1e10)
    weights = 1.0 / (knn_distances + eps)  # (M, K)
    weights = weights / weights.sum(dim=1, keepdims=True)  # (M, K)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (M, K, C)
    q_feats = (knn_feats * weights.unsqueeze(-1)).sum(dim=1)  # (M, C)
    return q_feats


def knn_interpolate_pack_mode(
    q_points: Tensor,
    s_points: Tensor,
    s_feats: Tensor,
    neighbor_indices: Tensor,
    k: Optional[int] = None,
    eps: float = 1e-8,
):
    """kNN interpolate in pack mode.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        q_points (Tensor): a Tensor of the query points in the shape of (N, 3).
        s_points (Tensor): a Tensor of the support points in the shape of (M, 3).
        s_feats (Tensor): a Tensor of the support features in the shape of (M, C).
        neighbor_indices (Tensor): a LongTensor of the neighbor indices for the query points in the shape of (N, K).
        k (int, optional): the number of the neighbors. If None, use all neighbors.
        eps (float): the safe number for division. Default: 1e-8.

    Returns:
        A Tensor of the features of the query points in the shape of (N, C).
    """
    s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (M + 1, 3)
    s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (M + 1, C)
    knn_indices = neighbor_indices if k is None else neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, k, 3)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, k, C)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)  # (N, k)
    knn_masks = torch.ne(knn_indices, s_points.shape[0] - 1).float()  # (N, k)
    knn_weights = knn_masks / (knn_sq_distances + eps)  # (N, k)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)  # (N, k)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return q_feats
