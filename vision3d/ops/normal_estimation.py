from typing import Optional

import torch
from torch import Tensor

from .index_select import index_select
from .knn import knn


def estimate_normals(
    q_points: Tensor, s_points: Optional[Tensor] = None, k: int = 50, disambiguate_directions: bool = True
):
    """Estimate normals for the query points from the support points.

    Modified from [PyTorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_normals.py).

    Args:
        q_points (Tensor): query points (N, 3)
        s_points (Tensor=None): support points. If None, use q_points (M, 3)
        k (int=50): number of neighbors
        disambiguate_directions (bool=True): disambiguate directions.

    Returns:
        q_normals (Tensor): normals of the query points (N, 3)
    """
    if s_points is None:
        s_points = q_points

    knn_indices = knn(q_points, s_points, k)  # (N, K)
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, K, 3)
    centroids = knn_points.mean(dim=1, keepdim=True)  # (N, 1, 3)
    knn_offsets = knn_points - centroids  # (N, K, 3)

    cov_mat = knn_offsets.unsqueeze(3) * knn_offsets.unsqueeze(2)  # (N, K, 3, 1) x (N, K, 1, 3)
    cov_mat = cov_mat.mean(dim=1)  # (N, 3, 3)
    _, eigenvectors = torch.symeig(cov_mat.cpu(), eigenvectors=True)  # (N, 3), (N, 3, 3)
    eigenvectors = eigenvectors.cuda()
    normals = eigenvectors[:, :, 0]  # (N, 3)

    if disambiguate_directions:
        knn_offsets = knn_points - q_points.unsqueeze(1)  # (N, K, 3) - (N, 3) -> (N, 1, 3)
        projections = (normals.unsqueeze(1) * knn_offsets).sum(2)  # (N, 3) -> (N, 1, 3) x (N, K, 3) -> (N, K)
        flips = torch.gt(projections.sum(dim=-1, keepdim=True), 0).float()  # (N, K) -> (N, 1)
        # flips = torch.lt(torch.gt(projections, 0).float().mean(dim=1, keepdim=True), 0.5).float()  # (N, K) -> (N, 1)
        normals = (1.0 - 2.0 * flips) * normals

    return normals
