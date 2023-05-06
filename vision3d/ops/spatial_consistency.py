import torch
from torch import Tensor

from vision3d.ops import pairwise_distance


def spatial_consistency(src_corr_points: Tensor, tgt_corr_points: Tensor, sigma: float) -> Tensor:
    """Compute spatial consistency.

    SC_{i,j} = max(1 - d_{i,j}^2 / sigma ^2, 0)
    d_{i,j} = \lvert \lVert p_i - p_j \rVert - \lVert q_i - q_j \rVert \rvert

    Args:
        src_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        tgt_corr_points (Tensor): The correspondence points in the source point cloud in the shape of (*, N, 3).
        sigma (float): The sensitivity factor.

    Returns:
        A Tensor of the spatial consistency between the correspondences in the shape of (*, N, N).
    """
    src_dist_mat = pairwise_distance(src_corr_points, src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(tgt_corr_points, tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)
    consistency_mat = torch.relu(1.0 - delta_mat.pow(2) / (sigma ** 2))
    return consistency_mat


def cross_spatial_consistency(
    q_src_corr_points: Tensor,
    q_tgt_corr_points: Tensor,
    s_src_corr_points: Tensor,
    s_tgt_corr_points: Tensor,
    sigma: float,
) -> Tensor:
    """Compute cross spatial consistency between two set of correspondences.

    SC_{i,j} = max(1 - d_{i,j}^2 / sigma ^2, 0)
    d_{i,j} = \lvert \lVert p_i - p_j \rVert - \lVert q_i - q_j \rVert \rvert

    Args:
        q_src_corr_points (Tensor): the query correspondence points in the source point cloud (*, N, 3).
        q_tgt_corr_points (Tensor): the query correspondence points in the target point cloud (*, N, 3).
        s_src_corr_points (Tensor): the support correspondence points in the source point cloud (*, M, 3).
        s_tgt_corr_points (Tensor): the support correspondence points in the target point cloud (*, M, 3).
        sigma (float): The sensitivity factor.

    Returns:
        A Tensor of the spatial consistency between the correspondences in the shape of (*, N, M).
    """
    src_dist_mat = pairwise_distance(q_src_corr_points, s_src_corr_points, squared=False)  # (*, N, N)
    tgt_dist_mat = pairwise_distance(q_tgt_corr_points, s_tgt_corr_points, squared=False)  # (*, N, N)
    delta_mat = torch.abs(src_dist_mat - tgt_dist_mat)
    consistency_mat = torch.relu(1.0 - delta_mat.pow(2) / (sigma ** 2))
    return consistency_mat
