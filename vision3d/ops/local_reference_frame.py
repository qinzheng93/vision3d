import torch
from torch.nn import functional as F

from .group_gather import group_gather
from .knn import knn


def build_local_reference_frame(points, normals, num_neighbors, return_axes=False):
    """Build Local Reference Frame.

    Refer to ``The Perfect Match: 3D Point Cloud Matching with Smoothed Densities``.

    Args:
        points (Tensor): (B, 3, N)
        normals (Tensor): (B, 3, N)
        num_neighbors (int): number of neighbors.
        return_axes (bool=False): If True, return the lrf axes.

    Returns:
        x_axes (Tensor): (B, 3, N)
        y_axes (Tensor): (B, 3, N)
        z_axes (Tensor): (B, 3, N)
        knn_lrf_points (Tensor): (B, 3, N, K)
    """
    knn_indices = knn(points, points, num_neighbors, transposed=True)  # (B, N, k)
    knn_points = group_gather(points, knn_indices)  # (B, 3, N, k)

    origins = points.unsqueeze(-1)  # (B, 3, N, 1)
    z_axes = F.normalize(normals, dim=1, p=2).unsqueeze(-1)  # (B, 3, N, 1)

    knn_offsets = knn_points - origins  # (B, 3, N, k)
    knn_distances = torch.linalg.norm(knn_offsets, dim=1, keepdims=True)  # (B, 1, N, k)
    knn_z_coords = (knn_offsets * z_axes).sum(dim=1, keepdims=True)  # (B, 1, N, k)
    knn_plane_offsets = knn_offsets - knn_z_coords * z_axes  # (B, 3, N, k)
    knn_weights = torch.abs(knn_z_coords) / (knn_distances + 1e-10)  # (B, 1, N, k)
    knn_weights = knn_weights / knn_weights.sum(dim=-1, keepdims=True)  # (B, 1, N, k)
    x_axes = F.normalize((knn_plane_offsets * knn_weights).sum(dim=-1, keepdims=True), dim=1, p=2)  # (B, 3, N, 1)
    y_axes = torch.cross(x_axes, z_axes, dim=1)  # (B, 3, N, 1)

    knn_x_coords = (knn_offsets * x_axes).sum(dim=1, keepdims=True)  # (B, 1, N, k)
    knn_y_coords = (knn_offsets * y_axes).sum(dim=1, keepdims=True)  # (B, 1, N, k)

    knn_lrf_points = torch.cat([knn_x_coords, knn_y_coords, knn_z_coords], dim=1)  # (B, 3, N, k)

    if return_axes:
        x_axes = x_axes.squeeze(-1)
        y_axes = y_axes.squeeze(-1)
        z_axes = z_axes.squeeze(-1)
        return x_axes, y_axes, z_axes, knn_lrf_points
    else:
        return knn_lrf_points
