import torch
from torch import Tensor

from .vector_angle import vector_angle
from .group_gather import group_gather


def local_ppf(
    q_points: Tensor,
    s_points: Tensor,
    q_normals: Tensor,
    s_normals: Tensor,
    neighbor_indices: Tensor,
    use_absolute_position: bool = False,
    use_relative_position: bool = False,
    use_degree: bool = False,
) -> Tensor:
    """Extract Point Pair Features for points in local region.

    The point pair features are in order: [<na, d>, <nr, d>, <na, nr>, ||d||]

    The output features are in order: [absolute_position, relative_position, point_pair_features]

    Args:
        q_points (Tensor): query point cloud to compute PPF. (B, 3, N)
        q_normals (Tensor): query normals of the point cloud. (B, 3, N)
        s_points (Tensor): support point cloud to compute PPF. (B, 3, N)
        s_normals (Tensor): support normals of the point cloud. (B, 3, N)
        neighbor_indices (Tensor): the indices of the neighbors for each point. (B, N, K)
        use_absolute_position (bool=False): If True, concat absolute position of the points.
        use_relative_position (bool=False): If True, concat relative position of the neighbors.
        use_degree (bool=False): If True, return angles in degree instead of rad.

    Returns:
        local_features (Tensor): output features. (B, 4, N, K) or (B, 7, N, K) or (B, 10, N, K)
    """
    num_neighbors = neighbor_indices.shape[-1]

    # 1. construct local region
    neighbor_points = group_gather(s_points, neighbor_indices)  # (B, 3, N, K)
    neighbor_normals = group_gather(s_normals, neighbor_indices)  # (B, 3, N, K)

    # 2. coordinates
    anchor_points = q_points.unsqueeze(3).expand(-1, -1, -1, num_neighbors)  # (B, 3, N, K)
    anchor_normals = q_normals.unsqueeze(3).expand(-1, -1, -1, num_neighbors)  # (B, 3, N, K)
    neighbor_offsets = neighbor_points - anchor_points  # (B, 3, N, K)

    # 3. point pair features
    ppf_0 = vector_angle(anchor_normals, neighbor_offsets, dim=1, use_degree=use_degree)  # <n0, d>
    ppf_1 = vector_angle(neighbor_normals, neighbor_offsets, dim=1, use_degree=use_degree)  # <n1, d>
    ppf_2 = vector_angle(anchor_normals, neighbor_normals, dim=1, use_degree=use_degree)  # <n0, n1>
    ppf_3 = torch.linalg.norm(neighbor_offsets, dim=1)  # ||d||
    features = torch.stack([ppf_0, ppf_1, ppf_2, ppf_3], dim=1)  # (B, 4, N, K)

    # 4. compose features
    if use_relative_position:
        features = torch.cat([neighbor_offsets, features], dim=1)
    if use_absolute_position:
        features = torch.cat([anchor_points, features], dim=1)

    return features


def global_ppf(points: Tensor, normals: Tensor, use_degree: bool = False) -> Tensor:
    """Pairwise Point Pair Features.

    Args:
        points (Tensor): (B, N, 3)
        normals (Tensor): (B, N, 3)
        use_degree (bool=False): If True, return angles in degree instead of rad.

    Returns:
        feats (Tensor): (B, N, N, 4)
    """
    pairwise_offsets = points.unsqueeze(2) - points.unsqueeze(1)  # (B, N, N, 3)
    anc_normals = normals.unsqueeze(2).expand_as(pairwise_offsets)  # (B, N, 3) -> (B, N, 1, 3) -> (B, N, N, 3)
    ref_normals = normals.unsqueeze(1).expand_as(pairwise_offsets)  # (B, N, 3) -> (B, 1, N, 3) -> (B, N, N, 3)
    ppf_0 = vector_angle(anc_normals, pairwise_offsets, dim=-1, use_degree=use_degree)  # (B, N, N)
    ppf_1 = vector_angle(ref_normals, pairwise_offsets, dim=-1, use_degree=use_degree)  # (B, N, N)
    ppf_2 = vector_angle(anc_normals, ref_normals, dim=-1, use_degree=use_degree)  # (B, N, N)
    ppf_3 = torch.linalg.norm(pairwise_offsets, dim=-1)  # (B, N, N)
    features = torch.stack([ppf_0, ppf_1, ppf_2, ppf_3], dim=-1)
    return features
