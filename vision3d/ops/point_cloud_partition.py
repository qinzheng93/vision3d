from typing import Optional, Tuple

import torch
from torch import Tensor

from vision3d.utils.misc import deprecated

from .index_select import index_select
from .knn import knn
from .pairwise_distance import pairwise_distance


@deprecated("point_to_node_partition")
def get_point_to_node_indices(points: Tensor, nodes: Tensor, return_counts: bool = False):
    """Compute Point-to-Node partition indices of the point cloud.

    Distribute points to the nearest node. Each point is is_distributed to only one node.

    Args:
        points (Tensor): point cloud (N, C)
        nodes (Tensor): node set (M, C)
        return_counts (bool=False): whether return the number of points in each node.

    Returns:
        indices (LongTensor): index of the node that each point belongs to (N,)
        node_sizes (longTensor): the number of points in each node.
    """
    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.min(dim=1)[1]

    if return_counts:
        unique_indices, unique_counts = torch.unique(indices, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes

    return indices


@torch.no_grad()
def point_to_node_partition(
    points: Tensor,
    nodes: Tensor,
    point_limit: Optional[int] = None,
    return_count: bool = False,
    gather_points: bool = True,
    inf: float = 1e12,
) -> Tuple[Tensor, ...]:
    """Point-to-Node partition to the point cloud.

    Fixed knn bug.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): The max number of points to each node.
        return_count (bool=False): If True, return the sizes of the nodes.
        gather_points (bool=True): If True, gather points for each node.
        inf (float=1e12): safe number

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): Only if return_count (M,)
        node_masks (BoolTensor): Only if gather_points (M,)
        node_knn_indices (LongTensor): Only if gather_points (M, K)
        node_knn_masks (BoolTensor): Only if gather_points (M, K)
    """
    return_list = []

    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    return_list.append(point_to_node)

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return_list.append(node_sizes)

    if gather_points:
        node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
        node_masks.index_fill_(0, point_to_node, True)
        return_list.append(node_masks)

        matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
        point_indices = torch.arange(points.shape[0]).cuda()  # (N,)
        matching_masks[point_to_node, point_indices] = True  # (M, N)
        sq_dist_mat.masked_fill_(~matching_masks, inf)  # (M, N)

        max_points_per_node = matching_masks.sum(dim=1).max().item()
        if point_limit is not None:
            max_points_per_node = min(max_points_per_node, point_limit)
        assert max_points_per_node > 0, "All nodes are empty."

        node_knn_indices = sq_dist_mat.topk(k=max_points_per_node, dim=1, largest=False)[1]  # (M, K)
        node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
        node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, max_points_per_node)  # (M, K)
        node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
        node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])
        return_list.append(node_knn_indices)
        return_list.append(node_knn_masks)

    return tuple(return_list)


@torch.no_grad()
def ball_query_partition(
    points: Tensor,
    nodes: Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
) -> Tuple[Tensor, ...]:
    node_knn_distances, node_knn_indices = knn(nodes, points, point_limit, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    node_knn_indices[~node_knn_masks] = points.shape[0]
    node_sizes = node_knn_masks.sum(1)  # (N,)
    node_masks = torch.gt(node_sizes, 0)  # (N,)

    if return_count:
        return node_knn_indices, node_knn_masks, node_masks, node_sizes

    return node_knn_indices, node_knn_masks, node_masks
