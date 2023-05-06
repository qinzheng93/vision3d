from typing import Tuple

import torch
from torch import Tensor

from .knn import knn
from .pairwise_distance import pairwise_distance


def compute_skinning_weights(distances: Tensor, node_coverage: float) -> Tensor:
    """Skinning weight proposed in DynamicFusion.

    w = exp(-d^2 / (2 * r^2))

    Args:
        distances (Tensor): The distance tensor in arbitrary shape.
        node_coverage (float): The node coverage.

    Returns:
        weights (Tensor): The skinning weights in arbitrary shape.
    """
    weights = torch.exp(-(distances ** 2) / (2.0 * node_coverage ** 2))
    return weights


def build_euclidean_deformation_graph(
    points: Tensor,
    nodes: Tensor,
    num_anchors: int,
    node_coverage: float,
    return_point_anchor: bool = True,
    return_node_graph: bool = True,
    return_distance: bool = False,
    return_adjacent_matrix: bool = False,
    eps: float = 1e-6,
) -> Tuple[Tensor, ...]:
    """Build deformation graph with euclidean distance.

    We use the method proposed in Embedded Deformation to construct the deformation graph:
        1. Each point is assigned to its k-nearest nodes.
        2. If two nodes cover the same point, there is an edge between them.
        3. We compute the skinning weights following DynamicFusion.

    Args:
        points (Tensor): the points in the shape of (N, 3).
        nodes (Tensor): the nodes in the shape of (M, 3).
        num_anchors (int): the number of anchors for each point.
        node_coverage (float): the node coverage to compute skinning weights.
        return_point_anchor (bool): if True, return the anchors for the points. Default: True.
        return_node_graph (bool): if True, return the edges between the nodes. Default: True.
        return_distance (bool): if True, return the distance. Default: False.
        return_adjacent_matrix (bool): if True, return the adjacent matrix instead of edge list. Default: False.
            Only take effect when 'return_node_graph' is True.
        eps (float): A safe number for division. Default: 1e-6.

    Returns:
        A LongTensor of the anchor node indices for the points in the shape of (N, K).
        A Tensor of the anchor node weights for the points in the shape of (N, K).
        A Tensor of the anchor node distances for the points in the shape of (N, K).
        A LongTensor of the endpoint indices of the edges in the shape of (E, 2).
        A Tensor of the weights of the edges in the shape of (E).
        A Tensor of the distances of the edges in the shape of (E).
        A BoolTensor of the adjacent matrix between nodes in the shape of (M, M).
        A Tensor of the skinning weight matrix between nodes of (M, M).
        A Tensor of the distance matrix between nodes of (M, M).
    """
    output_list = []

    anchor_distances, anchor_indices = knn(points, nodes, num_anchors, return_distance=True)  # (N, K)
    anchor_weights = compute_skinning_weights(anchor_distances, node_coverage)  # (N, K)
    anchor_weights = anchor_weights / anchor_weights.sum(dim=1, keepdim=True)  # (N, K)

    if return_point_anchor:
        output_list.append(anchor_indices)
        output_list.append(anchor_weights)
        if return_distance:
            output_list.append(anchor_distances)

    if return_node_graph:
        point_indices = torch.arange(points.shape[0]).cuda().unsqueeze(1).expand_as(anchor_indices)  # (N, K)
        node_to_point = torch.zeros(size=(nodes.shape[0], points.shape[0])).cuda()  # (N, M)
        node_to_point[anchor_indices, point_indices] = 1.0
        adjacent_mat = torch.gt(torch.einsum("nk,mk->nm", node_to_point, node_to_point), 0)
        distance_mat = pairwise_distance(nodes, nodes, squared=False)
        weight_mat = compute_skinning_weights(distance_mat, node_coverage)
        weight_mat = weight_mat * adjacent_mat.float()
        weight_mat = weight_mat / weight_mat.sum(dim=-1, keepdim=True).clamp(min=eps)
        if return_adjacent_matrix:
            output_list.append(adjacent_mat)
            output_list.append(weight_mat)
            if return_distance:
                distance_mat = distance_mat * adjacent_mat.float()
                output_list.append(distance_mat)
        else:
            edge_indices = torch.nonzero(adjacent_mat).contiguous()
            edge_weights = weight_mat[adjacent_mat].contiguous()
            output_list.append(edge_indices)
            output_list.append(edge_weights)
            if return_distance:
                edge_distances = distance_mat[adjacent_mat].contiguous()
                output_list.append(edge_distances)

    return tuple(output_list)
