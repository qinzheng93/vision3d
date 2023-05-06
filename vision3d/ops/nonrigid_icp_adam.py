import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from vision3d.ops import apply_deformation, apply_rotation, get_transform_from_rotation_translation


def compute_landmark_cost(
    src_nodes: Tensor,
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    transforms: Tensor,
):
    warped_src_corr_points = apply_deformation(src_corr_points, src_nodes, transforms, anchor_indices, anchor_weights)
    cost = (warped_src_corr_points - tgt_corr_points).pow(2).sum(1).mean()
    return cost


def compute_arap_cost(
    src_nodes: Tensor, node_edges: Tensor, node_edge_weights: Tensor, rotations: Tensor, translations: Tensor
):
    anchor_indices = node_edges[:, 0]
    neighbor_indices = node_edges[:, 1]
    anchor_nodes = src_nodes[anchor_indices]
    neighbor_nodes = src_nodes[neighbor_indices]
    anchor_rotations = rotations[anchor_indices]
    anchor_translations = translations[anchor_indices]
    neighbor_translations = translations[neighbor_indices]
    displacements = (neighbor_nodes - anchor_nodes).unsqueeze(1)
    warped_neighbor_nodes = apply_rotation(displacements, anchor_rotations).squeeze(1)
    warped_neighbor_nodes = warped_neighbor_nodes + anchor_nodes + anchor_translations
    tgt_neighbor_nodes = neighbor_nodes + neighbor_translations
    cost = ((warped_neighbor_nodes - tgt_neighbor_nodes).pow(2).sum(1) * node_edge_weights).mean()
    return cost


def compute_orthogonal_cost(rotations: Tensor):
    columns1 = rotations[:, :, 0]  # (N, 3)
    columns2 = rotations[:, :, 1]  # (N, 3)
    columns3 = rotations[:, :, 2]  # (N, 3)
    orthogonal12 = (columns1 * columns2).sum(1)  # (N)
    orthogonal13 = (columns1 * columns3).sum(1)  # (N)
    orthogonal23 = (columns2 * columns3).sum(1)  # (N)
    unit1 = (columns1 * columns1).sum(1) - 1
    unit2 = (columns2 * columns2).sum(1) - 1
    unit3 = (columns3 * columns3).sum(1) - 1
    cost = orthogonal12 ** 2 + orthogonal13 ** 2 + orthogonal23 ** 2 + unit1 ** 2 + unit2 ** 2 + unit3 ** 2
    cost = cost.mean()
    return cost


def compute_cost_function(
    rotations: Tensor,
    translations: Tensor,
    src_nodes: Tensor,
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    node_edges: Tensor,
    node_edge_weights: Tensor,
):
    transforms = get_transform_from_rotation_translation(rotations, translations)
    landmark_cost = compute_landmark_cost(
        src_nodes, src_corr_points, tgt_corr_points, anchor_indices, anchor_weights, transforms
    )
    arap_cost = compute_arap_cost(src_nodes, node_edges, node_edge_weights, rotations, translations)
    orthogonal_cost = compute_orthogonal_cost(rotations)
    cost = 1.0 * landmark_cost + 0.1 * arap_cost + 0.1 * orthogonal_cost
    return cost


def non_rigid_icp_adam(
    src_nodes: Tensor,
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    node_edges: Tensor,
    node_edge_weights: Tensor,
    num_iterations: int = 500,
):
    """Non-rigid ICP based on Embedded Deformation for non-rigid registration with Adam solver.

    Args:
        src_nodes (Tensor): The nodes in the source point cloud in the shape of (M, 3).
        src_corr_points (Tensor): The correspondences in the source point cloud in the shape of (N, 3).
        tgt_corr_points (Tensor): The correspondences in the target point cloud in the shape of (N, 3).
        anchor_indices (LongTensor): The indices of the anchor nodes for each correspondence in the shape of (N, K).
        anchor_weights (Tensor): The weights of the anchor nodes for each correspondence in the shape of (N, K).
        node_edges (LongTensor): The node indices of the edges in the shape of (E, 2)
        node_edge_weights (Tensor): The weights of the edges in the shape of (E,).
        num_iterations (int=400): The number of Adam iterations.

    Returns:
        transforms (Tensor): The transformations for each node in the shape of (M, 4, 4).
    """
    with torch.enable_grad():
        rotations = torch.eye(3).unsqueeze(0).repeat(src_nodes.shape[0], 1, 1).cuda()
        translations = torch.zeros_like(src_nodes).cuda()
        rotations = nn.Parameter(rotations)
        translations = nn.Parameter(translations)

        optimizer = optim.Adam([rotations, translations], lr=0.5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        optimizer.zero_grad()

        for i in range(num_iterations):
            cost = compute_cost_function(
                rotations,
                translations,
                src_nodes,
                src_corr_points,
                tgt_corr_points,
                anchor_indices,
                anchor_weights,
                node_edges,
                node_edge_weights,
            )

            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    transforms = get_transform_from_rotation_translation(rotations, translations)

    return transforms
