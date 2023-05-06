import torch
from torch import Tensor

from .se3 import apply_transform


def apply_embedded_deformation_v0(
    points: Tensor, nodes: Tensor, transforms: Tensor, weights: Tensor, eps: float = 1e-6
) -> Tensor:
    """Apply Embedded Deformation Warping Function.

    Args:
        points (Tensor): The points to warp in the shape of (N, 3).
        nodes (Tensor): The graph nodes in the shape of (M, 3).
        transforms (Tensor): The associated transformations for each node in the shape of (M, 4, 4).
        weights (Tensor): The skinning weights between the points and the nodes in the shape of (N, M).
        eps (float=1e-6): Safe number for division.

    Returns:
        warped_points (Tensor): The warped points in the shape of (N, 3).
    """
    points = points.unsqueeze(0)  # (1, N, 3)
    nodes = nodes.unsqueeze(1)  # (M, 1, 3)
    warped_points = apply_transform(points - nodes, transforms) + nodes  # (M, N, 3)
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # (N, M)
    weights = weights.transpose(0, 1).unsqueeze(-1)  # (M, N, 1)
    warped_points = torch.sum(warped_points * weights, dim=0)  # (N, 3)
    return warped_points


def apply_deformation(
    points: Tensor,
    nodes: Tensor,
    transforms: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Apply Embedded Deformation Warping Function.

    Args:
        points (Tensor): The points to warp in the shape of (N, 3).
        nodes (Tensor): The graph nodes in the shape of (M, 3).
        transforms (Tensor): The associated transformations for each node in the shape of (M, 4, 4).
        anchor_indices (LongTensor): The indices of the anchor nodes for the points in the shape of (N, K). If an index
            is -1, the corresponding anchor does not exist.
        anchor_weights (Tensor): The skinning weights of the anchor nodes for the point in the shape of (N, K).
        eps (float=1e-6): A safe number for division.

    Returns:
        warped_points (Tensor): The warped points in the shape of (N, 3).
    """
    anchor_weights = anchor_weights / (anchor_weights.sum(dim=1, keepdim=True) + eps)  # (N, K)
    anchor_masks = torch.ne(anchor_indices, -1)  # (N, K)
    p_indices, col_indices = torch.nonzero(anchor_masks, as_tuple=True)  # (C), (C)
    n_indices = anchor_indices[p_indices, col_indices]  # (C)
    weights = anchor_weights[p_indices, col_indices]  # (C)
    sel_points = points[p_indices]  # (C, 3)
    sel_nodes = nodes[n_indices]  # (C, 3)
    sel_transforms = transforms[n_indices]  # (C, 4, 4)
    sel_warped_points = apply_transform(sel_points - sel_nodes, sel_transforms) + sel_nodes  # (C, 3)
    sel_warped_points = sel_warped_points * weights.unsqueeze(1)  # (C, 3)
    warped_points = torch.zeros_like(points)  # (N, 3)
    p_indices = p_indices.unsqueeze(1).expand_as(sel_warped_points)  # (C, 3)
    warped_points.scatter_add_(dim=0, index=p_indices, src=sel_warped_points)  # (N, 3)
    return warped_points
