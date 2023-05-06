from typing import Optional

import torch.nn as nn
from torch import Tensor

from vision3d.ops import apply_transform, get_rotation_translation_from_transform


def as_rigid_as_possible_loss(
    nodes: Tensor, transforms: Tensor, edge_indices: Tensor, edge_weights: Optional[Tensor] = None
) -> Tensor:
    """As-rigid-as-possible loss.

    Args:
        nodes (Tensor): the nodes in the point clouds in the shape of (V, 3).
        transforms (Tensor): the corresponding transformations for the nodes in the shape of (V, 4, 4).
        edge_indices (LongTensor): the edge list between the nodes in the shape of (E, 2).
        edge_weights (Tensor, optional): the weights of the edges in the shape of (E).

    Returns:
        A Tensor of the as-rigid-as-possible-loss.
    """
    rotations, translations = get_rotation_translation_from_transform(transforms)  # (V, 3, 3), (V, 3)
    anc_indices = edge_indices[:, 0]  # (E)
    ref_indices = edge_indices[:, 1]  # (E)
    anc_nodes = nodes[anc_indices]  # (E, 3)
    ref_nodes = nodes[ref_indices]  # (E, 3)
    anc_transforms = transforms[anc_indices]  # (E, 4, 4)
    tgt_ref_nodes = translations[ref_indices] + ref_nodes  # (E, 3)
    warped_ref_nodes = apply_transform(ref_nodes - anc_nodes, anc_transforms) + anc_nodes
    loss_values = (warped_ref_nodes - tgt_ref_nodes).pow(2).sum(1)
    if edge_weights is not None:
        loss = (loss_values * edge_weights).mean()
    else:
        loss = loss_values.mean()
    return loss


class AsRigidAsPossibleLoss(nn.Module):
    @staticmethod
    def forward(nodes: Tensor, transforms: Tensor, edges: Tensor, edge_weights: Optional[Tensor] = None):
        loss = as_rigid_as_possible_loss(nodes, transforms, edges, edge_weights=edge_weights)
        return loss
