import torch

from .index_select import index_select


def nearest_interpolate_pack_mode(inputs, neighbor_indices):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        inputs: [n1, d] features matrix
        neighbor_indices: [n2, max_num] Only the first column is used for pooling

    Returns:
        outputs: [n2, d] pooled features matrix
    """
    padded_inputs = torch.cat([inputs, torch.zeros_like(inputs[:1])], dim=0)
    outputs = index_select(padded_inputs, neighbor_indices[:, 0], dim=0)
    return outputs
