from typing import Tuple

import torch
from torch import Tensor

from vision3d.utils.misc import load_ext


ext_module = load_ext("vision3d.ext", ["three_nn"])


def three_nn(q_points: Tensor, s_points: Tensor, transposed: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Compute the three nearest neighbors for the query points in the support points.

    Args:
        q_points (Tensor): The query points in shape of (B, N, 3) or (B, 3, N) if transposed.
        s_points (Tensor): The support points in shape of (B, M, 3) or (B, 3, M) if transposed.
        transposed (bool=True): If True, the points shape is (B, 3, N).

    Returns:
        A Tensor of the distances of the 3-NN in the shape of (B, N, 3).
        A LongTensor of the indices of the 3-NN in the shape of (B, N, 3).
    """
    if transposed:
        q_points = q_points.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        s_points = s_points.transpose(1, 2)  # (B, M, 3) -> (B, 3, M)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    assert s_points.shape[1] >= 3, "s_points must have at least 3 points."

    tnn_distances = q_points.new_zeros(size=(q_points.shape[0], q_points.shape[1], 3))  # (B, N, 3)
    tnn_indices = torch.zeros(size=(q_points.shape[0], q_points.shape[1], 3), dtype=torch.long).cuda()  # (B, N, 3)

    ext_module.three_nn(q_points, s_points, tnn_distances, tnn_indices)

    return tnn_distances, tnn_indices
