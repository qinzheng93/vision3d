import torch
from torch import Tensor

from vision3d.utils.misc import load_ext


ext_module = load_ext("vision3d.ext", ["ball_query"])


def ball_query(
    q_points: Tensor,
    s_points: Tensor,
    num_samples: int,
    radius: float,
    transposed: bool = False,
) -> Tensor:
    """Ball query.

    The first `num_samples` support points found in the local ball around each query points are returned.

    Args:
        q_points (Tensor): The query points in the shape of (B, N, 3).
        s_points (Tensor): The support points in the shape of (B, M, 3).
        num_samples (int): max neighbors.
        radius (float): search radius.
        transposed (bool=True): If True, the point tensor shapes are (B, 3, N) and (B, 3, M).

    Returns:
        A LongTensor of the indices of the sampled points in the shape of (B, M, K).
    """
    if transposed:
        q_points = q_points.transpose(1, 2)  # (B, 3, N) -> (B, N, 3)
        s_points = s_points.transpose(1, 2)  # (B, 3, M) -> (B, M, 3)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    # (B, M, K)
    indices = torch.zeros(size=(q_points.shape[0], q_points.shape[1], num_samples), dtype=torch.long).cuda()

    ext_module.ball_query(q_points, s_points, indices, radius, num_samples)

    return indices
