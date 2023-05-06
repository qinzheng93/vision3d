import torch
from torch import Tensor


def pairwise_distance(
    x: Tensor,
    y: Tensor,
    normalized: bool = False,
    transposed: bool = False,
    squared: bool = True,
    strict: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): the row tensor in the shape of (*, N, C).
        y (Tensor): the column tensor in the shape of (*, M, C).
        normalized (bool): if the points are normalized, we have "x^2 + y^2 = 1", so "d^2 = 2 - 2xy". Default: False.
        transposed (bool): if True, x and y are in the shapes of (*, C, N) and (*, C, M) respectively. Default: False.
        squared (bool): if True, return squared distance. Default: True.
        strict (bool): if True, use strict mode to guarantee precision. Default: False.
        eps (float): a safe number for sqrt. Default: 1e-8.

    Returns:
        dist: Tensor (*, N, M)
    """
    if strict:
        if transposed:
            displacements = x.unsqueeze(-1) - y.unsqueeze(-2)  # (*, C, N, 1) x (*, C, 1, M) -> (*, C, N, M)
            distances = torch.linalg.norm(displacements, dim=-3)  # (*, C, N, M) -> (*, N, M)
        else:
            displacements = x.unsqueeze(-2) - y.unsqueeze(-3)  # (*, N, 1, C) x (*, 1, M, C) -> (*, N, M, C)
            distances = torch.linalg.norm(displacements, dim=-1)  # (*, N, M, C) -> (*, N, M)

        if squared:
            distances = distances.pow(2)
    else:
        if transposed:
            channel_dim = -2
            xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
        else:
            channel_dim = -1
            xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]

        if normalized:
            distances = 2.0 - 2.0 * xy
        else:
            x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
            y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
            distances = x2 - 2 * xy + y2

        distances = distances.clamp(min=0.0)

        if not squared:
            distances = torch.sqrt(distances + eps)

    return distances
