from typing import Optional

import ipdb
import torch
from torch import Tensor

from vision3d.ops.se3 import get_transform_from_rotation_translation


def weighted_procrustes(
    src_points: Tensor,
    tgt_points: Tensor,
    weights: Optional[Tensor] = None,
    weight_threshold: float = 0.0,
    eps: float = 1e-5,
):
    """Compute rigid transformation from `src_points` to `tgt_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points (Tensor): (B, N, 3) or (N, 3)
        tgt_points (Tensor): (B, N, 3) or (N, 3)
        weights (Tensor, optional): (B, N) or (N,).
        weight_threshold (float): Default: 0.
        eps (float): Default: 1e-5.

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.dim() == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_threshold), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    tgt_centroid = torch.sum(tgt_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    tgt_points_centered = tgt_points - tgt_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * tgt_points_centered)

    if torch.isnan(H).any():
        ipdb.set_trace()

    U, _, V = torch.svd(H.cpu())  # H = USV^T
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    rotation = V @ eye @ Ut

    translation = tgt_centroid.permute(0, 2, 1) - rotation @ src_centroid.permute(0, 2, 1)
    translation = translation.squeeze(2)

    transform = get_transform_from_rotation_translation(rotation, translation)
    if squeeze_first:
        transform = transform.squeeze(0)
    return transform
