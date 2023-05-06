from typing import Optional

import ipdb
import torch
from torch import Tensor


def volume_render(
    ray_directions: Tensor,
    z_values: Tensor,
    point_sigmas: Tensor,
    point_colors: Optional[Tensor] = None,
    sigma_noise: Optional[float] = None,
    opacity_only: bool = False,
    white_background: bool = False,
    deterministic: bool = False,
    eps: float = 1e-10,
):
    """Volumetric Rendering.

    Note:
        1. Depth predicted here is not the strict "z" values, but the euclidean distances from the camera center.

    Args:
        ray_directions (tensor): the view directions of the rays in the shape of (N, 3).
        z_values (tensor): the depth values of the sampled points on the rays in the shape of (N, M).
        point_sigmas (tensor): the sigma values of the sampled points on the rays in the shape of (N, M).
        point_colors (tensor, optional): the color values of the sampled points on the rays in the shape of (N, M, 3).
        sigma_noise (float, optional): the noise on the sigma values of the sampled points.
        opacity_only (bool): If True, only return weights. Default: False.
        white_background (bool): If True, the background is white. Default: False.
        deterministic (bool): If True, do not apply noise. Default: False.
        eps (float): a safe number for multiplication. Default: 1e-10.

    Returns:
        A tensor of the rendered ray colors in the shape of (N, 3).
        A tensor of the rendered ray depths in the shape of (N).
        A tensor of the rendered point opacities of each sampled point in the shape of (N, M).

    """
    deltas = z_values[..., 1:] - z_values[..., :-1]  # (N, M-1)
    delta_inf = 1e10 * torch.ones_like(deltas[..., :1])  # (N, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], dim=-1)  # (N, M)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.linalg.norm(ray_directions, dim=-1).unsqueeze(1)

    if (not deterministic) and (sigma_noise is not None and sigma_noise > 0.0):
        point_sigmas = point_sigmas + torch.randn(point_sigmas.shape).cuda() * sigma_noise

    # compute alpha by the formula (3)
    alphas = 1.0 - torch.exp(-deltas * torch.relu(point_sigmas))  # (N, M)
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1.0 - alphas + eps], -1)  # [1, a1, a2, ...]
    point_opacities = alphas * torch.cumprod(alphas_shifted, -1)[..., :-1]  # (N, M)
    if opacity_only:
        return point_opacities

    # compute the accumulated opacity along the rays, equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    ray_opacities = point_opacities.sum(-1)  # (N)

    # compute final depth
    ray_depths = torch.sum(point_opacities * z_values, dim=-1)  # (N)

    # compute final weighted outputs
    if point_colors is not None:
        ray_colors = torch.sum(point_opacities.unsqueeze(-1) * point_colors, dim=-2)  # (N, 3)

        if white_background:
            ray_colors = ray_colors + 1.0 - ray_opacities.unsqueeze(-1)

        return ray_colors, ray_depths, point_opacities

    return ray_depths, point_opacities
