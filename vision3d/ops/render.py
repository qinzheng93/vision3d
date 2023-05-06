from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from .se3 import apply_transform


def render(
    points: Tensor,
    intrinsics: Tensor,
    extrinsics: Optional[Tensor] = None,
    rounding: bool = True,
    return_depth: bool = False,
    eps: float = 1e-8,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Render points into image according to intrinsics and extrinsic.

    The points and extrinsics should be broadcastable in the first dim.

    Args:
        points (Tensor): point cloud (N, 3) or (B, N, 3).
        intrinsics (Tensor): intrinsic matrix (3, 3) or (B, 3, 3).
        extrinsics (Tensor, optional): extrinsic matrix (4, 4) or (B, 4, 4).
        rounding (bool): if True, convert the pixels to long dtype. Default: True.
        return_depth (bool): if True, return depth values. Default: True.

    Returns:
        A Tensor of the pixel cloud [height, width] in the shape of (N, 2) of (B, N, 2).
        A Tensor of the depth values in the shape of (N) of (B, N).
    """
    assert points.dim() == intrinsics.dim()

    if extrinsics is not None:
        assert points.dim() == extrinsics.dim()
        points = apply_transform(points, extrinsics)  # (N, 3) or (B, N, 3)

    x_coords = points[..., 0]  # (N) or (B, N)
    y_coords = points[..., 1]  # (N) or (B, N)
    z_coords = points[..., 2]  # (N) or (B, N)

    focal_x = intrinsics[..., 0, 0].unsqueeze(-1)  # (1) or (B, 1)
    focal_y = intrinsics[..., 1, 1].unsqueeze(-1)  # (1) or (B, 1)
    center_x = intrinsics[..., 0, 2].unsqueeze(-1)  # (1) or (B, 1)
    center_y = intrinsics[..., 1, 2].unsqueeze(-1)  # (1) or (B, 1)

    w_coords = focal_x * x_coords / z_coords.clamp(min=eps) + center_x  # (N) or (B, N)
    h_coords = focal_y * y_coords / z_coords.clamp(min=eps) + center_y  # (N) or (B, N)
    if rounding:
        w_coords = w_coords.long()
        h_coords = h_coords.long()
    pixels = torch.stack([h_coords, w_coords], dim=-1)  # (N, 2) or (B, N, 2)

    if return_depth:
        return pixels, z_coords  # (N, 2) or (B, N, 2), (N) or (B, N)

    return pixels  # (N, 2) or (B, N, 2)


def mask_pixels_with_image_size(pixels: Tensor, image_h: int, image_w: int) -> Tensor:
    """Compute the masks of the pixels which are within the range of an image.

    Args:
        pixels (Tensor): the pixels in the shape of (..., 2). Note that the pixels are represented as (h, w).
        image_h (int): the height of the image.
        image_w (int): the height of the image.

    Returns:
        A BoolTensor of the masks of the pixels in the shape of (..., 2). A pixel is with the image if True.
    """
    masks = torch.logical_and(
        torch.logical_and(torch.ge(pixels[..., 0], 0), torch.lt(pixels[..., 0], image_h)),
        torch.logical_and(torch.ge(pixels[..., 1], 0), torch.lt(pixels[..., 1], image_w)),
    )
    return masks
