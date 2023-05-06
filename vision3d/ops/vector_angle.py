import torch
from torch import Tensor
import numpy as np


def rad2deg(rad: Tensor) -> Tensor:
    factor = 180.0 / np.pi
    deg = rad * factor
    return deg


def deg2rad(deg: Tensor) -> Tensor:
    factor = np.pi / 180.0
    rad = deg * factor
    return rad


def vector_angle(x: Tensor, y: Tensor, dim: int, use_degree: bool = False) -> Tensor:
    """Compute the angles between two set of 3D vectors.

    Args:
        x (Tensor): set of vectors (*, 3, *)
        y (Tensor): set of vectors (*, 3, *).
        dim (int): dimension index of the coordinates.
        use_degree (bool=False): If True, return angles in degree instead of rad.

    Returns:
        angle (Tensor): (*)
    """
    cross = torch.linalg.norm(torch.cross(x, y, dim=dim), dim=dim)  # (*, 3 *) x (*, 3, *) -> (*, 3, *) -> (*)
    dot = torch.sum(x * y, dim=dim)  # (*, 3 *) x (*, 3, *) -> (*)
    angle = torch.atan2(cross, dot)  # (*)
    if use_degree:
        angle = rad2deg(angle)
    return angle
