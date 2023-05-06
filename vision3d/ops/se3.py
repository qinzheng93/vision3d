from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def apply_transform(
    points: Tensor, transform: Tensor, normals: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are three cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points are automatically broadcast if B=1.
    3. points and normals are (B, 3), transform is (B, 4, 4), the output points are (B, 3).
       In this case, the points are automatically broadcast to (B, 1, 3) and the transform is applied batch-wise. The
       first dim of points/normals and transform must be the same.

    Args:
        points (Tensor): (*, 3) or (B, N, 3) or (B, 3).
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    assert transform.dim() == 2 or (
        transform.dim() == 3 and points.dim() in [2, 3]
    ), f"Incompatible shapes between points {tuple(points.shape)} and transform {tuple(transform.shape)}."

    if normals is not None:
        assert (
            points.shape == normals.shape
        ), f"The shapes of points {tuple(points.shape)} and normals {tuple(normals.shape)} must be the same."

    if transform.dim() == 2:
        # case 1: (*, 3) x (4, 4)
        input_shape = points.shape
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*input_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*input_shape)
    elif transform.dim() == 3 and points.dim() == 3:
        # case 2: (B, N, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    elif transform.dim() == 3 and points.dim() == 2:
        # case 3: (B, 3) x (B, 4, 4)
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points.unsqueeze(1)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.squeeze(1)
        if normals is not None:
            normals = normals.unsqueeze(1)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.squeeze(1)

    if normals is not None:
        return points, normals

    return points


def get_rotation_translation_from_transform(transform: Tensor) -> Tuple[Tensor, Tensor]:
    """Decompose transformation matrix into rotation matrix and translation vector.

    Args:
        transform (Tensor): (*, 4, 4)

    Returns:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)
    """
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return rotation, translation


def get_transform_from_rotation_translation(rotation: Tensor, translation: Tensor) -> Tensor:
    """Compose transformation matrix from rotation matrix and translation vector.

    Args:
        rotation (Tensor): (*, 3, 3)
        translation (Tensor): (*, 3)

    Returns:
        transform (Tensor): (*, 4, 4)
    """
    input_shape = rotation.shape
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)
    transform = torch.eye(4).to(rotation).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    output_shape = input_shape[:-2] + (4, 4)
    transform = transform.view(*output_shape)
    return transform


def inverse_transform(transform: Tensor) -> Tensor:
    """Inverse rigid transform.

    Args:
        transform (Tensor): (*, 4, 4)

    Return:
        inv_transform (Tensor): (*, 4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (*, 3, 3), (*, 3)
    inv_rotation = rotation.transpose(-1, -2)  # (*, 3, 3)
    inv_translation = -torch.matmul(inv_rotation, translation.unsqueeze(-1)).squeeze(-1)  # (*, 3)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (*, 4, 4)
    return inv_transform
