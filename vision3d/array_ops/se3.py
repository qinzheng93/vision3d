from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray


def apply_transform(
    points: ndarray, transform: ndarray, normals: Optional[ndarray] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    return points


def compose_transforms(*transforms: ndarray) -> ndarray:
    """
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation: Optional[ndarray], translation: Optional[ndarray]) -> ndarray:
    """Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: ndarray) -> Tuple[ndarray, ndarray]:
    """Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: ndarray) -> ndarray:
    """Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform
