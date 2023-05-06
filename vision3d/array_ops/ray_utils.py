from typing import Tuple

import numpy as np
from numpy import ndarray

from .se3 import get_rotation_translation_from_transform
from .so3 import apply_rotation


def get_camera_rays(
    image_h: int,
    image_w: int,
    focal_h: float,
    focal_w: float,
    center_h: float,
    center_w: float,
    convention: str = "opencv",
) -> ndarray:
    """Generate rays in camera coordinate system.

    Args:
        image_h (int): the image height.
        image_w (int): the image width.
        focal_h (float): the focal in y-dimension (height).
        focal_w (float): the focal in x-dimension (width).
        center_h (float): the center in y-dimension (height).
        center_w (float): the center in x-dimension (width).
        convention (str): the convention of camera coordinate system. Default: "opencv".
            1. "opencv": x-right, y-down, z-forward.
            2. "opengl": x-right, y-up, z-backward.

    Returns:
        A float array of the view directions of the rays in the shape of (H, W, 3).
    """
    assert convention in ["opencv", "opengl"]
    u_indices, v_indices = np.meshgrid(np.arange(image_w), np.arange(image_h), indexing="xy")  # (H, W)
    x_vals = (u_indices - center_w) / focal_w  # (H, W)
    y_vals = (v_indices - center_h) / focal_h  # (H, W)
    z_vals = np.ones_like(x_vals)  # (H, W)
    if convention == "opencv":
        ray_directions = np.stack([x_vals, y_vals, z_vals], axis=-1)  # (H, W, 3)
    else:
        ray_directions = np.stack([x_vals, -y_vals, -z_vals], axis=-1)  # (H, W, 3)
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)  # (H, W, 3)
    return ray_directions


def get_world_rays(camera_ray_directions: ndarray, pose: ndarray) -> Tuple[ndarray, ndarray]:
    """Generate rays in world coordinate system from camera pose.

    Args:
        camera_ray_directions (array): the view directions of the rays in camera coordinates in the shape of (H, W, 3).
        pose (array): the camera pose in the shape of (4, 4).

    Returns:
        A float array of the original points of the rays in the shape of (N, 3).
        A float array of the view directions of the rays in the shape of (N, 3).
    """
    camera_ray_directions = camera_ray_directions.reshape(-1, 3)  # (H, W, 3) -> (HxW, 3)
    rotation, translation = get_rotation_translation_from_transform(pose)  # (3, 3), (3)
    ray_directions = apply_rotation(camera_ray_directions, rotation)  # (HxW, 3)
    ray_centers = np.broadcast_to(translation[None, :], ray_directions.shape)  # (3) -> (1, 3) -> (HxW, 3)
    return ray_centers, ray_directions


def batch_get_world_rays(camera_ray_directions: ndarray, poses: ndarray) -> Tuple[ndarray, ndarray]:
    """Generate rays in world coordinate system from camera pose in batch.

    Args:
        camera_ray_directions (array): the view directions of the rays in camera coordinates in the shape of (N, 3).
        pose (array): the camera pose in the shape of (N, 4, 4).

    Returns:
        A float array of the original points of the rays in the shape of (N, 3).
        A float array of the view directions of the rays in the shape of (N, 3).
    """
    rotations = poses[:, :3, :3]  # (N, 3, 3)
    ray_centers = poses[:, :3, 3]  # (N, 3)
    ray_directions = np.matmul(camera_ray_directions[:, None, :], rotations.transpose(0, 2, 1)).reshape(-1, 3)  # (N, 3)
    return ray_centers, ray_directions
