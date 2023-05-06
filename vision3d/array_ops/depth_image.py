from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from .se3 import apply_transform


def back_project(
    depth_mat: ndarray,
    intrinsics: ndarray,
    scaling_factor: float = 1000.0,
    depth_limit: Optional[float] = None,
    return_matrix: bool = False,
    return_pixels: bool = False,
):
    """Back project depth image to point cloud.

    Args:
        depth_mat (array): depth image (H, W).
        intrinsics (array): intrinsics matrix (3, 3).
        scaling_factor (float=1000.0): depth scaling factor.
        depth_limit (float=None): ignore the pixels further than this value.
        return_matrix (bool=False): If True, return point matrix instead of point clouds.
        return_pixels (bool=False): If True, return pixel coordinates of the point clouds.

    Returns:
        point_mat (array): point image (H, W, 3)
        points (array): point cloud (N, 3)
    """
    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]
    height, width = depth_mat.shape
    coords = np.arange(height * width)
    u = coords % width
    v = coords / width
    depth = depth_mat.flatten()
    z = depth / scaling_factor
    if depth_limit is not None:
        z[z > depth_limit] = 0.0
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    points = np.stack([x, y, z], axis=1)

    output_list = []

    masks = z > 0
    if return_matrix:
        point_mat = points.reshape(height, width, 3)
        output_list.append(point_mat)
    else:
        points = points[masks]
        output_list.append(points)

    if return_pixels:
        masks = masks.reshape(height, width)
        v_indices, u_indices = np.nonzero(masks)
        pixel_indices = np.stack([v_indices, u_indices], axis=1)
        output_list.append(pixel_indices)

    if len(output_list) == 1:
        return output_list[0]
    return tuple(output_list)


def render(points: ndarray, intrinsics: ndarray, extrinsics: Optional[ndarray] = None, return_depth: bool = False):
    """Render points into image according to intrinsics and extrinsic.

    Args:
        points (array): the point cloud to render in the shape of (N, 3).
        intrinsics (array): the intrinsic matrix in the shape of (3, 3).
        extrinsics (array, optional): the extrinsic matrix in the shape of (3, 3).
        return_depth (bool): if True, return depth. Default: False.

    Returns:
        A float array of the pixel cloud [height, width] in the shape of (N, 2).
        (Optional) A float array of the depth values in meter of the pixel cloud in the shape of (N,).
    """
    if extrinsics is not None:
        points = apply_transform(points, extrinsics)

    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    w_coords = (focal_x * x_coords / z_coords + center_x).astype(int)
    h_coords = (focal_y * y_coords / z_coords + center_y).astype(int)
    pixels = np.stack([h_coords, w_coords], axis=-1)

    if return_depth:
        return pixels, z_coords

    return pixels


def render_with_z_buffer(
    points: ndarray,
    intrinsics: ndarray,
    image_h: int,
    image_w: int,
    extrinsics: Optional[ndarray] = None,
    return_depth: bool = False,
    threshold: float = 0.05,
):
    """Render points into image according to intrinsics and extrinsic.

    Args:
        points (array): the point cloud to render in the shape of (N, 3).
        intrinsics (array): the intrinsic matrix in the shape of (3, 3).
        image_h: the height of the image.
        image_w: the width of the image.
        extrinsics (array, optional): the extrinsic matrix in the shape of (3, 3).
        return_depth (bool): if True, return depth. Default: False.
        threshold (float): the depth threshold for acceptance. Default: 0.05.

    Returns:
        A float array of the pixel cloud [height, width] in the shape of (N, 2).
        (Optional) A float array of the depth values in meter of the pixel cloud in the shape of (N,).
    """
    if extrinsics is not None:
        points = apply_transform(points, extrinsics)

    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    w_coords = (focal_x * x_coords / z_coords + center_x).astype(int)
    h_coords = (focal_y * y_coords / z_coords + center_y).astype(int)
    pixels = np.stack([h_coords, w_coords], axis=-1)

    masks = (h_coords >= 0) & (h_coords < image_h) & (w_coords >= 0) & (w_coords < image_w) & (z_coords > 0)
    z_masks = np.ones_like(masks)

    sel_h_coords = h_coords[masks]
    sel_w_coords = w_coords[masks]
    sel_z_coords = z_coords[masks]
    indices = sel_h_coords * image_w + sel_w_coords
    unique_indices, inv_indices = np.unique(indices, return_inverse=True)
    unique_z_buffers = np.full(unique_indices.shape[0], fill_value=1e10)
    np.minimum.at(unique_z_buffers, inv_indices, sel_z_coords)
    sel_z_buffers = unique_z_buffers[inv_indices]
    sel_z_masks = np.abs(sel_z_buffers - sel_z_coords) < threshold
    z_masks[masks] = sel_z_masks
    masks = masks & z_masks

    if return_depth:
        return pixels, masks, z_coords

    return pixels, masks
