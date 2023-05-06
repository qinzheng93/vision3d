from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .open3d import draw_geometries, get_color, make_open3d_axes, make_open3d_corr_lines, make_open3d_point_cloud


def draw_point_to_node_partition(
    points: ndarray, nodes: ndarray, point_to_node: ndarray, node_colors: Optional[ndarray] = None
) -> None:
    if node_colors is None:
        node_colors = np.random.rand(*nodes.shape)

    point_colors = node_colors[point_to_node]
    node_colors = np.ones_like(nodes) * np.array([[1, 0, 0]])

    ncd = make_open3d_point_cloud(nodes, colors=node_colors)
    pcd = make_open3d_point_cloud(points, colors=point_colors)
    axes = make_open3d_axes()

    draw_geometries(pcd, ncd, axes)


def draw_node_correspondences(
    src_points: ndarray,
    src_nodes: ndarray,
    src_point_to_node: ndarray,
    tgt_points: ndarray,
    tgt_nodes: ndarray,
    tgt_point_to_node: ndarray,
    node_correspondences: ndarray,
    src_node_colors: Optional[ndarray] = None,
    tgt_node_colors: Optional[ndarray] = None,
    offsets: Tuple[float, float, float] = (0.0, 2.0, 0.0),
) -> None:
    src_nodes = src_nodes + offsets
    src_points = src_points + offsets

    if src_node_colors is None:
        src_node_colors = np.random.rand(*src_nodes.shape)
    # tgt_point_colors = tgt_node_colors[tgt_point_to_node] * make_scales_along_axis(tgt_points).reshape(-1, 1)
    src_point_colors = src_node_colors[src_point_to_node]
    src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])

    if tgt_node_colors is None:
        tgt_node_colors = np.random.rand(*tgt_nodes.shape)
    # src_point_colors = src_node_colors[src_point_to_node] * make_scales_along_axis(src_points).reshape(-1, 1)
    tgt_point_colors = tgt_node_colors[tgt_point_to_node]
    tgt_node_colors = np.ones_like(tgt_nodes) * np.array([[1, 0, 0]])

    src_ncd = make_open3d_point_cloud(src_nodes, colors=src_node_colors)
    src_pcd = make_open3d_point_cloud(src_points, colors=src_point_colors)
    tgt_ncd = make_open3d_point_cloud(tgt_nodes, colors=tgt_node_colors)
    tgt_pcd = make_open3d_point_cloud(tgt_points, colors=tgt_point_colors)
    corr_lines = make_open3d_corr_lines(src_nodes, tgt_nodes, node_correspondences)
    axes = make_open3d_axes(scale=0.1)

    draw_geometries(src_pcd, src_ncd, tgt_pcd, tgt_ncd, corr_lines, axes)


def draw_correspondences(
    src_points: ndarray,
    tgt_points: ndarray,
    src_corr_indices: ndarray,
    tgt_corr_indices: ndarray,
    offsets: Tuple[float, float, float] = (0.0, 2.0, 0.0),
) -> None:
    src_points = src_points + np.asarray(offsets)[None, :]
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_yellow"))
    tgt_pcd = make_open3d_point_cloud(tgt_points)
    tgt_pcd.estimate_normals()
    tgt_pcd.paint_uniform_color(get_color("custom_blue"))
    src_corr_points = src_points[src_corr_indices]
    tgt_corr_points = tgt_points[tgt_corr_indices]
    corr_lines = make_open3d_corr_lines(src_corr_points, tgt_corr_points, label="pos")
    axes = make_open3d_axes(scale=0.1)
    draw_geometries(src_pcd, tgt_pcd, corr_lines, axes)


def draw_registration(src_points: ndarray, tgt_points: ndarray, transform: ndarray) -> None:
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_yellow"))
    tgt_pcd = make_open3d_point_cloud(tgt_points)
    tgt_pcd.estimate_normals()
    tgt_pcd.paint_uniform_color(get_color("custom_blue"))
    draw_geometries(src_pcd, tgt_pcd)

    src_pcd = src_pcd.transform(transform)
    draw_geometries(src_pcd, tgt_pcd)
