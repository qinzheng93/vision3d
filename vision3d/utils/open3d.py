from typing import List, Optional

import ipdb
import matplotlib.colors
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.manifold import TSNE
from tqdm import tqdm

# color utilities


def get_color(color_name: str) -> ndarray:
    if color_name == "tgt_color":
        return np.asarray([0, 0.651, 0.929])
    if color_name == "src_color":
        return np.asarray([1, 0.706, 0])
    if color_name == "custom_yellow":
        return np.asarray([255.0, 204.0, 102.0]) / 255.0
    if color_name == "custom_blue":
        return np.asarray([102.0, 153.0, 255.0]) / 255.0
    assert color_name in matplotlib.colors.CSS4_COLORS
    return np.asarray(matplotlib.colors.to_rgb(matplotlib.colors.CSS4_COLORS[color_name]))


def get_colors_with_tsne(data: ndarray, num_iterations: int = 300) -> ndarray:
    """Use t-SNE to project high-dimension feats to RGB.

    Args:
        data (array): The data to process (N, C).
        num_iterations (int): The number of t-SNE iterations. Default: 300.

    Returns:
        colors (array): The colors for each data item (N, 3).
    """
    tsne = TSNE(n_components=1, perplexity=40, n_iter=num_iterations, random_state=0)
    tsne_results = tsne.fit_transform(data).reshape(-1)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    normalized_tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.jet(normalized_tsne_results)[:, :3]
    return colors


def get_colors_with_tsne_3dof(data: ndarray, num_iterations: int = 300) -> ndarray:
    """Use t-SNE to project high-dimension feats to RGB.

    Args:
        data (array): The data to process (N, C).
        num_iterations (int): The number of t-SNE iterations. Default: 300.

    Returns:
        colors (array): The colors for each data item (N, 3).
    """
    tsne = TSNE(n_components=3, perplexity=40, n_iter=num_iterations, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_min = np.min(tsne_results, axis=0, keepdims=True)
    tsne_max = np.max(tsne_results, axis=0, keepdims=True)
    colors = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    return colors


def make_scales_along_axis(points, axis=2, alpha=0):
    if isinstance(axis, int):
        new_scaling_axis = np.zeros(3)
        new_scaling_axis[axis] = 1
        axis = new_scaling_axis
    if not isinstance(axis, ndarray):
        axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    projections = np.matmul(points, axis)
    upper = np.amax(projections)
    lower = np.amin(projections)
    scales = 1 - ((projections - lower) / (upper - lower) * (1 - alpha) + alpha)
    return scales


def make_open3d_colors(points, base_color, scaling_axis=2, scaling_alpha=0):
    if not isinstance(base_color, ndarray):
        base_color = np.asarray(base_color)
    colors = np.ones_like(points) * base_color
    scales = make_scales_along_axis(points, axis=scaling_axis, alpha=scaling_alpha)
    colors = colors * scales.reshape(-1, 1)
    return colors


# geometry utilities


def make_open3d_point_cloud(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def make_open3d_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def make_open3d_axis(axis_vector=None, origin=None, scale=1.0):
    """Make an axis as Open3d LineSet.

    Args:
        axis_vector (array): The vector indicating the axis direction in shape of (3,). If None, use z+ (up-axis).
        origin (array): The original point of the axis in shape of (3,). If None, use (0, 0, 0).
        scale (float=1.0): The length of the axis.

    Returns:
        axis (LineSet): The axis as Open3d LineSet.
    """
    if origin is None:
        origin = np.zeros(3)
    if axis_vector is None:
        axis_vector = np.array([0, 0, 1], dtype=np.float)
    axis_vector = axis_vector * scale
    axis_point = origin + axis_vector
    points = np.stack([origin, axis_point], axis=0)
    line = np.array([[0, 1]], dtype=np.long)
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(points)
    axis.lines = o3d.utility.Vector2iVector(line)
    axis.paint_uniform_color(get_color("red"))
    return axis


def make_open3d_axes(axis_vectors=None, origin=None, scale=1.0):
    if origin is None:
        origin = np.zeros((1, 3))
    if axis_vectors is None:
        axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axis_vectors = axis_vectors * scale
    axis_points = origin + axis_vectors
    points = np.concatenate([origin, axis_points], axis=0)
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.long)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes


def make_open3d_corr_lines(src_corr_points, tgt_corr_points, label):
    num_correspondences = src_corr_points.shape[0]
    corr_points = np.concatenate([src_corr_points, tgt_corr_points], axis=0)
    corr_indices = [(i, i + num_correspondences) for i in range(num_correspondences)]
    corr_lines = o3d.geometry.LineSet()
    corr_lines.points = o3d.utility.Vector3dVector(corr_points)
    corr_lines.lines = o3d.utility.Vector2iVector(corr_indices)
    if label == "pos":
        corr_lines.paint_uniform_color(get_color("lime"))
    elif label == "neg":
        corr_lines.paint_uniform_color(get_color("red"))
    else:
        raise ValueError("Unsupported `label` {} for correspondences".format(label))
    return corr_lines


def make_open3d_rays(rays_loc, rays_points):
    num_rays = rays_loc.shape[0]
    num_points = rays_points.shape[1]
    line_indices = np.asarray(
        [(i * num_points + j, i * num_points + j + 1) for i in range(num_rays) for j in range(0, num_points)]
    )
    line_points = np.concatenate([rays_loc[:, None, :], rays_points], axis=1).reshape(-1, 3)

    # ipdb.set_trace()

    rays = o3d.geometry.LineSet()
    rays.points = o3d.utility.Vector3dVector(line_points)
    rays.lines = o3d.utility.Vector2iVector(line_indices)
    rays.paint_uniform_color(get_color("blue"))

    return rays


def make_open3d_lines(start_points, end_points):
    num_lines = start_points.shape[0]
    points = np.concatenate([start_points, end_points], axis=0)
    indices = np.asarray([[i, i + num_lines] for i in range(num_lines)])
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(indices)
    return lines


def make_open3d_graph(nodes, edges=None, adjacent_list=None, adjacent_mat=None):
    """Create Deformation Graph with open3d.

    The edge set can be in one of the follow forms:
      1. edge list (edges)
      2. adjacent list (adjacent_list)
      3. adjacent matrix (adjacent_mat).
    The priority is the same.

    Args:
        nodes (ndarray): graph nodes (N, 3).
        edges (ndarray): graph edge list (M, 2).
        adjacent_list (ndarray): use -1 to represent missing edges (N, K).
        adjacent_mat (ndarray): If True, there is an edge between two nodes (N, N).

    Returns:
        graph (LineSet): graph.
    """
    if edges is None:
        if adjacent_list is not None:
            num_points, num_neighbors = adjacent_list.shape
            s_indices = np.arange(adjacent_list.shape[0])[:, None].repeat(repeats=num_neighbors, axis=-1).reshape(-1)
            t_indices = adjacent_list.reshape(-1)
            masks = t_indices != -1
            s_indices = s_indices[masks]
            t_indices = t_indices[masks]
            edges = np.stack([s_indices, t_indices], axis=1)
        else:
            edges = np.nonzero(adjacent_mat)
    assert edges is not None

    graph = o3d.geometry.LineSet()
    graph.points = o3d.utility.Vector3dVector(nodes)
    graph.lines = o3d.utility.Vector2iVector(edges)
    graph.paint_uniform_color(get_color("lime"))

    return graph


def extend_point_to_ball(points, colors=None, radius=0.02, resolution=6):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)

    new_vertices = points[:, None, :] + vertices[None, :, :]  # (N, 3) -> (N, 1, 3) -> (N, V, 3)
    new_vertices = new_vertices.reshape(-1, 3)  # (N, V, 3) -> (NxV, 3)
    bases = np.arange(points.shape[0]) * vertices.shape[0]  # (N,)
    new_triangles = bases[:, None, None] + triangles[None, :, :]  # (N, T, 3)
    new_triangles = new_triangles.reshape(-1, 3)  # (NxT, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

    if colors is not None:
        new_vertex_colors = np.broadcast_to(colors[:, None, :], (points.shape[0], vertices.shape[0], 3))  # (N, V, 3)
        new_vertex_colors = new_vertex_colors.reshape(-1, 3)  # (NxV, 3)
        mesh.vertex_colors = o3d.utility.Vector3dVector(new_vertex_colors)

    return mesh


# point cloud utilities


def estimate_normals(points):
    pcd = make_open3d_point_cloud(points)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals


def voxel_down_sample(points, voxel_size, normals=None):
    pcd = make_open3d_point_cloud(points, normals=normals)
    pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)

    if normals is not None:
        normals = np.asarray(pcd.normals)
        return points, normals

    return points


def extract_fpfh(points, radius, max_nn=100, normalize=False):
    pcd = make_open3d_point_cloud(points)
    pcd.estimate_normals()
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    fpfh = pcd_fpfh.data.T
    if normalize:
        fpfh = fpfh / (np.linalg.norm(fpfh, axis=1, keepdims=True) + 1e-6)
    return fpfh


# visualization utilities


def draw_geometries(*geometries, **kwargs):
    o3d.visualization.draw_geometries(geometries, **kwargs)


# registration utilities


def make_open3d_registration_feature(data):
    """
    Make open3d registration features

    :param data: numpy.ndarray (N, C)
    :return feats: o3d.pipelines.registration.Feature
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = data.T
    return feats


def registration_with_ransac_from_feats(
    src_points,
    tgt_points,
    src_feats,
    tgt_feats,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=50000,
    val_iterations=1000,
):
    """Compute the transformation matrix from src_points to tgt_points."""
    src_pcd = make_open3d_point_cloud(src_points)
    tgt_pcd = make_open3d_point_cloud(tgt_points)
    src_feats = make_open3d_registration_feature(src_feats)
    tgt_feats = make_open3d_registration_feature(tgt_feats)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        tgt_pcd,
        src_feats,
        tgt_feats,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, val_iterations),
    )

    return result.transformation


def registration_with_ransac_from_correspondences(
    src_corr_points,
    tgt_corr_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=10000,
):
    """Compute the transformation matrix from src_corr_points to tgt_corr_points."""
    src_pcd = make_open3d_point_cloud(src_corr_points)
    tgt_pcd = make_open3d_point_cloud(tgt_corr_points)

    if correspondences is None:
        indices = np.arange(src_corr_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        tgt_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, num_iterations),
    )

    return result.transformation


def multi_scale_icp(src_points, tgt_points, init_transform, multi_scale_icp_cfgs, distance_threshold):
    src_pcd = make_open3d_point_cloud(src_points)
    tgt_pcd = make_open3d_point_cloud(tgt_points)
    transform = init_transform
    for icp_cfg in multi_scale_icp_cfgs:
        max_iteration = icp_cfg["max_iteration"]
        voxel_size = icp_cfg["voxel_size"]
        cur_src_pcd = src_pcd.voxel_down_sample(voxel_size)
        cur_tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size)
        result = o3d.pipelines.registration.registration_icp(
            cur_src_pcd,
            cur_tgt_pcd,
            distance_threshold,
            transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        )
        transform = result.transformation
    return transform


# fusion utilities


def read_rgbd_image(color_file, depth_file, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale, depth_trunc, convert_rgb_to_intensity
    )
    return rgbd_image


def tsdf_fusion(
    color_files: List[str],
    depth_files: List[str],
    pose_files: List[str],
    intrinsics: ndarray,
    width: int = 640,
    height: int = 480,
    voxel_length: float = 0.006,
    sdf_trunc: float = 0.04,
    depth_scale: float = 1000.0,
    depth_trunc: float = 4.0,
    voxel_size: Optional[float] = None,
    verbose: bool = False,
):
    """TSDF fusion from RGB-D images.

    Args:
        color_files (List[str]): A list of color image filenames. Give an empty list if not available.
        depth_files (List[str]): A list of depth image filenames.
        pose_files (List[str]): A list of pose filenames (cam-to-world).
        intrinsics (ndarray): The intrinsics matrix of the camera.
        width (int=640): The width of the RGB-D images.
        height (int=480): The height of the RGB-D images.
        voxel_length (float=0.006): The voxel length of TSDF fusion.
        sdf_trunc (float=0.04): The truncate value of SDF.
        depth_scale (float=1000.0): The depth scaling factor for depth images.
        depth_trunc (float=4.0): The pixels whose depth value larger than this value are ignores.
        voxel_size (float=None): If not None, the fused point cloud is voxelized to save memory.

    Returns:
        points (array): The fused point cloud in shape of (N, 3).
        base_pose (array): The pose of the point cloud in shape of (3, 3), i.e., the pose of the first frame.
    """
    if len(color_files) == 0:
        color_files = depth_files
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    else:
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=color_type
    )

    base_pose = np.loadtxt(pose_files[0])

    num_frames = len(depth_files)
    pbar = range(num_frames)
    if verbose:
        pbar = tqdm(pbar)
    for i in pbar:
        color_file = color_files[i]
        depth_file = depth_files[i]

        pose = np.loadtxt(pose_files[i])
        extrinsics = np.matmul(np.linalg.inv(pose), base_pose)

        rgbd = read_rgbd_image(color_file, depth_file, depth_scale=depth_scale, depth_trunc=depth_trunc)
        volume.integrate(rgbd, intrinsics, extrinsics)

    pcd = volume.extract_point_cloud()
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    return np.asarray(pcd.points), base_pose
