import warnings
from typing import Optional

import numpy as np
from numpy import ndarray

from vision3d.utils.misc import load_ext

from .depth_image import back_project

ext_module = load_ext(
    "vision3d.ext",
    [
        "build_deformation_graph_from_point_cloud",
        "compute_clusters",
        "compute_edges_geodesic",
        "compute_pixel_anchors_geodesic",
        "depth_to_mesh",
        "erode_mesh",
        "node_and_edge_clean_up",
        "sample_nodes",
        "update_pixel_anchors",
    ],
)


def embedded_deformation_warp(points: ndarray, nodes: ndarray, transforms: ndarray, weights: ndarray, eps=1e-6):
    """Warp a point cloud using the embedded deformation.

    Args:
        points (array<float>): The point cloud (N, 3).
        nodes (array<float>): The graph nodes (M, 3).
        transforms (array<float>): The transformations for the nodes (M, 4, 4).
        weights (array<float>): The skinning weights between nodes and points (M, N).
        eps (float=1e-6): safe number.

    Returns:
        deformed_points (array<float>): The deformed point cloud (N, 3).
    """
    rotations = transforms[:, :3, :3]  # (M, 3, 3)
    translations = transforms[:, :3, 3]  # (M, 3)
    points = points[None, :, :]  # (1, N, 3)
    nodes = nodes[:, None, :]  # (M, 1, 3)
    weights = weights[:, :, None]  # (M, N, 1)
    deformed_points = (points - nodes) @ rotations.T + nodes + translations[:, None, :]  # (M, N, 3)
    deformed_points = (deformed_points * weights).sum(0) / (weights.sum(0) + eps)  # (N, 3)
    return deformed_points


def build_mesh_from_depth_image(
    depth_image: ndarray,
    mask_image: ndarray,
    intrinsics: ndarray,
    scaling_factor: float = 1000.0,
    max_triangle_distance: float = 0.04,
    depth_limit: Optional[float] = None,
):
    """Build mesh from depth image.

    Args:
        depth_image (array<float>): depth image (H, W).
        mask_image (array<float>): mask image (H, W).
        intrinsics (array<float>): intrinsics matrix (3, 3).
        scaling_factor (float=1000.0): depth scaling factor.
        max_triangle_distance (float=0.04): maximal triangle distance in mesh.
        depth_limit (float=None): maximal depth.

    Returns:
        points (array<float>): points (N, 3)
        faces (array<float>): triangle faces (M, 3)
        pixels (array<float>): pixel coordinates of the points (N, 2)
        point_image (array<float>): point image (H, W, 3)
    """
    mask_image[mask_image > 0] = 1
    depth_image = depth_image * mask_image

    point_image = back_project(
        depth_image, intrinsics, scaling_factor=scaling_factor, depth_limit=depth_limit, return_matrix=True
    )
    point_image = point_image.astype(np.float32)  # (H, W, 3)

    points, faces, pixels = ext_module.depth_to_mesh(point_image, max_triangle_distance)

    return points, faces, pixels, point_image


def erode_mesh(points: ndarray, faces: ndarray, erosion_num_iterations: int, erosion_min_neighbors: int):
    """Erode mesh to erase the points with too few neighbors.

    Args:
        points (array<float>): The input point cloud (N, 3).
        faces (array<int>): The triangle faces (M, 3).
        erosion_num_iterations (int): The number of iterations for erosion.
        erosion_min_neighbors (int): The minimal number of neighbors for each point.

    Returns:
        point_masks (array<bool>): If True, the point is valid (N,).
    """
    point_masks = np.ones(shape=(points.shape[0],), dtype=np.bool)
    if erosion_num_iterations > 0:
        ext_module.erode_mesh(points, faces, point_masks, erosion_num_iterations, erosion_min_neighbors)
    return point_masks


def sample_nodes_from_point_cloud(
    points: ndarray,
    point_masks: ndarray,
    node_coverage,
    use_only_valid_points=True,
    random_shuffle=False,
):
    """Sample nodes according to node coverage.

    Args:
        points (array<float>): The point cloud (N, 3).
        point_masks (array<float>): If True, the point is valid (N,).
        node_coverage (float): minimal distance between two nodes.
        use_only_valid_points (bool=True): If True, use only valid points.
        random_shuffle (bool=False): If True, shuffle points before sampling.

    Returns:
        nodes (array<float>): nodes (M, 3).
        node_indices (array<float>): indices of the nodes in the point cloud (M,).
    """
    nodes, node_indices = ext_module.sample_nodes(
        points, point_masks, node_coverage, use_only_valid_points, random_shuffle
    )
    return nodes, node_indices


def build_graph_edges_from_geodesic_distance(
    points: ndarray,
    point_masks: ndarray,
    faces: ndarray,
    node_indices: ndarray,
    num_neighbors: int,
    node_coverage: float,
    use_only_valid_points: bool = True,
    enforce_total_num_neighbors: bool = False,
):
    """Build deformation graph edges from geodesic distance.

    Args:
        points (array<float>): The point cloud (N, 3).
        point_masks (array<bool>): The masks of the valid points (N,).
        faces (array<int>): The triangle mesh (F, 3).
        node_indices (array<int>): The indices of the nodes (M,)
        num_neighbors (int): The maximal number of neighbors for each node.
        node_coverage (float): The node coverage, the maximal influence R_max = 2 * node_coverage.
        use_only_valid_points (bool=True): If True, only valid points are used.
        enforce_total_num_neighbors (bool=False): If True, each node will have full neighbors (even further than R_max).

    Returns:
        graph_edge_indices (array<int>): The indices of the neighbors for each node (M, K), -1 if not valid.
        graph_edge_weights (array<float>): The skinning weights of the neighbors for each node (M, K).
        graph_edge_distances (array<float>): The geodesic distances of the neighbors for each node (M, K).
        node_to_point_distances (array<float>): The geodesic distance between nodes and points (M, N), -1 if too far.
    """
    num_nodes = node_indices.shape[0]
    num_points = points.shape[0]

    graph_edge_indices = -np.ones(shape=(num_nodes, num_neighbors), dtype=np.int32)
    graph_edge_weights = np.zeros(shape=(num_nodes, num_neighbors), dtype=np.float32)
    graph_edge_distances = np.zeros(shape=(num_nodes, num_neighbors), dtype=np.float32)
    node_to_point_distances = -np.ones(shape=(num_nodes, num_points), dtype=np.float32)

    visible_points = np.ones_like(point_masks)
    ext_module.compute_edges_geodesic(
        points,
        visible_points,
        faces,
        node_indices,
        graph_edge_indices,
        graph_edge_weights,
        graph_edge_distances,
        node_to_point_distances,
        num_neighbors,
        node_coverage,
        use_only_valid_points,
        enforce_total_num_neighbors,
    )

    return graph_edge_indices, graph_edge_weights, graph_edge_distances, node_to_point_distances


def clean_nodes_and_edges(graph_edge_indices: ndarray, remove_nodes_with_not_enough_neighbors: bool = True):
    """Remove nodes without enough neighbors.

    Args:
        graph_edge_indices (array<int>): The indices of the neighbors for the nodes (N, 2).
        remove_nodes_with_not_enough_neighbors (bool=True): If False, the cleaning operation is skipped.

    Returns:
        node_masks (array<bool>): If True, the node is valid (N,).
        node_id_black_list (List[int]): The removed node indices.
    """
    num_nodes = graph_edge_indices.shape[0]
    node_masks = np.ones(shape=(num_nodes,), dtype=np.bool)
    node_id_black_list = []
    if remove_nodes_with_not_enough_neighbors:
        ext_module.node_and_edge_clean_up(graph_edge_indices, node_masks)
        node_id_black_list = np.where(~node_masks)[0].tolist()
    else:
        warnings.warn("You are allowing nodes with not enough neighbors!")
    return node_masks, node_id_black_list


def compute_pixel_anchors_from_geodesic_distance(
    node_to_point_distances: ndarray,
    nodes_mask: ndarray,
    points: ndarray,
    pixels: ndarray,
    width: int,
    height: int,
    num_anchors: int,
    node_coverage: float,
):
    """Compute anchor nodes for each pixel.

    Args:
        node_to_point_distances (array<float>): The geodesic distance between nodes and points (M, N).
        nodes_mask (array<bool>): The masks for valid nodes (M,).
        points (array<float>): The point cloud (N, 3).
        pixels (array<int>): The pixel coordinates for the points (N, 2).
        width (int): The width of the image.
        height (int): The height of the image.
        num_anchors (int): The number of the anchors.
        node_coverage (float): The node coverage.

    Returns:
        pixel_anchors (array<int>): the indices of the anchor nodes for each pixel (H, W, K).
        pixel_weights (array<float>): the weights of the anchor nodes for each pixel (H, W, K).
    """
    pixel_anchors = -np.ones(shape=(height, width, num_anchors), dtype=np.int32)
    pixel_weights = np.zeros(shape=(height, width, num_anchors), dtype=np.float32)
    ext_module.compute_pixel_anchors_geodesic(
        node_to_point_distances,
        nodes_mask,
        points,
        pixels,
        pixel_anchors,
        pixel_weights,
        width,
        height,
        num_anchors,
        node_coverage,
    )
    return pixel_anchors, pixel_weights


def compute_clusters(graph_edge_indices):
    """Compute clusters in the graph.

    Args:
        graph_edge_indices (array<int>): The indices of neighbors for the nodes (M, 3).

    Returns:
        cluster_indices (array<int>): The indices of the cluster for the nodes (M,).
    """
    cluster_indices = -np.ones(shape=(graph_edge_indices.shape[0],), dtype=np.int32)
    ext_module.compute_clusters(graph_edge_indices, cluster_indices)
    return cluster_indices


def build_deformation_graph_from_depth_image(
    depth_image: ndarray,
    intrinsics: ndarray,
    node_coverage: float,
    depth_scale: float = 1000.0,
    max_triangle_distance: float = 0.04,
    erosion_num_iterations: int = 0,
    erosion_min_neighbors: int = 0,
    use_only_valid_points: bool = True,
    num_neighbors: int = 8,
    num_anchors: int = 6,
    enforce_total_num_neighbors: bool = False,
    sample_random_shuffle: bool = False,
    remove_nodes_with_not_enough_neighbors: bool = True,
):
    """Construct deformation graph from depth map.

    Modified from [NeuralTracking](https://github.com/DeformableFriends/NeuralTracking/blob/main/create_graph_data.py)

    Args:
        depth_image (array<float>): the depth image to convert.
        intrinsics (array<float>): the camera intrinsics.
        node_coverage (float): the minimal distance between nodes.
        depth_scale (float=1000.0): the sigma of the depth image.
        max_triangle_distance (float=0.04): only triangles whose edges are all smaller than this value are preserved.
        erosion_num_iterations (int=0): the number of erosion iterations.
        erosion_min_neighbors (int=0): the nodes with fewer neighbors than this value are filtered.
        use_only_valid_points (bool=True): If True, only points which belong to at least one face is preserved.
        num_neighbors (int=8): the number of neighbors for the nodes to build edges.
        num_anchors (int=6): the number of anchor nodes for the points.
        enforce_total_num_neighbors (bool=False):
        sample_random_shuffle (bool=False): If True, shuffle nodes after sampling.
        remove_nodes_with_not_enough_neighbors (bool=True): If True, remove nodes with no more than 1 neighbors.

    Returns:
        graph_nodes (array<float>): the xyz of the graph nodes (M, 3).
        graph_edges (array<int>): If -1, the neighbor is missing (M, K).
        graph_edges_weights (array<float>): the skinning weights of the edges (M, K).
        graph_clusters (array<int>): the cluster index of each node (M,).
        point_image (array<float>): the xyz of the pixels (H, W, 3).
        points (array<float>): the xyz of the valid points (N, 3).
        pixels (array<int>): the uv of the valid points (N, 2).
        pixel_anchors (array<int>): the k-nearest nodes for the pixels (H, W, K).
        pixel_weights (array<float>): the skinning weights of the pixel anchors (H, W, K).
    """
    # convert depth to mesh
    height, width = depth_image.shape
    mask_image = depth_image > 0
    points, faces, pixels, point_image = build_mesh_from_depth_image(
        depth_image, mask_image, intrinsics, max_triangle_distance=max_triangle_distance, scaling_factor=depth_scale
    )
    assert points.shape[0] > 0, "No points found."
    assert faces.shape[0] > 0, "No faces found."

    # Erode mesh, to not sample unstable nodes on the mesh boundary
    point_masks = erode_mesh(points, faces, erosion_num_iterations, erosion_min_neighbors)

    # Sample graph nodes
    nodes, node_indices = sample_nodes_from_point_cloud(
        points,
        point_masks,
        node_coverage,
        use_only_valid_points=use_only_valid_points,
        random_shuffle=sample_random_shuffle,
    )
    assert nodes.shape[0] > 0, "No nodes found."

    # Compute graph edges
    (
        graph_edge_indices,
        graph_edges_weights,
        graph_edges_distances,
        node_to_point_distances,
    ) = build_graph_edges_from_geodesic_distance(
        points,
        point_masks,
        faces,
        node_indices,
        num_neighbors,
        node_coverage,
        use_only_valid_points=use_only_valid_points,
        enforce_total_num_neighbors=enforce_total_num_neighbors,
    )

    # Remove nodes
    node_masks, node_id_black_list = clean_nodes_and_edges(
        graph_edge_indices, remove_nodes_with_not_enough_neighbors=remove_nodes_with_not_enough_neighbors
    )

    # Compute pixel anchors
    pixel_anchors, pixel_weights = compute_pixel_anchors_from_geodesic_distance(
        node_to_point_distances, node_masks, points, pixels, width, height, num_anchors, node_coverage
    )

    # filter invalid nodes
    nodes = nodes[node_masks]
    node_indices = node_indices[node_masks]
    graph_edge_indices = graph_edge_indices[node_masks]
    graph_edges_weights = graph_edges_weights[node_masks]
    graph_edges_distances = graph_edges_distances[node_masks]
    assert nodes.shape[0] > 0, "No valid nodes found."

    # Update node ids
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(node_masks):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edge_indices):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edge_indices[node_id] = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id] = np.zeros_like(graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edge_indices[node_id, count_valid_neighbors] = mapped_neighbor_id
                    graph_edges_weights[node_id, count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id, count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                warnings.warn("Zero graph edge weights.", graph_edges_weights[node_id])
                raise Exception("Not good")

        # 3. Update pixel anchors using the id mapping
        # note that, at this point, pixel_anchors is already free of "bad" nodes, since
        # 'compute_pixel_anchors_geodesic_c' was given 'valid_nodes_mask')
        ext_module.update_pixel_anchors(node_id_mapping, pixel_anchors)

    graph_clusters = compute_clusters(graph_edge_indices)

    data_dict = {
        "graph_nodes": nodes,
        "graph_edges": graph_edge_indices,
        "graph_edges_weights": graph_edges_weights,
        "graph_clusters": graph_clusters,
        "node_indices": node_indices,
        "point_image": point_image,
        "points": points,
        "pixels": pixels,
        "pixel_anchors": pixel_anchors,
        "pixel_weights": pixel_weights,
    }

    return data_dict


def build_deformation_graph_from_point_cloud(
    points: ndarray,
    node_indices: ndarray,
    num_neighbors: int,
    num_anchors: int,
    max_distance: float,
    node_coverage: float,
):
    """Build deformation graph from point clouds.

    This function is implemented on CPU.

    Args:
        points (array<float>): the point clouds (N, 3).
        node_indices (array<float>): the number of nodes in the batch (M).
        num_neighbors (int): the maximal number of neighbor nodes of each node, i.e., Kn.
        num_anchors (int): the maximal number of anchor nodes of each point, i.e., Ka.
        max_distance (float): the maximal distance of the edges between points.
        node_coverage (float): the coverage radius of the nodes, we only consider the nodes within 2xCoverage.

    Returns:
        A dict containing the information of the deformation graph. The keywords are:
            "neighbor_indices": an int array of the neighbor indices of each node, -1 if not exists. (M, Kn)
            "neighbor_distances": a float array of the neighbor distances of each node, 0 if not exists. (M, Kn)
            "neighbor_weights": a float array of the neighbor weights of each node, 0 if not exists. (M, Kn)
            "anchor_indices": an int array of the anchor indices of each point, -1 if not exists. (N, Ka)
            "anchor_distances": a float array of the anchor distances of each point, 0 if not exists. (N, Ka)
            "anchor_weights": a float array of the anchor weights of each point, 0 if not exists. (N, Ka)
    """
    (
        neighbor_indices,
        neighbor_distances,
        neighbor_weights,
        anchor_indices,
        anchor_distances,
        anchor_weights,
    ) = ext_module.build_deformation_graph_from_point_cloud(
        points, node_indices, num_neighbors, num_anchors, max_distance, node_coverage
    )

    return {
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
        "neighbor_weights": neighbor_weights,
        "anchor_indices": anchor_indices,
        "anchor_distances": anchor_distances,
        "anchor_weights": anchor_weights,
    }
