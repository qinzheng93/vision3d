from typing import List, Dict

from numpy import ndarray

from .grid_subsample import grid_subsample_pack_mode
from .radius_search import radius_search_pack_mode


def build_grid_and_radius_graph_pyramid_pack_mode(
    points: ndarray,
    lengths: ndarray,
    num_stages: int,
    voxel_size: float,
    radius: float,
    neighbor_limits: List[int],
) -> Dict:
    """Build graph pyramid with grid subsampling and radius searching in pack mode."""
    assert num_stages is not None
    assert voxel_size is not None
    assert radius is not None
    assert neighbor_limits is not None
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample_pack_mode(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search_pack_mode(
            cur_points, cur_points, cur_lengths, cur_lengths, radius, neighbor_limits[i]
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search_pack_mode(
                sub_points, cur_points, sub_lengths, cur_lengths, radius, neighbor_limits[i]
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search_pack_mode(
                cur_points, sub_points, cur_lengths, sub_lengths, radius * 2, neighbor_limits[i + 1]
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        "points": points_list,
        "lengths": lengths_list,
        "neighbors": neighbors_list,
        "subsampling": subsampling_list,
        "upsampling": upsampling_list,
    }
