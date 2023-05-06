from typing import List

import torch
from torch import Tensor

from vision3d.ops import grid_subsample_pack_mode, radius_search_pack_mode


@torch.no_grad()
def build_grid_and_radius_graph_pyramid(
    points: Tensor,
    lengths: Tensor,
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    neighbor_limits: List[int],
    voxelize_first: bool = False,
) -> dict:
    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    for i in range(num_stages):
        if voxelize_first or i > 0:
            points, lengths = grid_subsample_pack_mode(points, lengths, voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2.0

    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search_pack_mode(
            cur_points, cur_points, cur_lengths, cur_lengths, search_radius, neighbor_limits[i]
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search_pack_mode(
                sub_points, cur_points, sub_lengths, cur_lengths, search_radius, neighbor_limits[i]
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search_pack_mode(
                cur_points, sub_points, cur_lengths, sub_lengths, search_radius * 2, neighbor_limits[i + 1]
            )
            upsampling_list.append(upsampling)

        search_radius *= 2

    return {
        "points": points_list,
        "lengths": lengths_list,
        "neighbors": neighbors_list,
        "subsampling": subsampling_list,
        "upsampling": upsampling_list,
    }
