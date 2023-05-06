from typing import List

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm

from vision3d.ops import grid_subsample_pack_mode, radius_count_pack_mode
from vision3d.utils.tensor import move_to_cuda, tensor_to_array


@torch.no_grad()
def count_neighbors_pack_mode(
    points: Tensor,
    lengths: Tensor,
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    voxelize_first: bool = False,
) -> List[Tensor]:
    counts_list = []
    for i in range(num_stages):
        if voxelize_first or i > 0:
            points, lengths = grid_subsample_pack_mode(points, lengths, voxel_size)
        counts = radius_count_pack_mode(points, points, lengths, lengths, search_radius)
        counts_list.append(counts)
        voxel_size *= 2
        search_radius *= 2
    return counts_list


@torch.no_grad()
def calibrate_neighbors_pack_mode(
    dataset,
    collate_fn,
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    voxelize_first: bool = False,
    keep_ratio: float = 0.8,
    sample_threshold: int = 2000,
) -> ndarray:
    # Compute higher bound of neighbors number in a neighborhood
    neighbor_limit = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, neighbor_limit), dtype=np.int32)  # (S, H)

    for i in tqdm(range(len(dataset))):
        data_dict = collate_fn([dataset[i]])
        data_dict = move_to_cuda(data_dict)

        points = data_dict["points"]
        lengths = data_dict["lengths"]

        counts_list = count_neighbors_pack_mode(
            points, lengths, num_stages, voxel_size, search_radius, voxelize_first=voxelize_first
        )
        counts_list = tensor_to_array(counts_list)

        neighbor_hists += np.stack(
            [np.bincount(counts, minlength=neighbor_limit)[:neighbor_limit] for counts in counts_list], axis=0
        )  # (S, H)

        if sample_threshold is not None and np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[neighbor_limit - 1, :]), axis=0)

    return neighbor_limits
