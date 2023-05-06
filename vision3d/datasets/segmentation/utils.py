import itertools
from typing import List, Optional

import numpy as np
from numpy import ndarray

from vision3d.array_ops import (
    grid_reduce,
    random_rotate_points_along_up_axis,
    random_scale_points,
    random_jitter_points,
)


def random_sample_indices(indices: ndarray, num_samples: int, pad_points: bool) -> ndarray:
    num_points = indices.shape[0]
    if num_points >= num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)
    elif pad_points:
        padded_indices = np.random.choice(indices, num_samples - num_points, replace=True)
        indices = np.concatenate([indices, padded_indices], axis=0)
        np.random.shuffle(indices)
    return indices


def check_in_range_2d(points: ndarray, lower_bound: ndarray, upper_bound: ndarray):
    results = (
        (points[:, 0] >= lower_bound[0])
        & (points[:, 0] <= upper_bound[0])
        & (points[:, 1] >= lower_bound[1])
        & (points[:, 1] <= upper_bound[1])
    )
    return results


def random_sample_block(
    points: ndarray,
    scene_size: ndarray,
    block_size: float,
    num_samples: Optional[int] = 4096,
    point_threshold: int = 1024,
    pad_points: bool = True,
) -> ndarray:
    points_2d = points[:, :2]
    # scene_size_2d = scene_size[:2]
    while True:
        index = np.random.randint(0, points.shape[0])
        center = points_2d[index]
        # center = np.random.uniform(-0.5, 0.5) * block_size + points_2d[index]
        # center = np.clip(center, a_min=0.0, a_max=scene_size_2d)
        lower_bound = center - block_size / 2.0
        upper_bound = center + block_size / 2.0
        masks = check_in_range_2d(points_2d, lower_bound, upper_bound)
        if np.sum(masks) > point_threshold:
            break
    indices = np.nonzero(masks)[0]
    if num_samples is not None:
        indices = random_sample_indices(indices, num_samples, pad_points)
    return indices


def find_nearest_block(src_block, block_list, point_threshold):
    best_dist = np.inf
    best_index = -1
    for i, tgt_block in enumerate(block_list):
        total_points = tgt_block["indices"].shape[0] + src_block["indices"].shape[0]
        if total_points > point_threshold:
            dist = np.linalg.norm(tgt_block["barycenter_2d"] - src_block["barycenter_2d"])
            if dist < best_dist:
                best_dist = dist
                best_index = i
    return best_index


def divide_point_cloud_into_blocks(
    points: ndarray,
    scene_size: ndarray,
    block_size: float,
    block_stride: float,
    num_samples: Optional[int] = 4096,
    point_threshold: int = 2048,
    pad_points: bool = True,
) -> List[ndarray]:
    num_blocks_x = int(np.ceil((scene_size[0] - block_size) / block_stride)) + 1
    num_blocks_y = int(np.ceil((scene_size[1] - block_size) / block_stride)) + 1

    # select non-empty blocks
    points_2d = points[:, :2]
    block_list = []
    for i, j in itertools.product(range(num_blocks_x), range(num_blocks_y)):
        lower_bound = np.array([i, j]) * block_stride
        upper_bound = lower_bound + block_size
        masks = check_in_range_2d(points_2d, lower_bound, upper_bound)
        if not np.any(masks):
            continue
        indices = np.nonzero(masks)[0]
        barycenter_2d = np.mean(points_2d[indices], axis=0)
        block_list.append({"indices": indices, "barycenter_2d": barycenter_2d})

    # merge small blocks
    num_blocks = len(block_list)
    block_index = 0
    while block_index < num_blocks:
        if block_list[block_index]["indices"].shape[0] > point_threshold:
            block_index += 1
            continue

        src_block = block_list.pop(block_index)
        num_blocks -= 1

        tgt_block_index = find_nearest_block(src_block, block_list, point_threshold)
        tgt_block_dict = block_list[tgt_block_index]

        new_indices = np.concatenate([tgt_block_dict["indices"], src_block["indices"]], axis=0)
        new_barycenter_2d = np.mean(points[new_indices, :2], axis=0)
        block_list[tgt_block_index] = {
            "indices": new_indices,
            "barycenter_2d": new_barycenter_2d,
        }

    if num_samples is None:
        batch_indices_list = [block["indices"] for block in block_list]
        return batch_indices_list

    # divide blocks into batches
    batch_indices_list = []
    for block_dict in block_list:
        block_indices = block_dict["indices"]

        if pad_points:
            # pad to the smallest multiple of num_samples
            num_points = block_indices.shape[0]
            if num_points % num_samples != 0:
                padding_size = num_samples - num_points % num_samples
                padded_indices = np.random.choice(block_indices, padding_size, replace=True)
                block_indices = np.concatenate([block_indices, padded_indices], axis=0)
            np.random.shuffle(block_indices)

            # split block into batches
            cur_batch_indices_list = np.split(block_indices, block_indices.shape[0] // num_samples, axis=0)
        else:
            # merge last batch if too small
            num_points = block_indices.shape[0]
            num_batches = num_points // num_samples
            last_batch_size = num_points % num_samples
            if last_batch_size > num_samples // 4:
                num_batches += 1
            sections = np.arange(1, num_batches) * num_samples
            cur_batch_indices_list = np.split(block_indices, sections, axis=0)

        batch_indices_list += cur_batch_indices_list

    return batch_indices_list


def grid_subsample(points: ndarray, feats: ndarray, labels: ndarray, voxel_size: float, num_classes: int):
    onehot_labels = np.zeros(shape=(points.shape[0], num_classes))
    point_indices = np.arange(points.shape[0])
    onehot_labels[point_indices, labels] = 1.0

    s_points, s_feats, s_onehot_labels, inv_indices = grid_reduce(
        points, voxel_size, feats, onehot_labels, return_inverse=True
    )
    s_labels = np.argmax(s_onehot_labels, axis=1)

    return s_points, s_feats, s_labels, inv_indices


class SceneSegTransform:
    def __init__(
        self,
        use_augmentation=True,
        rotation_scale=1.0,
        min_scale=0.8,
        max_scale=1.2,
        noise_sigma=0.01,
        noise_scale=0.05,
        voxel_size=None,
        num_classes=None,
    ):
        self.use_augmentation = use_augmentation
        self.rotation_scale = rotation_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_sigma = noise_sigma
        self.noise_scale = noise_scale
        self.voxel_size = voxel_size
        self.num_classes = num_classes

    def __call__(self, data_dict):
        points = data_dict.pop("points")

        if self.use_augmentation:
            points = random_rotate_points_along_up_axis(points, rotation_scale=self.rotation_scale)
            points = random_scale_points(points, low=self.min_scale, high=self.max_scale)
            points = random_jitter_points(points, sigma=self.noise_sigma, scale=self.noise_scale)

        if self.voxel_size is not None:
            feats = data_dict.pop("feats")
            labels = data_dict.pop("labels")
            data_dict["raw_points"] = points.astype(np.float32)
            data_dict["raw_feats"] = feats.astype(np.float32)
            data_dict["raw_labels"] = labels.astype(np.int64)
            points, feats, labels, inv_indices = grid_subsample(
                points, feats, labels, self.voxel_size, self.num_classes
            )
            data_dict["points"] = points.astype(np.float32)
            data_dict["feats"] = feats.astype(np.float32)
            data_dict["labels"] = labels.astype(np.int64)
            data_dict["inv_indices"] = inv_indices.astype(np.int64)
        else:
            data_dict["points"] = points.astype(np.float32)

        return data_dict
