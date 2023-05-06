import os.path as osp
import random

import numpy as np
import torch
import torch.utils.data

from vision3d.array_ops import (
    apply_rotation,
    apply_transform,
    compose_transforms,
    get_correspondences,
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_rotation,
    random_sample_small_transform,
)
from vision3d.utils.io import load_pickle


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        subset,
        transform_fn=None,
        max_points=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        use_weak_augmentation=False,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
        deterministic=False,
        scaling_factor=None,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.data_dir = osp.join(self.dataset_dir, "data")

        self.transform_fn = transform_fn
        self.subset = subset
        self.max_points = max_points
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated
        self.deterministic = deterministic
        self.scaling_factor = scaling_factor

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices:
            assert self.matching_radius is not None, "'matching_radius' is None but 'return_corr_indices' is True."

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.use_weak_augmentation = use_weak_augmentation

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{subset}.pkl"))

        if self.overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)

    def load_point_cloud(self, file_name):
        """Load point cloud.
        Random sample if there are too many points.
        NOTE: setting "max_points" with "num_workers" > 1 will cause nondeterminism.
        """
        points = torch.load(osp.join(self.data_dir, file_name))
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]
        return points

    def weakly_augment_point_cloud(self, src_points, tgt_points, transform):
        """Weakly augment point clouds as proposed in RegTR.

        tgt_points = src_points @ rotation.T + translation

        1. Randomly sample a rotation axis and a small rotation angle.
        2. Randomly sample a small translation.
        3. Move point clouds to the original point.
        4. Apply random augmentation transformation.
        5. Move back to their initial positions.
        6. Random noise.
        """
        aug_transform = random_sample_small_transform()
        if random.random() > 0.5:
            tgt_center = tgt_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -tgt_center)
            add_center = get_transform_from_rotation_translation(None, tgt_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            tgt_points = apply_transform(tgt_points, aug_transform)
            transform = compose_transforms(transform, aug_transform)
        else:
            src_center = src_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -src_center)
            add_center = get_transform_from_rotation_translation(None, src_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            src_points = apply_transform(src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, transform

    def augment_point_cloud(self, src_points, tgt_points, transform):
        """Augment point clouds.

        tgt_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_transform = get_transform_from_rotation_translation(aug_rotation, None)
        if random.random() > 0.5:
            tgt_points = apply_transform(tgt_points, aug_transform)
            transform = compose_transforms(transform, aug_transform)
        else:
            src_points = apply_transform(src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, transform

    def __getitem__(self, index):
        # deterministic
        if self.deterministic:
            np.random.seed(index)

        data_dict = {}

        # metadata
        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["src_frame"] = metadata["src_frame"]
        data_dict["tgt_frame"] = metadata["tgt_frame"]
        data_dict["overlap"] = metadata["overlap"]

        # get transformation
        transform = metadata["transform"]

        # get point cloud
        src_points = self.load_point_cloud(metadata["src_path"])
        tgt_points = self.load_point_cloud(metadata["tgt_path"])

        # augmentation
        if self.use_augmentation:
            if self.use_weak_augmentation:
                src_points, tgt_points, transform = self.weakly_augment_point_cloud(src_points, tgt_points, transform)
            else:
                src_points, tgt_points, transform = self.augment_point_cloud(src_points, tgt_points, transform)

        # apply random rotation to both point clouds
        if self.rotated:
            # random rotate src point cloud
            src_rotation = random_sample_rotation()
            src_points = apply_rotation(src_points, src_rotation)

            # random rotate tgt point cloud
            tgt_rotation = random_sample_rotation()
            tgt_points = apply_rotation(tgt_points, tgt_rotation)

            # adjust ground-truth transformation
            src_transform = get_transform_from_rotation_translation(src_rotation, None)
            inv_src_transform = inverse_transform(src_transform)
            tgt_transform = get_transform_from_rotation_translation(tgt_rotation, None)
            transform = compose_transforms(inv_src_transform, transform, tgt_transform)

        # get correspondences
        if self.return_corr_indices:
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            data_dict["src_corr_indices"] = src_corr_indices
            data_dict["tgt_corr_indices"] = tgt_corr_indices

        if self.scaling_factor is not None:
            src_points = src_points * self.scaling_factor
            tgt_points = tgt_points * self.scaling_factor
            rotation, translation = get_rotation_translation_from_transform(transform)
            translation = translation * self.scaling_factor
            transform = get_transform_from_rotation_translation(rotation, translation)

        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["tgt_points"] = tgt_points.astype(np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones((tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)

        # post transform fn
        if self.transform_fn is not None:
            data_dict = self.transform_fn(data_dict)

        return data_dict
