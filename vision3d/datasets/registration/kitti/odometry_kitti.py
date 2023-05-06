import os.path as osp
import random

import numpy as np
import torch.utils.data

from vision3d.array_ops import (
    get_correspondences,
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
    random_sample_rotation,
)
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import voxel_down_sample


class OdometryKittiPairDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        "train": ["00", "01", "02", "03", "04", "05"],
        "val": ["06", "07"],
        "test": ["08", "09", "10"],
    }

    def __init__(
        self,
        dataset_dir,
        subset,
        max_points=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
        scaling_factor=None,
        use_raw_data=False,
        voxel_size=None,
        min_range=None,
        max_range=None,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.max_points = max_points

        self.scaling_factor = scaling_factor
        self.use_raw_data = use_raw_data
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"positive_radius" is None but "return_corr_indices" is set.')

        self.metadata = load_pickle(osp.join(self.dataset_dir, "metadata", f"{subset}.pkl"))

    def _augment_point_cloud(self, src_points, tgt_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        tgt_points = tgt_points + (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            tgt_points = np.matmul(tgt_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        src_points = src_points * scale
        tgt_points = tgt_points * scale
        translation = translation * scale
        # random shift
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        tgt_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_points = src_points + src_shift
        tgt_points = tgt_points + tgt_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + tgt_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return src_points, tgt_points, transform

    def _load_point_cloud(self, file_name):
        if self.use_raw_data:
            split_path = file_name.split("/")
            split_path[-3] = "sequences"
            split_path.insert(-1, "velodyne")
            split_path[-1] = split_path[-1].replace("npy", "bin")
            file_name = osp.join("/", *split_path)
            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]
            if self.voxel_size is not None:
                points = voxel_down_sample(points, self.voxel_size)
        else:
            points = np.load(file_name)
        if self.min_range is not None:
            masks = np.amin(points, axis=1) >= self.min_range
            points = points[masks]
        if self.max_range is not None:
            masks = np.amax(points, axis=1) <= self.max_range
            points = points[masks]
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict["seq_id"] = metadata["seq_id"]
        data_dict["src_frame"] = metadata["frame1"]
        data_dict["tgt_frame"] = metadata["frame0"]

        src_points = self._load_point_cloud(osp.join(self.dataset_dir, metadata["pcd1"]))
        tgt_points = self._load_point_cloud(osp.join(self.dataset_dir, metadata["pcd0"]))
        transform = metadata["transform"]

        if self.use_augmentation:
            src_points, tgt_points, transform = self._augment_point_cloud(src_points, tgt_points, transform)

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
        data_dict["src_feats"] = np.ones(shape=(src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)
