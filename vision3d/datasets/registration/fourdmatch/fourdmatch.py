import glob
import os.path as osp
import random

import numpy as np
import torch.utils.data

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_rotation,
    random_sample_small_transform,
)
from vision3d.utils.open3d import voxel_down_sample


class FourDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        subset,
        transform_fn=None,
        voxel_size=None,
        use_augmentation=False,
        augmentation_noise=0.002,
        augmentation_rotation=1.0,
        use_weak_augmentation=False,
        return_corr_indices=False,
        shape_names=None,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.subset = subset
        filenames = sorted(glob.glob(osp.join(self.dataset_dir, subset, "*", "*.npz")))
        if shape_names is not None:
            filenames = [filename for filename in filenames if filename.split("/")[-2] in shape_names]
        self.filenames = filenames

        self.transform_fn = transform_fn
        self.voxel_size = voxel_size
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.use_weak_augmentation = use_weak_augmentation
        self.return_corr_indices = return_corr_indices

    def __len__(self):
        return len(self.filenames)

    def weakly_augment_point_cloud(self, src_points, tgt_points, scene_flows, transform):
        deformed_src_points = src_points + scene_flows
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
            deformed_src_points = apply_transform(deformed_src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise
        scene_flows = deformed_src_points - src_points

        return src_points, tgt_points, scene_flows, transform

    def augment_point_cloud(self, src_points, tgt_points, scene_flows, transform):
        deformed_src_points = src_points + scene_flows
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_transform = get_transform_from_rotation_translation(aug_rotation, None)
        if random.random() > 0.5:
            tgt_points = apply_transform(tgt_points, aug_transform)
            transform = compose_transforms(transform, aug_transform)
        else:
            src_points = apply_transform(src_points, aug_transform)
            deformed_src_points = apply_transform(deformed_src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise
        scene_flows = deformed_src_points - src_points

        return src_points, tgt_points, scene_flows, transform

    def __getitem__(self, index):
        new_data_dict = {}
        filename = self.filenames[index]
        data_dict = np.load(filename)

        split_filename = filename.split("/")
        new_data_dict["shape_name"] = split_filename[-2]
        new_data_dict["src_frame"] = split_filename[-1][:9]
        new_data_dict["tgt_frame"] = split_filename[-1][10:19]
        new_data_dict["overlap"] = data_dict["s_overlap_rate"]
        if "metric_index" in data_dict:
            new_data_dict["test_indices"] = data_dict["metric_index"].flatten()
        if self.return_corr_indices:
            corr_indices = data_dict["correspondences"]
            new_data_dict["gt_src_corr_indices"] = corr_indices[:, 0]
            new_data_dict["gt_tgt_corr_indices"] = corr_indices[:, 1]
        new_data_dict["filename"] = "/".join(split_filename[-2:])

        src_points = data_dict["s_pc"]
        tgt_points = data_dict["t_pc"]
        scene_flows = data_dict["s2t_flow"]

        #  tgt_points = rotation * (src_points + scene_flows) + translation
        rotation = data_dict["rot"]
        translation = data_dict["trans"].flatten()
        transform = get_transform_from_rotation_translation(rotation, translation)

        if self.use_augmentation:
            if self.use_weak_augmentation:
                src_points, tgt_points, scene_flows, transform = self.weakly_augment_point_cloud(
                    src_points, tgt_points, scene_flows, transform
                )
            else:
                src_points, tgt_points, scene_flows, transform = self.augment_point_cloud(
                    src_points, tgt_points, scene_flows, transform
                )

        if self.voxel_size is not None:
            new_data_dict["raw_src_points"] = src_points.astype(np.float32)
            new_data_dict["raw_tgt_points"] = tgt_points.astype(np.float32)
            new_data_dict["raw_scene_flows"] = scene_flows.astype(np.float32)
            src_points = voxel_down_sample(src_points, self.voxel_size)
            tgt_points = voxel_down_sample(tgt_points, self.voxel_size)
            new_data_dict["src_points"] = src_points.astype(np.float32)
            new_data_dict["tgt_points"] = tgt_points.astype(np.float32)
        else:
            new_data_dict["src_points"] = src_points.astype(np.float32)
            new_data_dict["tgt_points"] = tgt_points.astype(np.float32)
            new_data_dict["scene_flows"] = scene_flows.astype(np.float32)

        new_data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        new_data_dict["tgt_feats"] = np.ones((tgt_points.shape[0], 1), dtype=np.float32)
        new_data_dict["transform"] = transform.astype(np.float32)

        if self.transform_fn is not None:
            new_data_dict = self.transform_fn(new_data_dict)

        return new_data_dict
