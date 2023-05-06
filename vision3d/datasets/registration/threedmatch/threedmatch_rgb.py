import os.path as osp
import random
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.utils.data

from vision3d.array_ops import (
    apply_rotation,
    apply_transform,
    compose_transforms,
    get_correspondences,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_rotation,
    random_sample_small_transform,
)
from vision3d.utils.io import load_pickle, read_image


class ThreeDMatchRgbPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: Optional[Callable] = None,
        max_points: Optional[int] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1.0,
        use_weak_augmentation: bool = False,
        overlap_threshold: Optional[float] = None,
        return_corr_indices: bool = False,
        matching_radius: Optional[float] = None,
        deterministic: bool = False,
        image_h: int = 240,
        image_w: int = 320,
        image_frames: tuple = (0, 4),
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.frame_dir = osp.join(self.data_dir, "frames")

        self.transform_fn = transform_fn
        self.subset = subset
        self.max_points = max_points
        self.overlap_threshold = overlap_threshold
        self.deterministic = deterministic
        self.image_h = image_h
        self.image_w = image_w
        self.original_h = 480
        self.original_w = 640
        self.scale_h = self.image_h / self.original_h
        self.scale_w = self.image_w / self.original_w
        self.image_frames = image_frames

        assert len(image_frames) > 0, "No image frames are selected."

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices:
            assert self.matching_radius is not None, "'matching_radius' is None but 'return_corr_indices' is True."

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.use_weak_augmentation = use_weak_augmentation

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{subset}-rgb.pkl"))

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

    def weakly_augment_point_cloud(self, src_points, tgt_points, src_pose, tgt_pose):
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
            # transform tgt
            tgt_center = tgt_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -tgt_center)
            add_center = get_transform_from_rotation_translation(None, tgt_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            tgt_points = apply_transform(tgt_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            tgt_pose = compose_transforms(inv_aug_transform, tgt_pose)
        else:
            # transform src
            src_center = src_points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -src_center)
            add_center = get_transform_from_rotation_translation(None, src_center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            src_points = apply_transform(src_points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            src_pose = compose_transforms(inv_aug_transform, src_pose)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, src_pose, tgt_pose

    def augment_point_cloud(self, src_points, tgt_points, src_pose, tgt_pose):
        """Augment point clouds.

        tgt_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_transform = get_transform_from_rotation_translation(aug_rotation, None)
        inv_aug_transform = inverse_transform(aug_transform)
        if random.random() > 0.5:
            # transform tgt
            tgt_points = apply_rotation(tgt_points, aug_rotation)
            tgt_pose = compose_transforms(inv_aug_transform, tgt_pose)
        else:
            # transform src
            src_points = apply_transform(src_points, aug_transform)
            src_pose = compose_transforms(inv_aug_transform, src_pose)

        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        tgt_points += (np.random.rand(tgt_points.shape[0], 3) - 0.5) * self.aug_noise

        return src_points, tgt_points, src_pose, tgt_pose

    def __getitem__(self, index):
        # deterministic
        if self.deterministic:
            np.random.seed(index)

        data_dict = {}

        # metadata
        metadata: dict = self.metadata_list[index]

        scene_name = metadata["scene_name"]
        src_frame = metadata["src_frame"]
        tgt_frame = metadata["tgt_frame"]
        intrinsics = metadata["intrinsics"]

        intrinsics[0, 0] *= self.scale_w
        intrinsics[0, 2] *= self.scale_w
        intrinsics[1, 1] *= self.scale_h
        intrinsics[1, 2] *= self.scale_h

        data_dict["scene_name"] = scene_name
        data_dict["src_frame"] = src_frame
        data_dict["tgt_frame"] = tgt_frame
        data_dict["overlap"] = metadata["overlap"]

        # get point cloud
        src_points = self.load_point_cloud(metadata["src_path"])
        tgt_points = self.load_point_cloud(metadata["tgt_path"])
        src_pose = metadata["src_pose"]
        tgt_pose = metadata["tgt_pose"]

        # augmentation
        if self.use_augmentation:
            if self.use_weak_augmentation:
                src_points, tgt_points, src_pose, tgt_pose = self.weakly_augment_point_cloud(
                    src_points, tgt_points, src_pose, tgt_pose
                )
            else:
                src_points, tgt_points, src_pose, tgt_pose = self.augment_point_cloud(
                    src_points, tgt_points, src_pose, tgt_pose
                )

        # get transform
        transform = compose_transforms(src_pose, inverse_transform(tgt_pose))

        # get correspondences
        if self.return_corr_indices:
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            data_dict["src_corr_indices"] = src_corr_indices
            data_dict["tgt_corr_indices"] = tgt_corr_indices

        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["tgt_points"] = tgt_points.astype(np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones((tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["src_pose"] = src_pose.astype(np.float32)
        data_dict["tgt_pose"] = tgt_pose.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)

        # get images
        src_images = []
        tgt_images = []
        src_transforms = []
        tgt_transforms = []
        frame_dir = osp.join(self.frame_dir, scene_name)
        for image_frame in self.image_frames:
            src_image = read_image(osp.join(frame_dir, f"cloud_bin_{src_frame}_{image_frame}.color.png"))
            tgt_image = read_image(osp.join(frame_dir, f"cloud_bin_{tgt_frame}_{image_frame}.color.png"))
            src_image = cv2.resize(src_image, (self.image_w, self.image_h))
            tgt_image = cv2.resize(tgt_image, (self.image_w, self.image_h))

            src_img_pose = np.loadtxt(osp.join(frame_dir, f"cloud_bin_{src_frame}_{image_frame}.pose.txt"))
            tgt_img_pose = np.loadtxt(osp.join(frame_dir, f"cloud_bin_{tgt_frame}_{image_frame}.pose.txt"))
            src_transform = compose_transforms(src_pose, inverse_transform(src_img_pose))
            tgt_transform = compose_transforms(tgt_pose, inverse_transform(tgt_img_pose))

            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_transforms.append(src_transform)
            tgt_transforms.append(tgt_transform)

        src_images = np.stack(src_images, axis=0)  # (I, H, W, 3)
        tgt_images = np.stack(tgt_images, axis=0)  # (I, H, W, 3)
        src_transforms = np.stack(src_transforms, axis=0)  # (I, 4, 4)
        tgt_transforms = np.stack(tgt_transforms, axis=0)  # (I, 4, 4)

        data_dict["src_images"] = src_images.astype(np.float32)
        data_dict["tgt_images"] = tgt_images.astype(np.float32)
        data_dict["src_transforms"] = src_transforms.astype(np.float32)
        data_dict["tgt_transforms"] = tgt_transforms.astype(np.float32)
        data_dict["intrinsics"] = intrinsics.astype(np.float32)

        # post transform fn
        if self.transform_fn is not None:
            data_dict = self.transform_fn(data_dict)

        return data_dict
