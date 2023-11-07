import os.path as osp
import random
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    get_2d3d_correspondences_mutual,
    get_2d3d_correspondences_radius,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from vision3d.utils.io import load_pickle, read_depth_image, read_image


class RGBDScenes2D3DHardPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.0375,
        scene_name: Optional[str] = None,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        scale_augmentation: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"], f"Bad subset name: {subset}."
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))

        if scene_name is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

        if overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] >= overlap_threshold]

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_method = matching_method
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.scale_augmentation = scale_augmentation

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index: int):
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["cloud_file"] = metadata["cloud_file"]
        data_dict["overlap"] = metadata["overlap"]
        data_dict["image_id"] = osp.basename(metadata["image_file"]).split(".")[0].split("_")[1]
        data_dict["cloud_id"] = osp.basename(metadata["cloud_file"]).split(".")[0].split("_")[1]

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)
        transform = metadata["cloud_to_image"]

        # read image
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read points
        points = np.load(osp.join(self.data_dir, metadata["cloud_file"]))
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        if self.scale_augmentation and random.random() > 0.5:
            # augment image
            scale = random.uniform(1.0, 1.2)
            raw_image_h = image.shape[0]
            raw_image_w = image.shape[1]
            new_image_h = int(raw_image_h * scale)
            new_image_w = int(raw_image_w * scale)
            start_h = new_image_h // 2 - raw_image_h // 2
            end_h = start_h + raw_image_h
            start_w = new_image_w // 2 - raw_image_w // 2
            end_w = start_w + raw_image_w
            image = cv2.resize(image, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            image = image[start_h:end_h, start_w:end_w]
            depth = cv2.resize(depth, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            depth = depth[start_h:end_h, start_w:end_w]
            intrinsics[0, 0] = intrinsics[0, 0] * scale
            intrinsics[1, 1] = intrinsics[1, 1] * scale

        image -= image.mean()

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        return data_dict
