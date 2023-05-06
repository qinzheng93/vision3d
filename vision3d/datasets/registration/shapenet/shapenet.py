import os.path as osp
from typing import Optional

import numpy as np
import torch.utils.data

from vision3d.array_ops import (
    apply_transform,
    inverse_transform,
    normalize_points,
    point_cloud_overlap,
    random_crop_points_from_viewpoint,
    random_crop_points_with_plane,
    random_jitter_points,
    random_sample_points,
    random_sample_transform,
    random_sample_viewpoint,
    random_shuffle_points,
    regularize_normals,
)
from vision3d.utils.open3d import estimate_normals, voxel_down_sample


class ShapeNetPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = "plane",
        asymmetric: bool = True,
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"]
        assert crop_method in ["plane", "point"]

        self.dataset_dir = dataset_dir
        self.subset = subset

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal

        with open(osp.join(dataset_dir, "metadata", f"{subset}.txt")) as f:
            filenames = f.readlines()
            filenames = [filename.strip() for filename in filenames]
        self.filenames = filenames

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_dict: dict = np.load(osp.join(self.dataset_dir, "data", filename))
        raw_points = data_dict["points"].copy()
        raw_normals = data_dict["normals"].copy()
        label = filename.split("/")[0]

        # swap axes to make (x->front, y->right, z->up) from (x->left, y->up, z->back)
        raw_points = np.stack([-raw_points[:, 2], -raw_points[:, 0], raw_points[:, 1]], axis=1)
        raw_normals = np.stack([-raw_normals[:, 2], -raw_normals[:, 0], raw_normals[:, 1]], axis=1)

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # once sample on raw point cloud
        if not self.twice_sample:
            raw_points, raw_normals = random_sample_points(raw_points, self.num_points, normals=raw_normals)

        # split target and source point cloud
        tgt_points = raw_points.copy()
        tgt_normals = raw_normals.copy()

        # twice transform
        if self.twice_transform:
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            tgt_points, tgt_normals = apply_transform(tgt_points, transform, normals=tgt_normals)

        src_points = tgt_points.copy()
        src_normals = tgt_normals.copy()

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        raw_src_points = src_points
        raw_src_normals = src_normals
        raw_tgt_points = tgt_points
        raw_tgt_normals = tgt_normals

        while True:
            src_points = raw_src_points
            src_normals = raw_src_normals
            tgt_points = raw_tgt_points
            tgt_normals = raw_tgt_normals
            # crop
            if self.keep_ratio is not None:
                if self.crop_method == "plane":
                    src_points, src_normals = random_crop_points_with_plane(
                        src_points, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                    tgt_points, tgt_normals = random_crop_points_with_plane(
                        tgt_points, keep_ratio=self.keep_ratio, normals=tgt_normals
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    src_points, src_normals = random_crop_points_from_viewpoint(
                        src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                    tgt_points, tgt_normals = random_crop_points_from_viewpoint(
                        tgt_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=tgt_normals
                    )

            # data check
            is_available = True
            # check overlap
            if self.check_overlap:
                overlap = point_cloud_overlap(src_points, tgt_points, transform, positive_radius=0.05)
                if self.min_overlap is not None:
                    is_available = is_available and overlap >= self.min_overlap
                if self.max_overlap is not None:
                    is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)
            tgt_points, tgt_normals = random_sample_points(tgt_points, self.num_points, normals=tgt_normals)

        # random jitter
        if self.noise_magnitude is not None:
            src_points = random_jitter_points(src_points, sigma=0.01, scale=self.noise_magnitude)
            tgt_points = random_jitter_points(tgt_points, sigma=0.01, scale=self.noise_magnitude)

        # random shuffle
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)
        tgt_points, tgt_normals = random_shuffle_points(tgt_points, normals=tgt_normals)

        if self.voxel_size is not None:
            # voxel downsample target point cloud
            src_points, src_normals = voxel_down_sample(src_points, self.voxel_size, normals=src_normals)
            tgt_points, tgt_normals = voxel_down_sample(tgt_points, self.voxel_size, normals=tgt_normals)

        new_data_dict = {
            "raw_points": raw_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "tgt_points": tgt_points.astype(np.float32),
            "transform": transform.astype(np.float32),
            "label": int(label),
            "index": int(index),
        }

        if self.estimate_normal:
            tgt_normals = estimate_normals(tgt_points)
            tgt_normals = regularize_normals(tgt_points, tgt_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict["raw_normals"] = raw_normals.astype(np.float32)
            new_data_dict["src_normals"] = src_normals.astype(np.float32)
            new_data_dict["tgt_normals"] = tgt_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict["src_feats"] = np.ones_like(src_points[:, :1]).astype(np.float32)
            new_data_dict["tgt_feats"] = np.ones_like(tgt_points[:, :1]).astype(np.float32)

        return new_data_dict

    def __len__(self):
        return len(self.filenames)
