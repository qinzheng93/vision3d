import os.path as osp
from typing import Optional

import numpy as np
import torch.utils.data

from vision3d.array_ops import (
    apply_transform,
    point_cloud_overlap,
    get_correspondences,
    inverse_transform,
    normalize_points,
    random_crop_points_from_viewpoint,
    random_crop_points_with_plane,
    random_jitter_points,
    random_sample_points,
    random_sample_transform,
    random_sample_viewpoint,
    random_shuffle_points,
    regularize_normals,
)
from vision3d.utils.io import load_pickle
from vision3d.utils.open3d import estimate_normals, voxel_down_sample


class ModelNetPairDataset(torch.utils.data.Dataset):
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]
    # fmt: on

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
        class_indices: str = "all",
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        return_corr_indices: bool = False,
        matching_radius: float = 0.05,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        use_estimated_normals: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super().__init__()

        assert subset in ["trainval", "train", "val", "test"]
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
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.use_estimated_normals = use_estimated_normals
        self.overfitting_index = overfitting_index

        data_list = load_pickle(osp.join(dataset_dir, f"{subset}.pkl"))
        data_list = [x for x in data_list if x["label"] in self.class_indices]
        if overfitting_index is not None and deterministic:
            data_list = [data_list[overfitting_index]]
        self.data_list = data_list

    def get_class_indices(self, class_indices, asymmetric):
        """Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if isinstance(class_indices, str):
            assert class_indices in ["all", "seen", "unseen"]
            if class_indices == "all":
                class_indices = list(range(40))
            elif class_indices == "seen":
                class_indices = list(range(20))
            else:
                class_indices = list(range(20, 40))
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        return class_indices

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        data_dict: dict = self.data_list[index]
        raw_points = data_dict["points"].copy()
        raw_normals = data_dict["normals"].copy()
        label = data_dict["label"]

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # once sample on raw point cloud
        if not self.twice_sample:
            raw_points, raw_normals = random_sample_points(raw_points, self.num_points, normals=raw_normals)

        # split tgterence and source point cloud
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

            # check overlap
            is_available = True
            overlap = point_cloud_overlap(src_points, tgt_points, transform, positive_radius=self.matching_radius)
            if self.min_overlap is not None:
                is_available = is_available and overlap >= self.min_overlap
            else:
                is_available = is_available and overlap > 0
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
            # voxel downsample tgterence point cloud
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

        if self.use_estimated_normals:
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)
            tgt_normals = estimate_normals(tgt_points)
            tgt_normals = regularize_normals(tgt_points, tgt_normals)

        if self.return_normals:
            new_data_dict["raw_normals"] = raw_normals.astype(np.float32)
            new_data_dict["src_normals"] = src_normals.astype(np.float32)
            new_data_dict["tgt_normals"] = tgt_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict["src_feats"] = np.ones_like(src_points[:, :1]).astype(np.float32)
            new_data_dict["tgt_feats"] = np.ones_like(tgt_points[:, :1]).astype(np.float32)

        if self.return_corr_indices:
            src_corr_indices, tgt_corr_indices = get_correspondences(
                src_points, tgt_points, transform, self.matching_radius
            )
            new_data_dict["src_corr_indices"] = src_corr_indices
            new_data_dict["tgt_corr_indices"] = tgt_corr_indices

        return new_data_dict

    def __len__(self):
        return len(self.data_list)
