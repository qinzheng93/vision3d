import os.path as osp
from typing import Optional

import numpy as np
import torch.utils.data

from vision3d.array_ops.point_cloud_utils import (
    normalize_points,
    random_crop_points_with_plane,
    random_jitter_points,
    random_sample_points,
    random_shuffle_points,
)
from vision3d.utils.io import load_pickle


class ModelNetDataset(torch.utils.data.Dataset):
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
        keep_ratio: float = 0.7,
        noise_magnitude: Optional[float] = None,
        asymmetric: bool = True,
        class_indices: str = "all",
        deterministic: bool = False,
    ):
        super().__init__()

        assert subset in ["trainval", "train", "val", "test"]

        self.dataset_dir = dataset_dir
        self.subset = subset

        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.asymmetric = asymmetric
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
        self.deterministic = deterministic
        self.num_points = int(np.floor(num_points * keep_ratio + 0.5))

        data_list = load_pickle(osp.join(dataset_dir, f"{subset}.pkl"))
        data_list = [x for x in data_list if x["label"] in self.class_indices]
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
        data_dict: dict = self.data_list[index]
        raw_points = data_dict["points"].copy()
        raw_normals = data_dict["normals"].copy()
        label = data_dict["label"]

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # split reference and source point cloud
        points = raw_points.copy()
        normals = raw_normals.copy()

        # crop
        if self.keep_ratio is not None:
            points, normals = random_crop_points_with_plane(points, keep_ratio=self.keep_ratio, normals=normals)

        points, normals = random_sample_points(points, self.num_points, normals=normals)

        # random jitter
        if self.noise_magnitude is not None:
            points = random_jitter_points(points, sigma=0.01, scale=self.noise_magnitude)

        # random shuffle
        points, normals = random_shuffle_points(points, normals=normals)

        return {
            "raw_points": raw_points.astype(np.float32),
            "raw_normals": raw_normals.astype(np.float32),
            "points": points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "label": int(label),
            "index": int(index),
        }

    def __len__(self):
        return len(self.data_list)
