import os.path as osp

import numpy as np
import torch.utils.data

from vision3d.array_ops import normalize_points_on_xy_plane

from ...utils.io import dump_pickle, ensure_dir, load_pickle
from .utils import divide_point_cloud_into_blocks, random_sample_block

NUM_CLASSES = 13
CLASS_NAMES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)
AREA_NAMES = ("Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6")


def load_s3dis_scene(data_file: str, use_normalized_location: bool = True, use_z_coordinate=False) -> dict:
    data_dict = np.load(data_file)
    points = data_dict["points"].astype(np.float32)
    feats = data_dict["colors"].astype(np.float32)
    min_point = np.amin(points, axis=0)
    max_point = np.amax(points, axis=0)
    points -= min_point
    scene_size = max_point - min_point
    labels = data_dict["labels"].astype(np.int64)
    if use_normalized_location:
        normalized_locations = points / scene_size
        feats = np.concatenate([feats, normalized_locations], axis=1)
    if use_z_coordinate:
        feats = np.concatenate([feats, points[:, 2:]], axis=1)
    return {"points": points, "feats": feats, "labels": labels, "scene_size": scene_size}


class S3DISBlockWiseTrainingDataset(torch.utils.data.Dataset):
    """
    'S3DISBlockWiseTrainingDataset' uses pre-computed npy files from the official PointNet implementation and performs
    data augmentation on the fly. Read <https://github.com/charlesq34/pointnet/blob/master/sem_seg/README.md> for more
    details.

    The code is modified from <https://github.com/yanx27/Pointnet_Pointnet2_pytorch>.
    """

    def __init__(
        self,
        dataset_dir,
        transform=None,
        test_area="Area_5",
        num_samples=4096,
        block_size=1.0,
        use_normalized_location=True,
        use_z_coordinate=False,
        point_threshold=1024,
        pad_points=True,
        training=True,
    ):
        super().__init__()

        assert test_area in AREA_NAMES, f"Invalid test_area: {test_area}."

        self.test_area = test_area
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.num_samples = num_samples
        self.block_size = block_size
        self.use_normalized_location = use_normalized_location
        self.use_z_coordinate = use_z_coordinate
        self.point_threshold = point_threshold
        self.pad_points = pad_points
        self.training = training

        with open(osp.join(self.dataset_dir, "scene_names.txt")) as f:
            lines = f.readlines()
        if self.training:
            self.scene_names = [line.rstrip() for line in lines if self.test_area not in line]
        else:
            self.scene_names = [line.rstrip() for line in lines if self.test_area in line]
        self.num_scenes = len(self.scene_names)
        self.data_files = [osp.join(self.dataset_dir, scene_name + ".npz") for scene_name in self.scene_names]
        self.data_list = [
            load_s3dis_scene(
                data_file, use_normalized_location=self.use_normalized_location, use_z_coordinate=self.use_z_coordinate
            )
            for data_file in self.data_files
        ]

        # resample scenes according to the number of points
        total_points = np.sum([data_dict["points"].shape[0] for data_dict in self.data_list])
        if self.num_samples is not None:
            num_iterations = total_points // self.num_samples
        else:
            # fallback: use 5000 iterations per epoch
            num_iterations = 5000
        self.scene_indices = []
        for i, data_dict in enumerate(self.data_list):
            num_blocks = int(np.floor(data_dict["points"].shape[0] / total_points * num_iterations)) + 1
            self.scene_indices += [i] * num_blocks

    def __getitem__(self, index):
        index = self.scene_indices[index]
        data_dict = self.data_list[index]
        points = data_dict["points"]
        feats = data_dict["feats"]
        labels = data_dict["labels"]
        scene_size = data_dict["scene_size"]

        if not self.training:
            np.random.seed(index)

        indices = random_sample_block(
            points,
            scene_size,
            self.block_size,
            num_samples=self.num_samples,
            point_threshold=self.point_threshold,
            pad_points=self.pad_points,
        )

        points = points[indices].copy()
        points = normalize_points_on_xy_plane(points)
        feats = feats[indices].copy()
        labels = labels[indices].copy()

        new_data_dict = {
            "points": points.astype(np.float32),
            "feats": feats.astype(np.float32),
            "labels": labels.astype(np.int64),
        }

        if self.transform is not None:
            new_data_dict = self.transform(new_data_dict)

        return new_data_dict

    def __len__(self):
        return len(self.scene_indices)


class S3DISBlockWiseTestingDataset(torch.utils.data.Dataset):
    """
    S3DISBlockWiseTestingDataset uses pre-computed npy files from the official PointNet implementation.
    All points in a scene are used for evaluation.
    Small blocks are merged with the nearest block and large blocks are divided into batches.

    The code is modified from <https://github.com/yanx27/Pointnet_Pointnet2_pytorch>.
    """

    def __init__(
        self,
        dataset_dir,
        transform=None,
        test_area="Area_5",
        num_samples=4096,
        block_size=1.0,
        block_stride=0.5,
        use_normalized_location=True,
        use_z_coordinate=False,
        point_threshold=2048,
        pad_points=True,
        cache_data=True,
    ):
        super().__init__()

        assert test_area in AREA_NAMES, f"Invalid test_area: {test_area}."

        self.test_area = test_area
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.num_samples = num_samples
        self.block_size = block_size
        self.block_stride = block_stride
        self.use_normalized_location = use_normalized_location
        self.use_z_coordinate = use_z_coordinate
        self.point_threshold = point_threshold
        self.pad_points = pad_points

        self.cache_data = cache_data
        cache_name = f"s3dis_testing_cached_{block_size:g}b_{block_stride:g}s"
        if num_samples is not None:
            cache_name = cache_name + f"_{num_samples}n"
        if point_threshold is not None:
            cache_name = cache_name + f"_{point_threshold}t"
        self.cache_dir = osp.join(self.dataset_dir, "..", cache_name)
        if self.cache_data:
            print(f"Testing data is cached in '{self.cache_dir}'.")
            ensure_dir(self.cache_dir)

        with open(osp.join(self.dataset_dir, "scene_names.txt")) as f:
            lines = f.readlines()
        self.scene_names = [line.rstrip() for line in lines if self.test_area in line]
        self.num_scenes = len(self.scene_names)
        self.data_files = [osp.join(self.dataset_dir, scene_name + ".npz") for scene_name in self.scene_names]

    def process_one_scene(self, data_file):
        data_dict = load_s3dis_scene(
            data_file, use_normalized_location=self.use_normalized_location, use_z_coordinate=self.use_z_coordinate
        )

        points = data_dict["points"]
        feats = data_dict["feats"]
        labels = data_dict["labels"]
        scene_size = data_dict["scene_size"]

        batch_indices_list = divide_point_cloud_into_blocks(
            points,
            scene_size,
            self.block_size,
            self.block_stride,
            num_samples=self.num_samples,
            point_threshold=self.point_threshold,
            pad_points=self.pad_points,
        )

        batch_list = []
        for batch_indices in batch_indices_list:
            batch_points = points[batch_indices].copy()
            batch_points = normalize_points_on_xy_plane(batch_points)
            batch_feats = feats[batch_indices].copy()
            batch_labels = labels[batch_indices].copy()
            batch_list.append(
                {
                    "points": batch_points.astype(np.float32),
                    "feats": batch_feats.astype(np.float32),
                    "labels": batch_labels.astype(np.int64),
                    "indices": batch_indices.astype(np.int64),
                }
            )

        return batch_list, labels

    def __getitem__(self, index):
        scene_name = self.scene_names[index]
        cache_file = osp.join(self.cache_dir, f"{scene_name}.pkl")

        if osp.exists(cache_file):
            batch_list, labels = load_pickle(cache_file)
        else:
            data_file = self.data_files[index]
            batch_list, labels = self.process_one_scene(data_file)

            if self.cache_data:
                dump_pickle([batch_list, labels], cache_file)

        if self.transform is not None:
            batch_list = [self.transform(batch_dict) for batch_dict in batch_list]

        return {
            "batch_list": batch_list,
            "labels": labels.astype(np.int64),
        }

    def __len__(self):
        return len(self.data_files)
