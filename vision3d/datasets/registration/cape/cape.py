import json
import os.path as osp
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

_subset_names = [
    "train",
    "val",
    "tet",
    "train-f-5cm",
    "val-f-5cm",
    "test-f-5cm",
    "train-f-10cm",
    "val-f-10cm",
    "test-f-10cm",
    "all-f-10cm",
]


class CapePairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        transform_fn: Optional[Callable] = None,
        voxel_size: Optional[float] = None,
        return_corr_indices: bool = False,
        matching_radius: float = 0.0375,
    ):
        super().__init__()

        assert subset in _subset_names, f"Bad subset name: {subset}."

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset

        self.transform_fn = transform_fn
        self.voxel_size = voxel_size
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius

        with open(osp.join(self.metadata_dir, f"{self.subset}.json"), "r") as f:
            self.metadata_list = json.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index):
        new_data_dict = {}

        filepath = self.metadata_list[index]
        data_dict = np.load(osp.join(self.dataset_dir, filepath))

        new_data_dict["filename"] = osp.basename(filepath)

        src_points = data_dict["src_points"]
        tgt_points = data_dict["tgt_points"]
        scene_flows = data_dict["scene_flows"]
        flow_norms = np.linalg.norm(scene_flows, axis=1)
        mean_flow = flow_norms.mean()
        max_flow = flow_norms.max()

        new_data_dict["src_points"] = src_points.astype(np.float32)
        new_data_dict["tgt_points"] = tgt_points.astype(np.float32)
        new_data_dict["src_feats"] = np.ones(shape=(src_points.shape[0], 1), dtype=np.float32)
        new_data_dict["tgt_feats"] = np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32)
        new_data_dict["scene_flows"] = scene_flows.astype(np.float32)
        new_data_dict["mean_flow"] = mean_flow
        new_data_dict["max_flow"] = max_flow
        new_data_dict["transform"] = np.eye(4).astype(np.float32)

        if self.transform_fn is not None:
            new_data_dict = self.transform_fn(new_data_dict)

        return new_data_dict
