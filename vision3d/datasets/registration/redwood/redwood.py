import os
import os.path as osp
from typing import Optional

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

from vision3d.array_ops import inverse_transform, point_cloud_overlap
from vision3d.utils.open3d import voxel_down_sample


class RedwoodPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        scene_name: str,
        voxel_size: Optional[float] = 0.025,
        max_points: Optional[int] = None,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.scene_name = scene_name
        self.fragment_dir = osp.join(self.dataset_dir, f"{scene_name}-simulated", "fragments")
        self.voxel_size = voxel_size
        self.max_points = max_points

        filenames = os.listdir(self.fragment_dir)
        filenames = [filename for filename in filenames if filename.endswith(".ply")]
        filenames = sorted(filenames, key=lambda x: x[:-4].split("_")[-1])
        self.filenames = filenames
        self.num_fragments = len(self.filenames)

        pose_list = []
        metadata_list = []
        for i in range(self.num_fragments):
            src_pose = np.load(osp.join(self.fragment_dir, self.filenames[i].replace("ply", "npy")))
            pose_list.append(src_pose)
            for j in range(i + 1, self.num_fragments):
                tgt_pose = np.load(osp.join(self.fragment_dir, self.filenames[j].replace("ply", "npy")))
                transform = inverse_transform(tgt_pose) @ src_pose
                metadata = {
                    "scene_name": self.scene_name,
                    "src_frame": i,
                    "tgt_frame": j,
                    "src_pcd": self.filenames[i],
                    "tgt_pcd": self.filenames[j],
                    "src_pose": src_pose,
                    "tgt_pose": tgt_pose,
                    "transform": transform,
                }
                metadata_list.append(metadata)
        self.pose_list = pose_list
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def read_point_cloud(self, filename):
        pcd = o3d.io.read_point_cloud(osp.join(self.fragment_dir, filename))
        points = np.asarray(pcd.points)
        if self.voxel_size is not None:
            points = voxel_down_sample(points, self.voxel_size)
        if self.max_points is not None:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["src_frame"] = metadata["src_frame"]
        data_dict["tgt_frame"] = metadata["tgt_frame"]

        src_points = self.read_point_cloud(metadata["src_pcd"])
        tgt_points = self.read_point_cloud(metadata["tgt_pcd"])
        transform = metadata["transform"].astype(np.float32)

        overlap = point_cloud_overlap(src_points, tgt_points, transform=transform, positive_radius=0.0375)

        data_dict["src_points"] = src_points.astype(np.float32)
        data_dict["tgt_points"] = tgt_points.astype(np.float32)
        data_dict["src_feats"] = np.ones(shape=(src_points.shape[0], 1), dtype=np.float32)
        data_dict["tgt_feats"] = np.ones(shape=(tgt_points.shape[0], 1), dtype=np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["overlap"] = overlap

        return data_dict
