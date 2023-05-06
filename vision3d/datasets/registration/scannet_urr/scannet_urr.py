import os.path as osp
from typing import Optional

import cv2
import ipdb
import numpy as np
import torch.utils.data
from skimage.transform import resize

from vision3d.array_ops import back_project, compose_transforms, inverse_transform
from vision3d.utils.io import load_pickle, read_depth_image, read_image
from vision3d.utils.open3d import voxel_down_sample


class ScanNetPairURRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        voxel_size: float = 0.025,
        depth_limit: Optional[float] = 6.0,
        scaling_factor: float = 1000.0,
        image_h: int = 480,
        image_w: int = 640,
        max_points: Optional[int] = None,
        overfitting: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "train-10", "train-5", "val", "test"], f"Unsupported data subset: {subset}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.voxel_size = voxel_size
        self.depth_limit = depth_limit
        self.scaling_factor = scaling_factor
        self.image_h = image_h
        self.image_w = image_w
        self.max_points = max_points
        self.overfitting = overfitting

        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))

    def __len__(self):
        return len(self.metadata_list)

    def read_frame(self, scene_name, frame_id):
        # read data
        scene_dir = osp.join(self.data_dir, self.subset, scene_name)
        color_img = cv2.imread(osp.join(scene_dir, "color", f"{frame_id}.jpg"))  # (H, W, 3)
        depth_img = read_depth_image(osp.join(scene_dir, "depth", f"{frame_id}.png"))  # (H, W)
        pose = np.loadtxt(osp.join(scene_dir, "pose", f"{frame_id}.txt"))

        return color_img, depth_img, pose

    def get_points(self, depth_img, intrinsics):
        # convert to point cloud
        points = back_project(depth_img, intrinsics, scaling_factor=self.scaling_factor, depth_limit=self.depth_limit)

        # voxelize
        if self.voxel_size is not None:
            points = voxel_down_sample(points, self.voxel_size)

        # random sample
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[indices]

        return points

    def color_transform(self, src_img, tgt_img, intrinsics):
        # TODO: center crop as in UR&R code
        # 1. resize
        scale_h = self.image_h / src_img.shape[0]
        scale_w = self.image_w / src_img.shape[1]

        new_src_img = cv2.resize(src_img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        new_tgt_img = cv2.resize(tgt_img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

        new_intrinsics = np.eye(3)
        new_intrinsics[0, 0] = scale_w * intrinsics[0, 0]
        new_intrinsics[0, 2] = scale_w * intrinsics[0, 2]
        new_intrinsics[1, 1] = scale_h * intrinsics[1, 1]
        new_intrinsics[1, 2] = scale_h * intrinsics[1, 2]

        # 2. normalize color image
        new_src_img = 2.0 * (new_src_img / 255.0 - 0.5)
        new_tgt_img = 2.0 * (new_tgt_img / 255.0 - 0.5)

        return new_src_img, new_tgt_img, new_intrinsics

    def depth_transform(self, src_img, tgt_img, intrinsics):
        # TODO: center crop as in UR&R code
        # 1. resize
        scale_h = self.image_h / src_img.shape[0]
        scale_w = self.image_w / src_img.shape[1]

        new_src_img = cv2.resize(src_img, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)
        new_tgt_img = cv2.resize(tgt_img, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)

        new_intrinsics = np.eye(3)
        new_intrinsics[0, 0] = scale_w * intrinsics[0, 0]
        new_intrinsics[0, 2] = scale_w * intrinsics[0, 2]
        new_intrinsics[1, 1] = scale_h * intrinsics[1, 1]
        new_intrinsics[1, 2] = scale_h * intrinsics[1, 2]

        return new_src_img, new_tgt_img, new_intrinsics

    def pcd_transform(self, points):
        # TODO: point transform
        pass

    def __getitem__(self, index):
        if self.overfitting:
            index = 100000

        metadata_dict: dict = self.metadata_list[index]

        scene_name = metadata_dict["scene_name"]
        src_frame = metadata_dict["src_frame"]
        tgt_frame = metadata_dict["tgt_frame"]

        color_intrinsics = metadata_dict["depth_intrinsics"]
        depth_intrinsics = metadata_dict["depth_intrinsics"]

        src_color_img, src_depth_img, src_pose = self.read_frame(scene_name, src_frame)
        tgt_color_img, tgt_depth_img, tgt_pose = self.read_frame(scene_name, tgt_frame)

        src_points = self.get_points(src_depth_img, depth_intrinsics)
        tgt_points = self.get_points(tgt_depth_img, depth_intrinsics)

        tgt_inv_pose = inverse_transform(tgt_pose)
        transform = compose_transforms(src_pose, tgt_inv_pose)

        src_color_img, tgt_color_img, color_intrinsics = self.color_transform(
            src_color_img, tgt_color_img, color_intrinsics
        )

        src_depth_img, tgt_depth_img, depth_intrinsics = self.depth_transform(
            src_depth_img, tgt_depth_img, depth_intrinsics
        )

        src_feats = np.ones(shape=(src_points.shape[0], 1))
        tgt_feats = np.ones(shape=(tgt_points.shape[0], 1))

        output_dict = {
            "scene_name": scene_name,
            "src_frame": src_frame,
            "tgt_frame": tgt_frame,
            "color_intrinsics": color_intrinsics.astype(np.float32),
            "depth_intrinsics": depth_intrinsics.astype(np.float32),
            "src_color_img": src_color_img.astype(np.float32),
            "tgt_color_img": tgt_color_img.astype(np.float32),
            "src_depth_img": src_depth_img.astype(np.float32),
            "tgt_depth_img": tgt_depth_img.astype(np.float32),
            "src_pose": src_pose.astype(np.float32),
            "tgt_pose": tgt_pose.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "tgt_points": tgt_points.astype(np.float32),
            "src_feats": src_feats.astype(np.float32),
            "tgt_feats": tgt_feats.astype(np.float32),
            "transform": transform.astype(np.float32),
        }

        return output_dict
