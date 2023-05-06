import os.path as osp

import h5py
import numpy as np
import torch.utils.data


NUM_CLASSES = 40
# fmt: off
CLASS_NAMES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
    'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
    'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
    'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]
# fmt: on


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, transform):
        super().__init__()

        assert split in ["train", "test"]

        self.dataset_dir = dataset_dir
        self.transform = transform

        with open(osp.join(self.dataset_dir, "{}_files.txt".format(split))) as f:
            lines = f.readlines()
        data_files = [line.rstrip() for line in lines]

        self.data = self.read_data(data_files)

    def __getitem__(self, index):
        data_dict = {
            "points": self.data["points"][index].copy(),
            "normals": self.data["normals"][index].copy(),
            "label": self.data["labels"][index],
        }
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        return self.data["points"].shape[0]

    def read_data(self, data_files):
        all_points = []
        all_normals = []
        all_labels = []
        for data_file in data_files:
            h5file = h5py.File(osp.join(self.dataset_dir, data_file), "r")
            points = h5file["data"][:]
            normals = h5file["normal"][:]
            labels = h5file["label"][:].flatten().astype(np.int)
            all_points.append(points)
            all_normals.append(normals)
            all_labels.append(labels)
        all_points = np.concatenate(all_points, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return {"points": all_points, "normals": all_normals, "labels": all_labels}
