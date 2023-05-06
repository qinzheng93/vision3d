import os.path as osp

import torch.utils.data

from vision3d.utils.io import load_pickle

NUM_CLASSES = 16

NUM_PARTS = 50

# fmt: off
CLASS_NAMES = (
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug',
    'Pistol', 'Rocket', 'Skateboard', 'Table'
)

PART_NAMES = (
    'Airplane0', 'Airplane1', 'Airplane2', 'Airplane3',
    'Bag0', 'Bag1',
    'Cap0', 'Cap1', 'Car0', 'Car1', 'Car2', 'Car3',
    'Chair0', 'Chair1', 'Chair2', 'Chair3',
    'Earphone0', 'Earphone1', 'Earphone2',
    'Guitar0', 'Guitar1', 'Guitar2',
    'Knife0', 'Knife1',
    'Lamp0', 'Lamp1', 'Lamp2', 'Lamp3',
    'Laptop0', 'Laptop1',
    'Motorbike0', 'Motorbike1', 'Motorbike2', 'Motorbike3', 'Motorbike4', 'Motorbike5',
    'Mug0', 'Mug1',
    'Pistol0', 'Pistol1', 'Pistol2',
    'Rocket0', 'Rocket1', 'Rocket2',
    'Skateboard0', 'Skateboard1', 'Skateboard2',
    'Table0', 'Table1', 'Table2'
)
# fmt: on

CLASS_ID_TO_PART_IDS = tuple(
    [(i for i, part_name in enumerate(PART_NAMES) if class_name in part_name) for class_name in CLASS_NAMES]
)

PART_ID_TO_CLASS_ID = tuple([CLASS_NAMES.index(part_name[:-1]) for part_name in PART_NAMES])

SYNSET_TO_CLASS_NAME = {
    "02691156": "Airplane",
    "02773838": "Bag",
    "02954340": "Cap",
    "02958343": "Car",
    "03001627": "Chair",
    "03261776": "Earphone",
    "03467517": "Guitar",
    "03624134": "Knife",
    "03636649": "Lamp",
    "03642806": "Laptop",
    "03790512": "Motorbike",
    "03797390": "Mug",
    "03948459": "Pistol",
    "04099429": "Rocket",
    "04225987": "Skateboard",
    "04379243": "Table",
}


class ShapeNetPartDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        if split not in ["train", "val", "trainval", "test"]:
            raise ValueError(f'Invalid split "{split}"!')

        self.data = load_pickle(osp.join(dataset_dir, f"{split}.pickle"))

        self.transform = transform

    def __getitem__(self, index):
        data_dict = {
            "points": self.data["points"][index].copy(),
            "normals": self.data["normals"][index].copy(),
            "labels": self.data["labels"][index].copy(),
            "class_ids": self.data["class_ids"][index].copy(),
        }
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        return self.data["points"].shape[0]
