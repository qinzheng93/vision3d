import numpy as np
from numpy import ndarray


class IntersectOverUnionMeter:
    """Intersect over Union metric for scene segmentation."""

    def __init__(self, num_classes: int, eps: float = 1e-6):
        self.num_classes = num_classes
        self.intersects = np.zeros(num_classes)
        self.unions = np.zeros(num_classes)
        self.eps = eps

    def update(self, outputs: ndarray, targets: ndarray):
        class_indices = np.arange(self.num_classes)[None, :]  # (1, C)
        outputs = outputs.flatten()[:, None]  # (N, 1)
        targets = targets.flatten()[:, None]  # (N, 1)
        output_mat = outputs == class_indices  # (N, C)
        target_mat = targets == class_indices  # (N, C)
        intersects = np.sum(output_mat & target_mat, axis=0)  # (C,)
        unions = np.sum(output_mat | target_mat, axis=0)  # (C,)
        self.intersects += intersects
        self.unions += unions

    def iou(self, class_id: int):
        return self.intersects[class_id] / (self.unions[class_id] + self.eps)

    def mean_iou(self):
        return np.mean(self.intersects / (self.unions + self.eps))

    def overall_iou(self):
        return np.sum(self.intersects) / (np.sum(self.unions) + self.eps)
