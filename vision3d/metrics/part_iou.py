from typing import List
import numpy as np
from numpy import ndarray


class PartIntersectOverUnionMeter:
    """Intersect over Union metric for part segmentation."""

    def __init__(self, num_classes: int, num_parts: int, class_id_to_part_ids: List, eps: float = 1e-6):
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.class_id_to_part_ids = class_id_to_part_ids
        self.iou_sums = np.zeros(self.num_classes)
        self.counts = np.zeros(self.num_classes)
        self.eps = eps

    def update(self, outputs: ndarray, targets: ndarray, class_id: int):
        class_indices = np.asarray(self.class_id_to_part_ids[class_id])[None, :]
        outputs = outputs.flatten()[:, None]
        targets = targets.flatten()[:, None]
        output_mat = outputs == class_indices
        target_mat = targets == class_indices
        intersects = np.sum(output_mat & target_mat, axis=0)
        unions = np.sum(output_mat | target_mat, axis=0)
        ious = intersects / (unions + self.eps)
        ious[unions == 0] = 1.0
        iou = np.mean(ious)
        self.iou_sums[class_id] += iou
        self.counts[class_id] += 1

    def iou(self, class_id: int):
        return self.iou_sums[class_id] / (self.counts[class_id] + self.eps)

    def mean_iou(self):
        """Mean IoU over all classes."""
        return np.mean(self.iou_sums / (self.counts + self.eps))

    def overall_iou(self):
        """Overall average IoU over all instances."""
        return np.sum(self.iou_sums) / np.sum(self.counts + self.eps)
