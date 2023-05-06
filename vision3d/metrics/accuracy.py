import numpy as np
from numpy import ndarray


class AccuracyMeter:
    """Accuracy metric for multi-class classification."""

    def __init__(self, num_classes: int, eps: float = 1e-6):
        self.num_classes = num_classes
        self.positives = np.zeros(num_classes)
        self.counts = np.zeros(num_classes)
        self.eps = eps

    def update(self, outputs: ndarray, targets: ndarray):
        class_indices = np.arange(self.num_classes)[None, :]  # (1, C)
        outputs = outputs.flatten()[:, None]  # (N, 1)
        targets = targets.flatten()[:, None]  # (N, 1)
        output_mat = outputs == class_indices  # (N, C)
        target_mat = targets == class_indices  # (N, C)
        positives = np.sum(output_mat & target_mat, axis=0)  # (C,)
        counts = np.sum(target_mat, axis=0)  # (C,)
        self.positives += positives
        self.counts += counts

    def accuracy(self, class_id: int):
        return self.positives[class_id] / (self.counts[class_id] + self.eps)

    def mean_accuracy(self):
        return np.mean(self.positives / (self.counts + self.eps))

    def overall_accuracy(self):
        return np.sum(self.positives) / (np.sum(self.counts) + self.eps)
