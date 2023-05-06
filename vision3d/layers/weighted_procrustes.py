from typing import Optional

import torch.nn as nn
from torch import Tensor

from vision3d.ops.weighted_procrustes import weighted_procrustes


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_threshold: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.weight_threshold = weight_threshold
        self.eps = eps

    def forward(self, src_points: Tensor, tgt_points: Tensor, weights: Optional[Tensor] = None):
        return weighted_procrustes(
            src_points, tgt_points, weights=weights, weight_threshold=self.weight_threshold, eps=self.eps
        )

    def extra_repr(self) -> str:
        param_strings = [f"weight_threshold={self.weight_threshold:g}", f"eps={self.eps:g}"]
        format_string = ", ".join(param_strings)
        return format_string
