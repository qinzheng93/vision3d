import torch
import torch.nn as nn
from IPython import embed
from torch import Tensor

from vision3d.ops import spatial_consistency


class SpatialConsistencyFiltering(nn.Module):
    def __init__(self, num_correspondences, sigma):
        super().__init__()
        self.sigma = sigma
        self.num_correspondences = num_correspondences

    def forward(self, src_corr_points: Tensor, tgt_corr_points: Tensor) -> Tensor:
        sc_score_mat = spatial_consistency(src_corr_points, tgt_corr_points, self.sigma)  # (N, N)
        corr_sc_counts = sc_score_mat.sum(dim=1)  # (N)
        _, topk_indices = corr_sc_counts.topk(k=self.num_correspondences, largest=True)
        return topk_indices

    def extra_repr(self) -> str:
        param_strings = [f"num_correspondences={self.num_correspondences}", f"sigma={self.sigma:g}"]
        format_string = ", ".join(param_strings)
        return format_string
