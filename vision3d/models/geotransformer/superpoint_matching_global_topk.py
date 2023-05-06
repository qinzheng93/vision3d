from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import pairwise_distance


class SuperPointMatchingGlobalTopk(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super().__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(
        self,
        src_feats: Tensor,
        tgt_feats: Tensor,
        src_masks: Optional[Tensor] = None,
        tgt_masks: Optional[Tensor] = None,
        src_weights: Optional[Tensor] = None,
        tgt_weights: Optional[Tensor] = None,
    ):
        """Extract superpoint correspondences.

        Args:
            src_feats (Tensor): features of the superpoints in source point cloud.
            tgt_feats (Tensor): features of the superpoints in target point cloud.
            src_masks (BoolTensor, optional): masks of the superpoints in source point cloud (False if empty).
            tgt_masks (BoolTensor, optional): masks of the superpoints in target point cloud (False if empty).
            src_weights (Tensor, optional): weights of the superpoints in source point cloud.
            tgt_weights (Tensor, optional): weights of the superpoints in target point cloud.

        Returns:
            A LongTensor of the indices of the matched superpoints in source point cloud.
            A LongTensor of the indices of the matched superpoints in target point cloud.
            A Tensor of the scores of the correspondences.
        """
        # remove empty patch
        if src_masks is not None:
            src_feats = src_feats[src_masks]
            if src_weights is not None:
                src_weights = src_weights[src_masks]

        if tgt_masks is not None:
            tgt_feats = tgt_feats[tgt_masks]
            if tgt_weights is not None:
                tgt_weights = tgt_weights[tgt_masks]

        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(src_feats, tgt_feats, normalized=True))
        if self.dual_normalization:
            src_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            tgt_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = src_matching_scores * tgt_matching_scores
        if src_weights is not None:
            matching_scores = matching_scores * src_weights.unsqueeze(1)
        if tgt_weights is not None:
            matching_scores = matching_scores * tgt_weights.unsqueeze(0)
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        src_corr_indices = torch.div(corr_indices, matching_scores.shape[1], rounding_mode="floor")
        tgt_corr_indices = corr_indices % matching_scores.shape[1]

        # recover original indices
        if src_masks is not None:
            src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
            src_corr_indices = src_indices[src_corr_indices]

        if tgt_masks is not None:
            tgt_indices = torch.nonzero(tgt_masks, as_tuple=True)[0]
            tgt_corr_indices = tgt_indices[tgt_corr_indices]

        return src_corr_indices, tgt_corr_indices, corr_scores

    def extra_repr(self) -> str:
        param_strings = [
            f"num_correspondences={self.num_correspondences}",
            f"dual_normalization={self.dual_normalization}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
