import numpy as np
import torch
import torch.nn as nn

from vision3d.ops import random_choice


class SuperPointProposalGenerator(nn.Module):
    def __init__(self, num_proposals, overlap_threshold, decay_step=0.05, probabilistic=False):
        super().__init__()
        self.num_proposals = num_proposals
        self.overlap_threshold = overlap_threshold
        self.decay_step = decay_step
        self.probabilistic = probabilistic

    @torch.no_grad()
    def forward(self, gt_src_corr_indices, gt_tgt_corr_indices, gt_corr_overlaps):
        """Generate ground truth superpoint (patch) correspondences.

        Randomly select "num_proposals" correspondences whose overlap is above "overlap_threshold".

        Args:
            gt_src_corr_indices (LongTensor): ground truth superpoint correspondences (N,)
            gt_tgt_corr_indices (LongTensor): ground truth superpoint correspondences (N,)
            gt_corr_overlaps (Tensor): ground truth superpoint correspondences overlap (N,)

        Returns:
            gt_src_corr_indices (LongTensor): selected superpoints in source point cloud.
            gt_tgt_corr_indices (LongTensor): selected superpoints in target point cloud.
            gt_corr_overlaps (LongTensor): overlaps of the selected superpoint correspondences.
        """
        overlap_threshold = self.overlap_threshold
        while True:
            gt_corr_masks = torch.gt(gt_corr_overlaps, overlap_threshold)
            if gt_corr_masks.sum() > 0:
                break
            overlap_threshold -= self.decay_step

        gt_corr_overlaps = gt_corr_overlaps[gt_corr_masks]
        gt_src_corr_indices = gt_src_corr_indices[gt_corr_masks]
        gt_tgt_corr_indices = gt_tgt_corr_indices[gt_corr_masks]

        if gt_corr_overlaps.shape[0] > self.num_proposals:
            indices = torch.arange(gt_corr_overlaps.shape[0]).cuda()
            if self.probabilistic:
                probabilities = gt_corr_overlaps / gt_corr_overlaps.sum()
                sel_indices = random_choice(indices, size=self.num_proposals, replace=False, p=probabilities)
            else:
                sel_indices = random_choice(indices, size=self.num_proposals, replace=False)
            gt_src_corr_indices = gt_src_corr_indices[sel_indices]
            gt_tgt_corr_indices = gt_tgt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        return gt_src_corr_indices, gt_tgt_corr_indices, gt_corr_overlaps

    def extra_repr(self) -> str:
        param_strings = [
            f"num_proposals={self.num_proposals}",
            f"overlap_threshold={self.overlap_threshold:g}",
            f"decay_step={self.decay_step:g}",
            f"probabilistic={self.probabilistic:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
