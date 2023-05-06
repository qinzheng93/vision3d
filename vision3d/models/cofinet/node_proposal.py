import torch
import torch.nn as nn

from vision3d.ops import random_choice


class NodeProposalGenerator(nn.Module):
    def __init__(self, num_proposals):
        super().__init__()
        self.num_proposals = num_proposals

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
        if gt_corr_overlaps.shape[0] > self.num_proposals:
            scores = gt_corr_overlaps / gt_corr_overlaps.sum()
            sel_indices = random_choice(gt_corr_overlaps.shape[0], self.num_proposals, replace=False, p=scores)
            gt_src_corr_indices = gt_src_corr_indices[sel_indices]
            gt_tgt_corr_indices = gt_tgt_corr_indices[sel_indices]
            gt_corr_overlaps = gt_corr_overlaps[sel_indices]

        return gt_src_corr_indices, gt_tgt_corr_indices, gt_corr_overlaps

    def extra_repr(self) -> str:
        param_strings = [f"num_proposals={self.num_proposals}"]
        format_string = ", ".join(param_strings)
        return format_string
