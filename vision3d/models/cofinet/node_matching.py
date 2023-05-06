import ipdb
import torch
import torch.nn as nn

from vision3d.ops import pairwise_distance


class NodeMatching(nn.Module):
    def __init__(self, min_num_correspondences, confidence_threshold):
        super().__init__()
        self.min_num_correspondences = min_num_correspondences
        self.confidence_threshold = confidence_threshold

    def forward(self, score_mat, src_masks=None, tgt_masks=None):
        """Extract superpoint correspondences.

        Args:
            score_mat (Tensor): the log matching score matrix between two node sets in the shape of (N, M).
            src_masks (BoolTensor, optional): the masks of the superpoints in source point cloud (False if empty).
            tgt_masks (BoolTensor, optional): the masks of the superpoints in target point cloud (False if empty).

        Returns:
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            tgt_corr_indices (LongTensor): indices of the corresponding superpoints in target point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        score_mat = torch.exp(score_mat)
        if src_masks is not None:
            score_mat = score_mat * src_masks.unsqueeze(1).float()
        if tgt_masks is not None:
            score_mat = score_mat * tgt_masks.unsqueeze(0).float()

        # select proposals
        min_num_correspondences = min(self.min_num_correspondences, score_mat.numel())
        mask_mat = torch.gt(score_mat, self.confidence_threshold)
        if mask_mat.sum() < min_num_correspondences:
            _, corr_indices = score_mat.view(-1).topk(k=min_num_correspondences, largest=True)
            src_corr_indices = torch.div(corr_indices, score_mat.shape[1], rounding_mode="floor")
            tgt_corr_indices = corr_indices % score_mat.shape[1]
        else:
            src_corr_indices, tgt_corr_indices = torch.nonzero(mask_mat, as_tuple=True)
        corr_scores = score_mat[src_corr_indices, tgt_corr_indices]

        # select only non-masked entries
        corr_masks = torch.gt(corr_scores, 0)
        src_corr_indices = src_corr_indices[corr_masks]
        tgt_corr_indices = tgt_corr_indices[corr_masks]
        corr_scores = corr_scores[corr_masks]

        return src_corr_indices, tgt_corr_indices, corr_scores

    def extra_repr(self) -> str:
        param_strings = [
            f"min_num_correspondences={self.min_num_correspondences}",
            f"confidence_threshold={self.confidence_threshold:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
