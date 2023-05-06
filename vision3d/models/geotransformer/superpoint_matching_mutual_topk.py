import torch
import torch.nn as nn

from vision3d.ops import mutual_topk_select, pairwise_distance


class SuperPointMatchingMutualTopk(nn.Module):
    def __init__(self, num_correspondences, k, threshold=None, mutual=True, eps=1e-8):
        super().__init__()
        self.num_correspondences = num_correspondences
        self.k = k
        self.threshold = threshold
        self.mutual = mutual
        self.eps = eps

    def forward(self, src_feats, tgt_feats, src_masks=None, tgt_masks=None):
        """Extract superpoint correspondences.

        Args:
            src_feats (Tensor): features of the superpoints in source point cloud.
            tgt_feats (Tensor): features of the superpoints in target point cloud.
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).
            tgt_masks (BoolTensor=None): masks of the superpoints in target point cloud (False if empty).

        Returns:
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            tgt_corr_indices (LongTensor): indices of the corresponding superpoints in target point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        # remove empty patch
        if src_masks is not None:
            src_feats = src_feats[src_masks]

        if tgt_masks is not None:
            tgt_feats = tgt_feats[tgt_masks]

        # select top-k proposals for each superpoint
        score_mat = torch.sqrt(pairwise_distance(src_feats, tgt_feats, normalized=True) + self.eps)

        src_corr_indices, tgt_corr_indices, corr_scores = mutual_topk_select(
            score_mat, self.k, largest=False, threshold=None, mutual=self.mutual
        )

        # threshold
        if self.threshold is not None:
            num_correspondences = min(self.num_correspondences, corr_scores.numel())
            masks = torch.le(corr_scores, self.threshold)
            # print(masks.sum().item())
            if masks.sum().item() < num_correspondences:
                # not enough good correspondences, fallback to topk selection
                corr_scores, topk_indices = corr_scores.topk(k=num_correspondences, largest=False)
                src_corr_indices = src_corr_indices[topk_indices]
                tgt_corr_indices = tgt_corr_indices[topk_indices]
            else:
                src_corr_indices = src_corr_indices[masks]
                tgt_corr_indices = tgt_corr_indices[masks]
                corr_scores = corr_scores[masks]

        # recover original indices
        if src_masks is not None:
            src_valid_indices = torch.nonzero(src_masks, as_tuple=True)[0]
            src_corr_indices = src_valid_indices[src_corr_indices]

        if tgt_masks is not None:
            tgt_valid_indices = torch.nonzero(tgt_masks, as_tuple=True)[0]
            tgt_corr_indices = tgt_valid_indices[tgt_corr_indices]

        return src_corr_indices, tgt_corr_indices, corr_scores

    def extra_repr(self) -> str:
        param_strings = [
            f"num_correspondences={self.num_correspondences}",
            f"k={self.k}",
            f"threshold={self.threshold}",
            f"mutual={self.mutual}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
