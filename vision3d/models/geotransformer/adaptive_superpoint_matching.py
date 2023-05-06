import torch
import torch.nn as nn

from vision3d.ops import pairwise_distance


class AdaptiveSuperPointMatching(nn.Module):
    def __init__(self, min_num_correspondences, similarity_threshold):
        super().__init__()
        self.min_num_correspondences = min_num_correspondences
        self.similarity_threshold = similarity_threshold

    def forward(self, src_feats, tgt_feats, src_masks=None, tgt_masks=None):
        """Extract superpoint correspondences.

        Args:
            src_feats (Tensor): features of the superpoints in source point cloud.
            tgt_feats (Tensor): features of the superpoints in target point cloud.
            src_masks (BoolTensor, optional): masks of the superpoints in source point cloud (False if empty).
            tgt_masks (BoolTensor, optional): masks of the superpoints in target point cloud (False if empty).

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

        # select proposals
        similarity_mat = pairwise_distance(src_feats, tgt_feats, normalized=True, squared=False)
        min_num_correspondences = min(self.min_num_correspondences, similarity_mat.numel())
        masks = torch.le(similarity_mat, self.similarity_threshold)
        if masks.sum() < min_num_correspondences:
            corr_distances, corr_indices = similarity_mat.view(-1).topk(k=min_num_correspondences, largest=False)
            src_corr_indices = torch.div(corr_indices, similarity_mat.shape[1], rounding_mode="floor")
            tgt_corr_indices = corr_indices % similarity_mat.shape[1]
        else:
            src_corr_indices, tgt_corr_indices = torch.nonzero(masks, as_tuple=True)
            corr_distances = similarity_mat[src_corr_indices, tgt_corr_indices]
        corr_scores = torch.exp(-corr_distances)

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
            f"min_num_correspondences={self.min_num_correspondences}",
            f"similarity_threshold={self.similarity_threshold:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
