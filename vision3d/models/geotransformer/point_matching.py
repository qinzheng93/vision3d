import torch
import torch.nn as nn

from vision3d.ops import batch_mutual_topk_select


class PointMatching(nn.Module):
    """Point Matching with Local-to-Global Registration.

    Args:
        k (int): top-k selection for matching.
        mutual (bool=True): mutual or non-mutual matching.
        confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
        use_dustbin (bool=False): whether the dustbin row/column is used in the score matrix.
        use_global_score (bool=False): whether use patch correspondence scores.
    """

    def __init__(
        self,
        k: int,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
    ):
        super().__init__()
        self.k = k
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score

    def forward(
        self,
        src_knn_points,
        tgt_knn_points,
        src_knn_masks,
        tgt_knn_masks,
        src_knn_indices,
        tgt_knn_indices,
        score_mat,
        global_scores,
    ):
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_knn_points (Tensor): (B, K, 3)
            tgt_knn_points (Tensor): (B, K, 3)
            src_knn_masks (BoolTensor): (B, K)
            tgt_knn_masks (BoolTensor): (B, K)
            src_knn_indices (LongTensor): (B, K)
            tgt_knn_indices (LongTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            src_corr_points (Tensor): (C, 3)
            tgt_corr_points (Tensor): (C, 3)
            src_corr_indices (LongTensor): (C,)
            tgt_corr_indices (LongTensor): (C,)
            corr_scores (Tensor): (C,)
        """
        score_mat = torch.exp(score_mat)

        corr_mat = batch_mutual_topk_select(
            score_mat,
            self.k,
            row_masks=src_knn_masks,
            col_masks=tgt_knn_masks,
            threshold=self.confidence_threshold,
            mutual=self.mutual,
            reduce_result=False,
        )

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)
        src_corr_indices = src_knn_indices[batch_indices, src_indices]
        tgt_corr_indices = tgt_knn_indices[batch_indices, tgt_indices]
        src_corr_points = src_knn_points[batch_indices, src_indices]
        tgt_corr_points = tgt_knn_points[batch_indices, tgt_indices]
        corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        return src_corr_points, tgt_corr_points, src_corr_indices, tgt_corr_indices, corr_scores

    def extra_repr(self) -> str:
        param_strings = [
            f"k={self.k}",
            f"mutual={self.mutual}",
            f"threshold={self.confidence_threshold:g}",
            f"use_dustbin={self.use_dustbin}",
            f"use_global_score={self.use_global_score}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
