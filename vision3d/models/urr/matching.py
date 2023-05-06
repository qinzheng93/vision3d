from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import index_select, knn


class CorrespondenceExtractor(nn.Module):
    def __init__(self, num_correspondences: int, ratio_test: bool = True, eps=1e-8):
        super().__init__()
        self.num_correspondences = num_correspondences
        self.ratio_test = ratio_test  # no use
        self.eps = eps

    def match_one_side(
        self, q_points: Tensor, s_points: Tensor, q_feats: Tensor, s_feats: Tensor
    ) -> Tuple[Tensor, ...]:
        knn_indices = knn(q_feats, s_feats, k=2)  # (N, 2)
        knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, 2, C)
        knn_similarities = 1.0 - torch.sum(knn_feats * q_feats.unsqueeze(1), dim=-1)  # (N, 2)
        weights = 1.0 - knn_similarities[:, 0] / (knn_similarities[:, 1] + self.eps)  # (N)
        q_corr_indices = weights.topk(self.num_correspondences, largest=True)[1]
        s_corr_indices = knn_indices[q_corr_indices, 0]
        q_corr_points = q_points[q_corr_indices]
        s_corr_points = s_points[s_corr_indices]
        q_corr_feats = q_feats[q_corr_indices]
        s_corr_feats = s_feats[s_corr_indices]
        corr_weights = weights[q_corr_indices]
        return q_corr_points, s_corr_points, q_corr_feats, s_corr_feats, corr_weights

    def forward(
        self, src_points: Tensor, tgt_points: Tensor, src_feats: Tensor, tgt_feats: Tensor
    ) -> Tuple[Tensor, ...]:
        """Extract correspondences with optional Lowe's ratio test.

        Args:
            src_points (Tensor): the coordinates of the source point cloud in the shape of (N, 3).
            tgt_points (Tensor): the coordinates of the target point cloud in the shape of (M, 3).
            src_feats (Tensor): the features of the source point cloud in the shape of (N, C).
            tgt_feats (Tensor): the features of the target point cloud in the shape of (M, C).

        Returns:
            A float Tensor of the source correspondence points in the shape of (C, 3).
            A float Tensor of the target correspondence points in the shape of (C, 3).
            A float Tensor of the source correspondence features in the shape of (C, 3).
            A float Tensor of the target correspondence features in the shape of (C, 3).
            A float Tensor of the correspondence weights in the shape of (C).
        """
        (
            src_corr_points_1,
            tgt_corr_points_1,
            src_corr_feats_1,
            tgt_corr_feats_1,
            corr_weights_1,
        ) = self.match_one_side(src_points, tgt_points, src_feats, tgt_feats)

        (
            tgt_corr_points_2,
            src_corr_points_2,
            tgt_corr_feats_2,
            src_corr_feats_2,
            corr_weights_2,
        ) = self.match_one_side(tgt_points, src_points, tgt_feats, src_feats)

        src_corr_points = torch.cat([src_corr_points_1, src_corr_points_2], dim=0)
        tgt_corr_points = torch.cat([tgt_corr_points_1, tgt_corr_points_2], dim=0)
        src_corr_feats = torch.cat([src_corr_feats_1, src_corr_feats_2], dim=0)
        tgt_corr_feats = torch.cat([tgt_corr_feats_1, tgt_corr_feats_2], dim=0)
        corr_weights = torch.cat([corr_weights_1, corr_weights_2], dim=0)

        return src_corr_points, tgt_corr_points, src_corr_feats, tgt_corr_feats, corr_weights
