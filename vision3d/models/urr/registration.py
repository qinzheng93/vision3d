import ipdb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import apply_transform, index_select, weighted_procrustes


class RandomizedWeightedProcrustes(nn.Module):
    def __init__(self, num_subsets: int, num_samples: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.num_samples = num_samples

    def forward(self, src_corr_points: Tensor, tgt_corr_points: Tensor, corr_weights: Tensor) -> Tensor:
        """Randomized weighted procrustes.

        Args:
            src_corr_points (Tensor): the correspondence points in the source point cloud in the shape of (B, N, 3).
            tgt_corr_points (Tensor): the correspondence points in the target point cloud in the shape of (B, N, 3).
            corr_weights (Tensor): the weights of the correspondences in the shape of (B, N).

        Returns:
            A tensor of the estimated transformation in the shape of (B, 3, 3).
        """
        batch_size, num_correspondences = corr_weights.shape

        # NOTE: for simplicity, we use the same indices for all samples in the batch
        sel_indices = [np.random.permutation(num_correspondences)[: self.num_samples] for _ in range(self.num_subsets)]
        sel_indices = np.stack(sel_indices, axis=0)
        sel_indices = torch.from_numpy(sel_indices).cuda()  # (T, K)

        # indexing: (BxT, K, 3) (BxT, K, 3) (BxT, K)
        sel_src_corr_points = index_select(src_corr_points, sel_indices, dim=1).view(-1, self.num_samples, 3)
        sel_tgt_corr_points = index_select(tgt_corr_points, sel_indices, dim=1).view(-1, self.num_samples, 3)
        sel_corr_weights = index_select(corr_weights, sel_indices, dim=1).view(-1, self.num_samples)

        # weighted SVD
        transforms = weighted_procrustes(sel_src_corr_points, sel_tgt_corr_points, sel_corr_weights)  # (BxT, 4, 4)

        # select best transformation
        src_corr_points = src_corr_points.unsqueeze(1).repeat(1, self.num_subsets, 1, 1)  # (B, T, N, 3)
        src_corr_points = src_corr_points.view(-1, num_correspondences, 3)  # (BxT, N, 3)
        aligned_src_corr_points = apply_transform(src_corr_points, transforms)  # (BxT, N, 3)
        aligned_src_corr_points = aligned_src_corr_points.view(batch_size, -1, num_correspondences, 3)  # (B, T, N, 3)
        tgt_corr_points = tgt_corr_points.view(batch_size, 1, num_correspondences, 3)  # (B, 1, N, 3)
        corr_errors = torch.linalg.norm(aligned_src_corr_points - tgt_corr_points, dim=-1)  # (B, T, N)
        weighted_corr_errors = torch.mean(corr_errors * corr_weights.unsqueeze(1), dim=2)  # (B, T)
        _, best_indices = weighted_corr_errors.min(dim=1)  # (B), (B)
        batch_indices = torch.arange(batch_size).cuda()  # (B)
        transforms = transforms.view(batch_size, self.num_subsets, 4, 4)  # (B, T, 4, 4)
        best_transform = transforms[batch_indices, best_indices]  # (B, 4, 4)

        return best_transform
