from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vision3d.layers import WeightedProcrustes
from vision3d.ops import apply_transform


class LocalGlobalRegistrationThreshold(nn.Module):
    def __init__(
        self,
        acceptance_radius: float,
        confidence_threshold: float = 0.2,
        min_local_correspondences: int = 3,
        min_global_correspondences: int = 256,
        max_global_correspondences: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        """Point Matching with Local-to-Global Registration.

        Args:
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            min_local_correspondences (int=3): minimal number of correspondences for each patch correspondence.
            max_global_correspondences (optional[int]=None): maximal number of global correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super().__init__()
        self.acceptance_radius = acceptance_radius
        self.confidence_threshold = confidence_threshold
        self.min_local_correspondences = min_local_correspondences
        self.min_global_correspondences = min_global_correspondences
        self.max_global_correspondences = max_global_correspondences
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes()

    @staticmethod
    def convert_to_batch(src_corr_points, tgt_corr_points, corr_scores, chunks):
        """Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            src_corr_points (Tensor): (C, 3)
            tgt_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_tgt_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        src_corr_points = src_corr_points[indices]  # (total, 3)
        tgt_corr_points = tgt_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_tgt_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_tgt_corr_points.index_put_([indices0, indices1], tgt_corr_points)
        batch_tgt_corr_points = batch_tgt_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_src_corr_points, batch_tgt_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, src_corr_points, tgt_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(tgt_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores

    def local_to_global_registration(self, src_knn_points, tgt_knn_points, score_mat, corr_mat):
        # extract point correspondences
        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)
        all_src_corr_points = src_knn_points[batch_indices, src_indices]
        all_tgt_corr_points = tgt_knn_points[batch_indices, tgt_indices]
        all_corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        # build verification set
        if self.max_global_correspondences is not None and all_corr_scores.shape[0] > self.max_global_correspondences:
            global_corr_scores, sel_indices = all_corr_scores.topk(k=self.max_global_correspondences, largest=True)
            global_src_corr_points = all_src_corr_points[sel_indices]
            global_tgt_corr_points = all_tgt_corr_points[sel_indices]
        else:
            global_src_corr_points = all_src_corr_points
            global_tgt_corr_points = all_tgt_corr_points
            global_corr_scores = all_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.min_local_correspondences
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_src_corr_points, batch_tgt_corr_points, batch_corr_scores = self.convert_to_batch(
                all_src_corr_points, all_tgt_corr_points, all_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_tgt_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                global_tgt_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = global_corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: setup_engine transformation with all correspondences
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, global_corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(global_src_corr_points, global_tgt_corr_points, cur_corr_scores)

        return global_src_corr_points, global_tgt_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        src_knn_points,
        tgt_knn_points,
        src_knn_masks,
        tgt_knn_masks,
        score_mat,
    ):
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_knn_points (Tensor): The points in the source patches (B, K, 3).
            tgt_knn_points (Tensor): The points in the target patches (B, K, 3).
            src_knn_masks (BoolTensor): The point masks in the source patches, True if valid (B, K).
            tgt_knn_masks (BoolTensor): The point masks in the target patches, True if valid (B, K).
            score_mat (Tensor): The log likelihood matrix for each patch pair (B, K, K) or (B, K + 1, K + 1).

        Returns:
            src_corr_indices (LongTensor): The indices of the correspondence points in the source point cloud (C,).
            tgt_corr_indices (LongTensor): The indices of the correspondence points in the target point cloud (C,).
            src_corr_points (Tensor): The correspondence points in the source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in the target point cloud (C, 3).
            corr_scores (Tensor): The matching scores of the correspondences (C,).
            estimated_transform (Tensor): The estimated transformation from source to target (4, 4).
        """
        score_mat = torch.sigmoid(score_mat)
        mask_mat = torch.logical_and(src_knn_masks.unsqueeze(2), tgt_knn_masks.unsqueeze(1))

        confidence_threshold = self.confidence_threshold
        min_global_correspondences = min(score_mat.shape[1], score_mat.shape[2]) * 3
        min_global_correspondences = min(min_global_correspondences, self.min_global_correspondences)
        min_global_correspondences = min(min_global_correspondences, mask_mat.sum().item())
        while True:
            corr_mat = torch.gt(score_mat, confidence_threshold)
            corr_mat = torch.logical_and(corr_mat, mask_mat)
            if corr_mat.sum() >= min_global_correspondences:
                break
            confidence_threshold -= 0.05

        score_mat = score_mat * corr_mat.float()

        src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            src_knn_points, tgt_knn_points, score_mat, corr_mat
        )

        return src_corr_points, tgt_corr_points, corr_scores, estimated_transform

    def extra_repr(self) -> str:
        param_strings = [
            f"radius={self.acceptance_radius:g}",
            f"threshold={self.confidence_threshold:g}",
            f"min_local_corr={self.min_local_correspondences}",
            f"min_global_corr={self.min_global_correspondences}",
            f"max_global_corr={self.max_global_correspondences}",
            f"refinement={self.num_refinement_steps}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
