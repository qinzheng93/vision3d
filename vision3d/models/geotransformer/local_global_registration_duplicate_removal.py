from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vision3d.layers import WeightedProcrustes
from vision3d.ops import apply_transform, batch_mutual_topk_select, index_select


class LocalGlobalRegistrationWithDuplicateRemoval(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        min_local_correspondences: int = 3,
        max_global_correspondences: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        """Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
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
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.min_local_correspondences = min_local_correspondences
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

    def local_to_global_registration(
        self, src_points, tgt_points, src_knn_indices, tgt_knn_indices, score_mat, corr_mat
    ):
        # extract point correspondences
        batch_indices, src_indices, tgt_indices = torch.nonzero(corr_mat, as_tuple=True)

        all_src_corr_indices = src_knn_indices[batch_indices, src_indices]
        all_tgt_corr_indices = tgt_knn_indices[batch_indices, tgt_indices]
        all_src_corr_points = index_select(src_points, all_src_corr_indices, dim=0)
        all_tgt_corr_points = index_select(tgt_points, all_tgt_corr_indices, dim=0)
        all_corr_scores = score_mat[batch_indices, src_indices, tgt_indices]

        # duplicate removal
        all_corr_hash_values = all_src_corr_indices * tgt_points.shape[0] + all_tgt_corr_indices
        unique_corr_hash_values, inv_indices, unique_counts = torch.unique(
            all_corr_hash_values, return_inverse=True, return_counts=True
        )
        unique_src_corr_indices = torch.div(unique_corr_hash_values, tgt_points.shape[0], rounding_mode="floor")
        unique_tgt_corr_indices = unique_corr_hash_values % tgt_points.shape[0]
        unique_src_corr_points = index_select(src_points, unique_src_corr_indices, dim=0)
        unique_tgt_corr_points = index_select(tgt_points, unique_tgt_corr_indices, dim=0)
        unique_corr_scores = torch.zeros(size=(unique_corr_hash_values.shape[0],)).cuda()
        unique_corr_scores.scatter_add_(dim=0, index=inv_indices, src=all_corr_scores)
        unique_corr_scores /= unique_counts.float()

        # build global correspondences
        if (
            self.max_global_correspondences is not None
            and unique_corr_scores.shape[0] > self.max_global_correspondences
        ):
            global_corr_scores, sel_indices = unique_corr_scores.topk(k=self.max_global_correspondences, largest=True)
            global_src_corr_points = unique_src_corr_points[sel_indices]
            global_tgt_corr_points = unique_tgt_corr_points[sel_indices]
        else:
            global_src_corr_points = unique_src_corr_points
            global_tgt_corr_points = unique_tgt_corr_points
            global_corr_scores = unique_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        all_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        all_indices = torch.nonzero(all_masks, as_tuple=True)[0] + 1
        all_indices = all_indices.detach().cpu().numpy().tolist()
        all_indices = [0] + all_indices + [batch_indices.shape[0]]
        chunks = [(x, y) for x, y in zip(all_indices[:-1], all_indices[1:]) if y - x >= self.min_local_correspondences]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            local_src_corr_points, local_tgt_corr_points, local_corr_scores = self.convert_to_batch(
                all_src_corr_points, all_tgt_corr_points, all_corr_scores, chunks
            )
            local_transforms = self.procrustes(local_src_corr_points, local_tgt_corr_points, local_corr_scores)
            local_aligned_src_corr_points = apply_transform(global_src_corr_points.unsqueeze(0), local_transforms)
            batch_corr_residuals = torch.linalg.norm(
                global_tgt_corr_points.unsqueeze(0) - local_aligned_src_corr_points, dim=2
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
        src_points,
        tgt_points,
        src_knn_indices,
        tgt_knn_indices,
        src_knn_masks,
        tgt_knn_masks,
        score_mat,
        global_scores,
    ):
        """Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            src_points (Tensor): The points in the source point cloud (N, 3).
            tgt_points (Tensor): The points in the target point cloud (M, 3).
            src_knn_indices (LongTensor): The indices of the points in the source patches (B, K).
            tgt_knn_indices (LongTensor): The indices of the points in the target patches (B, K).
            src_knn_masks (BoolTensor): The point masks in the source patches, True if valid (B, K).
            tgt_knn_masks (BoolTensor): The point masks in the target patches, True if valid (B, K).
            score_mat (Tensor): The log likelihood matrix for each patch pair (B, K, K) or (B, K + 1, K + 1).
            global_scores (Tensor): The matching scores for each patch pair (B,).

        Returns:
            src_corr_indices (LongTensor): The indices of the correspondence points in the source point cloud (C,).
            tgt_corr_indices (LongTensor): The indices of the correspondence points in the target point cloud (C,).
            src_corr_points (Tensor): The correspondence points in the source point cloud (C, 3).
            tgt_corr_points (Tensor): The correspondence points in the target point cloud (C, 3).
            corr_scores (Tensor): The matching scores of the correspondences (C,).
            estimated_transform (Tensor): The estimated transformation from source to target (4, 4).
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
        )  # (B, K, K)

        if self.use_dustbin:
            corr_mat = corr_mat[:, :-1, :-1]
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        src_corr_points, tgt_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            src_points, tgt_points, src_knn_indices, tgt_knn_indices, score_mat, corr_mat
        )

        return src_corr_points, tgt_corr_points, corr_scores, estimated_transform

    def extra_repr(self) -> str:
        param_strings = [
            f"k={self.k}",
            f"radius={self.acceptance_radius:g}",
            f"mutual={self.mutual}",
            f"threshold={self.confidence_threshold:g}",
            f"use_dustbin={self.use_dustbin}",
            f"use_global_score={self.use_global_score}",
            f"min_local_corr={self.min_local_correspondences}",
            f"max_global_corr={self.max_global_correspondences}",
            f"refinement={self.num_refinement_steps}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
