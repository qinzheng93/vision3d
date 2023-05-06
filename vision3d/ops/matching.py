import torch
from torch import Tensor

from .mutual_topk_select import mutual_topk_select
from .pairwise_distance import pairwise_distance

# Extract correspondences


@torch.no_grad()
def extract_correspondences_from_scores(
    score_mat: Tensor,
    mutual: bool = False,
    has_dustbin: bool = False,
    threshold: float = 0.0,
    return_score: bool = False,
):
    """Extract the indices of correspondences from matching score matrix (max selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M) or (N + 1, M + 1) according to `has_dustbin`
        mutual (bool = False): whether to get mutual correspondences.
        has_dustbin (bool = False): If true, use dustbin variables.
        threshold (float = 0): confidence threshold.
        return_score (bool = False): return correspondence scores.

    Returns:
        row_corr_indices (LongTensor): (C,)
        col_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    score_mat = torch.exp(score_mat)
    row_length, col_length = score_mat.shape

    row_max_scores, row_max_indices = torch.max(score_mat, dim=1)
    row_indices = torch.arange(row_length).cuda()
    row_corr_scores_mat = torch.zeros_like(score_mat)
    row_corr_scores_mat[row_indices, row_max_indices] = row_max_scores
    row_corr_masks_mat = torch.gt(row_corr_scores_mat, threshold)

    col_max_scores, col_max_indices = torch.max(score_mat, dim=0)
    col_indices = torch.arange(col_length).cuda()
    col_corr_scores_mat = torch.zeros_like(score_mat)
    col_corr_scores_mat[col_max_indices, col_indices] = col_max_scores
    col_corr_masks_mat = torch.gt(col_corr_scores_mat, threshold)

    if mutual:
        corr_masks_mat = torch.logical_and(row_corr_masks_mat, col_corr_masks_mat)
    else:
        corr_masks_mat = torch.logical_or(row_corr_masks_mat, col_corr_masks_mat)

    if has_dustbin:
        corr_masks_mat = corr_masks_mat[:-1, :-1]

    row_corr_indices, col_corr_indices = torch.nonzero(corr_masks_mat, as_tuple=True)

    if return_score:
        corr_scores = score_mat[row_corr_indices, col_corr_indices]
        return row_corr_indices, col_corr_indices, corr_scores

    return row_corr_indices, col_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_threshold(score_mat: Tensor, threshold: float, return_score: bool = False):
    """Extract the indices of correspondences from matching score matrix (thresholding selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M).
        threshold (float = 0): confidence threshold
        return_score (bool = False): return correspondence scores

    Returns:
        row_corr_indices (LongTensor): (C,)
        col_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    score_mat = torch.exp(score_mat)
    masks = torch.gt(score_mat, threshold)
    row_corr_indices, col_corr_indices = torch.nonzero(masks, as_tuple=True)

    if return_score:
        corr_scores = score_mat[row_corr_indices, col_corr_indices]
        return row_corr_indices, col_corr_indices, corr_scores

    return row_corr_indices, col_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_topk(
    score_mat: Tensor, k: int, has_dustbin: bool = False, largest: bool = True, return_score: bool = False
):
    """Extract the indices of correspondences from matching score matrix (global top-k selection).

    Args:
        score_mat (Tensor): the scores (N, M) or (N + 1, M + 1) according to `has_dustbin`.
        k (int): top-k.
        has_dustbin (bool = False): whether to use the slack variables.
        largest (bool = True): whether to choose the largest ones.
        return_score (bool = False): return correspondence scores.

    Returns:
        row_corr_indices (LongTensor): (C,)
        col_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    corr_indices = score_mat.view(-1).topk(k=k, largest=largest)[1]
    row_corr_indices = corr_indices // score_mat.shape[1]
    col_corr_indices = corr_indices % score_mat.shape[1]
    if has_dustbin:
        row_masks = torch.ne(row_corr_indices, score_mat.shape[0] - 1)
        col_masks = torch.ne(col_corr_indices, score_mat.shape[1] - 1)
        masks = torch.logical_and(row_masks, col_masks)
        row_corr_indices = row_corr_indices[masks]
        col_corr_indices = col_corr_indices[masks]

    if return_score:
        corr_scores = score_mat[row_corr_indices, col_corr_indices]
        return row_corr_indices, col_corr_indices, corr_scores

    return row_corr_indices, col_corr_indices


@torch.no_grad()
def extract_correspondences_from_feats(
    src_feats: Tensor, tgt_feats: Tensor, mutual: bool = False, return_feat_dist: bool = False, normalized: bool = True
):
    """Extract the indices of correspondences from feature distances (nn selection).

    Args:
        src_feats (Tensor): features of source point cloud (M, C).
        tgt_feats (Tensor): features of target point cloud (N, C).
        mutual (bool): if True, get mutual correspondences. Default: False.
        return_feat_dist (bool): if True, return feature distances. Default: False.
        normalized (bool): if True, the features are normalized. Default: True.

    Returns:
        src_corr_indices (LongTensor): (C,)
        tgt_corr_indices (LongTensor): (C,)
        corr_feat_dists (Tensor): (C,)
    """
    fdist_mat = pairwise_distance(src_feats, tgt_feats, normalized=normalized)

    src_corr_indices, tgt_corr_indices, corr_feat_dists = mutual_topk_select(
        fdist_mat, k=1, mutual=mutual, largest=False
    )

    if return_feat_dist:
        return src_corr_indices, tgt_corr_indices, corr_feat_dists

    return src_corr_indices, tgt_corr_indices
