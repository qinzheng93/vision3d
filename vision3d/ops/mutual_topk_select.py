from typing import Optional

import torch
from torch import Tensor


def mutual_topk_select(
    score_mat: Tensor,
    k: int,
    largest: bool = True,
    threshold: Optional[float] = None,
    mutual: bool = True,
    reduce_result: bool = True,
):
    """Mutual Top-k Selection.

    Args:
        score_mat (Tensor): score matrix. (N, M)
        k (int): the top-k entries from both sides are selected.
        largest (bool=True): use largest top-k.
        threshold (float=0.0): only scores >(<) threshold are selected if (not) largest.
        mutual (bool=True): If True, only entries that are within the top-k of both sides are selected.
        reduce_result (bool=True): If True, return correspondences indices and scores. If False, return corr_mat.

    Returns:
        row_corr_indices (LongTensor): row indices of the correspondences.
        col_corr_indices (LongTensor): col indices of the correspondences.
        corr_scores (Tensor): scores of the correspondences.
        corr_mat (BoolTensor): correspondences matrix.  (N, M)
    """
    num_rows, num_cols = score_mat.shape

    row_topk_indices = score_mat.topk(k=k, largest=largest, dim=1)[1]  # (N, K)
    row_indices = torch.arange(num_rows).cuda().view(num_rows, 1).expand(-1, k)  # (N, K)
    row_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)  # (N, M)
    row_corr_mat[row_indices, row_topk_indices] = True

    col_topk_indices = score_mat.topk(k=k, largest=largest, dim=0)[1]  # (K, M)
    col_indices = torch.arange(num_cols).cuda().view(1, num_cols).expand(k, -1)  # (K, M)
    col_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)  # (N, M)
    col_corr_mat[col_topk_indices, col_indices] = True

    if mutual:
        corr_mat = torch.logical_and(row_corr_mat, col_corr_mat)
    else:
        corr_mat = torch.logical_or(row_corr_mat, col_corr_mat)

    if threshold is not None:
        if largest:
            masks = torch.gt(score_mat, threshold)
        else:
            masks = torch.lt(score_mat, threshold)
        corr_mat = torch.logical_and(corr_mat, masks)

    if reduce_result:
        row_corr_indices, col_corr_indices = torch.nonzero(corr_mat, as_tuple=True)
        corr_scores = score_mat[row_corr_indices, col_corr_indices]
        return row_corr_indices, col_corr_indices, corr_scores

    return corr_mat


def batch_mutual_topk_select(
    score_mat: Tensor,
    k: int,
    row_masks: Optional[Tensor] = None,
    col_masks: Optional[Tensor] = None,
    largest: bool = True,
    threshold: Optional[float] = None,
    mutual: bool = True,
    reduce_result: bool = True,
):
    """Batched Mutual Top-k Selection.

    Args:
        score_mat (Tensor): score matrix. (B, N, M)
        k (int): the top-k entries from both sides are selected.
        row_masks (Tensor): row masks. (B, N)
        col_masks (Tensor): col masks. (B, M)
        largest (bool=True): use largest top-k.
        threshold (float=0.0): only scores >(<) threshold are selected if (not) largest.
        mutual (bool=True): If True, only entries that are within the top-k of both sides are selected.
        reduce_result (bool=True): If True, return correspondences indices and scores. If False, return corr_mat.

    Returns:
        batch_corr_indices (LongTensor): row indices of the correspondences.
        row_corr_indices (LongTensor): row indices of the correspondences.
        col_corr_indices (LongTensor): col indices of the correspondences.
        corr_scores (Tensor): scores of the correspondences.
        corr_mat (BoolTensor): correspondences matrix. (B, N, M)
    """
    batch_size, num_rows, num_cols = score_mat.shape
    batch_indices = torch.arange(batch_size).cuda()

    # correspondences from row-to-col side
    row_topk_indices = score_mat.topk(k=k, largest=largest, dim=2)[1]  # (B, N, K)
    row_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, num_rows, k)  # (B, N, K)
    row_indices = torch.arange(num_rows).cuda().view(1, num_rows, 1).expand(batch_size, -1, k)  # (B, N, K)
    row_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)
    row_corr_mat[row_batch_indices, row_indices, row_topk_indices] = True

    # correspondences from col-to-row side
    col_topk_indices = score_mat.topk(k=k, largest=largest, dim=1)[1]  # (B, K, N)
    col_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, k, num_cols)  # (B, K, N)
    col_indices = torch.arange(num_cols).cuda().view(1, 1, num_cols).expand(batch_size, k, -1)  # (B, K, N)
    col_corr_mat = torch.zeros_like(score_mat, dtype=torch.bool)
    col_corr_mat[col_batch_indices, col_topk_indices, col_indices] = True

    # merge results from two sides
    if mutual:
        corr_mat = torch.logical_and(row_corr_mat, col_corr_mat)
    else:
        corr_mat = torch.logical_or(row_corr_mat, col_corr_mat)

    # threshold
    if threshold is not None:
        if largest:
            masks = torch.gt(score_mat, threshold)
        else:
            masks = torch.lt(score_mat, threshold)
        corr_mat = torch.logical_and(corr_mat, masks)

    if row_masks is not None:
        corr_mat = torch.logical_and(corr_mat, row_masks.unsqueeze(2).expand_as(score_mat))
    if col_masks is not None:
        corr_mat = torch.logical_and(corr_mat, col_masks.unsqueeze(1).expand_as(score_mat))

    if reduce_result:
        batch_corr_indices, row_corr_indices, col_corr_indices = torch.nonzero(corr_mat, as_tuple=True)
        corr_scores = score_mat[batch_corr_indices, row_corr_indices, col_corr_indices]
        return batch_corr_indices, row_corr_indices, col_corr_indices, corr_scores

    return corr_mat
