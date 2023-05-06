from typing import Optional

import torch
from torch import Tensor


def dual_softmax(score_mat: Tensor, row_masks: Optional[Tensor] = None, col_masks: Optional[Tensor] = None) -> Tensor:
    """Dual softmax.

    output = softmax(input, dim=-1) * softmax(input, dim=-2)

    Args:
        score_mat (Tensor): the score matrix in the shape of (*, N, M).
        row_masks (BoolTensor, optional): the masks for the rows in the shape of (*, N).
        col_masks (BoolTensor, optional): the masks for the columns in the shape of (*, M).

    Returns:
        A Tensor of the dual softmax score matrix in the shape of (*, N, M).
    """
    output_mat = torch.softmax(score_mat, dim=-1) * torch.softmax(score_mat, dim=-2)
    if row_masks is not None:
        output_mat = torch.where(row_masks.unsqueeze(-1), output_mat, torch.zeros_like(output_mat))
    if col_masks is not None:
        output_mat = torch.where(col_masks.unsqueeze(-2), output_mat, torch.zeros_like(output_mat))
    return output_mat
