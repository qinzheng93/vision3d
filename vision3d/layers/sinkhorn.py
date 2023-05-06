from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LearnableLogSinkhorn(nn.Module):
    def __init__(self, num_iterations: int, inf: float = 1e12):
        """Sinkhorn Optimal transport with dust-bin parameter (SuperGlue style)."""
        super().__init__()
        self.num_iterations = num_iterations
        self.register_parameter("alpha", torch.nn.Parameter(torch.tensor(1.0)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(
        self,
        scores: Tensor,
        row_masks: Optional[Tensor] = None,
        col_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Sinkhorn Optimal Transport (SuperGlue style) forward.

        Args:
            scores (Tensor): log scores (B, M, N)
            row_masks (Tensor): If False, the row is invalid. (B, M)
            col_masks (Tensor): If False, the column is invalid. (B, N)

        Returns:
            matching_scores (Tensor): log matching scores (B, M+1, N+1)
        """
        batch_size, num_row, num_col = scores.shape

        padded_row_masks = torch.zeros(size=(batch_size, num_row + 1), dtype=torch.bool).cuda()
        padded_col_masks = torch.zeros(size=(batch_size, num_col + 1), dtype=torch.bool).cuda()
        if row_masks is not None:
            padded_row_masks[:, :num_row] = ~row_masks
        if col_masks is not None:
            padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores.masked_fill_(padded_score_masks, -self.inf)

        num_valid_row = float(num_row) - padded_row_masks.sum(1).float()
        num_valid_col = float(num_col) - padded_col_masks.sum(1).float()
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

        return outputs

    def extra_repr(self) -> str:
        format_string = f"num_iterations={self.num_iterations}"
        return format_string


class LogSinkhorn(nn.Module):
    def __init__(self, num_iterations: int, inf: float = 1e12):
        """Sinkhorn Optimal Transport (RPM-Net style)."""
        super().__init__()
        self.num_iterations = num_iterations
        self.inf = inf

    def forward(
        self,
        score_mat: Tensor,
        row_masks: Optional[Tensor] = None,
        col_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Sinkhorn Optimal Transport (RPM-Net style) forward.

        Args:
            score_mat (Tensor): log scores (B, N, M)
            row_masks (Tensor=None): If False, the row is invalid (B, N)
            col_masks (Tensor=None): If False, the column is invalid (B, M)

        Returns:
            matching_scores (Tensor): log matching scores (B, N, M)
        """
        mask_mat = torch.ones_like(score_mat, dtype=torch.bool)
        if row_masks is not None:
            mask_mat = torch.logical_and(mask_mat, row_masks.unsqueeze(2))
        if col_masks is not None:
            mask_mat = torch.logical_and(mask_mat, col_masks.unsqueeze(1))
        score_mat = torch.where(mask_mat, score_mat, torch.full_like(score_mat, -self.inf))

        padded_score_mat = F.pad(score_mat, (0, 1, 0, 1), "constant", value=0)

        for i in range(self.num_iterations):
            rows = padded_score_mat[:, :-1, :]
            rows = rows - torch.logsumexp(rows, dim=2, keepdim=True)
            padded_score_mat = torch.cat([rows, padded_score_mat[:, -1:, :]], dim=1)

            cols = padded_score_mat[:, :, :-1]
            cols = cols - torch.logsumexp(cols, dim=1, keepdim=True)
            padded_score_mat = torch.cat([cols, padded_score_mat[:, :, -1:]], dim=2)

        matching_scores = padded_score_mat[:, :-1, :-1]

        return matching_scores

    def extra_repr(self) -> str:
        format_string = f"num_iterations={self.num_iterations}"
        return format_string
