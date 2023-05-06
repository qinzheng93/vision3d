from typing import Optional

import ipdb
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


def circle_loss(
    feat_dists: Tensor,
    pos_masks: Tensor,
    neg_masks: Tensor,
    pos_margin: float,
    neg_margin: float,
    pos_optimal: float,
    neg_optimal: float,
    log_scale: float,
    pos_scales: Optional[Tensor] = None,
    neg_scales: Optional[Tensor] = None,
) -> Tensor:
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class CircleLoss(nn.Module):
    def __init__(
        self,
        pos_margin: float,
        neg_margin: float,
        pos_optimal: float,
        neg_optimal: float,
        log_scale: float,
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(
        self,
        pos_masks: Tensor,
        neg_masks: Tensor,
        feat_dists: Tensor,
        pos_scales: Optional[Tensor] = None,
        neg_scales: Optional[Tensor] = None,
    ) -> Tensor:
        """Circle loss forward.

        Args:
            pos_masks (BoolTensor): If True, the entry is a positive pair (*, N, M).
            neg_masks (BoolTensor): If True, the entry is a negative pair (*, N, M).
            feat_dists (Tensor): the euclidean distance between two sets of features (*, N, M).
            pos_scales (Tensor): the weights of the positive pairs (*. N, M).
            neg_scales (Tensor): the weights of the negative pairs (*. N, M).

        Returns:
            loss (Tensor): the final loss.
        """
        return circle_loss(
            feat_dists,
            pos_masks,
            neg_masks,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )

    def extra_repr(self) -> str:
        param_strings = [
            f"pos_margin={self.pos_margin:g}",
            f"neg_margin={self.neg_margin:g}",
            f"pos_optimal={self.pos_optimal:g}",
            f"neg_optimal={self.neg_optimal:g}",
            f"log_scale={self.log_scale:g}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
