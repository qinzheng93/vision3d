# Sigmoid focal loss implementation from FVCore.
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> Tensor:
    """Sigmoid focal loss.

    Paper: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): the predictions for each example.
        targets (Tensor): the binary classification label for each element in inputs.
            (0 for the negative class and 1 for the positive class).
        alpha (float): the weighting factor in range (0, 1) to balance pos/neg examples. Default: -1 (no weighting).
        gamma (float): the exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: 2.
        reduction (str): the reduction method: "none", "mean", "sum". Default: "none".
            "none": No reduction will be applied to the output.
            "mean": The output will be averaged.
            "sum": The output will be summed.

    Returns:
        A Tensor of the loss value(s) with the reduction option applied.
    """
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss_with_logits(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> Tensor:
    """Sigmoid focal loss with logits.

    This function combines `sigmoid` and `sigmoid_focal_loss`.

    Paper: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): the predictions for each example.
        targets (Tensor): the binary classification label for each element in inputs.
            (0 for the negative class and 1 for the positive class).
        alpha (float): the weighting factor in range (0, 1) to balance pos/neg examples. Default: -1 (no weighting).
        gamma (float): the exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default: 2.
        reduction (str): the reduction method: "none", "mean", "sum". Default: "none".
            "none": No reduction will be applied to the output.
            "mean": The output will be averaged.
            "sum": The output will be summed.

    Returns:
        A Tensor of the loss value(s) with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        param_strings = [
            f"alpha={self.alpha:g}",
            f"gamma={self.gamma:g}",
            f"reduction={self.reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string


class SigmoidFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss_with_logits(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        param_strings = [
            f"alpha={self.alpha:g}",
            f"gamma={self.gamma:g}",
            f"reduction={self.reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
