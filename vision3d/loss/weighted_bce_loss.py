import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


def weighted_bce_loss(inputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Weighted BCE loss (function).

    Args:
        inputs (Tensor): predicted values. (*)
        targets (Tensor): target values. (*)
        reduction (str="mean"): reduction method, available: "mean", "sum", "none".

    Return:
        loss (Tensor): loss value.
    """
    # generate weights
    negative_weights = targets.mean()
    positive_weights = 1.0 - negative_weights
    weights = targets * positive_weights + (1.0 - targets) * negative_weights
    weights = weights.detach()

    # weighted bce loss
    loss_values = F.binary_cross_entropy(inputs, targets, reduction="none")
    loss = weights * loss_values

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


def weighted_bce_loss_with_logits(inputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Weighted BCE loss with logits.

    This function combines a sigmoid function and a weighted bce loss.

    Args:
        inputs (Tensor): predicted values. (*)
        targets (Tensor): target values. (*)
        reduction (str="mean"): reduction method, available: "mean", "sum", "none".

    Return:
        loss (Tensor): loss value.
    """
    # generate weights
    negative_weights = targets.mean()
    positive_weights = 1.0 - negative_weights
    weights = targets * positive_weights + (1.0 - targets) * negative_weights
    weights = weights.detach()

    # weighted bce loss
    loss_values = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = weights * loss_values

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Weighted BCE loss forward.

        Args:
            inputs (Tensor): predicted values. (*)
            targets (Tensor): target values. (*)

        Return:
            loss (Tensor): loss value.
            precision (Tensor): predicted precision.
            recall (Tensor): predicted recall.
        """
        return weighted_bce_loss(inputs, targets, reduction=self.reduction)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Weighted BCE loss with logits forward.

        This function combines a sigmoid function and a weighted bce loss.

        Args:
            inputs (Tensor): predicted values. (*)
            targets (Tensor): target values. (*)

        Return:
            loss (Tensor): loss value.
            precision (Tensor): predicted precision.
            recall (Tensor): predicted recall.
        """
        return weighted_bce_loss_with_logits(inputs, targets, reduction=self.reduction)
