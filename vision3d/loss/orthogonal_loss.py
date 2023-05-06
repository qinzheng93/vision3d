from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


def orthogonal_loss(inputs, targets: Tensor = None, reduction: str = "mean") -> Tensor:
    """Identity loss for rigid transformation.

    If target is None: loss = || R - I ||_F^2,
    otherwise: loss = || R^T R* - I ||_F^2

    Args:
        inputs (Tensor): input rotations (*, 3, 3)
        targets (Tensor=None): target rotations (*, 3, 3)
        reduction (str='mean'): reduction method

    Returns:
        loss (Tensor): identity loss.
    """
    if targets is not None:
        inputs = torch.matmul(inputs.transpose(-1, -2), targets)  # (*, 3, 3) x (*, 3, 3) -> (*, 3, 3)
    inputs = inputs.view(-1, 3, 3)  # (*, 3, 3) -> (B, 3, 3)
    identity = torch.eye(3).to(inputs).unsqueeze(0).expand_as(inputs)  # (*, 3, 3)
    loss = F.mse_loss(inputs, identity, reduction=reduction)
    return loss


class OrthogonalLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Tensor:
        return orthogonal_loss(inputs, targets=targets, reduction=self.reduction)
