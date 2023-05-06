from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Smooth Cross-entropy Loss.

        Args:
            inputs (Tensor): (B, C, *)
            targets (LongTensor): (B, *)

        Returns:
            loss (Tensor)
        """
        batch_size = inputs.shape[0]
        num_classes = inputs.shape[1]
        inputs = inputs.view(batch_size, num_classes, -1)  # (B, C, N)
        targets = targets.view(batch_size, -1)  # (B, N)
        one_hot = F.one_hot(targets, num_classes=num_classes).float().transpose(1, 2)  # (B, N, C) -> (B, C, N)
        targets = one_hot * (1 - self.eps) + self.eps / inputs.shape[1]
        log_inputs = F.log_softmax(inputs, dim=1)
        loss = -(targets * log_inputs).sum(dim=1).mean()
        return loss
