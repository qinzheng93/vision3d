import torch
from torch import Tensor


def masked_sum(inputs: Tensor, masks: Tensor, transposed: bool = False) -> Tensor:
    """Masked sum.

    Args:
        inputs (Tensor): The input tensor in the shape of (B, N, C).
        masks (Tensor): The input masks in the shape of (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.

    Returns:
        A Tensor of the masked sum.
    """
    if transposed:
        inputs = inputs.transpose(1, 2)  # (B, N, C)
    masks = masks.float().unsqueeze(-1)  # (B, N, 1)
    masked_sum_values = (inputs * masks).sum(dim=1)  # (B, C)
    return masked_sum_values


def masked_mean(inputs: Tensor, masks: Tensor, transposed: bool = False, eps: float = 1e-6) -> Tensor:
    """Masked mean.

    Args:
        inputs (Tensor): The input tensor in the shape of (B, N, C).
        masks (Tensor): The input masks in the shape of (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.
        eps (float): A safe number for division. Default: 1e-6.

    Returns:
        A Tensor of the masked mean.
    """
    if transposed:
        inputs = inputs.transpose(1, 2)  # (B, N, C)
    masks = masks.float().unsqueeze(-1)  # (B, N, 1)
    masked_sum_values = (inputs * masks).sum(dim=1)  # (B, C)
    counts = masks.sum(1)  # (B, 1)
    masked_mean_values = masked_sum_values / (counts + eps)  # (B, C)
    return masked_mean_values


def masked_normalize(inputs: Tensor, masks: Tensor, transposed: bool = False) -> Tensor:
    """Masked normalize.

    Args:
        inputs (Tensor): The input tensor in the shape of (B, N, C).
        masks (Tensor): The input masks in the shape of (B, N).
        transposed (bool): If True, the input shape if (B, C, N). Default: False.

    Returns:
        A Tensor of the masked normlized outputs.
    """
    if transposed:
        inputs = inputs.transpose(1, 2)  # (B, N, C)
    masked_mean_values = masked_mean(inputs, masks)  # (B, C)
    masked_normalized_values = inputs - masked_mean_values.unsqueeze(1)  # (B, N, C)
    masked_normalized_values = torch.where(masks.unsqueeze(-1), masked_normalized_values, inputs)  # (B, N, C)
    if transposed:
        masked_normalized_values = masked_normalized_values.transpose(1, 2).contiguous()
    return masked_normalized_values
