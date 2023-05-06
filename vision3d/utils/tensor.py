import numpy as np
import torch
from torch import Tensor


def tensor_to_array(x):
    """Release all pytorch tensors to item or numpy arrays."""
    if isinstance(x, list):
        x = [tensor_to_array(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([tensor_to_array(item) for item in x])
    elif isinstance(x, dict):
        x = {key: tensor_to_array(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def array_to_tensor(x):
    """Convert all numpy arrays to pytorch tensors."""
    if isinstance(x, list):
        x = [array_to_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([array_to_tensor(item) for item in x])
    elif isinstance(x, dict):
        x = {key: array_to_tensor(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def move_to_cpu(x):
    """Move all tensors to cpu."""
    if isinstance(x, list):
        x = [move_to_cpu(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([move_to_cpu(item) for item in x])
    elif isinstance(x, dict):
        x = {key: move_to_cpu(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = x.cpu()
    return x


def move_to_cuda(x):
    """Move all tensors to cuda."""
    if isinstance(x, list):
        x = [move_to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([move_to_cuda(item) for item in x])
    elif isinstance(x, dict):
        x = {key: move_to_cuda(value) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = x.cuda()
    return x
