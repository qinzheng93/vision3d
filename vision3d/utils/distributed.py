import os
from functools import wraps

import torch
import torch.distributed as dist
from torch import Tensor

# distributed env


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        # single-gpu mode, use cuda:0
        torch.cuda.set_device(0)


def is_distributed() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_master() -> bool:
    return get_local_rank() == 0


def master_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)

    return wrapper


# reduce tensor


def all_reduce_tensor(tensor, world_size=None):
    """Average reduce a tensor across all workers."""
    if world_size is None:
        world_size = get_world_size()
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_tensors(x, world_size=None):
    """Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = tuple([all_reduce_tensors(item, world_size=world_size) for item in x])
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x
