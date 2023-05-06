from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


def random_sample_from_scores(scores: Tensor, size: int, replace: bool = False) -> Tensor:
    """Random sample with `scores` as probability.

    Args:
        scores: Tensor (N, *)
        size: int
        replace: bool

    Returns:
        sel_indices: torch.LongTensor (size,)
    """
    if scores.shape[0] > size or replace:
        probs = scores / scores.sum()
        probs = probs.detach().cpu().numpy()
        sel_indices = np.random.choice(scores.shape[0], size=size, p=probs, replace=replace)
        sel_indices = torch.from_numpy(sel_indices).cuda()
    else:
        sel_indices = torch.arange(scores.shape[0]).cuda()
    return sel_indices


def random_choice(
    a: Union[Tensor, int],
    size: int = None,
    replace: bool = True,
    p: Optional[Tensor] = None,
) -> Tensor:
    """Numpy-style random choice.

    Args:
        a (Tensor|int): 1-D tensor or int.
        size (int=None): the number of choices.
        replace (bool=True): True if sample with replacement.
        p (Tensor=None): probabilities.

    Returns:
        selected (Tensor): selected items if ``a`` is a tensor or selected indices if ``a`` is int.
    """
    assert isinstance(a, (Tensor, int)), "'a' must be a 1-D tensor or int."
    if p is not None:
        assert isinstance(p, Tensor) and p.dim() == 1, "'p' must be a 1-D tensor or None."
        p = p.detach().cpu().numpy()
    if isinstance(a, Tensor):
        assert a.dim() == 1, "'a' must be a 1-D tensor or int."
        sel_indices = np.random.choice(a.shape[0], size=size, replace=replace, p=p)
        sel_indices = torch.from_numpy(sel_indices).cuda()
        return a[sel_indices]
    else:
        if p is not None:
            assert a == p.shape[0], f"'a' must be equal to the length of 'p' ({a} vs {p.shape[0]})."
        sel_indices = np.random.choice(a, size=size, replace=replace, p=p)
        sel_indices = torch.from_numpy(sel_indices).cuda()
        return sel_indices
