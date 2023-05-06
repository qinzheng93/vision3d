import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_similarity(x: Tensor, y: Tensor, dim: int, normalized: bool = False) -> Tensor:
    """Cosine similarity between two set of feature vectors.

    The cosine similarity between two feature vector is defined as:

    s_{i} = \frac{\mathbf{x}_i \cdot \mathbf{y}_i}{\lVert \mathbf{x}_i \rVert \cdot \lVert \mathbf{y}_i \rVert}.

    The values of s_{i} are within [-1, 1]. We rescaled them to [0, 1] (larger is better):

    s*_{i} = 0.5 * (s_{i} + 1).

    Args:
        x (Tensor): (*, C, *)
        y (Tensor): (*, C, *)
        dim (int): the channel dim
        normalized (bool=False): If True, x and y is normalized.

    Returns:
        similarity (Tensor): (*)
    """
    if not normalized:
        x = F.normalize(x, p=2, dim=dim)
        y = F.normalize(y, p=2, dim=dim)
    similarity = (x * y).sum(dim)
    similarity = 0.5 * (similarity + 1.0)
    return similarity


def pairwise_cosine_similarity(x: Tensor, y: Tensor, normalized: bool = False, transposed: bool = False) -> Tensor:
    """Pairwise cosine similarity.

    The cosine similarity between two feature vector is defined as:

    s_{i, j} = \frac{\mathbf{x}_i \cdot \mathbf{y}_j}{\lVert \mathbf{x}_i \rVert \cdot \lVert \mathbf{y}_j \rVert}.

    The values of s_{i, j} are within [-1, 1]. We rescaled them to [0, 1] (larger is better):

    s*_{i, j} = 0.5 * (s_{i, j} + 1).

    Args:
        x (Tensor): (*, N, C) or (*, C, N) if transposed is True.
        y (Tensor): (*, M, C) or (*, C, M) if transposed is True.
        normalized (bool=False): If True, x and y is normalized.
        transposed (bool=False): If True, channel_dim is before length_dim.

    Returns:
        similarity_mat (Tensor): (*, N, M)
    """
    if transposed:
        channel_dim = -2
    else:
        channel_dim = -1
    if not normalized:
        x = F.normalize(x, p=2, dim=channel_dim)
        y = F.normalize(y, p=2, dim=channel_dim)
    if transposed:
        similarity_mat = torch.matmul(x.transpose(-1, -2), y)
    else:
        similarity_mat = torch.matmul(x, y.transpose(-1, -2))
    similarity_mat = 0.5 * (similarity_mat + 1.0)
    return similarity_mat
