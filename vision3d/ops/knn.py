from typing import Tuple

import torch
from pykeops.torch import LazyTensor
from torch import Tensor

from .conversion import batch_to_pack, pack_to_batch


def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    return knn_distances, knn_indices


def knn(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Use KeOps to accelerate computation.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are ignored according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): the padding mode for neighbors further than distance radius. ('nearest', 'empty').
        padding_value (float=1e10): the value for padding.
        squeeze (bool=False): if True, the distance and the indices are squeezed if k=1.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = torch.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = torch.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points

    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices


def knn_pack_mode(
    q_points: Tensor,
    s_points: Tensor,
    q_lengths: Tensor,
    s_lengths: Tensor,
    k: int,
    return_distance: bool = False,
    inf: float = 1e10,
):
    """Compute the kNN of the query points in the support points in pack mode.

    Note:
        1. This function requires the number of support points is larger than k.

    Args:
        q_points (Tensor): the query points in the shape of (N, 3).
        s_points (Tensor): the support points in the shape of (M, 3).
        q_lengths (Tensor): the number of the query points in the batch in the shape of (B).
        s_lengths (Tensor): the number of the support points in the batch in the shape of (B).
        k (int): the k-nearest neighbors are computed.
        return_distance (bool): if True, return the distance of the neighbors. Default: False.
        inf (float): the infinity value for padding. Default: 1e10.

    Returns:
        A Tensor of the distances of the k-nearest neighbors in the shape of (N, k).
        A LongTensor of the indices of the k-nearest neighbors in the shape of (N, k).
    """
    assert torch.all(torch.ge(s_lengths, k)), f"The number of support points less than {k}."
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, k)
    knn_indices, _ = batch_to_pack(batch_knn_indices, masks=batch_q_masks)
    if not return_distance:
        return knn_indices
    knn_distances, _ = batch_to_pack(batch_knn_distances, masks=batch_q_masks)
    return knn_distances, knn_indices
