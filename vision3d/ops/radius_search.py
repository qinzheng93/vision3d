import torch
from pykeops.torch import LazyTensor

from .knn import keops_knn
from .conversion import batch_to_pack, pack_to_batch


def naive_radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    """Radius search in pack mode (naive implementation).

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.

    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """
    batch_size = q_lengths.shape[0]
    q_start_index = 0
    s_start_index = 0
    knn_indices_list = []
    for i in range(batch_size):
        cur_q_length = q_lengths[i]
        cur_s_length = s_lengths[i]
        q_end_index = q_start_index + cur_q_length
        s_end_index = s_start_index + cur_s_length
        cur_q_points = q_points[q_start_index:q_end_index]
        cur_s_points = s_points[s_start_index:s_end_index]
        knn_distances, knn_indices = keops_knn(cur_q_points, cur_s_points, neighbor_limit)
        knn_indices = knn_indices + s_start_index
        knn_masks = torch.gt(knn_distances, radius)
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])
        knn_indices_list.append(knn_indices)
        q_start_index = q_end_index
        s_start_index = s_end_index
    knn_indices = torch.cat(knn_indices_list, dim=0)
    return knn_indices


def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, inf=1e10):
    """Radius search in pack mode (fast version).

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.
        inf (float=1e10): infinity value.

    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """
    # pack to batch
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)  # (B, M', 3)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)  # (B, N', 3)
    # knn
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, neighbor_limit)  # (B, M', K)
    # accumulate index
    batch_start_index = torch.cumsum(s_lengths, dim=0) - s_lengths
    batch_knn_indices += batch_start_index.view(-1, 1, 1)
    batch_knn_masks = torch.gt(batch_knn_distances, radius)
    batch_knn_indices.masked_fill_(batch_knn_masks, s_points.shape[0])  # (B, M', K)
    # batch to pack
    knn_indices, _ = batch_to_pack(batch_knn_indices, batch_q_masks)  # (M, K)
    return knn_indices


def keops_radius_count(q_points, s_points, radius):
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    vij = (radius - dij).relu().sign()  # (*, N, M)
    radius_counts = vij.sum(dim=num_batch_dims + 1)  # (*, N)
    return radius_counts


def radius_count_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, inf=1e10):
    """Count neighbors within radius in pack mode (fast version).

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        inf (float=1e10): infinity value.

    Returns:
        radius_counts (LongTensor): the numbers of neighbors within radius for the query points.
    """
    # pack to batch
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)  # (B, M', 3)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)  # (B, N', 3)
    # radius count
    batch_radius_counts = keops_radius_count(batch_q_points, batch_s_points, radius)  # (B, M', 1)
    batch_radius_counts = batch_radius_counts.long()
    # batch to pack
    radius_counts, _ = batch_to_pack(batch_radius_counts, batch_q_masks)  # (M, 1)
    radius_counts = radius_counts.squeeze(-1)
    return radius_counts
