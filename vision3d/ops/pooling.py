import torch

from .index_select import index_select


def local_maxpool_pack_mode(feats, neighbor_indices):
    """Max pooling from neighbors in pack mode.

    Args:
        feats (Tensor): The input features in the shape of (N, C).
        neighbor_indices (LongTensor): The neighbor indices in the shape of (M, K).

    Returns:
        pooled_feats (Tensor): The pooled features in the shape of (M, C).
    """
    feats = torch.cat((feats, torch.zeros_like(feats[:1, :])), 0)  # (N+1, C)
    neighbor_feats = index_select(feats, neighbor_indices, dim=0)  # (M, K, C)
    pooled_feats = neighbor_feats.max(1)[0]  # (M, K)
    return pooled_feats


def global_avgpool_pack_mode(feats, lengths):
    """Global average pooling over batch.

    Args:
        feats (Tensor): The input features in the shape of (N, C).
        lengths (LongTensor): The length of each sample in the batch in the shape of (B).

    Returns:
        feats (Tensor): The pooled features in the shape of (B, C).
    """
    feats_list = []
    start_index = 0
    for batch_index in range(lengths.shape[0]):
        end_index = start_index + lengths[batch_index].item()
        feats_list.append(torch.mean(feats[start_index:end_index], dim=0))
        start_index = end_index
    feats = torch.stack(feats_list, dim=0)
    return feats
