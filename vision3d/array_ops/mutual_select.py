import numpy as np
from numpy import ndarray

from .knn import knn


def mutual_select(src_feats: ndarray, tgt_feats: ndarray, mutual: bool = False, bidirectional: bool = False):
    """Extract correspondence indices from features.

    Args:
        tgt_feats (array): (N, C)
        src_feats (array): (M, C)
        mutual (bool = False): If True, use mutual matching
        bidirectional (bool = False): If True, use directional non-mutual matching, ignored if `mutual` is True.

    Returns:
        tgt_corr_indices: (M,)
        src_corr_indices: (M,)
    """
    src_nn_indices = knn(src_feats, tgt_feats, k=1)
    if mutual or bidirectional:
        tgt_nn_indices = knn(tgt_feats, src_feats, k=1)
        src_indices = np.arange(src_feats.shape[0])
        if mutual:
            src_masks = np.equal(tgt_nn_indices[src_nn_indices], src_indices)
            src_corr_indices = src_indices[src_masks]
            tgt_corr_indices = src_nn_indices[src_corr_indices]
        else:
            tgt_indices = np.arange(tgt_feats.shape[0])
            src_corr_indices = np.concatenate([src_indices, tgt_nn_indices], axis=0)
            tgt_corr_indices = np.concatenate([src_nn_indices, tgt_indices], axis=0)
    else:
        src_corr_indices = np.arange(src_feats.shape[0])
        tgt_corr_indices = src_nn_indices
    return src_corr_indices, tgt_corr_indices
