import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

from vision3d.ops import pairwise_distance


def _hash(pairs: ndarray, hash_seed: int) -> ndarray:
    hash_vec = pairs[:, 0] + pairs[:, 1] * hash_seed
    return hash_vec


class HardestContrastiveLoss(nn.Module):
    def __init__(self, pos_thresh: float, neg_thresh: float, num_pos_pairs: int, num_candidates: int):
        super().__init__()
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.num_pos_pairs = num_pos_pairs
        self.num_candidates = num_candidates

    def forward(self, feats0: Tensor, feats1: Tensor, pos_pairs: Tensor) -> dict:
        """Hardest-in-Batch Contrastive Loss.

        Args:
            feats0 (Tensor): (N, C)
            feats1 (Tensor): (M, C)
            pos_pairs (LongTensor): (P, 2)

        Return:
            loss_dict (dict): 'loss', 'pos_loss', 'neg_loss'.
        """
        length0 = feats0.shape[0]
        length1 = feats1.shape[0]
        num_pos_pairs = pos_pairs.shape[0]
        hash_seed = max(length0, length1)
        num_candidates0 = min(length0, self.num_candidates)
        num_candidates1 = min(length1, self.num_candidates)

        # sample positive pairs as anchors
        if self.num_pos_pairs < num_pos_pairs:
            sel_indices = np.random.choice(num_pos_pairs, self.num_pos_pairs, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            sampled_pos_pairs = pos_pairs[sel_indices]
        else:
            sampled_pos_pairs = pos_pairs
        anchor_indices0 = sampled_pos_pairs[:, 0].long()
        anchor_indices1 = sampled_pos_pairs[:, 1].long()
        anchor_feats0 = feats0[anchor_indices0]
        anchor_feats1 = feats1[anchor_indices1]

        # sample candidates
        candidate_indices0 = np.random.choice(length0, num_candidates0, replace=False)
        candidate_indices1 = np.random.choice(length1, num_candidates1, replace=False)
        candidate_indices0 = torch.from_numpy(candidate_indices0).cuda()
        candidate_indices1 = torch.from_numpy(candidate_indices1).cuda()
        candidate_feats0 = feats0[candidate_indices0]
        candidate_feats1 = feats1[candidate_indices1]

        # compute feature distance for anchors
        dist_mat0 = torch.sqrt(pairwise_distance(anchor_feats0, candidate_feats1, normalized=True))
        dist_mat1 = torch.sqrt(pairwise_distance(anchor_feats1, candidate_feats0, normalized=True))
        nn_distances0, nn_indices0 = dist_mat0.min(1)
        nn_distances1, nn_indices1 = dist_mat1.min(1)
        nn_candidate_indices0 = candidate_indices1[nn_indices0]
        nn_candidate_indices1 = candidate_indices0[nn_indices1]

        # select negative masks
        pos_pairs = pos_pairs.detach().cpu().numpy().astype(np.int64)
        pos_keys = _hash(pos_pairs, hash_seed)
        anchor_candidate_pairs0 = torch.stack([anchor_indices0, nn_candidate_indices0], dim=1)
        anchor_candidate_pairs1 = torch.stack([nn_candidate_indices1, anchor_indices1], dim=1)
        anchor_candidate_keys0 = _hash(anchor_candidate_pairs0.detach().cpu().numpy(), hash_seed)
        anchor_candidate_keys1 = _hash(anchor_candidate_pairs1.detach().cpu().numpy(), hash_seed)
        neg_masks0 = torch.from_numpy(~np.isin(anchor_candidate_keys0, pos_keys, assume_unique=False)).cuda()
        neg_masks1 = torch.from_numpy(~np.isin(anchor_candidate_keys1, pos_keys, assume_unique=False)).cuda()

        # compute loss
        pos_distances = torch.linalg.norm(anchor_feats0 - anchor_feats1, dim=1)
        pos_loss = F.relu(pos_distances - self.pos_thresh).pow(2).mean()
        neg_loss0 = F.relu(self.neg_thresh - nn_distances0[neg_masks0]).pow(2).mean()
        neg_loss1 = F.relu(self.neg_thresh - nn_distances1[neg_masks1]).pow(2).mean()
        neg_loss = (neg_loss0 + neg_loss1) / 2
        total_loss = pos_loss + neg_loss

        return {
            "loss": total_loss,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }
