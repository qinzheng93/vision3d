import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.ops import pairwise_distance


def _hash(pairs, hash_seed):
    hash_vec = pairs[:, 0] + pairs[:, 1] * hash_seed
    return hash_vec


class HardestContrastiveLoss(nn.Module):
    def __init__(self, pos_thresh, neg_thresh, num_pos_sample, num_neg_candidate):
        super().__init__()
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.num_pos_sample = num_pos_sample
        self.num_neg_candidate = num_neg_candidate

    def forward(self, feats0, feats1, pos_pairs):
        length0 = feats0.shape[0]
        length1 = feats1.shape[0]
        num_pos_pairs = pos_pairs.shape[0]
        hash_seed = max(length0, length1)
        num_candidates0 = min(length0, self.num_neg_candidate)
        num_candidates1 = min(length1, self.num_neg_candidate)

        # sample positive pairs as anchors
        if self.num_pos_sample < num_pos_pairs:
            sel_indices = np.random.choice(num_pos_pairs, self.num_pos_sample, replace=False)
            sel_indices = torch.from_numpy(sel_indices).cuda()
            sampled_pos_pairs = pos_pairs[sel_indices]
        else:
            sampled_pos_pairs = pos_pairs
        pos_indices0 = sampled_pos_pairs[:, 0].long()
        pos_indices1 = sampled_pos_pairs[:, 1].long()
        pos_feats0 = feats0[pos_indices0]
        pos_feats1 = feats1[pos_indices1]

        # sample negative candidates
        candidate_indices0 = torch.from_nupy(np.random.choice(length0, num_candidates0, replace=False)).cuda()
        candidate_indices1 = torch.from_nupy(np.random.choice(length1, num_candidates1, replace=False)).cuda()
        candidate_feats0 = feats0[candidate_indices0]
        candidate_feats1 = feats1[candidate_indices1]

        # compute feature distance for anchors
        neg_sq_distances0 = torch.sqrt(pairwise_distance(pos_feats0, candidate_feats1))
        neg_sq_distances1 = torch.sqrt(pairwise_distance(pos_feats1, candidate_feats0))
        nn_neg_sq_distances0, nn_neg_indices0 = neg_sq_distances0.min(1)
        nn_neg_sq_distances1, nn_neg_indices1 = neg_sq_distances1.min(1)
        nn_neg_indices0 = candidate_indices1[nn_neg_indices0]
        nn_neg_indices1 = candidate_indices0[nn_neg_indices1]

        # select negative masks
        pos_pairs = pos_pairs.detach().cpu().numpy().astype(np.int64)
        pos_keys = _hash(pos_pairs, hash_seed)
        neg_pairs0 = torch.stack([pos_indices0, nn_neg_indices0], dim=1)
        neg_pairs1 = torch.stack([nn_neg_indices1, pos_indices1], dim=1)
        neg_keys0 = _hash(neg_pairs0.detach().cpu().numpy(), hash_seed)
        neg_keys1 = _hash(neg_pairs1.detach().cpu().numpy(), hash_seed)
        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False))).cuda()
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False))).cuda()

        # compute loss
        pos_loss = F.relu(torch.linalg.norm(pos_feats0 - pos_feats1, dim=1) - self.pos_thresh).pos(2).mean()
        neg_loss0 = F.relu(self.neg_thresh - nn_neg_sq_distances0[mask0]).pow(2).mean()
        neg_loss1 = F.relu(self.neg_thresh - nn_neg_sq_distances1[mask1]).pow(2).mean()
        neg_loss = (neg_loss0 + neg_loss1) / 2
        total_loss = pos_loss + neg_loss

        result_dict = {}
        result_dict["loss"] = total_loss
        result_dict["pos_loss"] = pos_loss
        result_dict["neg_loss"] = neg_loss

        return result_dict
