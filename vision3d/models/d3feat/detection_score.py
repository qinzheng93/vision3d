import torch
import torch.nn.functional as F
from torch import Tensor

from vision3d.ops import index_select


def compute_detection_scores_baixuyang(features, batch, training):
    neighbors = batch["neighbors"][0]  # [n_points, n_neighbors]
    length0, length1 = batch["stack_lengths"][0]
    total_length = length0 + length1

    # add a fake point in the last row for shadow neighbors
    shadow_features = torch.zeros_like(features[:1, :])
    features = torch.cat([features, shadow_features], dim=0)
    shadow_neighbor = torch.ones_like(neighbors[:1, :]) * total_length
    neighbors = torch.cat([neighbors, shadow_neighbor], dim=0)

    # #  normalize the feature to avoid overflow
    features = features / (torch.max(features) + 1e-6)

    # local max score (saliency score)
    neighbor_features = features[neighbors, :]  # [n_points, n_neighbors, 64]
    neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
    neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdim=True)  # [n_points, 1]
    neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, 64]
    local_max_score = F.softplus(features - mean_features)  # [n_points, 64]

    # calculate the depth-wise max score
    depth_wise_max = torch.max(features, dim=1, keepdim=True)[0]  # [n_points, 1]
    depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]

    all_scores = local_max_score * depth_wise_max_score
    # use the max score among channel to be the score of a single point.
    scores = torch.max(all_scores, dim=1)[0]  # [n_points]

    # hard selection (used during test)
    if not training:
        local_max = torch.max(neighbor_features, dim=1)[0]
        is_local_max = (features == local_max).float()
        # print(f"Local Max Num: {float(is_local_max.sum().detach().cpu())}")
        detected = torch.max(is_local_max, dim=1)[0]
        scores = scores * detected

    return scores[:-1]


def compute_detection_scores(feats: Tensor, neighbor_indices: Tensor, training: bool, eps: float = 1e-6) -> Tensor:
    # normalize the feature to avoid overflow
    feats = feats / (torch.max(feats) + eps)  # (N, C)

    # add a fake point in the last row for shadow neighbors
    padded_feats = torch.cat([feats, torch.zeros_like(feats[:1, :])], dim=0)  # (N + 1, C)

    # local max score
    neighbor_feats = index_select(padded_feats, neighbor_indices, dim=0)  # (N, k, C)
    neighbor_masks = torch.ne(neighbor_indices, feats.shape[0])  # (N, k)
    num_neighbors = neighbor_masks.sum(dim=-1, keepdim=True)  # (N, 1)
    local_mean_feats = neighbor_feats.sum(dim=1) / (num_neighbors.float() + eps)  # (N, C)
    local_max_scores = F.softplus(feats - local_mean_feats)  # (N, C)

    # depth max score
    depth_mean_feats = feats.mean(dim=1, keepdim=True)  # (N, 1)
    depth_max_scores = F.softplus(feats - depth_mean_feats)  # (N, C)

    # saliency scores
    scores = (local_max_scores * depth_max_scores).max(dim=1)[0]  # (N)

    # hard selection during testing
    if not training:
        local_max_feats = neighbor_feats.max(dim=1)[0]  # (N, C)
        masks = torch.isclose(feats, local_max_feats).float().max(dim=1)[0]  # (N)
        scores = scores * masks

    return scores
