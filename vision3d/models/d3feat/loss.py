import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.ops import apply_transform, pairwise_distance


class DetectionLoss(nn.Module):
    def __init__(self, pos_radius, neg_radius):
        super().__init__()
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius

    def forward(self, coordinate_distances, feature_distances, scores0, scores1):
        indices = torch.arange(scores0.shape[0]).cuda()
        pos_mask = indices.unsqueeze(1) == indices.unsqueeze(0)
        neg_mask = coordinate_distances > self.neg_radius

        furthest_positive = torch.max(feature_distances - 1e5 * (~pos_mask).float(), dim=1)[0]
        closest_negative = torch.min(feature_distances + 1e5 * (~neg_mask).float(), dim=1)[0]

        loss = (furthest_positive - closest_negative) * (scores0 + scores1)
        return torch.mean(loss)


class D3FeatLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.log_scale = config.circle_loss_log_scale
        self.pos_optimal = config.circle_loss_positive_optimal
        self.neg_optimal = config.circle_loss_negative_optimal
        self.pos_margin = config.circle_loss_positive_margin
        self.neg_margin = config.circle_loss_negative_margin
        self.max_points = config.circle_loss_max_correspondence

        self.pos_radius = config.loss_positive_radius + 0.001
        self.neg_radius = config.loss_negative_radius

        self.alpha_circle_loss = config.alpha_circle_loss
        self.alpha_detection_loss = config.alpha_detection_loss

    def get_recall(self, coordinate_distances, feature_distances):
        # Get feature match recall, divided by number of true inliers
        with torch.no_grad():
            pos_mask = (coordinate_distances < self.pos_radius).sum(-1) > 0
            num_gt_pos = pos_mask.float().sum() + 1e-12
            _, nn_indices0 = torch.min(feature_distances, -1)
            nn_distances0 = torch.gather(coordinate_distances, dim=-1, index=nn_indices0[:, None])[pos_mask]
            num_pred_pos = (nn_distances0 < self.pos_radius).float().sum()
            recall = num_pred_pos / num_gt_pos
        return recall

    def get_circle_loss(self, coordinate_distances, feature_distances):
        pos_mask = coordinate_distances < self.pos_radius
        neg_mask = coordinate_distances > self.neg_radius

        # get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feature_distances - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weight = pos_weight - self.pos_optimal  # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feature_distances + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weight = self.neg_optimal - neg_weight  # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

        loss_pos_row = torch.logsumexp(self.log_scale * (feature_distances - self.pos_margin) * pos_weight, dim=-1)
        loss_pos_col = torch.logsumexp(self.log_scale * (feature_distances - self.pos_margin) * pos_weight, dim=-2)

        loss_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feature_distances) * neg_weight, dim=-1)
        loss_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feature_distances) * neg_weight, dim=-2)

        loss_row = F.softplus(loss_pos_row + loss_neg_row) / self.log_scale
        loss_col = F.softplus(loss_pos_col + loss_neg_col) / self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_detection_loss(self, feature_distances, scores0, scores1):
        indices = torch.arange(scores0.shape[0]).cuda()
        pos_mask = indices.unsqueeze(1) == indices.unsqueeze(0)

        furthest_positive = torch.max(feature_distances * pos_mask.float(), dim=1)[0]
        closest_negative = torch.min(feature_distances + 1e5 * pos_mask.float(), dim=1)[0]

        loss = (furthest_positive - closest_negative) * (scores0 + scores1)

        # closest_negative_row = torch.min(feature_distances + 1e5 * pos_mask.float(), dim=1)[0]
        # closest_negative_col = torch.min(feature_distances + 1e5 * pos_mask.float(), dim=0)[0]
        # closest_negative = closest_negative_col + closest_negative_row

        # loss = (2 * furthest_positive - closest_negative) * (scores0 + scores1)

        return torch.mean(loss)

    def forward(self, points0, points1, feats0, feats1, scores0, scores1, correspondences, transform):
        points1 = apply_transform(points1, transform)

        if correspondences.shape[0] > self.max_points:
            choice = np.random.permutation(correspondences.shape[0])[: self.max_points]
            correspondences = correspondences[choice]

        indices0, indices1 = correspondences[:, 0], correspondences[:, 1]
        points0, points1 = points0[indices0], points1[indices1]
        feats0, feats1 = feats0[indices0], feats1[indices1]
        scores0, scores1 = scores0[indices0], scores1[indices1]

        coordinate_distances = torch.sqrt(pairwise_distance(points0, points1))
        feature_distances = torch.sqrt(pairwise_distance(feats0, feats1))

        recall = self.get_recall(coordinate_distances, feature_distances)
        circle_loss = self.get_circle_loss(coordinate_distances, feature_distances)
        detection_loss = self.get_detection_loss(feature_distances, scores0, scores1)

        overall_loss = self.alpha_circle_loss * circle_loss + self.alpha_detection_loss * detection_loss

        result_dict = {}
        result_dict["recall"] = recall
        result_dict["circle_loss"] = circle_loss
        result_dict["detection_loss"] = detection_loss
        result_dict["overall_loss"] = overall_loss

        return result_dict
