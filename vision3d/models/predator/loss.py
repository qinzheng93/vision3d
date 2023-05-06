import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn as nn
from torch.nn import functional as F

from vision3d.ops.se3 import apply_transform
from vision3d.ops import pairwise_distance


class PredatorLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.log_scale = config.circle_loss_log_scale
        self.pos_optimal = config.circle_loss_positive_optimal
        self.neg_optimal = config.circle_loss_negative_optimal
        self.pos_margin = config.circle_loss_positive_margin
        self.neg_margin = config.circle_loss_negative_margin
        self.max_points = config.circle_loss_max_correspondence

        self.pos_radius = config.loss_positive_radius
        self.neg_radius = config.loss_negative_radius

        self.saliency_loss_positive_radius = config.saliency_loss_positive_radius

        self.alpha_circle_loss = config.alpha_circle_loss
        self.alpha_overlap_loss = config.alpha_overlap_loss
        self.alpha_saliency_loss = config.alpha_saliency_loss

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

    def get_weighted_bce_loss(self, preds, labels):
        loss_func = nn.BCELoss(reduction="none")

        loss = loss_func(preds, labels)

        weights = torch.ones_like(labels)
        negative_weights = labels.sum() / labels.size(0)
        positive_weights = 1 - negative_weights

        weights[labels >= 0.5] = positive_weights
        weights[labels < 0.5] = negative_weights
        weighted_loss = torch.mean(weights * loss)

        #######################################
        # get classification precision and recall
        labels_array = labels.cpu().numpy()
        preds_array = preds.detach().cpu().round().numpy()
        precision, recall, _, _ = precision_recall_fscore_support(labels_array, preds_array, average="binary")

        precision = torch.tensor(precision, dtype=torch.float32).cuda()
        recall = torch.tensor(recall, dtype=torch.float32).cuda()

        return weighted_loss, precision, recall

    def forward(self, points0, points1, feats0, feats1, correspondences, transform, overlap_scores, saliency_scores):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        length0, length1 = points0.shape[0], points1.shape[0]
        points1 = apply_transform(points1, transform)
        stats = {}

        indices0 = list(set(correspondences[:, 0].int().tolist()))
        indices1 = list(set(correspondences[:, 1].int().tolist()))

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        selected_feats0, selected_points0 = feats0[indices0], points0[indices0]
        selected_feats1, selected_points1 = feats1[indices1], points1[indices1]
        scores = torch.matmul(selected_feats0, selected_feats1.transpose(0, 1))
        nn_indices0 = scores.max(1)[1]
        distance_0 = torch.linalg.norm(selected_points0 - selected_points1[nn_indices0], dim=1)
        nn_indices1 = scores.max(0)[1]
        distance_1 = torch.linalg.norm(selected_points1 - selected_points0[nn_indices1], dim=1)

        gt_labels0 = (distance_0 < self.saliency_loss_positive_radius).float()
        gt_labels1 = (distance_1 < self.saliency_loss_positive_radius).float()
        gt_labels = torch.cat([gt_labels0, gt_labels1])

        saliency_scores0 = saliency_scores[:length0][indices0]
        saliency_scores1 = saliency_scores[length0:][indices1]
        saliency_scores = torch.cat([saliency_scores0, saliency_scores1])

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(saliency_scores, gt_labels)
        stats["saliency_loss"] = class_loss
        stats["saliency_recall"] = cls_recall
        stats["saliency_precision"] = cls_precision

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        gt_labels0 = torch.zeros(length0).cuda()
        gt_labels0[indices0] = 1.0
        gt_labels1 = torch.zeros(length1).cuda()
        gt_labels1[indices1] = 1.0
        gt_labels = torch.cat([gt_labels0, gt_labels1])

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(overlap_scores, gt_labels)
        stats["overlap_loss"] = class_loss
        stats["overlap_recall"] = cls_recall
        stats["overlap_precision"] = cls_precision

        #######################################
        # filter some of correspondence
        if correspondences.shape[0] > self.max_points:
            choice = np.random.permutation(correspondences.shape[0])[: self.max_points]
            correspondences = correspondences[choice]
        indices0, indices1 = correspondences[:, 0], correspondences[:, 1]
        points0, points1 = points0[indices0], points1[indices1]
        feats0, feats1 = feats0[indices0], feats1[indices1]

        #######################
        # get L2 distance between source / target point cloud
        coordinate_distances = torch.sqrt(pairwise_distance(points0, points1))
        feature_distances = torch.sqrt(pairwise_distance(feats0, feats1, normalized=True))

        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coordinate_distances, feature_distances)
        circle_loss = self.get_circle_loss(coordinate_distances, feature_distances)

        stats["circle_loss"] = circle_loss
        stats["recall"] = recall

        overall_loss = (
            stats["circle_loss"] * self.alpha_circle_loss
            + stats["overlap_loss"] * self.alpha_overlap_loss
            + stats["saliency_loss"] * self.alpha_saliency_loss
        )
        stats["overall_loss"] = overall_loss

        return stats
