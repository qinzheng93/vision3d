import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.ops import gather, group_gather, furthest_point_sample, knn


class PointTransformerLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.position_encoding = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )
        self.q_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.k_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.v_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.attention_encoding = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

    def forward(self, feats, grouped_feats, points, grouped_points):
        feats = feats.unsqueeze(3)
        points = points.unsqueeze(3)
        delta = self.position_encoding(points - grouped_points)

        k = self.k_layer(feats)
        q = self.q_layer(grouped_feats)
        v = self.v_layer(grouped_feats) + delta
        attention_scores = self.attention_encoding(k - q + delta)
        attention_scores = F.softmax(attention_scores, dim=3)
        output = torch.sum(attention_scores * v, dim=3)

        return output


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_neighbors):
        super().__init__()
        self.num_neighbor = num_neighbors
        self.r_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.e_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.point_transformer = PointTransformerLayer(hidden_dim)

    def forward(self, feats, points):
        identity = feats
        feats = self.r_layer(feats)
        indices = knn(points, points, self.num_neighbor, transposed=True)
        grouped_feats = group_gather(feats, indices)
        grouped_points = group_gather(points, indices)
        feats = self.point_transformer(feats, grouped_feats, points, grouped_points)
        feats = self.e_layer(feats)
        feats = feats + identity
        return feats, points


class TransitionDownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsample_ratio, num_neighbors):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.num_neighbor = num_neighbors
        self.transition_layer = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, feats, points):
        feats = self.transition_layer(feats)
        num_sample = int(np.ceil(points.shape[2] / self.downsample_ratio))
        indices = furthest_point_sample(points, num_sample, gather_points=False, transposed=False)
        centroids = gather(points, indices)
        indices = knn(centroids, points, self.num_neighbor, transposed=True)
        grouped_feats = group_gather(feats, indices)
        feats = grouped_feats.mean(dim=3)
        return feats, centroids
