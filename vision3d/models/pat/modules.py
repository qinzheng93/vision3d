import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.models.pat.functional import (
    create_pat_conv1d_blocks,
    create_pat_conv2d_blocks,
    k_nearest_neighbors_graph,
)
from vision3d.loss.smooth_ce_loss import SmoothCrossEntropyLoss

__all__ = [
    "AbsoluteRelativePositionEmbedding",
    "GroupShuffleAttention",
    "GumbelSubsetSampling",
    "AttentionSubsetSampling",
    "ElementwiseLoss",
]


class AbsoluteRelativePositionEmbedding(nn.Module):
    def __init__(self, input_dim, output_dims1, output_dims2, num_neighbor, dilation=1, ignore_nearest=True):
        super().__init__()
        self.num_neighbor = num_neighbor
        self.ignore_nearest = ignore_nearest
        self.dilation = dilation

        layers = create_pat_conv2d_blocks(input_dim, output_dims1, groups=8)
        self.pointnet1 = nn.Sequential(OrderedDict(layers))

        input_dim = output_dims1[-1]
        layers = create_pat_conv1d_blocks(input_dim, output_dims2, groups=8)
        self.pointnet2 = nn.Sequential(OrderedDict(layers))

    def forward(self, points):
        neighbors = k_nearest_neighbors_graph(
            points,
            self.num_neighbor,
            dilation=self.dilation,
            training=self.training,
            ignore_nearest=self.ignore_nearest,
        )
        num_neighbor = neighbors.shape[3]
        points = points.unsqueeze(3).repeat(1, 1, 1, num_neighbor)
        points = torch.cat([points, neighbors - points], dim=1)
        points = self.pointnet1(points)
        points, _ = points.max(dim=3)
        points = self.pointnet2(points)
        return points


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.has_dropout = dropout is not None
        if self.has_dropout:
            self.dp = nn.Dropout(dropout)

    def forward(self, q, k, v):
        num_channel = q.shape[-1]
        attention = torch.matmul(q, k) / math.sqrt(num_channel)
        attention = torch.softmax(attention, dim=-2)
        if self.has_dropout:
            attention = self.dp(attention)
        v = torch.matmul(v, attention)
        return v


class GroupShuffleAttention(nn.Module):
    def __init__(self, feature_dim, groups, dropout=None):
        super().__init__()
        self.num_group = groups
        self.transform = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, groups=groups)
        self.attention = ScaledDotProductAttention(dropout)
        self.gn = nn.GroupNorm(groups, feature_dim)

    def forward(self, points):
        identity = points
        batch_size, num_channel, num_point = points.shape
        num_channel_per_group = num_channel // self.num_group
        points = self.transform(points)
        points = points.view(batch_size, self.num_group, num_channel_per_group, num_point)
        points = self.attention(points.transpose(2, 3), points, F.elu(points))
        points = points.transpose(1, 2).contiguous().view(batch_size, num_channel, num_point)
        points += identity
        points = self.gn(points)

        return points


class GumbelSubsetSampling(nn.Module):
    """
    Gumbel Subset Sampling proposed in `Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling`.

    Use soft sampling in training and hard sampling in testing.
    """

    def __init__(self, input_dim, num_sample, tau=1.0):
        super().__init__()
        self.num_sample = num_sample
        self.tau = tau
        self.layer = nn.Conv1d(input_dim, num_sample, kernel_size=1)

    def forward(self, points):
        weight = self.layer(points)
        weight = F.gumbel_softmax(weight, tau=self.tau, hard=self.hard, dim=2).transpose(1, 2)
        points = torch.matmul(points, weight)
        return points

    @property
    def hard(self):
        return not self.training


class AttentionSubsetSampling(nn.Module):
    def __init__(self, input_dim, num_sample):
        super().__init__()
        self.num_sample = num_sample
        self.layer = nn.Conv1d(input_dim, num_sample, kernel_size=1)

    def forward(self, points):
        weight = self.layer(points)
        weight = torch.softmax(weight, dim=2).transpose(1, 2)
        points = torch.matmul(points, weight)
        return points


class ElementwiseLoss(nn.Module):
    def __init__(self, label_smoothing_eps=None):
        super().__init__()
        if label_smoothing_eps is None:
            self.cls_loss = nn.CrossEntropyLoss()
        else:
            self.cls_loss = SmoothCrossEntropyLoss(eps=label_smoothing_eps)

    def forward(self, inputs, targets):
        num_point = inputs.shape[2]
        targets = targets.unsqueeze(1).repeat(1, num_point)
        cls_loss = self.cls_loss(inputs, targets)
        return cls_loss
