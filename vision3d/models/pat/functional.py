from collections import OrderedDict

import torch.nn as nn

from vision3d.ops import gather, group_gather, furthest_point_sample, knn


def create_pat_conv1d_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("conv", nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("conv{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_conv2d_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("conv", nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("conv{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def create_pat_linear_blocks(input_dim, output_dims, groups, dropout=None):
    layers = []
    for i, output_dim in enumerate(output_dims):
        block = [
            ("fc", nn.Linear(input_dim, output_dim, bias=False)),
            ("gn", nn.GroupNorm(groups, output_dim)),
            ("elu", nn.ELU(inplace=True)),
        ]
        if dropout is not None:
            block.append(("dp", nn.Dropout(dropout)))
        layers.append(("fc{}".format(i + 1), nn.Sequential(OrderedDict(block))))
        input_dim = output_dim
    return layers


def k_nearest_neighbors_graph(points, num_neighbor, dilation=1, training=True, ignore_nearest=True):
    num_neighbor_dilated = num_neighbor * dilation + int(ignore_nearest)
    indices = knn(points, points, num_neighbor_dilated, transposed=True)
    start_index = 1 if ignore_nearest else 0
    if training:
        indices = indices[:, :, start_index::dilation].contiguous()
    else:
        indices = indices[:, :, start_index:].contiguous()
    points = group_gather(points, indices)
    return points


def furthest_point_sampling_and_gather(points, features, num_sample):
    indices = furthest_point_sample(points, num_sample, gather_points=False, transposed=False)
    points = gather(points, indices)
    if features is not None:
        features = gather(features, indices)
    return points, features
