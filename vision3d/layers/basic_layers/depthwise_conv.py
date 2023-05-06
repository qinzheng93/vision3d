import torch.nn as nn


def _check_depth_multiplier(x):
    if not isinstance(x, int) or x <= 0:
        raise ValueError(f'"depth_multiplier" ({x}) must be a positive integer.')


class DepthwiseConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        depth_multiplier=1,
        bias=True,
    ):
        _check_depth_multiplier(depth_multiplier)
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        depth_multiplier=1,
        bias=True,
    ):
        _check_depth_multiplier(depth_multiplier)
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
