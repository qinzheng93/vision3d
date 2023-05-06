import torch.nn as nn


def _check_depth_multiplier(x):
    if not isinstance(x, int) or x <= 0:
        raise ValueError(f'"depth_multiplier" ({x}) must be a positive integer.')


class SeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        depth_multiplier=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        _check_depth_multiplier(depth_multiplier)
        hidden_dim = in_channels * depth_multiplier
        self.dwconv = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
            padding_mode=padding_mode,
        )
        self.pwconv = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        depth_multiplier=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        _check_depth_multiplier(depth_multiplier)
        dw_channels = in_channels * depth_multiplier
        self.dwconv = nn.Conv2d(
            in_channels,
            dw_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
            padding_mode=padding_mode,
        )
        self.pwconv = nn.Conv2d(dw_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x
