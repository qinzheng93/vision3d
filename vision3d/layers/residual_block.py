from typing import Union

import torch.nn as nn

from .basic_layers import build_act_layer
from .conv_block import ConvBlock


class BasicConvResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        conv_cfg: str = "None",
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        assert conv_cfg in ["Conv1d", "Conv2d", "Conv3d"]

        self.residual = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg="None",
            ),
        )

        if stride > 1 or in_channels != out_channels:
            self.identity = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg="None",
            )
        else:
            self.identity = nn.Identity()

        self.act = build_act_layer(act_cfg)

    def forward(self, x):
        residual = self.residual(x)
        identity = self.identity(x)
        output = self.act(identity + residual)
        return output


class BottleneckConvResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        conv_cfg: str = "None",
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        assert conv_cfg in ["Conv1d", "Conv2d", "Conv3d"]

        mid_channels = out_channels // 4
        self.residual = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg="None",
            ),
        )

        if stride > 1 or in_channels != out_channels:
            self.identity = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg="None",
            )
        else:
            self.identity = nn.Identity()

        self.act = build_act_layer(act_cfg)

    def forward(self, x):
        residual = self.residual(x)
        identity = self.identity(x)
        x = self.act(residual + identity)
        return x
