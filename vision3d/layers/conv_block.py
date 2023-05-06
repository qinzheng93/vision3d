import warnings
from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from .basic_layers import build_act_layer, build_conv_layer, build_dropout_layer, build_norm_layer


class ConvBlock(nn.Module):
    """Conv-Norm-Act-Dropout Block.

    Build a conv-norm-act-dropout block.

    Available conv layers: "Conv1d", "Conv2d", "Conv3d", "SeparableConv1d", "SeparableConv2d", "Linear". For depthwise
        convolutions, please specify the "groups" argument. For "Linear", only two arguments "in_channels" and
        "out_channels" take effect and all other convolution arguments are ignored.
    Available norm layers: "BatchNorm", "InstanceNorm", "GroupNorm", "LayerNorm".
    Available act layers: "ReLU", "LeakyReLU", "ELU", "GELU", "Sigmoid", "Softplus", "Tanh", "Identity".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[Union[int, Tuple[int, ...]]] = None,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[str, int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        depth_multiplier: Optional[int] = None,
        conv_cfg: Union[str, dict] = "None",
        norm_cfg: Union[str, dict] = "None",
        act_cfg: Union[str, dict] = "None",
        dropout: Optional[float] = None,
        act_before_norm: bool = False,
    ):
        """Initialize a ConvBlock.

        Args:
            in_channels (int): The number of the input channels.
            out_channels (int): The number of the input channels.
            kernel_size (int or tuple, optional): The kernel size of the convolution.
            stride (int or tuple): The stride of the convolution. Default: 1.
            padding (str or int or tuple): The padding of the convolution. Default: 0.
            dilation (int or tuple): The dilation of the convolution. Default: 1.
            groups (int): The groups of the convolution. Default: 1.
            padding_mode (str): The padding mode of the convolution: "zeros", "reflect", "replicate", "circular".
                Default: "zeros".
            depth_multiplier (int, optional): The depth multiplier of the convolution. Only for "SeparableConv*d".
            conv_cfg (str or dict, optional): The config of the convolution. This argument must be specified.
            norm_cfg (str or dict, optional): The config of the normalization.
            act_cfg (str or dict, optional): The config of the activation.
            dropout (float, optional): The dropout rate.
            act_before_norm (bool): If True, the activation is before the normalization (Conv-Act-Norm-Dropout).
                Default: False.
        """
        super().__init__()

        assert conv_cfg != "None", "'conv_cfg' must be specified."

        if isinstance(conv_cfg, str):
            conv_cfg = {"type": conv_cfg}
        conv_type = conv_cfg["type"]

        if isinstance(norm_cfg, str):
            norm_cfg = {"type": norm_cfg}
        if isinstance(act_cfg, str):
            act_cfg = {"type": act_cfg}

        if norm_cfg is not None:
            norm_type = norm_cfg["type"]
            if norm_type in ["BatchNorm", "InstanceNorm"]:
                norm_cfg["type"] = norm_type + conv_type[-2:]

        self.act_before_norm = act_before_norm

        bias = True
        if norm_cfg is not None and not self.act_before_norm:
            # conv-norm-act
            norm_type = norm_cfg["type"]
            if norm_type.startswith("BatchNorm") or norm_type.startswith("InstanceNorm"):
                bias = False
        if conv_type == "Linear":
            conv_cfg["in_features"] = in_channels
            conv_cfg["out_features"] = out_channels
        else:
            conv_cfg["in_channels"] = in_channels
            conv_cfg["out_channels"] = out_channels
            conv_cfg["kernel_size"] = kernel_size
            conv_cfg["stride"] = stride
            conv_cfg["padding"] = padding
            conv_cfg["dilation"] = dilation
            conv_cfg["padding_mode"] = padding_mode
            if conv_type.startswith("SeparableConv"):
                if groups != 1:
                    warnings.warn(f"Argument 'groups={groups}' ignored for {conv_type} layer.")
                conv_cfg["depth_multiplier"] = depth_multiplier
            else:
                if depth_multiplier is not None:
                    warnings.warn(f"Argument 'depth_multiplier={depth_multiplier}' ignored for {conv_type} layer.")
                conv_cfg["groups"] = groups
        conv_cfg["bias"] = bias
        self.conv = build_conv_layer(conv_cfg)

        norm_layer = build_norm_layer(out_channels, norm_cfg)
        act_layer = build_act_layer(act_cfg)
        # change the registration order to make repr correct
        if self.act_before_norm:
            self.act = act_layer
            self.norm = norm_layer
        else:
            self.norm = norm_layer
            self.act = act_layer
        self.dropout = build_dropout_layer(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_before_norm:
            x = self.norm(self.act(x))
        else:
            x = self.act(self.norm(x))
        x = self.dropout(x)
        return x
