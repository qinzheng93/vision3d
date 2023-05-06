import torch.nn as nn
from torch import Tensor

from .basic_layers import build_act_layer, build_norm_layer_pack_mode, check_bias_from_norm_cfg


class UnaryBlockPackMode(nn.Module):
    """Unary block with normalization and activation in pack mode.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(self, in_channels, out_channels, norm_cfg="GroupNorm", act_cfg="LeakyReLU"):
        super().__init__()

        bias = check_bias_from_norm_cfg(norm_cfg)
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

        self.norm = build_norm_layer_pack_mode(out_channels, norm_cfg)
        self.act = build_act_layer(act_cfg)

    def forward(self, feats: Tensor) -> Tensor:
        feats = self.mlp(feats)
        feats = self.norm(feats)
        feats = self.act(feats)
        return feats
