from typing import Optional, Tuple, Union

import torch.nn as nn

from .depthwise_conv import DepthwiseConv1d, DepthwiseConv2d
from .norm import BatchNormPackMode, GroupNormPackMode, InstanceNormPackMode
from .separable_conv import SeparableConv1d, SeparableConv2d

NORM_LAYERS = {
    "None": nn.Identity,
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
}


NORM_LAYERS_PACK_MODE = {
    "None": nn.Identity,
    "BatchNorm": BatchNormPackMode,
    "InstanceNorm": InstanceNormPackMode,
    "GroupNorm": GroupNormPackMode,
    "LayerNorm": nn.LayerNorm,
}


ACT_LAYERS = {
    "None": nn.Identity,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Identity": nn.Identity,
}


CONV_LAYERS = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "DepthwiseConv1d": DepthwiseConv1d,
    "DepthwiseConv2d": DepthwiseConv2d,
    "SeparableConv1d": SeparableConv1d,
    "SeparableConv2d": SeparableConv2d,
}


def parse_cfg(cfg: Union[str, dict]) -> Tuple[str, dict]:
    if isinstance(cfg, str):
        return cfg, {}
    assert isinstance(cfg, dict), "Illegal cfg type: {}.".format(type(cfg))
    layer = cfg["type"]
    kwargs = {key: value for key, value in cfg.items() if key != "type"}
    return layer, kwargs


def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    """Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    return nn.Dropout(p=p, **kwargs)


def _find_optimal_num_groups(num_channels: int) -> int:
    """Find the optimal number of groups for GroupNorm."""
    # strategy: at most 32 groups, at least 8 channels per group
    num_groups = 32
    while num_groups > 1:
        if num_channels % num_groups == 0:
            num_channels_per_group = num_channels // num_groups
            if num_channels_per_group >= 8:
                break
        num_groups = num_groups // 2
    assert num_groups != 1, (
        f"Cannot find 'num_groups' in GroupNorm with 'num_channels={num_channels}' automatically. "
        "Please manually specify 'num_groups'."
    )
    return num_groups


def _configure_norm_args(layer: str, kwargs: dict, num_features: int) -> dict:
    """Configure norm args."""
    if layer == "GroupNorm":
        kwargs["num_channels"] = num_features
        if "num_groups" not in kwargs:
            kwargs["num_groups"] = _find_optimal_num_groups(num_features)
    elif layer == "LayerNorm":
        kwargs["normalized_shape"] = num_features
    elif layer != "None":
        kwargs["num_features"] = num_features
    return kwargs


def build_norm_layer(num_features, norm_cfg: Union[str, dict]) -> nn.Module:
    """Factory function for normalization layers."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, num_features)
    return NORM_LAYERS[layer](**kwargs)


def build_norm_layer_pack_mode(num_features, norm_cfg: Union[str, dict]) -> nn.Module:
    """Factory function for normalization layers in pack mode."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS_PACK_MODE, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, num_features)
    return NORM_LAYERS_PACK_MODE[layer](**kwargs)


def _configure_act_args(layer: str, kwargs: dict) -> dict:
    """Configure activation args."""
    if layer == "LeakyReLU":
        if "negative_slope" not in kwargs:
            kwargs["negative_slope"] = 0.2
    return kwargs


def build_act_layer(act_cfg: Union[str, dict]) -> nn.Module:
    """Factory function for activation functions."""
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f"Illegal activation: {layer}."
    kwargs = _configure_act_args(layer, kwargs)
    return ACT_LAYERS[layer](**kwargs)


def build_conv_layer(conv_cfg: Union[str, dict]) -> nn.Module:
    """Factory function for convolution or linear layers."""
    layer, kwargs = parse_cfg(conv_cfg)
    assert layer in CONV_LAYERS, f"Illegal conv layer: {layer}."
    return CONV_LAYERS[layer](**kwargs)
