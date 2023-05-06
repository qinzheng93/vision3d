from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import group_gather, index_select

from .conv_block import ConvBlock


class EdgeConv(nn.Module):
    """EdgeConv in batch mode.

    Paper: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

    Args:
        input_dim (int): The dimension of the input features.
        output_dims (List[int]): The dimension of the output features of each level.
        norm_cfg (str or dict): The configurations of the normalization layer. Default: "BatchNorm".
        act_cfg (str or dict): The configuration of the activation layer. Default: "ReLU".
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        norm_cfg: Union[str, dict] = "BatchNorm",
        act_cfg: Union[str, dict] = "ReLU",
    ):
        super().__init__()
        layers = []
        input_dim = input_dim * 2
        for output_dim in output_dims:
            layers.append(
                ConvBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=1,
                    conv_cfg="Conv2d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            input_dim = output_dim
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, q_feats: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """EdgeConv forward in batch mode.

        Args:
            q_feats (Tensor): (B, C_in, M)
            s_feats (Tensor): (B, C_in, N)
            neighbor_indices (LongTensor): (B, M, k)

        Returns:
            feats (Tensor): (B, C_out, M)
        """
        neighbor_feats = group_gather(s_feats, neighbor_indices)  # (B, C_in, N) -> (B, C_in, N, k)
        q_feats = q_feats.unsqueeze(3).expand_as(neighbor_feats)  # (B, C_in, N) -> (B, C_in, N, 1) -> (B, C_in, N, k)
        neighbor_feats = torch.cat([q_feats, neighbor_feats - q_feats], dim=1)  # (B, C_in, N, k) -> (B, 2 * C_in, N, k)
        q_feats = self.shared_mlp(neighbor_feats)  # (B, C_in, N, k) -> (B, C_out, N, k)
        q_feats = q_feats.max(dim=3)[0]  # (B, C_out, N, k) -> (B, C_out, N)
        return q_feats


class EdgeConvPackMode(nn.Module):
    """EdgeConv in pack mode.

    Paper: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

    Args:
        input_dim (int): The dimension of the input features.
        output_dims (List[int]): The dimension of the output features of each level.
        norm_cfg (str or dict): The configurations of the normalization layer. Default: "BatchNorm".
        act_cfg (str or dict): The configuration of the activation layer. Default: "ReLU".
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: List[int],
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()
        layers = []
        input_dim = input_dim * 2
        for output_dim in output_dims:
            layers.append(
                ConvBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=1,
                    conv_cfg="Conv2d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, q_feats: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """EdgeConv forward in pack mode.

        Args:
            q_feats (Tensor): (M, C_in)
            s_feats (Tensor): (N, C_in)
            neighbor_indices (LongTensor): (M, k)

        Returns:
            feats (Tensor): (M, C_out)
        """
        expanded_s_feats = torch.cat([s_feats, torch.zeros_like(s_feats[:, :1])], dim=1)  # (N+1, C)
        neighbor_feats = index_select(expanded_s_feats, neighbor_indices, dim=0)  # (N, C) -> (M, k, C)
        neighbor_feats = neighbor_feats.permute(2, 0, 1).unsqueeze(0)  # (M, k, C) -> (1, C, M, k)
        q_feats = q_feats.transpose(0, 1).unsqueeze(0).unsqueeze(3).expand_as(neighbor_feats)  # (M, C) -> (1, C, M, k)
        neighbor_feats = torch.cat([q_feats, neighbor_feats], dim=1)  # (1, 2C, M, k)
        q_feats = self.shared_mlp(neighbor_feats)  # (1, C, M, k)
        q_feats = q_feats.squeeze(0).permute(1, 2, 0)  # (1, C, M, k) -> (M, k, C)
        neighbor_masks = torch.ne(neighbor_indices, s_feats.shape[0]).unsqueeze(-1).expand_as(q_feats)  # (M, k, C)
        q_feats = torch.where(neighbor_masks, q_feats, torch.full_like(q_feats, fill_value=-1e10))  # (M, k, C)
        q_feats = q_feats.max(dim=1)[0]  # (M, k, C) -> (M, C)
        return q_feats
