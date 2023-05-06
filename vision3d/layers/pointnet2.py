import torch
import torch.nn as nn

from vision3d.ops import group_gather

from .conv_block import ConvBlock


class SAConv(nn.Module):
    """Set Abstraction Convolution in batch mode.

    Proposed in PointNet++.
    """

    def __init__(self, input_dim, output_dims, norm_cfg="BatchNorm", act_cfg="ReLU"):
        super().__init__()
        layers = []
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
        self.local_mlp = nn.Sequential(*layers)

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        """Set Abstract forward.

        Args:
            q_points (Tensor): (B, 3, M)
            s_points (Tensor): (B, 3, N)
            s_feats (Tensor): (B, C_i, N)
            neighbor_indices (LongTensor): (B, N, K)

        Returns:
            q_feats (Tensor): (B, C_o, M)
        """
        neighbor_points = group_gather(s_points, neighbor_indices)  # (B, 3, N) -> (B, 3, M, K)
        neighbor_offsets = neighbor_points - q_points.unsqueeze(3)  # (B, 3, M, K)
        neighbor_feats = group_gather(s_feats, neighbor_indices)  # (B, C_i, N) -> (B, C_i, M, K)
        neighbor_feats = torch.cat([neighbor_feats, neighbor_offsets], dim=1)
        neighbor_feats = self.local_mlp(neighbor_feats)
        q_feats, _ = neighbor_feats.max(dim=3)
        return q_feats


class GSAConv(nn.Module):
    """Global Set Abstraction Convolution in batch mode.

    Proposed in PointNet++.
    """

    def __init__(self, input_dim, output_dims, norm_cfg="BatchNorm", act_cfg="ReLU"):
        super().__init__()
        layers = []
        for output_dim in output_dims:
            layers.append(
                ConvBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=1,
                    conv_cfg="Conv1d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            input_dim = output_dim
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, points, feats):
        feats = torch.cat([feats, points], dim=1)
        feats = self.shared_mlp(feats)
        feats = feats.max(dim=2)[0]
        return feats
