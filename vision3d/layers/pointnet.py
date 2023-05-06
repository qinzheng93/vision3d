import torch.nn as nn
from torch.nn import functional as F

from .conv_block import ConvBlock


class PNConv(nn.Module):
    """PointNet Convolution with a local shared MLP and a global shared MLP.

    Proposed in PointNet.
    """

    def __init__(self, input_dim, local_dims, global_dims, norm_cfg="BatchNorm", act_cfg="ReLU", normalize=False):
        super().__init__()

        self.normalize = normalize

        layers = []
        for output_dim in local_dims:
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

        layers = []
        for output_dim in global_dims:
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
        self.global_mlp = nn.Sequential(*layers)

    def forward(self, group_feats):
        """PointNet forward.

        Args:
            group_feats (Tensor): input features (B, C_in, N, K)

        Returns:
            feats (Tensor): output features (B, C_out, N)
        """
        group_feats = self.local_mlp(group_feats)
        feats = group_feats.max(dim=-1)[0]
        feats = self.global_mlp(feats)
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)
        return feats
