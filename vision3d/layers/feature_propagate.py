import torch
import torch.nn as nn

from vision3d.ops import three_interpolate, three_nn

from .conv_block import ConvBlock


class FeaturePropagate(nn.Module):
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

    def forward(self, q_points, s_points, q_feats, s_feats):
        """
        Feature Propagation Module.

        Args:
            q_points (Tensor): coordinates of the query points (B, 3, Q).
            s_points (Tensor): coordinates of the support points (B, 3, S).
            q_feats (Tensor): features of the query points (B, Cq, Q).
            s_feats (Tensor): features of the support points (B, Cs, S).

        Returns:
            q_features (Tensor): output features of the query points (B, CQ + CS, Q).
        """
        assert s_points.shape[2] >= 3, f"Too few support points: {s_points.shape}"

        distances, indices = three_nn(q_points, s_points, transposed=True)
        weights = torch.div(1.0, distances + 1e-5)
        weights = weights / torch.sum(weights, dim=2, keepdim=True)
        s_feats = three_interpolate(s_feats, indices, weights)

        if q_feats is not None:
            q_feats = torch.cat([q_feats, s_feats], dim=1)
        else:
            q_feats = s_feats

        q_feats = self.shared_mlp(q_feats)

        return q_feats
