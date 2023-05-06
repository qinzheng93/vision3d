import torch
import torch.nn as nn

from vision3d.ops import group_gather

from .conv_block import ConvBlock


class XReshape(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(2).view(batch_size, self.kernel_size, self.kernel_size, -1).transpose(1, 2)
        return x


class XSharedMLP(nn.Module):
    def __init__(self, input_dim, num_layers, kernel_size, norm_cfg="BatchNorm", act_cfg="ELU"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stem = ConvBlock(
            in_channels=input_dim,
            out_channels=kernel_size * kernel_size,
            kernel_size=(1, kernel_size),
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            act_before_norm=True,
        )
        layers = []
        for i in range(num_layers):
            layers.append(
                ConvBlock(
                    in_channels=kernel_size,
                    out_channels=kernel_size * kernel_size,
                    kernel_size=(kernel_size, 1),
                    groups=kernel_size,
                    conv_cfg="Conv2d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    act_before_norm=True,
                )
            )
            layers.append(XReshape(self.kernel_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, points):
        batch_size = points.shape[0]
        points = self.stem(points)
        points = points.squeeze(3).view(batch_size, self.kernel_size, self.kernel_size, -1)
        points = self.layers(points)
        return points


class XConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        kernel_size,
        dilation=1,
        depth_multiplier=1,
        norm_cfg="BatchNorm",
        act_cfg="ELU",
        with_global=False,
    ):
        # TODO: not tested.
        super().__init__()

        self.kernel_size = kernel_size
        self.num_neighbors = kernel_size
        self.dilation = dilation
        self.depth_multiplier = depth_multiplier
        self.with_global = with_global

        self.f_shared_mlp = nn.Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                act_before_norm=True,
            ),
            ConvBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                act_before_norm=True,
            ),
        )

        self.x_shared_mlp = XSharedMLP(3, num_layers=2, kernel_size=self.kernel_size)

        self.conv = ConvBlock(
            in_channels=input_dim + hidden_dim,
            out_channels=output_dim,
            kernel_size=(1, self.kernel_size),
            depth_multiplier=self.depth_multiplier,
            conv_cfg="SeparableConv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            act_before_norm=True,
        )

        if self.with_global:
            global_dim = output_dim // 4
            self.g_conv = nn.Sequential(
                ConvBlock(
                    in_channels=3,
                    out_channels=global_dim,
                    kernel_size=1,
                    conv_cfg="Conv1d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    act_before_norm=True,
                ),
                ConvBlock(
                    in_channels=global_dim,
                    out_channels=global_dim,
                    kernel_size=1,
                    conv_cfg="Conv1d",
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    act_before_norm=True,
                ),
            )

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        aligned_points = group_gather(s_points, neighbor_indices) - q_points.unsqueeze(3)
        new_features = self.f_shared_mlp(aligned_points)
        if s_feats is not None:
            neighbor_feats = group_gather(s_feats, neighbor_indices)
            neighbor_feats = torch.cat([neighbor_feats, new_features], dim=1)
        else:
            neighbor_feats = new_features
        x = self.x_shared_mlp(aligned_points).transpose(1, 3)
        q_feats = torch.matmul(neighbor_feats.transpose(1, 2), x).transpose(1, 2)
        q_feats = self.conv(q_feats).squeeze(3)
        if self.with_global:
            g_feats = self.g_conv(q_points)
            q_feats = torch.cat([q_feats, g_feats], dim=1)
        return q_feats
