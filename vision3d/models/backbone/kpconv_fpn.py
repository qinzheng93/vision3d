import torch
import torch.nn as nn

from vision3d.layers import KPConvBlock, KPResidualBlock
from vision3d.layers.unary_block import UnaryBlockPackMode
from vision3d.ops import nearest_interpolate_pack_mode


class KPConvFPN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        first_dim,
        kernel_size,
        voxel_size,
        kpconv_radius,
        kpconv_sigma,
        encoder_stages,
        decoder_stages,
        norm_act_last=False,
        norm_cfg="GroupNorm",
        act_cfg="LeakyReLU",
    ):
        super().__init__()

        assert (
            encoder_stages >= 2
        ), f"[{self.__class__.__name__}] At least TWO stages required in encoder ({encoder_stages} given)."
        assert (
            decoder_stages >= 1
        ), f"[{self.__class__.__name__}] At least ONE stage required in decoder ({decoder_stages} given)."
        assert (
            encoder_stages >= decoder_stages
        ), f"[{self.__class__.__name__}] Encoder has less stages than decoder ({encoder_stages} vs {decoder_stages})."

        self.voxel_size = voxel_size
        self.kpconv_radius = kpconv_radius
        self.kpconv_sigma = kpconv_sigma
        self.encoder_stages = encoder_stages
        self.decoder_stages = decoder_stages

        radius = voxel_size * kpconv_radius
        sigma = voxel_size * kpconv_sigma

        # encoder
        self.add_module(
            "encoder1_1",
            KPConvBlock(input_dim, first_dim, kernel_size, radius, sigma, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.add_module(
            "encoder1_2",
            KPResidualBlock(first_dim, first_dim * 2, kernel_size, radius, sigma, norm_cfg=norm_cfg, act_cfg=act_cfg),
        )

        feat_dim = first_dim * 2
        for i in range(1, self.encoder_stages):
            self.add_module(
                "encoder{}_1".format(i + 1),
                KPResidualBlock(
                    feat_dim, feat_dim, kernel_size, radius, sigma, strided=True, norm_cfg=norm_cfg, act_cfg=act_cfg
                ),
            )
            self.add_module(
                "encoder{}_2".format(i + 1),
                KPResidualBlock(
                    feat_dim, feat_dim * 2, kernel_size, radius * 2, sigma * 2, norm_cfg=norm_cfg, act_cfg=act_cfg
                ),
            )
            self.add_module(
                "encoder{}_3".format(i + 1),
                KPResidualBlock(
                    feat_dim * 2, feat_dim * 2, kernel_size, radius * 2, sigma * 2, norm_cfg=norm_cfg, act_cfg=act_cfg
                ),
            )
            feat_dim *= 2
            radius *= 2
            sigma *= 2

        # decoder
        for i in range(self.encoder_stages - 1, self.encoder_stages - self.decoder_stages, -1):
            if not norm_act_last and i == self.encoder_stages - self.decoder_stages + 1:
                self.add_module("decoder{}".format(i), nn.Linear(feat_dim + feat_dim // 2, output_dim))
            else:
                self.add_module("decoder{}".format(i), UnaryBlockPackMode(feat_dim + feat_dim // 2, feat_dim // 2))
            feat_dim //= 2

    def forward(self, feats, data_dict):
        points = data_dict["points"]
        neighbors = data_dict["neighbors"]
        subsamplings = data_dict["subsampling"]
        upsamplings = data_dict["upsampling"]

        # encoder
        encoder_feats_list = []
        feats = getattr(self, "encoder1_1")(points[0], points[0], feats, neighbors[0])
        feats = getattr(self, "encoder1_2")(points[0], points[0], feats, neighbors[0])
        encoder_feats_list.append(feats)
        for i in range(1, self.encoder_stages):
            feats = getattr(self, "encoder{}_1".format(i + 1))(points[i], points[i - 1], feats, subsamplings[i - 1])
            feats = getattr(self, "encoder{}_2".format(i + 1))(points[i], points[i], feats, neighbors[i])
            feats = getattr(self, "encoder{}_3".format(i + 1))(points[i], points[i], feats, neighbors[i])
            encoder_feats_list.append(feats)

        # encoder
        decoder_feats_list = []
        decoder_feats_list.append(feats)
        for i in range(self.encoder_stages - 1, self.encoder_stages - self.decoder_stages, -1):
            feats = nearest_interpolate_pack_mode(feats, upsamplings[i - 1])
            feats = torch.cat([feats, encoder_feats_list[i - 1]], dim=1)
            feats = getattr(self, "decoder{}".format(i))(feats)
            decoder_feats_list.append(feats)

        decoder_feats_list.reverse()

        return decoder_feats_list
