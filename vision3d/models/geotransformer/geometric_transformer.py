import torch.nn as nn

from vision3d.layers import TransformerLayer

from .geometric_structure_embedding import GeometricStructureEmbedding


class GeometricTransformer(nn.Module):
    """Geometric Transformer (GeoTransformer).

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        hidden_dim: hidden feature dimension
        num_heads: number of head in transformer
        blocks: list of 'self' or 'cross'
        sigma_d: temperature of distance
        sigma_a: temperature of angles
        angle_k: number of nearest neighbors for angular embedding
        angle_reduction: reduction mode of angular embedding ['max', 'mean']
        use_angle_embed: If True, use angular embedding
        dropout: dropout ratio in transformer
        activation_fn: activation function
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a=15.0,
        angle_k=3,
        angle_reduction="max",
        use_angle_embed=True,
        dropout=None,
        activation_fn="ReLU",
    ):
        super().__init__()

        self.embedding = GeometricStructureEmbedding(
            hidden_dim,
            sigma_d,
            sigma_a=sigma_a,
            angle_k=angle_k,
            angle_reduction=angle_reduction,
            use_angle_embed=use_angle_embed,
        )

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["self", "cross"]
            if block == "self":
                layers.append(
                    TransformerLayer(
                        hidden_dim,
                        num_heads,
                        qk_embed_proj=True,
                        dropout=dropout,
                        act_cfg=activation_fn,
                    )
                )
            else:
                layers.append(
                    TransformerLayer(
                        hidden_dim,
                        num_heads,
                        dropout=dropout,
                        act_cfg=activation_fn,
                    )
                )
        self.transformer = nn.ModuleList(layers)

    def forward(
        self,
        src_points,
        tgt_points,
        src_feats,
        tgt_feats,
        src_masks=None,
        tgt_masks=None,
    ):
        """Geometric Transformer

        Args:
            src_points (Tensor): (B, N, 3)
            tgt_points (Tensor): (B, M, 3)
            src_feats (Tensor): (B, N, C)
            tgt_feats (Tensor): (B, M, C)
            src_masks (BoolTensor, optional): (B, N)
            tgt_masks (BoolTensor, optional): (B, M)

        Returns:
            src_feats: torch.Tensor (B, N, C)
            tgt_feats: torch.Tensor (B, M, C)
        """
        src_embeddings = self.embedding(src_points)
        tgt_embeddings = self.embedding(tgt_points)

        src_tokens = self.in_proj(src_feats)
        tgt_tokens = self.in_proj(tgt_feats)

        for i, block in enumerate(self.blocks):
            if block == "self":
                src_tokens = self.transformer[i](
                    src_tokens, src_tokens, src_tokens, qk_embeds=src_embeddings, k_masks=src_masks
                )
                tgt_tokens = self.transformer[i](
                    tgt_tokens, tgt_tokens, tgt_tokens, qk_embeds=tgt_embeddings, k_masks=tgt_masks
                )
            else:
                src_tokens = self.transformer[i](src_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
                tgt_tokens = self.transformer[i](tgt_tokens, src_tokens, src_tokens, k_masks=src_masks)

        src_feats = self.out_proj(src_tokens)
        tgt_feats = self.out_proj(tgt_tokens)

        return src_feats, tgt_feats
