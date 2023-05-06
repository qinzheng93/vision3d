from typing import List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

from vision3d.layers import TransformerLayer


class VanillaSiameseTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        blocks: List[str],
        dropout: Optional[float] = None,
        activation_fn: str = "ReLU",
    ):
        super().__init__()

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["self", "cross"]
            layers.append(TransformerLayer(hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        self.transformer = nn.ModuleList(layers)

    def forward(
        self,
        src_feats: Tensor,
        tgt_feats: Tensor,
        src_masks: Optional[Tensor] = None,
        tgt_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        src_tokens = self.in_proj(src_feats)
        tgt_tokens = self.in_proj(tgt_feats)

        for i, block in enumerate(self.blocks):
            if block == "self":
                src_tokens = self.transformer[i](src_tokens, src_tokens, src_tokens, k_masks=src_masks)
                tgt_tokens = self.transformer[i](tgt_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
            else:
                src_tokens = self.transformer[i](src_tokens, tgt_tokens, tgt_tokens, k_masks=tgt_masks)
                tgt_tokens = self.transformer[i](tgt_tokens, src_tokens, src_tokens, k_masks=src_masks)

        src_feats = self.out_proj(src_tokens)
        tgt_feats = self.out_proj(tgt_tokens)

        return src_feats, tgt_feats
