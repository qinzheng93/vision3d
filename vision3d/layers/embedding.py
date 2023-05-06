from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .basic_layers import build_dropout_layer


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding.

    Paper: [Attention is all you need](https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, d_model: int):
        """Initialize a Sinusoidal embedding function.

        Args:
            d_model (int): the dimension of the embedding.
        """
        super().__init__()
        assert d_model % 2 == 0, f"Sinusoidal positional encoding with odd d_model: {d_model}."
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, emb_indices: Tensor) -> Tensor:
        """Sinusoidal positional embedding forward.

        Args:
            emb_indices (Tensor): (*)

        Returns:
            embeddings (Tensor): (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

    def extra_repr(self) -> str:
        format_string = f"d_model: {self.d_model}"
        return format_string


class FourierEmbedding(nn.Module):
    """Fourier positional embedding.

    Emb(x) = [sin(2^k Pi x), cos(2^k Pi x), sin(2^(k+1) Pi x), cos(2^(k+1) Pi x), ..., sin(2^(k+L-1) Pi x), cos(2^(k+L-1) Pi x)],
    where x is the input tensor.
    """

    def __init__(self, length: int, k0: float = 0.0, use_pi: bool = True, use_input: bool = False) -> None:
        """Initialize a Fourier embedding function.

        Args:
            length (float): the length of the embedding.
            k0 (float): the starting exponential of the embedding. Default: 0.
            use_pi (bool): if True, use pi in the embedding. Default: True.
            use_input (bool): if True, return the input vector in the embedding. Default: False.
        """
        super().__init__()
        self.length = length
        self.k0 = k0
        self.use_pi = use_pi
        self.use_input = use_input

    def forward(self, inputs: Tensor) -> Tensor:
        """Fourier embedding forward.

        Args:
            inputs (Tensor): the input tensor in the shape of (*, N).

        Returns:
            A Tensor of the embeddings in the shape of (*, Lx2xN) or (*, (2L+1)xN) if use_input.
        """
        batch_shape = inputs.shape[:-1]
        num_inputs = inputs.shape[-1]
        inputs = inputs.view(-1, 1, num_inputs)  # (B, 1, N)
        factors = (2.0 ** torch.arange(self.k0, self.k0 + self.length).float().cuda()).view(1, -1, 1)  # (1, L, 1)
        if self.use_pi:
            factors = factors * np.pi
        thetas = factors * inputs  # (B, L, N)
        sin_values = torch.sin(thetas)  # (B, L, N)
        cos_values = torch.cos(thetas)  # (B, L, N)
        embeddings = torch.cat([sin_values, cos_values], dim=-1)  # (B, L, 2xN)
        output_shape = batch_shape + (2 * self.length * num_inputs,)
        embeddings = embeddings.view(*output_shape)  # (*, Lx2xN)
        if self.use_input:
            input_shape = batch_shape + (num_inputs,)
            inputs = inputs.view(*input_shape)
            embeddings = torch.cat([inputs, embeddings], dim=-1)  # (*, (2L+1)xN)
        return embeddings


class LearnableEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dropout: Optional[float] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices: Tensor) -> Tensor:
        """Learnable Positional Embedding.

        `emb_indices` are truncated to fit the finite embedding space.

        Args:
            emb_indices (LongTensor): (*)

        Returns:
            embeddings (Tensor): (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings
