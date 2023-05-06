from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers import SinusoidalEmbedding
from vision3d.ops import index_select, pairwise_distance


class GeometricStructureEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sigma_d: float,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        angle_reduction: str = "max",
        use_angle_embed: bool = True,
    ):
        super().__init__()

        assert angle_reduction in ["max", "mean"], f"Unsupported reduction mode: {angle_reduction}."

        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k
        self.angle_reduction = angle_reduction
        self.use_angle_embed = use_angle_embed

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim) if self.use_angle_embed else None

    @torch.no_grad()
    def get_embedding_indices(self, points: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_points, _ = points.shape
        assert num_points > 1, "Too few superpoints."

        # distance indices
        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        if not self.use_angle_embed:
            return d_indices, None

        # angular indices
        k = min(self.angle_k, num_points - 1)
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_points, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points: Tensor) -> Tensor:
        d_indices, a_indices = self.get_embedding_indices(points)

        # distance embeddings
        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)
        embeddings = d_embeddings

        # angular embeddings
        if a_indices is not None:
            a_embeddings = self.embedding(a_indices)
            a_embeddings = self.proj_a(a_embeddings)
            if self.angle_reduction == "max":
                if self.training:
                    a_embeddings = a_embeddings.max(dim=3)[0]
                else:
                    # use amax to reduce memory footprint during testing
                    a_embeddings = a_embeddings.amax(dim=3)
            else:
                a_embeddings = a_embeddings.mean(dim=3)
            embeddings = embeddings + a_embeddings

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [f"sigma_d={self.sigma_d:g}"]
        if self.use_angle_embed:
            param_strings.append(f"sigma_a={self.sigma_a:g}")
            param_strings.append(f"angle_k={self.angle_k}")
            param_strings.append(f"angle_reduction={self.angle_reduction}")
        param_strings.append(f"use_angle_embed={self.use_angle_embed}")
        format_string = ", ".join(param_strings)
        return format_string


class GeometricStructureEmbeddingV2(nn.Module):
    def __init__(
        self, hidden_dim: int, sigma_d: float, sigma_a: float = 15.0, angle_k: int = 3, angle_reduction: str = "max"
    ):
        super().__init__()

        assert angle_reduction in ["max", "mean"], f"Unsupported reduction mode: {angle_reduction}."

        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.angle_k = angle_k
        self.angle_reduction = angle_reduction

        self.embedding = SinusoidalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.proj_e = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    @torch.no_grad()
    def get_embedding_indices(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        # distance indices
        dist_map = pairwise_distance(points, points, squared=False)  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        # angular indices
        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = torch.round(angles * (180.0 / np.pi)).long()  # in degrees

        return d_indices, a_indices

    def forward(self, points: Tensor) -> Tensor:
        d_indices, a_indices = self.get_embedding_indices(points)

        # distance embeddings
        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        # angular embeddings
        all_a_indices = torch.arange(181).cuda().float() / self.sigma_a  # (181)
        all_a_embeddings = self.embedding(all_a_indices)  # (181, C)
        all_a_embeddings = self.proj_a(all_a_embeddings)  # (181, C)
        a_embeddings = index_select(all_a_embeddings, a_indices, dim=0)  # (B, N, N, k, C)
        if self.angle_reduction == "max":
            if self.training:
                a_embeddings = a_embeddings.max(dim=3)[0]
            else:
                a_embeddings = a_embeddings.amax(dim=3)
        else:
            a_embeddings = a_embeddings.mean(dim=3)

        # geometric structure embeddings
        embeddings = d_embeddings + a_embeddings
        embeddings = self.act(embeddings)
        embeddings = self.proj_e(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def extra_repr(self) -> str:
        param_strings = [
            f"sigma_d={self.sigma_d:g}",
            f"sigma_a={self.sigma_a:g}",
            f"angle_k={self.angle_k}",
            f"angle_reduction={self.angle_reduction}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
