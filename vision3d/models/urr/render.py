from typing import Optional, Tuple

import ipdb
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.engine import get_context_manager
from vision3d.ops import create_meshgrid, index_select, knn, render
from vision3d.utils.profiling import profile_cuda_runtime


def rasterize(
    pcd_pixels: Tensor,
    pcd_depths: Tensor,
    image_h: int,
    image_w: int,
    num_samples: int,
    radius: float,
) -> Tuple[Tensor, ...]:
    # render pixels
    img_pixels = create_meshgrid(image_h, image_w, flatten=True, centering=True)  # (HxW, 2)

    # compute knn pixels and z-buffer
    indices = knn(img_pixels, pcd_pixels, k=num_samples)  # (HxW, K)
    zbuffers = index_select(pcd_depths, indices, dim=0)  # (HxW, K)
    # ipdb.set_trace()

    # sort with z-buffer
    zbuffers, sorted_indices = torch.sort(zbuffers, dim=-1)  # (HxW, K), (HxW, K)
    pixel_indices = torch.arange(image_h * image_w).cuda().unsqueeze(1)  # (HxW, 1)
    indices = indices[pixel_indices, sorted_indices]  # (HxW, K)

    # compute distances and masking
    knn_pixels = index_select(pcd_pixels, indices, dim=0)  # (HxW, K, 2)
    distances = torch.linalg.norm(knn_pixels - img_pixels.unsqueeze(1), dim=-1)  # (HxW, K)
    masks = torch.lt(distances, radius)

    # reshape
    distances = distances.view(image_h, image_w, num_samples)
    indices = indices.view(image_h, image_w, num_samples)
    zbuffers = zbuffers.view(image_h, image_w, num_samples)
    masks = masks.view(image_h, image_w, num_samples)

    return distances, indices, zbuffers, masks


class DifferentiableRenderer(nn.Module):
    def __init__(
        self,
        image_h: int,
        image_w: int,
        num_samples: int,
        radius: float,
        sigma: float,
        weighting_fn: str = "exponential",
        compositing_fn: str = "weighted",
        eps: float = 1e-10,
        min_depth: float = 1e-6,
    ):
        super().__init__()

        assert weighting_fn in ["linear", "exponential"], f"Unsupported weighting_fn: {weighting_fn}."
        assert compositing_fn in ["weighted", "alpha"], f"Unsupported compositing_fn: {compositing_fn}."

        self.image_h = image_h
        self.image_w = image_w
        self.num_samples = num_samples
        self.radius = radius
        self.sigma = sigma
        self.weighting_fn = weighting_fn
        self.compositing_fn = compositing_fn
        self.eps = eps
        self.min_depth = min_depth

    def forward(
        self, pcd_points: Tensor, pcd_colors: Optional[Tensor], pcd_feats: Optional[Tensor], intrinsics: Tensor
    ) -> Tuple[Tensor, ...]:
        # remove the points behind the camera
        pcd_depths = pcd_points[:, 2]
        pcd_masks = torch.gt(pcd_depths, self.min_depth)
        assert pcd_masks.any(), "All points are behind the camera!"
        pcd_points = pcd_points[pcd_masks]
        pcd_depths = pcd_depths[pcd_masks]

        # render points
        pcd_pixels = render(pcd_points, intrinsics, rounding=False)

        # rasterize
        distances, indices, _, masks = rasterize(
            pcd_pixels, pcd_depths, self.image_h, self.image_w, self.num_samples, self.radius
        )

        # compute weights
        if self.weighting_fn == "exponential":
            weights = torch.exp(-distances.pow(2) / self.sigma ** 2) * masks.float()  # (H, W, K)
        elif self.weighting_fn == "linear":
            weights = (1.0 - distances.pow(2) / self.sigma ** 2) * masks.float()  # (H, W, K)
        else:
            raise ValueError(f"Unsupported weighting_fn: {self.weighting_fn}.")

        if self.compositing_fn == "weighted":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)  # (H, W, K)
        elif self.compositing_fn == "alpha":
            alphas = torch.cat([torch.ones_like(weights[..., :1]), 1.0 - weights[..., :-1] + self.eps], dim=-1)
            weights = weights * torch.cumprod(alphas, dim=-1)  # (H, W, K)
        else:
            raise ValueError(f"Unsupported compositing_fn: {self.compositing_fn}.")

        # render depth
        sel_depths = index_select(pcd_depths, indices, dim=0)  # (H, W, K)
        rendered_depths = torch.sum(sel_depths * weights, dim=2)  # (H, W)

        # render colors
        if pcd_colors is not None:
            sel_colors = index_select(pcd_colors, indices, dim=0)  # (H, W, K, 3)
            rendered_colors = torch.sum(sel_colors * weights.unsqueeze(-1), dim=2)  # (H, W, 3)
        else:
            rendered_colors = None

        # render features
        if pcd_feats is not None:
            sel_feats = index_select(pcd_feats, indices, dim=0)  # (H, W, K, C)
            rendered_feats = torch.sum(sel_feats * weights.unsqueeze(-1), dim=2)  # (H, W, C)
        else:
            rendered_feats = None

        # compute masks
        rendered_masks = torch.any(masks, dim=-1)  # (H, W)

        return rendered_depths, rendered_colors, rendered_feats, rendered_masks
