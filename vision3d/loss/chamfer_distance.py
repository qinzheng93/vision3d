from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import pairwise_distance


def chamfer_distance_loss(
    src_points: Tensor,
    tgt_points: Tensor,
    src_masks: Optional[Tensor],
    tgt_masks: Optional[Tensor],
    transposed: bool = False,
    squared: bool = False,
    truncate: Optional[float] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
    inf: float = 1e10,
) -> Tensor:
    """Chamfer distance loss (truncated).

    Args:
        src_points (Tensor): the source point cloud in the shape of (*, N, 3).
        tgt_points (Tensor): the source point cloud in the shape of (*, M, 3).
        src_masks (BoolTensor, optional): the masks of the source point cloud in the shape of (*, N).
        tgt_masks (BoolTensor, optional): the masks of the target point cloud in the shape of (*, M).
        transposed (bool): if True, the shape of the point cloud is (*, 3, N) and (*, 3, M). Default: False.
        squared (bool): if True, use the squared distance. Default: False.
        truncate (float, optional): if not None, use truncated chamfer distance.
        reduction (str): reduction method: "mean", "sum". Default: "mean".
        eps (float): a safe number for sqrt. Default: 1e-12.
        inf (float): a safe number for the masked distances. Default: 1e10.

    Returns:
        A Tensor of the chamfer distance loss.
    """
    assert reduction in ["mean", "sum"], f"Unsupported reduction: {reduction}."

    dist_mat = pairwise_distance(src_points, tgt_points, transposed=transposed, squared=squared, eps=eps)  # (*, N, M)
    if src_masks is not None:
        dist_mat = dist_mat.masked_fill(src_masks.unsqueeze(-1), inf)
    if tgt_masks is not None:
        dist_mat = dist_mat.masked_fill(tgt_masks.unsqueeze(-2), inf)

    src_nn_distances = dist_mat.min(dim=-1)[0]  # (*, N)
    tgt_nn_distances = dist_mat.min(dim=-2)[0]  # (*, M)

    if truncate is not None:
        if squared:
            truncate = truncate ** 2
        src_trunc_masks = torch.lt(src_nn_distances, truncate)
        tgt_trunc_masks = torch.lt(tgt_nn_distances, truncate)
        src_masks = torch.logical_and(src_masks, src_trunc_masks) if src_masks is not None else src_trunc_masks
        tgt_masks = torch.logical_and(tgt_masks, tgt_trunc_masks) if tgt_masks is not None else tgt_trunc_masks

    src_nn_distances = src_nn_distances[src_masks]
    tgt_nn_distances = tgt_nn_distances[tgt_masks]

    if reduction == "mean":
        src_loss = src_nn_distances.mean()
        tgt_loss = tgt_nn_distances.mean()
    else:
        src_loss = src_nn_distances.sum()
        tgt_loss = tgt_nn_distances.sum()

    loss = src_loss + tgt_loss

    return loss


class ChamferDistanceLoss(nn.Module):
    def __init__(
        self,
        squared: bool = False,
        truncate: Optional[float] = None,
        reduction: str = "mean",
        eps: float = 1e-12,
        inf: float = 1e10,
    ):
        super().__init__()
        assert reduction in ["mean", "sum"]
        self.truncate = truncate
        self.squared = squared
        self.reduction = reduction
        self.eps = eps
        self.inf = inf

    def forward(
        self,
        src_points: Tensor,
        tgt_points: Tensor,
        src_masks: Optional[Tensor] = None,
        tgt_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Chamfer Distance forward.

        Args:
            src_points (Tensor): the source point cloud (*, N, C).
            tgt_points (Tensor): the target point cloud (*, M, C).
            src_masks (BoolTensor): the masks of the source point cloud (*, N).
            tgt_masks (BoolTensor): the masks of the target point cloud (*, M).

        Returns:
            A Tensor of the chamfer distance loss.
        """
        loss = chamfer_distance_loss(
            src_points,
            tgt_points,
            src_masks=src_masks,
            tgt_masks=tgt_masks,
            squared=self.squared,
            truncate=self.truncate,
            reduction=self.reduction,
            eps=self.eps,
            inf=self.inf,
        )

        return loss

    def extra_repr(self) -> str:
        param_strings = [f"squared: {self.squared}"]
        if self.truncate is not None:
            param_strings.append(f"truncate: {self.truncate:g}")
        param_strings.append(f"reduction: {self.reduction}")
        param_strings.append(f"eps: {self.eps:g}")
        format_string = ", ".join(param_strings)
        return format_string
