import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.ops import get_rotation_translation_from_transform


def rotation_loss(estimated_rotations, gt_rotations):
    """Rotation loss.

    Args:
        estimated_rotations (Tensor): (B, 3, 3)
        gt_rotations (Tensor): (B, 3, 3)

    Returns:
        r_loss (Tensor): rotation loss
    """
    identity = torch.eye(3).to(estimated_rotations).unsqueeze(0).expand_as(estimated_rotations)
    r_loss = F.mse_loss(torch.matmul(estimated_rotations.transpose(-1, -2), gt_rotations), identity)
    return r_loss


def translation_loss(estimated_translations, gt_translations):
    """Translation loss.

    Args:
        estimated_translations (Tensor): (B, 3)
        gt_translations (Tensor): (B, 3)

    Returns:
        t_loss (Tensor): translation loss
    """
    t_loss = F.mse_loss(estimated_translations, gt_translations)
    return t_loss


class TransformationLoss(nn.Module):
    f"""Rigid Transformation Loss.

    Rigid transformation loss is comprised of two parts:
    1. Rotation loss:
        L_R = \lVert R^T \cdot R^* - I \rVert_2^2
    2. Translation loss:\
        L_t = \lVert t - t^* \rVert_2^2\
    And the final transformation loss is:\
        L = w^R \cdot L_R + w^t \cdot L_t
    """

    def __init__(self, weight_r_loss=1.0, weight_t_loss=1.0):
        super().__init__()
        self.weight_r_loss = weight_r_loss
        self.weight_t_loss = weight_t_loss

    def forward(self, estimated_transforms, gt_transforms):
        """Transformation loss forward.

        Args:
            estimated_transforms (Tensor): (B, 4, 4)
            gt_transforms (Tensor): (B, 4, 4)

        Return:
            loss (Tensor): overall transformation loss
            r_loss (Tensor): rotation loss
            t_loss (Tensor): translation loss
        """
        estimated_rotations, estimated_translations = get_rotation_translation_from_transform(estimated_transforms)
        gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)

        r_loss = rotation_loss(estimated_rotations, gt_rotations)
        t_loss = translation_loss(estimated_translations, gt_translations)

        loss = self.weight_r_loss * r_loss + self.weight_t_loss * t_loss

        return loss, r_loss, t_loss
