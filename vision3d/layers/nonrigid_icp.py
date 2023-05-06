from typing import Optional

import ipdb
import torch
import torch.nn as nn
from torch import Tensor

from vision3d.ops import (
    apply_deformation,
    apply_rotation,
    apply_transform,
    axis_angle_to_rotation_matrix,
    get_transform_from_rotation_translation,
    skew_symmetric_matrix,
)


def non_rigid_icp_gauss_newton(
    src_nodes: Tensor,
    src_corr_points: Tensor,
    tgt_corr_points: Tensor,
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    edge_indices: Tensor,
    corr_weights: Optional[Tensor] = None,
    edge_weights: Optional[Tensor] = None,
    corr_lambda: float = 1.0,
    arap_lambda: float = 0.1,
    lm_lambda: float = 0.1,
    num_iterations: int = 5,
) -> Tensor:
    """Non-rigid ICP based on Embedded Deformation for non-rigid registration with Gauss-Newton solver.

    Args:
        src_nodes (Tensor): the nodes in the source point cloud in the shape of (M, 3).
        src_corr_points (Tensor): the correspondences in the source point cloud in the shape of (N, 3).
        tgt_corr_points (Tensor): the correspondences in the target point cloud in the shape of (N, 3).
        anchor_indices (LongTensor): the indices of the anchor nodes for each correspondence in the shape of (N, K).
        anchor_weights (Tensor): the weights of the anchor nodes for each correspondence in the shape of (N, K).
        edge_indices (LongTensor): the node indices of the edges in the shape of (E, 2).
        corr_weights (Tensor, optional): the weights of the correspondences in the shape of (N,).
        edge_weights (Tensor, optional): the weights of the edges in the shape of (E,).
        corr_lambda (float): the lambda for the correspondence term. Default: 1.0.
        arap_lambda (float): the lambda for the arap regularization term. Default: 0.1.
        lm_lambda (float): the lambda for Levenberg–Marquardt algorithm. Default: 0.1.
        num_iterations (int): the number of DFP iterations. Default: 5.

    Returns:
        transforms (Tensor): The transformations for each node in the shape of (M, 4, 4).
    """
    # ARAP: convert edge matrix to [u_index, v_index, uv_weight] pairs
    arap_u_indices = edge_indices[:, 0]  # (E')
    arap_v_indices = edge_indices[:, 1]  # (E')
    edge_masks = torch.ne(arap_u_indices, arap_v_indices)  # remove self edge
    arap_u_indices = arap_u_indices[edge_masks]  # (E)
    arap_v_indices = arap_v_indices[edge_masks]  # (E)
    if edge_weights is not None:
        arap_weights = edge_weights[edge_masks]  # (E)
        arap_weights = torch.sqrt(arap_weights)  # (E) arap weight is applied to final cost, so use sqrt here
    else:
        arap_weights = torch.ones(size=(arap_u_indices.shape[0],)).cuda()  # (E) fallback: all-one weights
    arap_u_nodes = src_nodes[arap_u_indices]  # (E, 3)
    arap_v_nodes = src_nodes[arap_v_indices]  # (E, 3)
    arap_deltas = arap_v_nodes - arap_u_nodes  # (E, 3)
    num_edges = arap_u_indices.shape[0]
    arap_e_indices = torch.arange(num_edges).cuda()  # (E)

    # CORR: convert anchor indices to [corr_index, node_index] pairs
    num_nodes = src_nodes.shape[0]
    num_matches = src_corr_points.shape[0]
    anchor_masks = (anchor_indices != -1) & (anchor_weights > 0.0)  # (N, K)
    corr_c_indices, col_indices = torch.nonzero(anchor_masks, as_tuple=True)  # (tot_C), (tot_C)
    corr_n_indices = anchor_indices[corr_c_indices, col_indices]  # (tot_C)
    corr_points = src_corr_points[corr_c_indices]  # (tot_C, 3)
    corr_nodes = src_nodes[corr_n_indices]  # (tot_C, 3)
    corr_deltas = corr_points - corr_nodes  # (tot_C, 3)
    if corr_weights is not None:
        # composed_anchor_weights = corr_weights.unsqueeze(1) * anchor_weights
        composed_anchor_weights = torch.sqrt(corr_weights).unsqueeze(1) * anchor_weights
        corr_weights = composed_anchor_weights[corr_c_indices, col_indices]  # (tot_C)
    else:
        corr_weights = anchor_weights[corr_c_indices, col_indices]  # (tot_C)
    total_matches = corr_c_indices.shape[0]

    # rotations, translations
    rotations = torch.eye(3).cuda().unsqueeze(0).repeat(num_nodes, 1, 1)  # (M, 3, 3)
    translations = torch.zeros(size=(num_nodes, 3)).cuda()  # (M, 3)

    # identities
    corr_identities = torch.eye(3).cuda().unsqueeze(0).repeat(total_matches, 1, 1)  # (tot_C, 3, 3)
    arap_identities = torch.eye(3).cuda().unsqueeze(0).repeat(num_edges, 1, 1)  # (E, 3, 3)

    for _ in range(num_iterations):
        transforms = get_transform_from_rotation_translation(rotations, translations)

        # compute corr jacobian matrix
        # jacobian dim: 2 (phi or t), N, M, 3 (xyz in residual), 3 (dims in phi or t)
        corr_rot_deltas = apply_rotation(corr_deltas, rotations[corr_n_indices])  # (tot_C, 3)
        corr_skew_matrices = skew_symmetric_matrix(corr_rot_deltas)  # (tot_C, 3, 3)
        corr_r_derivatives = -corr_lambda * corr_weights.view(-1, 1, 1) * corr_skew_matrices  # (tot_C, 3, 3)
        corr_t_derivatives = corr_lambda * corr_weights.view(-1, 1, 1) * corr_identities  # (tot_C, 3, 3)
        corr_jacobian = torch.zeros(size=(2, num_matches, num_nodes, 3, 3)).cuda()
        corr_jacobian[0, corr_c_indices, corr_n_indices] = corr_r_derivatives
        corr_jacobian[1, corr_c_indices, corr_n_indices] = corr_t_derivatives
        corr_jacobian = corr_jacobian.permute(1, 3, 0, 2, 4).contiguous().view(num_matches * 3, num_nodes * 6)

        # compute corr residual vector
        warped_src_corr_points = apply_deformation(
            src_corr_points, src_nodes, transforms, anchor_indices, anchor_weights
        )
        corr_residuals = corr_lambda * (warped_src_corr_points - tgt_corr_points)  # (C, 3)
        corr_residuals = corr_residuals.flatten()

        jacobian = corr_jacobian  # (3C, 6M)
        residuals = corr_residuals  # (3C)

        if arap_lambda > 0:
            # compute arap jacobian matrix
            # jacobian dim: 2 (phi or t), E, M, 3 (xyz in residual), 3 (dims in phi or t)
            arap_rot_deltas = apply_rotation(arap_deltas, rotations[arap_u_indices])  # (E, 3)
            arap_skew_matrices = skew_symmetric_matrix(arap_rot_deltas)  # (E, 3, 3)
            arap_r_derivatives = -arap_lambda * arap_weights.view(-1, 1, 1) * arap_skew_matrices  # (E, 3, 3)
            arap_t_derivatives = arap_lambda * arap_weights.view(-1, 1, 1) * arap_identities  # (E, 3, 3)
            arap_jacobian = torch.zeros(size=(2, num_edges, num_nodes, 3, 3)).cuda()
            arap_jacobian[0, arap_e_indices, arap_u_indices] = arap_r_derivatives
            arap_jacobian[1, arap_e_indices, arap_u_indices] = arap_t_derivatives
            arap_jacobian[1, arap_e_indices, arap_v_indices] = -arap_t_derivatives
            arap_jacobian = arap_jacobian.permute(1, 3, 0, 2, 4).contiguous().view(num_edges * 3, num_nodes * 6)

            # compute arap residual vector
            warped_v_nodes = apply_transform(arap_deltas, transforms[arap_u_indices]) + arap_u_nodes
            arap_residuals = warped_v_nodes - arap_v_nodes - translations[arap_v_indices]
            arap_residuals = arap_lambda * arap_weights.unsqueeze(1) * arap_residuals
            arap_residuals = arap_residuals.flatten()

            jacobian = torch.cat([jacobian, arap_jacobian], dim=0)  # (3C+3E, 6M)
            residuals = torch.cat([residuals, arap_residuals], dim=0)  # (3C+3E)

        # solve
        jacobian_t = jacobian.transpose(0, 1)
        a = torch.matmul(jacobian_t, jacobian)  # (6M, 6M)
        a = a + torch.eye(a.shape[0]).cuda() * lm_lambda
        b = -torch.matmul(jacobian_t, residuals)  # (6M)
        x = torch.linalg.solve(a, b)  # (6M)

        inc_rotations = axis_angle_to_rotation_matrix(x[: num_nodes * 3].view(num_nodes, 3))
        inc_translations = x[num_nodes * 3 :].view(num_nodes, 3)

        rotations = torch.matmul(inc_rotations, rotations)
        translations = translations + inc_translations

    transforms = get_transform_from_rotation_translation(rotations, translations)

    return transforms


class NonRigidICP(nn.Module):
    """Non-rigid ICP for non-rigid point cloud registration.

    This module solves for an Embedded Deformation warping function with Gauss-Newton solver.
    """

    def __init__(
        self, corr_lambda: float = 1.0, arap_lambda: float = 1.0, lm_lambda: float = 0.01, num_iterations: int = 5
    ):
        super().__init__()
        self.corr_lambda = corr_lambda
        self.arap_lambda = arap_lambda
        self.lm_lambda = lm_lambda
        self.num_iterations = num_iterations

    def forward(
        self,
        src_nodes: Tensor,
        src_corr_points: Tensor,
        tgt_corr_points: Tensor,
        anchor_indices: Tensor,
        anchor_weights: Tensor,
        edge_indices: Tensor,
        corr_weights: Optional[Tensor] = None,
        edge_weights: Optional[Tensor] = None,
    ) -> Tensor:
        transforms = non_rigid_icp_gauss_newton(
            src_nodes,
            src_corr_points,
            tgt_corr_points,
            anchor_indices,
            anchor_weights,
            edge_indices,
            corr_weights=corr_weights,
            edge_weights=edge_weights,
            corr_lambda=self.corr_lambda,
            arap_lambda=self.arap_lambda,
            lm_lambda=self.lm_lambda,
            num_iterations=self.num_iterations,
        )
        return transforms

    def extra_repr(self) -> str:
        param_strings = [
            f"corr_lambda={self.corr_lambda:g}",
            f"arap_lambda={self.arap_lambda:g}",
            f"lm_lambda={self.lm_lambda:g}",
            f"num_iterations={self.num_iterations}",
        ]
        format_string = ", ".join(param_strings)
        return format_string
