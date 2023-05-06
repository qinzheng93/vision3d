from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

from .numeric import safe_divide


def apply_rotation(
    points: Tensor, rotation: Tensor, normals: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are three cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), rotation is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.
    3. points and normals are (B, 3), rotation is (B, 3, 3), the output points are (B, 3).
       In this case, the points are automatically broadcast to (B, 1, 3) and the rotation is applied batch-wise. The
       first dim of points/normals and rotation must be the same.

    Args:
        points (Tensor): (*, 3) or (B, N, 3), or (B, 3)
        normals (Tensor=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    assert rotation.dim() == 2 or (
        rotation.dim() == 3 and points.dim() in (2, 3)
    ), f"Incompatible shapes between points {tuple(points.shape)} and rotation {tuple(rotation.shape)}."

    if normals is not None:
        assert (
            points.shape == normals.shape
        ), f"The shapes of points {tuple(points.shape)} and normals {tuple(normals.shape)} must be the same."

    if rotation.dim() == 2:
        # case 1
        input_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*input_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*input_shape)
    elif rotation.dim() == 3 and points.dim() == 3:
        # case 2
        points = torch.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    elif rotation.dim() == 3 and points.dim() == 2:
        # case 3
        points = points.unsqueeze(1)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.squeeze(1)
        if normals is not None:
            normals = normals.unsqueeze(1)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.squeeze(1)

    if normals is not None:
        return points, normals

    return points


def skew_symmetric_matrix(vector: Tensor) -> Tensor:
    """Compute Skew-symmetric Matrix.

    [v]_{\times} =  0 -z  y
                    z  0 -x
                   -y  x  0

    Note: Use matrix multiplication to make the computation differentiable.

    Args:
        vector (Tensor): input vectors (*, 3)

    Returns:
        skew (Tensor): output skew-symmetric matrix (*, 3, 3)
    """
    vector_shape = vector.shape
    matrix_shape = vector_shape[:-1] + (9, 3)
    vector_to_skew = torch.zeros(size=matrix_shape).cuda()  # (*, 9, 3)
    vector_to_skew[..., 1, 2] = -1.0
    vector_to_skew[..., 2, 1] = 1.0
    vector_to_skew[..., 3, 2] = 1.0
    vector_to_skew[..., 5, 0] = -1.0
    vector_to_skew[..., 6, 1] = -1.0
    vector_to_skew[..., 7, 0] = 1.0
    skew_shape = vector_shape[:-1] + (3, 3)
    skew = torch.matmul(vector_to_skew, vector.unsqueeze(-1)).view(*skew_shape)
    return skew


def rodrigues_rotation_formula(omega: Tensor, theta: Tensor) -> Tensor:
    """Compute rotation matrix from axis-angle with Rodrigues' Rotation Formula.

    R = I + \sin{\theta} K + (1 - \cos{\theta}) K^2,
    where K is the skew-symmetric matrix of the axis vector.

    Note:
        If omega is a zero vector, the rotation matrix is always an identity matrix.

    Args:
        omega (Tensor): The unit rotation axis vector in the shape of (*, 3).
        theta (Tensor): The rotation angles (rad) in right-hand direction in the shape of (*).

    Returns:
        rotations (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).
    """
    input_shape = omega.shape
    omega = omega.view(-1, 3)
    theta = theta.view(-1)
    skew = skew_symmetric_matrix(omega)  # (B, 3, 3)
    sin_value = torch.sin(theta).view(-1, 1, 1)  # (B, 1, 1)
    cos_value = torch.cos(theta).view(-1, 1, 1)  # (B, 1, 1)
    eye = torch.eye(3).cuda().unsqueeze(0).expand_as(skew)  # (B, 3, 3)
    rotation = eye + sin_value * skew + (1.0 - cos_value) * torch.matmul(skew, skew)
    output_shape = input_shape[:-1] + (3, 3)
    rotation = rotation.view(*output_shape)
    return rotation


def alignment_rotation_matrix(src_vector: Tensor, tgt_vector: Tensor, eps: float = 1e-5) -> Tensor:
    """Compute the rotation matrix aligning the source vector to the target vector using Rodrigues' Rotation Formula.

    Args:
        src_vector (Tensor): The source vectors (*, 3)
        tgt_vector (Tensor): The target vectors (*, 3)
        eps (float=1e-5): A safe number.

    Returns:
        rotation (Tensor): rotation matrix (*, 3, 3)
    """
    # check norm
    src_normal = torch.linalg.norm(src_vector, dim=-1)  # (*, 3)
    tgt_normal = torch.linalg.norm(tgt_vector, dim=-1)  # (*, 3)
    assert torch.all(torch.gt(src_normal, eps)), "Zero vector found in src_vector."
    assert torch.all(torch.gt(tgt_normal, eps)), "Zero vector found in tgt_vector."

    # normalize
    src_vector = F.normalize(src_vector, p=2, dim=-1)  # (*, 3)
    tgt_vector = F.normalize(tgt_vector, p=2, dim=-1)  # (*, 3)

    # compute axis
    src_skew = skew_symmetric_matrix(src_vector)  # (*, 3, 3)
    phi = torch.matmul(src_skew, tgt_vector.unsqueeze(-1)).squeeze(-1)  # (*, 3)

    # handle src and tgt are in opposite directions: generate a random axis orthogonal to src/tgt vector.
    # To achieve this, we try (1, 0, 0) and (0, 1, 0) as pseudo tgt vectors and use compute the axis by cross-product.
    opposite = torch.lt((src_vector * tgt_vector).sum(dim=-1), 0.0)  # (*)
    for i in range(2):
        masks = (torch.linalg.norm(phi, dim=-1) < eps) & opposite  # (*)
        if not torch.any(masks):
            break
        aux_vector = torch.zeros_like(tgt_vector)  # (*, 3)
        aux_vector[..., i] = 1.0  # (1, 0, 0), (0, 1, 0)
        new_phi = torch.matmul(src_skew, aux_vector.unsqueeze(-1)).squeeze(-1)  # (*, 3)
        new_phi = F.normalize(new_phi, p=2, dim=-1) * np.pi  # set norm to pi
        phi = torch.where(masks, new_phi, phi)

    # compute rodrigues formula
    rotation = axis_angle_to_rotation_matrix(phi)

    return rotation


# Quaternion


def quaternion_product(x: Tensor, y: Tensor) -> Tensor:
    """Quaternion multiplication.

    z = x * y = (x0*y0 - x1*y1 - x2*y2 - x3*y3,
                 x1*y0 + x0*y1 + x2*y3 - x3*y2,
                 x2*y0 + x0*y2 + x3*y1 - x1*y3,
                 x3*y0 + x0*y3 + x1*y2 - x2*y1)

    Args:
        x (Tensor): The left quaterion in the shape of (*, 4).
        y (Tensor): The right quaterion in the shape of (*, 4).

    Returns:
        z (Tensor): The product quaterion of x and y in the shape of (*, 4).
    """
    xw, xx, xy, xz = torch.split(x, 1, dim=-1)  # (*, 1)
    yw, yx, yy, yz = torch.split(y, 1, dim=-1)  # (*, 1)
    zw = xw * yw - xx * yx - xy * yy - xz * yz
    zx = xx * yw + xw * yx + xy * yz - xz * yy
    zy = xy * yw + xw * yy + xz * yx - xx * yz
    zz = xz * yw + xw * yz + xx * yy - xy * yx
    z = torch.cat([zw, zx, zy, zz], dim=-1)  # (*, 4)
    return z


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Quaternion conjugate.

    z = q^(-1) = [qw, -qx, -qy, -qz]

    Args:
        q (Tensor): The input quaternion in the shape of (*, 4).

    Returns:
        y (Tensor): The conjugate quaternion in the shape of (*, 4).
    """
    qw, qx, qy, qz = torch.split(q, 1, dim=-1)  # (*, 1)
    conjugate = torch.cat([qw, -qx, -qy, -qz], dim=-1)  # (*, 4)
    return conjugate


def quaternion_rotate(q: Tensor, x: Tensor) -> Tensor:
    """Rotate point p using the rotation matrix of quaternion q.

    \hat{y} = q * \hat{x} * q^{-1}

    \hat{*} indicates pure quaternion.

    Args:
        q (Tensor): The quaternion of rotation in the shape of (*, 4).
        x (Tensor): The point to be rotated in the shape of (*, 3).

    Returns:
        y (Tensor): The rotated point in the shape of (*, 3).
    """
    q = F.normalize(q, p=2, dim=-1)
    quaternion_x = torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)
    conj_q = quaternion_conjugate(q)
    quaternion_y = quaternion_product(quaternion_product(q, quaternion_x), conj_q)
    y = quaternion_y[..., 1:]
    return y


# Conversions


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        q (Tensor): The quaternion of rotation in the shape of (*, 4).

    Returns:
        rotation (Tensor): The rotation matrix in the shape of (*, 3, 3).
    """
    q = F.normalize(q, p=2, dim=-1)
    qw, qx, qy, qz = torch.split(q, 1, dim=-1)  # (*, 1)
    r00 = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx ** 2 + qz ** 2)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
    rotation_shape = q.shape[:-1] + (3, 3)
    rotation = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).view(*rotation_shape)  # (*, 3, 3)
    return rotation


def rotation_matrix_to_quaternion(rotation: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert rotation matrix to quaternion.

    Use the algorithm from [Eigen](https://gitlab.com/libeigen/eigen).

    References:
        1. Quaternion Calculus and Fast Animation, Ken Shoemake, 1987 SIGGRAPH course notes

    Notes:
        1. There could be NaN and inf in the computation (sqrt and div), but the final results should be right.

    Args:
        rotation (Tensor): The rotation matrix in the shape of (*, 3, 3).
        eps (float=1e-6): A safe number.

    Returns:
        q (Tensor): The quaternion of rotation in the shape of (*, 4).
    """
    flatten_shape = rotation.shape[:-2] + (9,)
    rotation = rotation.view(*flatten_shape)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = torch.split(rotation, 1, dim=-1)  # (*, 1)

    # conditions
    trace = r00 + r11 + r22  # (*, 1)
    condition1 = trace > eps  # (*, 1)
    condition2 = (~condition1) & ((r00 - r11) > eps) & ((r00 - r22) > eps)  # (*, 1)
    condition3 = (~condition1) & ((r11 - r00) > eps) & ((r11 - r22) > eps)  # (*, 1)
    condition4 = (~condition1) & ((r22 - r00) > eps) & ((r22 - r11) > eps)  # (*, 1)

    # 1. trace > 0
    t = torch.sqrt(1.0 + trace)
    s = safe_divide(0.5, t)
    qw = 0.5 * t
    qx = (r21 - r12) * s
    qy = (r02 - r20) * s
    qz = (r10 - r01) * s
    q1 = torch.cat([qw, qx, qy, qz], dim=-1)  # (*, 4)

    # 2. trace <= 0, R00 > R11 and R00 > R22
    t = torch.sqrt(1.0 + r00 - r11 - r22)
    s = safe_divide(0.5, t)
    qw = (r21 - r12) * s
    qx = 0.5 * t
    qy = (r10 + r01) * s
    qz = (r02 + r20) * s
    q2 = torch.cat([qw, qx, qy, qz], dim=-1)  # (*, 4)

    # 3. trace <= 0, R11 > R00 and R11 > R22
    t = torch.sqrt(1.0 + r11 - r00 - r22)
    s = safe_divide(0.5, t)
    qw = (r02 - r20) * s
    qx = (r10 + r01) * s
    qy = 0.5 * t
    qz = (r21 + r12) * s
    q3 = torch.cat([qw, qx, qy, qz], dim=-1)  # (*, 4)

    # 4. trace <= 0, R22 > R00 and R22 > R11
    t = torch.sqrt(1.0 + r22 - r00 - r11)
    s = safe_divide(0.5, t)
    qw = (r10 - r01) * s
    qx = (r02 + r20) * s
    qy = (r21 + r12) * s
    qz = 0.5 * t
    q4 = torch.cat([qw, qx, qy, qz], dim=-1)  # (*, 4)

    # case differentiation
    q = q1
    q = torch.where(condition2, q2, q)
    q = torch.where(condition3, q3, q)
    q = torch.where(condition4, q4, q)

    return q


def quaternion_to_axis_angle(q: Tensor) -> Tensor:
    """Convert quaternion to axis angle.

    q: (cos(theta/2), sin(theta/2) * vx, sin(theta/2) * vy, sin(theta/2) * vz)

    Notes:
        We would like to force the angle in [0, pi]. If cos_half < 0, where theta in [pi, 2pi], we convert it
        to [-pi, 0] by atan2(-sin_half, -cos_half). This will invert the axis in phi.

    Args:
        q (Tensor): The quaternion of the rotation in the shape of (*, 4).

    Returns:
        phi (Tensor): The axis-angle representation of the rotation in the shape of (*, 3).
    """
    q = F.normalize(q, p=2, dim=-1)
    cos_half = q[..., :1]  # (*, 1)
    sin_half = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)  # (*, 1)
    theta = 2.0 * torch.where(cos_half < 0.0, torch.atan2(-sin_half, -cos_half), torch.atan2(sin_half, cos_half))
    omega = F.normalize(q[..., 1:], dim=-1)  # (*, 3)
    phi = omega * theta
    return phi


def axis_angle_to_quaternion(phi: Tensor) -> Tensor:
    """Convert axis angle to quaternion.

    q: (cos(theta/2), sin(theta/2) * vx, sin(theta/2) * vy, sin(theta/2) * vz)

    Args:
        phi (Tensor): The axis-angle representation of the rotation in the shape of (*, 3).

    Returns:
        q (Tensor): The quaternion of the rotation in the shape of (*, 4).
    """
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)  # (*, 1)
    omega = safe_divide(phi, theta)  # (*, 3)
    q = torch.cat([torch.cos(0.5 * theta), omega * torch.sin(0.5 * theta)], dim=-1)  # (*, 4)
    return q


def axis_angle_to_rotation_matrix(phi: Tensor) -> Tensor:
    """Convert axis angle to rotation matrix.

    A.k.a., exponential map on Lie group, which is implemented with Rodrigues' rotation formula.

    Note:
        If phi is a zero vector, the rotation matrix is an identity matrix.

    Args:
        phi (Tensor): The so(3) exponential coordinates in the shape of (*, 3).

    Returns:
        rotation (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).
    """
    theta = torch.linalg.norm(phi, dim=-1)
    omega = safe_divide(phi, theta.unsqueeze(-1))
    rotation = rodrigues_rotation_formula(omega, theta)
    return rotation


def rotation_matrix_to_axis_angle(rotation: Tensor) -> Tensor:
    """Convert rotation matrix to axis angle.

    A.k.a., logarithmic map or inverse Rodrigues' rotation formula on Lie group.

    Note:
        1. If rotation is an identity matrix, the phi is a zero vector.
        2. theta is in the range of [0, pi]

    Args:
        rotation (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).

    Returns:
        phi (Tensor): The so(3) exponential coordinates in the shape of (*, 3).
    """
    q = rotation_matrix_to_quaternion(rotation)
    phi = quaternion_to_axis_angle(q)
    return phi


def rotation_matrix_to_axis_angle_naive(rotation: Tensor, eps: float = 1e-5) -> Tensor:
    """Convert rotation matrix to axis angle.

    A.k.a., logarithmic map or inverse Rodrigues' rotation formula on Lie group.

    Note:
        1. If rotation is an identity matrix, the phi is a zero vector.
        2. theta is in the range of [0, pi]

    Args:
        rotation (Tensor): The SO(3) rotation matrix in the shape of (*, 3, 3).
        eps (float=1e-6): A safe number for division.

    Returns:
        phi (Tensor): The so(3) exponential coordinates in the shape of (*, 3).
    """
    phi_shape = rotation.shape[:-2] + (3,)

    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]  # (*)
    cos_theta = torch.clamp(0.5 * (trace - 1.0), min=-1.0, max=1.0)  # (*)
    theta = torch.acos(cos_theta)  # (*)

    # condition
    condition2 = torch.gt(cos_theta, 1.0 - eps)  # (*)
    condition3 = torch.lt(cos_theta, -1.0 + eps)  # (*)

    # 1. 0 < theta < pi
    r_x = rotation[..., 2, 1] - rotation[..., 1, 2]  # (*)
    r_y = rotation[..., 0, 2] - rotation[..., 2, 0]  # (*)
    r_z = rotation[..., 1, 0] - rotation[..., 0, 1]  # (*)
    omega = safe_divide(torch.stack([r_x, r_y, r_z], dim=-1), 2.0 * torch.sin(theta).unsqueeze(-1))  # (*, 3)
    phi_1 = theta * omega

    # 2. theta == 0, phi is a zero vector
    phi_2 = torch.zeros(size=phi_shape).to(rotation.device)

    # 3. theta == pi, rotation matrix -> unit quaternion -> axis-angle
    q = rotation_matrix_to_quaternion(rotation)
    phi_3 = theta * F.normalize(q[..., 1:], p=2, dim=-1)

    # case differentiation
    phi = phi_1
    phi = torch.where(condition2.unsqueeze(-1), phi_2, phi)
    phi = torch.where(condition3.unsqueeze(-1), phi_3, phi)

    return phi
