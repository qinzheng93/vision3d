import torch
import torch.nn.functional as F
from torch import Tensor


def leading_eigenvector(mat: Tensor, method: str = "power", num_iterations: int = 10) -> Tensor:
    """Compute the leading eigenvector of the matrix.

    Source: https://github.com/XuyangBai/PointDSC/blob/master/models/PointDSC.py

    Args:
        mat (Tensor): the input matrix in the shape of (*, M, M).
        method (str): the method to use: "power", "torch". Default: "power".
        num_iterations (int): the number of iterations for "power" method. Default: 10.

    Returns:
        A Tensor of the leading eigenvector.
    """
    assert method in ["power", "torch"], f"Unsupported method: {method}."

    if method == "power":
        # power iteration algorithm
        leading_eig = torch.ones_like(mat[..., 0:1])  # (*, M, 1)
        leading_eig_last = leading_eig
        for i in range(num_iterations):
            leading_eig = torch.matmul(mat, leading_eig)
            leading_eig = F.normalize(leading_eig, p=2, dim=-2)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
    else:
        _, eigenvectors = torch.symeig(mat, eigenvectors=True)
        leading_eig = eigenvectors[:, :, -1]

    return leading_eig
