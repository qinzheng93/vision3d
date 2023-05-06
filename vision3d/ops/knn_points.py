from typing import Tuple, Union
import torch
from torch import Tensor

from vision3d.utils.misc import load_ext


ext_module = load_ext("vision3d.ext", ["knn_points"])


def knn_points(
    q_points: Tensor, s_points: Tensor, k: int, transposed: bool = False, return_distance: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Heap sort based kNN search for point cloud.

    Args:
        q_points (Tensor): The query points in shape of (B, N, 3) or (B, 3, N) if transposed.
        s_points (Tensor): The support points in shape of (B, M, 3) or (B, 3, M) if transposed.
        k (int): The number of neighbors.
        transposed (bool=False): If True, the points are in shape of (B, 3, N).
        return_distance (bool=False): If True, return the distances of the kNN.

    Returns:
        A Tensor of the distances of the kNN in shape of (B, N, k).
        A LongTensor of the indices of the kNN in shape of (B, N, k).
    """
    if transposed:
        q_points = q_points.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        s_points = s_points.transpose(1, 2)  # (B, M, 3) -> (B, 3, M)
    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    knn_distances = q_points.new_zeros(size=(q_points.shape[0], q_points.shape[1], k))  # (B, N, k)
    knn_indices = torch.zeros(size=(q_points.shape[0], q_points.shape[1], k), dtype=torch.long).cuda()  # (B, N, k)

    ext_module.knn(q_points, s_points, knn_distances, knn_indices, k)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices
