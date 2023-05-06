import torch
from torch import Tensor

from vision3d.utils.misc import load_ext

from .gather import gather


ext_module = load_ext("vision3d.ext", ["furthest_point_sample"])


def furthest_point_sample(
    points: Tensor, num_samples: int, *input_tensors: Tensor, gather_points: bool = True, transposed: bool = False
):
    """
    Furthest point sampling and gather the coordinates of the sampled points.

    Args:
        points (Tensor): The original points in shape of (B, N, 3).
        num_samples (int): The number of samples.
        input_tensors (List[Tensor]]): The features of the points (B, C, N).
        gather_points (bool=True): If True, gather points and features.
        transposed (bool=True): If True, the points shape is (B, 3, N).

    Returns:
        sampled_points (Tensor): The sampled points in shape of (B, M, 3) or (B, 3, M) if transposed.
        indices (LongTensor): The indices of the sample points in shape of (B, M).
        output_tensors (List[Tensor]): The list of sampled features of in shape of (B, C, M).
    """
    if transposed:
        points = points.transpose(1, 2)  # (B, 3, N) -> (B, N, 3)
    points = points.contiguous()

    distances = torch.full(size=(points.shape[0], points.shape[1]), fill_value=1e10).cuda()  # (B, N)
    indices = torch.zeros(size=(points.shape[0], num_samples), dtype=torch.long).cuda()  # (B, M)

    ext_module.furthest_point_sample(points, distances, indices, num_samples)

    if not gather_points:
        return indices

    points = points.transpose(1, 2).contiguous()  # (B, N, 3) -> (B, 3, N)
    sampled_points = gather(points, indices)  # (B, 3, M)
    if not transposed:
        sampled_points = sampled_points.transpose(1, 2).contiguous()  # (B, 3, M) -> (B, M, 3)

    output_tensors = [gather(input_tensor, indices) for input_tensor in input_tensors]

    return sampled_points, *output_tensors
