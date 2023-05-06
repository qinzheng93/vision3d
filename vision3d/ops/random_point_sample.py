import torch
from torch import Tensor

from .gather import gather


def random_point_sample(points: Tensor, num_samples: int, gather_points: bool = True):
    batch_size, _, num_points = points.shape
    weights = torch.ones(size=(batch_size, num_points)).cuda()
    indices = torch.multinomial(weights, num_samples)
    if gather_points:
        samples = gather(points, indices)
        return samples
    else:
        return indices
