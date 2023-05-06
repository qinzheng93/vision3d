import ipdb
import torch
from torch import Tensor


def sample_pdf(
    locations: Tensor, weights: Tensor, num_samples: int, deterministic: bool = False, eps: float = 1e-5
) -> Tensor:
    """Sampling from probabilistic distribution function from the bins.

    The bins are represented by the locations x_i: { [x_0, x_1], [x_1, x_2], ..., [x_{B-1}, x_B] }.

    Args:
        locations (tensor): The boundary points between the bins in the shape of (N, B+1).
        weights (tensor): The weights of each bin in the shape of (N, B).
        num_samples (int): The number of samples.
        deterministic (bool): If True, generate deterministic sampling results. Default: False.
        eps (float): A safe number for division. Default: 1e-5.

    Returns:
        A float tensor of the sampled points from the bins.
    """
    num_rays, num_bins = weights.shape

    # compute probabilistic distribution function
    weights = weights + eps  # prevent nans, prevent division by zero (don't do inplace op!)
    pdf_values = weights / torch.sum(weights, -1, keepdim=True)
    cdf_values = torch.cumsum(pdf_values, dim=-1)  # (N, B)
    cdf_values = torch.cat([torch.zeros_like(cdf_values[..., :1]), cdf_values], dim=-1)  # padded to [0, 1], (N, B+1)

    # create sample values
    if deterministic:
        # generate deterministic samples
        t_values = torch.linspace(0.0, 1.0, steps=num_samples).cuda()
        t_values = t_values.expand(num_rays, num_samples)
    else:
        # generate random samples
        t_values = torch.rand(size=(num_rays, num_samples)).cuda()
    t_values = t_values.contiguous()

    # CDF sampling
    indices = torch.searchsorted(cdf_values.detach(), t_values, right=True)  # (N, M)
    lower_bounds = torch.max(torch.zeros_like(indices - 1), indices - 1)  # (N, M)
    upper_bounds = torch.min(num_bins * torch.ones_like(indices), indices)  # (N, M)
    bound_indices = torch.stack([lower_bounds, upper_bounds], dim=-1)  # (N, M, 2)

    bound_cdf_values = torch.gather(cdf_values.unsqueeze(1).expand(-1, num_samples, -1), 2, bound_indices)  # (N, M, 2)
    bound_locations = torch.gather(locations.unsqueeze(1).expand(-1, num_samples, -1), 2, bound_indices)  # (N, M, 2)

    # denominator equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)
    denominators = bound_cdf_values[..., 1] - bound_cdf_values[..., 0]  # (N, M)
    denominators = torch.where(torch.lt(denominators, eps), torch.ones_like(denominators), denominators)

    t_values = (t_values - bound_cdf_values[..., 0]) / denominators
    samples = bound_locations[..., 0] + t_values * (bound_locations[..., 1] - bound_locations[..., 0])

    return samples
