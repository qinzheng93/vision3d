import torch


def ravel_hash_func(voxels, dimensions):
    dimension = voxels.shape[1]
    hash_values = voxels[:, 0].clone()
    for i in range(1, dimension):
        hash_values *= dimensions[i]
        hash_values += voxels[:, i]
    return hash_values


def naive_grid_subsample_pack_mode(points, lengths, voxel_size):
    """Grid subsample in pack mode (naive version).

    Args:
        points (Tensor): the original points (N, 3).
        lengths (LongTensor): the numbers of points in the batch (B,).
        voxel_size (float): the voxel size.

    Returns:
        sampled_points (Tensor): the subsampled points (M, 3).
        sampled_lengths (Tensor): the numbers of subsampled points in the batch (B,).
    """
    inv_voxel_size = 1.0 / voxel_size
    batch_size = lengths.shape[0]
    start_index = 0
    sampled_points_list = []
    for i in range(batch_size):
        cur_length = lengths[i]
        end_index = start_index + cur_length
        cur_points = points[start_index:end_index]
        voxels = torch.floor(cur_points * inv_voxel_size).long()
        voxels -= voxels.amin(0, keepdim=True)
        dimensions = voxels.amax(0) + 1
        hash_values = ravel_hash_func(voxels, dimensions)
        _, inv_indices, unique_counts = torch.unique(hash_values, return_inverse=True, return_counts=True)
        inv_indices = inv_indices.unsqueeze(1).expand(-1, 3)
        cur_sampled_points = torch.zeros(unique_counts.shape[0], 3).to(cur_points.device)
        cur_sampled_points.scatter_add_(0, inv_indices, cur_points)
        cur_sampled_points /= unique_counts.unsqueeze(1).float()
        sampled_points_list.append(cur_sampled_points)
        start_index = end_index
    sampled_points = torch.cat(sampled_points_list, dim=0)
    sampled_lengths = torch.LongTensor([x.shape[0] for x in sampled_points_list]).to(lengths.device)
    return sampled_points, sampled_lengths


def grid_subsample_pack_mode(points, lengths, voxel_size):
    """Grid subsample in pack mode (fast version).

    Args:
        points (Tensor): the original points (N, 3).
        lengths (LongTensor): the numbers of points in the batch (B,).
        voxel_size (float): the voxel size.

    Returns:
        sampled_points (Tensor): the subsampled points (M, 3).
        sampled_lengths (Tensor): the numbers of subsampled points in the batch (B,).
    """
    batch_size = lengths.shape[0]

    # voxelize
    inv_voxel_size = 1.0 / voxel_size
    voxels = torch.floor(points * inv_voxel_size).long()

    # normalize, pad with batch indices
    start_index = 0
    voxels_list = []
    for i in range(batch_size):
        cur_length = lengths[i].item()
        end_index = start_index + cur_length
        cur_voxels = voxels[start_index:end_index]  # (L, 3)
        cur_voxels -= cur_voxels.amin(0, keepdim=True)  # (L, 3)
        cur_voxels = torch.cat([torch.full_like(cur_voxels[:, :1], fill_value=i), cur_voxels], dim=1)  # (L, 4)
        voxels_list.append(cur_voxels)
        start_index = end_index
    voxels = torch.cat(voxels_list, dim=0)  # (N, 4)

    # scatter
    dimensions = voxels.amax(0) + 1  # (4)
    hash_values = ravel_hash_func(voxels, dimensions)  # (N)
    unique_values, inv_indices, unique_counts = torch.unique(
        hash_values, return_inverse=True, return_counts=True
    )  # (M) (N) (M)
    inv_indices = inv_indices.unsqueeze(1).expand(-1, 3)  # (N, 3)
    s_points = torch.zeros(size=(unique_counts.shape[0], 3)).cuda()  # (M, 3)
    s_points.scatter_add_(0, inv_indices, points)  # (M, 3)
    s_points /= unique_counts.unsqueeze(1).float()  # (M, 3)

    # compute lengths
    total_dimension = torch.cumprod(dimensions, dim=0)[-1] / dimensions[0]
    s_batch_indices = torch.div(unique_values, total_dimension, rounding_mode="floor").long()
    s_lengths = torch.bincount(s_batch_indices, minlength=batch_size)
    assert (
        s_lengths.shape[0] == batch_size
    ), f"Invalid length of s_lengths ({batch_size} expected, {s_lengths.shape} got)."

    return s_points, s_lengths
