from vision3d.utils.misc import load_ext


ext_module = load_ext("vision3d.ext", ["grid_subsampling"])


def grid_subsample_pack_mode(points, lengths, voxel_size):
    """Grid subsampling in pack mode.

    This function is implemented on CPU.

    Args:
        points (array): points in pack mode. (N, 3)
        lengths (array): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (array): sampled points in pack mode (M, 3)
        s_lengths (array): numbers of sampled points in the batch. (B,)
    """
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths
