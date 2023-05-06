import numpy as np
from numpy import ndarray


def ravel_hash_func(data: ndarray, dimensions: ndarray) -> ndarray:
    assert data.shape[1] == dimensions.shape[0]
    hash_values = data[:, 0].copy()
    for i in range(1, dimensions.shape[0]):
        hash_values *= dimensions[i]
        hash_values += data[:, i]
    return hash_values


def grid_reduce(
    points: ndarray, voxel_size: float, *input_arrays: ndarray, method: str = "average", return_inverse: bool = False
):
    """Grid reduce a point cloud.

    We provide two reduction methods:

    1. "random": We randomly sample a point in each voxel. The whole point cloud is first shuffled and the first
                 occurrence of each voxel is selected.
    2. "average": We average all points and properties in each voxel.

    Args:
        points (array): The point cloud in shape of (N, 3).
        input_arrays (List[array]): The properties of the point cloud in shape of (N, *).
        voxel_size (float): The voxel size.
        method (str="random"): The method used for reduction. Available choices: ["random", "average"]
        return_inverse (bool=False): If True, return inv_indices.

    Returns:
        points (array): The sampled point cloud in shape of (M, 3)
        output_array (List[array]): The properties of the sampled point cloud in the same order as 'input_arrays'.
        inv_indices (array<int>): The mapping indices from original point cloud to sampled point cloud.
    """
    assert method in ["random", "average"]

    # 1. voxelize
    voxels = np.floor(points / voxel_size).astype(np.int32)
    voxels -= np.amin(voxels, axis=0, keepdims=True)
    dimensions = np.amax(voxels, axis=0) + 1
    hash_values = ravel_hash_func(voxels, dimensions)

    if method == "random":
        num_points = points.shape[0]
        # 2. random shuffle
        shuffled_indices = np.random.permutation(num_points)
        hash_values = hash_values[shuffled_indices]
        # 3. remove duplicates
        _, sel_indices, inv_indices = np.unique(hash_values, return_index=True, return_inverse=True)
        # 4. recover shuffle
        sel_indices = shuffled_indices[sel_indices]
        sampled_points = points[sel_indices]
        output_arrays = [input_array[sel_indices] for input_array in input_arrays]
        # 5. recover inverse indices
        true_inv_indices = np.empty_like(inv_indices)
        true_inv_indices[shuffled_indices] = inv_indices
        inv_indices = true_inv_indices
    else:
        # 2. remove duplicates
        _, inv_indices, unique_counts = np.unique(hash_values, return_inverse=True, return_counts=True)
        # 3. average reduction
        num_voxels = unique_counts.shape[0]
        sampled_points = np.zeros(shape=(num_voxels, 3), dtype=points.dtype)
        np.add.at(sampled_points, inv_indices, points)
        sampled_points /= unique_counts[:, None]
        output_arrays = []
        for input_array in input_arrays:
            output_shape = (num_voxels, input_array.shape[1])
            output_array = np.zeros(shape=output_shape, dtype=input_array.dtype)
            np.add.at(output_array, inv_indices, input_array)
            output_array /= unique_counts[:, None]
            output_arrays.append(output_array)

    output_list = [sampled_points] + output_arrays
    if return_inverse:
        output_list.append(inv_indices)

    return output_list
