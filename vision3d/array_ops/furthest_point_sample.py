from typing import Optional

import numpy as np
from numpy import ndarray

from vision3d.utils.misc import load_ext

ext_module = load_ext("vision3d.ext", ["sample_nodes_with_fps"])


def furthest_point_sample(
    points: ndarray, min_distance: Optional[float] = None, num_samples: Optional[int] = None
) -> ndarray:
    if min_distance is None:
        min_distance = -1.0
    if num_samples is None:
        num_samples = -1
    node_indices = ext_module.sample_nodes_with_fps(points, min_distance, num_samples)
    return node_indices


def native_furthest_point_sample(
    points: ndarray, min_distance: Optional[float] = None, num_samples: Optional[int] = None
) -> ndarray:
    assert min_distance is not None or num_samples is not None, "Both 'min_distance' and 'num_samples' are None."

    num_points = points.shape[0]
    selected_indices = []
    unselected_indices = list(range(num_points))
    distances = np.full(shape=(num_points,), fill_value=np.inf)

    best_index = 0
    while True:
        sel_index = unselected_indices[best_index]
        selected_indices.append(sel_index)
        unselected_indices.pop(best_index)

        best_index = -1
        best_distance = 0.0
        indices_to_remove = []
        for index, point_index in enumerate(unselected_indices):
            distance = np.linalg.norm(points[point_index] - points[sel_index])
            distance = min(distances[point_index], distance)
            distances[point_index] = distance
            if distance > best_distance:
                best_index = index
                best_distance = distance
            if distance < min_distance:
                indices_to_remove.append(index)

        if indices_to_remove:
            indices_to_remove = reversed(indices_to_remove)
            for index in indices_to_remove:
                unselected_indices.pop(index)
                if index < best_index:
                    best_index -= 1

        if min_distance is not None and best_distance < min_distance:
            break

        if num_samples is not None and len(selected_indices) >= num_samples:
            break

    sel_indices = np.asarray(selected_indices, dtype=np.int64)

    return sel_indices
