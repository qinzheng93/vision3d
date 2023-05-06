from typing import Optional

import numpy as np
from numpy import ndarray


def radius_nms(points: ndarray, scores: ndarray, radius: float, num_samples: Optional[int] = None) -> ndarray:
    sorted_indices = np.argsort(-scores)
    kept_indices = []
    while sorted_indices.size > 0:
        i = sorted_indices[0]
        kept_indices.append(i)
        if num_samples is not None:
            if len(kept_indices) >= num_samples:
                break
        if sorted_indices.size == 1:
            break
        distances = np.linalg.norm(points[i][None] - points[sorted_indices[1:]], axis=1)
        rest_indices = np.nonzero(distances > radius)[0]
        if rest_indices.size == 0:
            break
        sorted_indices = sorted_indices[rest_indices + 1]
    return np.asarray(kept_indices)
