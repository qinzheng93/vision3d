from numpy import ndarray

from .knn import knn


def knn_interpolate(
    q_points: ndarray,
    s_points: ndarray,
    s_feats: ndarray,
    k: int = 3,
    distance_limit: float = 0.1,
    eps: float = 1e-10,
    inf: float = 1e10,
) -> ndarray:
    distances, indices = knn(q_points, s_points, k=k, return_distance=True)  # (N, 3)
    if distance_limit is not None:
        distances[distances > distance_limit] = inf
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)  # (N, 3)
    knn_feats = s_feats[indices]  # (N, 3, C)
    q_feats = (knn_feats * weights[:, :, None]).sum(axis=1)
    return q_feats
