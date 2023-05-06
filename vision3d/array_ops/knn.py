from numpy import ndarray
from scipy.spatial import cKDTree


def knn(q_points: ndarray, s_points: ndarray, k=1, return_distance: bool = False, n_jobs=-1):
    """Compute the nearest neighbor for the query points in support points.

    Note:
        If k=1, the return arrays are squeezed.
    """
    s_tree = cKDTree(s_points)
    knn_distances, knn_indices = s_tree.query(q_points, k=k, n_jobs=n_jobs)
    if return_distance:
        return knn_distances, knn_indices
    else:
        return knn_indices
