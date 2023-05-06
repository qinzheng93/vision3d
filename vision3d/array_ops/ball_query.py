from typing import List

from numpy import ndarray
from scipy.spatial.ckdtree import cKDTree


def ball_query(q_points: ndarray, s_points: ndarray, radius: float) -> List[ndarray]:
    s_tree = cKDTree(s_points)
    indices_list = s_tree.query_ball_point(q_points, radius)
    return indices_list
