from typing import Tuple

import numpy as np
from numpy import ndarray

from .radius_nms import radius_nms


def random_sample_keypoints(
    points: ndarray,
    feats: ndarray,
    num_keypoints: int,
) -> Tuple[ndarray, ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_topk_keypoints_with_scores(
    points: ndarray,
    feats: ndarray,
    scores: ndarray,
    num_keypoints: int,
) -> Tuple[ndarray, ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(-scores)[:num_keypoints]
        points = points[indices]
        feats = feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(
    points: ndarray,
    feats: ndarray,
    scores: ndarray,
    num_keypoints: int,
) -> Tuple[ndarray, ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        probs = scores / np.sum(scores)
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_topk_keypoints_with_nms(
    points: ndarray,
    feats: ndarray,
    scores: ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[ndarray, ndarray]:
    if points.shape[0] > num_keypoints:
        sel_indices = radius_nms(points, scores, radius, num_samples=num_keypoints)
        points = points[sel_indices]
        feats = feats[sel_indices]
    return points, feats


def random_sample_keypoints_with_nms(
    points: ndarray,
    feats: ndarray,
    scores: ndarray,
    num_keypoints: int,
    radius: float,
) -> Tuple[ndarray, ndarray]:
    if points.shape[0] > num_keypoints:
        sel_indices = radius_nms(points, scores, radius)
        if sel_indices.size > num_keypoints:
            scores = scores[sel_indices]
            probs = scores / np.sum(scores)
            sel_indices = np.random.choice(sel_indices, num_keypoints, replace=False, p=probs)
        points = points[sel_indices]
        feats = feats[sel_indices]
    return points, feats
