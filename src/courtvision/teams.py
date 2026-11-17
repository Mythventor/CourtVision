"""Split player tracks into two teams by jersey color."""

from __future__ import annotations

import numpy as np


def torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Upper-middle part of a player box, where the jersey is."""
    x1, y1, x2, y2 = xyxy.astype(int)
    w, h = x2 - x1, y2 - y1
    return frame[y1 + int(0.2 * h) : y1 + int(0.55 * h), x1 + int(0.25 * w) : x2 - int(0.25 * w)]


def color_feature(crop: np.ndarray) -> np.ndarray:
    """Mean color of a crop in Lab space (more perceptually uniform than BGR)."""
    import cv2

    if crop.size == 0:
        return np.zeros(3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)


def cluster_teams(features_by_track: dict[int, list[np.ndarray]]) -> dict[int, int]:
    """Assign each track to cluster 0 or 1 from its color features.

    The user then says which cluster is which team (one thumbnail per cluster).
    """
    from sklearn.cluster import KMeans

    track_ids = [t for t, feats in features_by_track.items() if feats]
    if len(track_ids) < 2:
        return {t: 0 for t in track_ids}
    X = np.stack([np.median(np.stack(features_by_track[t]), axis=0) for t in track_ids])
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    return {t: int(label) for t, label in zip(track_ids, labels)}
