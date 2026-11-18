"""Split player tracks into two teams by jersey color."""

from __future__ import annotations

import numpy as np


def torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Upper-middle part of a player box, where the jersey is."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = np.clip(xyxy, 0, [w_img, h_img, w_img, h_img]).astype(int)
    w, h = x2 - x1, y2 - y1
    return frame[y1 + int(0.2 * h) : y1 + int(0.55 * h), x1 + int(0.25 * w) : x2 - int(0.25 * w)]


def color_feature(crop: np.ndarray) -> np.ndarray | None:
    """Mean color of a crop in Lab space (more perceptually uniform than BGR)."""
    import cv2

    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)


def cluster_teams(
    features_by_track: dict[int, list[np.ndarray]]
) -> tuple[dict[int, int], np.ndarray]:
    """Assign each track to color cluster 0 or 1.

    Returns (track id -> cluster, (2, 3) cluster centers in Lab). The user then says
    which cluster is which team.
    """
    track_ids = [t for t, feats in features_by_track.items() if feats]
    if not track_ids:
        return {}, np.zeros((2, 3))
    X = np.stack([np.median(np.stack(features_by_track[t]), axis=0) for t in track_ids])
    if len(track_ids) < 2:
        return {track_ids[0]: 0}, np.stack([X[0], X[0]])

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    labels = {t: int(label) for t, label in zip(track_ids, km.labels_)}
    return labels, km.cluster_centers_


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """Convert one Lab color (OpenCV 0-255 scale) to a BGR tuple for drawing."""
    import cv2

    px = np.clip(np.asarray(lab, dtype=float), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)
