"""Court geometry: map image points to court coordinates and draw the 2D court panel.

Court coordinates are in feet, origin at a corner, x along the length, y across the width.
The camera pans and zooms with the ball, so the image -> court homography is estimated
per frame from court keypoints detected by a Roboflow keypoint model.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# NBA-style markings; only used for drawing the panel.
HOOP_FROM_BASELINE = 5.25
THREE_RADIUS = 23.75
CORNER_THREE_Y = 3.0   # distance of the corner three line from the sideline
LANE_WIDTH = 16.0
FT_LINE = 19.0         # baseline to free-throw line
CIRCLE_RADIUS = 6.0


@dataclass
class CourtSpec:
    length: float = 94.0
    width: float = 50.0

    @classmethod
    def from_config(cls, court_cfg: dict[str, Any]) -> "CourtSpec":
        return cls(court_cfg.get("length_ft", 94.0), court_cfg.get("width_ft", 50.0))


def homography_from_keypoints(
    image_xy: np.ndarray,
    confidence: np.ndarray,
    template: list,
    min_confidence: float = 0.5,
) -> np.ndarray | None:
    """Image -> court homography from detected keypoints, or None if under 4 usable points.

    `template[i]` is the court position [x_ft, y_ft] of keypoint i, or None to ignore it.
    """
    import cv2

    src, dst = [], []
    for i, court_xy in enumerate(template[: len(image_xy)]):
        if court_xy is None or confidence[i] < min_confidence:
            continue
        src.append(image_xy[i])
        dst.append(court_xy)
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 1.0)
    return H


def to_court(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map (N, 2) image points to court feet with homography H."""
    import cv2

    if len(points) == 0:
        return np.zeros((0, 2))
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


class CourtMapper:
    """Per-frame homography from the court keypoint model; reuses the last good one."""

    def __init__(self, court_cfg: dict[str, Any], model=None):
        self.template = court_cfg.get("template") or []
        self.enabled = sum(p is not None for p in self.template) >= 4
        self.confidence = court_cfg.get("confidence", 0.3)
        self.min_kp_conf = court_cfg.get("keypoint_confidence", 0.5)
        self.model = model
        if self.enabled and self.model is None:
            self.model = load_keypoint_model(court_cfg)
        self.last_H: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        xy, conf = detect_keypoints(self.model, frame, self.confidence)
        if xy is not None:
            H = homography_from_keypoints(xy, conf, self.template, self.min_kp_conf)
            if H is not None:
                self.last_H = H
        return self.last_H


def load_keypoint_model(court_cfg: dict[str, Any]):
    from .detection import prepare_gpu

    prepare_gpu()
    from inference import get_model

    return get_model(model_id=court_cfg["model_id"], api_key=os.environ.get("ROBOFLOW_API_KEY"))


def detect_keypoints(model, frame: np.ndarray, confidence: float):
    """(K, 2) keypoint positions and (K,) confidences of the best court detection."""
    import supervision as sv

    result = model.infer(frame, confidence=confidence)[0]
    kps = sv.KeyPoints.from_inference(result)
    if len(kps) == 0:
        return None, None
    # supervision >= 0.29 renamed `confidence` to `keypoint_confidence`
    conf = getattr(kps, "keypoint_confidence", None)
    if conf is None:
        conf = kps.confidence
    best = int(np.argmax(conf.mean(axis=1)))
    return kps.xy[best], conf[best]


# --- drawing -----------------------------------------------------------------

class CourtCanvas:
    """Maps court feet to panel pixels and draws the court."""

    def __init__(self, spec: CourtSpec, panel_width: int, margin: int = 16):
        self.spec = spec
        self.margin = margin
        self.scale = (panel_width - 2 * margin) / spec.length
        self.width = panel_width
        self.height = int(round(spec.width * self.scale + 2 * margin))

    def px(self, x_ft: float, y_ft: float) -> tuple[int, int]:
        return (int(round(self.margin + x_ft * self.scale)),
                int(round(self.margin + y_ft * self.scale)))

    def inside(self, x_ft: float, y_ft: float, slack: float = 3.0) -> bool:
        return (-slack <= x_ft <= self.spec.length + slack
                and -slack <= y_ft <= self.spec.width + slack)

    def draw(self) -> np.ndarray:
        import cv2

        s, L, W = self.scale, self.spec.length, self.spec.width
        img = np.full((self.height, self.width, 3), (60, 110, 165), np.uint8)  # wood, BGR
        line, t = (245, 245, 245), 2
        cv2.rectangle(img, self.px(0, 0), self.px(L, W), line, t)
        cv2.line(img, self.px(L / 2, 0), self.px(L / 2, W), line, t)
        cv2.circle(img, self.px(L / 2, W / 2), int(CIRCLE_RADIUS * s), line, t)

        arc_x = HOOP_FROM_BASELINE + math.sqrt(THREE_RADIUS**2 - (W / 2 - CORNER_THREE_Y) ** 2)
        half_angle = math.degrees(math.acos((arc_x - HOOP_FROM_BASELINE) / THREE_RADIUS))
        for side in (0, 1):
            def X(x):  # mirror for the right half
                return L - x if side else x

            hoop = self.px(X(HOOP_FROM_BASELINE), W / 2)
            cv2.circle(img, hoop, max(2, int(0.75 * s)), (0, 90, 255), -1)
            lane_top, lane_bot = W / 2 - LANE_WIDTH / 2, W / 2 + LANE_WIDTH / 2
            cv2.rectangle(img, self.px(X(0), lane_top), self.px(X(FT_LINE), lane_bot), line, t)
            cv2.circle(img, self.px(X(FT_LINE), W / 2), int(CIRCLE_RADIUS * s), line, t)
            for y in (CORNER_THREE_Y, W - CORNER_THREE_Y):
                cv2.line(img, self.px(X(0), y), self.px(X(arc_x), y), line, t)
            start = 180 - half_angle if side else -half_angle
            cv2.ellipse(img, hoop, (int(THREE_RADIUS * s),) * 2, 0, start, start + 2 * half_angle,
                        line, t)
        return img
