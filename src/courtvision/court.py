"""Court geometry: map image points to court coordinates and draw the 2D court panel.

Court coordinates are in feet: origin at the far-left corner as seen from the camera,
x along the length (left baseline -> right baseline), y across (far sideline -> near).

The image -> court mapping (a homography) comes either from a manual calibration
(calibration.py: the user clicks known court points, and camera pans are followed by
matching the background between frames) or from the Roboflow court keypoint model.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


@dataclass
class CourtSpec:
    """Court size and markings, in feet."""

    length: float = 94.0
    width: float = 50.0
    lane_width: float = 16.0
    ft_line: float = 19.0             # baseline to free-throw line
    hoop_from_baseline: float = 5.25  # baseline to the center of the rim
    three_radius: float = 23.75
    corner_three_y: float = 3.0       # sideline to the straight corner-three line
    circle_radius: float = 6.0

    @classmethod
    def preset(cls, name: str) -> "CourtSpec":
        try:
            return replace(PRESETS[name])
        except KeyError:
            raise ValueError(f"Unknown court standard {name!r}; use one of {sorted(PRESETS)}")

    @classmethod
    def from_config(cls, court_cfg: dict[str, Any]) -> "CourtSpec":
        """`court.standard` preset, with `length_ft` / `width_ft` overriding when set."""
        spec = cls.preset(court_cfg.get("standard", "nba"))
        if court_cfg.get("length_ft"):
            spec.length = float(court_cfg["length_ft"])
        if court_cfg.get("width_ft"):
            spec.width = float(court_cfg["width_ft"])
        return spec

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


PRESETS = {
    "nba": CourtSpec(),
    # NFHS: 84 x 50 ft, 12 ft lane, 19 ft 9 in three-point arc
    "high_school": CourtSpec(length=84.0, width=50.0, lane_width=12.0, three_radius=19.75,
                             corner_three_y=5.25),
}


def landmarks(spec: CourtSpec) -> dict[str, tuple[float, float]]:
    """Named court points for manual calibration, as seen from the camera."""
    L, W = spec.length, spec.width
    far_lane, near_lane = W / 2 - spec.lane_width / 2, W / 2 + spec.lane_width / 2
    points: dict[str, tuple[float, float]] = {}
    for side in ("Left", "Right"):
        def X(x):
            return L - x if side == "Right" else x

        points.update({
            f"{side} baseline x far sideline (corner)": (X(0), 0),
            f"{side} baseline x near sideline (corner)": (X(0), W),
            f"{side} lane: baseline corner, far side": (X(0), far_lane),
            f"{side} lane: baseline corner, near side": (X(0), near_lane),
            f"{side} free-throw line: far end": (X(spec.ft_line), far_lane),
            f"{side} free-throw line: near end": (X(spec.ft_line), near_lane),
            f"{side} free-throw line: center": (X(spec.ft_line), W / 2),
            f"{side} three-point arc: top": (X(spec.hoop_from_baseline + spec.three_radius), W / 2),
        })
    points.update({
        "Half-court line x far sideline": (L / 2, 0),
        "Half-court line x near sideline": (L / 2, W),
        "Center circle: center": (L / 2, W / 2),
        "Center circle x half-court line, far side": (L / 2, W / 2 - spec.circle_radius),
        "Center circle x half-court line, near side": (L / 2, W / 2 + spec.circle_radius),
    })
    return points


def homography_from_points(image_xy, court_xy) -> np.ndarray | None:
    """Image -> court homography from 4+ point pairs (least squares), or None."""
    import cv2

    image_xy, court_xy = np.float32(image_xy), np.float32(court_xy)
    if len(image_xy) < 4:
        return None
    H, _ = cv2.findHomography(image_xy, court_xy, 0)
    return H


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

def court_polylines(spec: CourtSpec, step_deg: float = 5.0) -> list[np.ndarray]:
    """Court markings as polylines in court feet (outline, half line, circles, lanes, 3s)."""
    L, W = spec.length, spec.width

    def arc(cx, cy, r, a0, a1):
        a = np.radians(np.arange(a0, a1 + step_deg / 2, step_deg))
        return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])

    hoop, R, c = spec.hoop_from_baseline, spec.three_radius, spec.corner_three_y
    lines = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
        np.array([[L / 2, 0], [L / 2, W]], float),
        arc(L / 2, W / 2, spec.circle_radius, 0, 360),
    ]
    arc_x = hoop + math.sqrt(max(R**2 - (W / 2 - c) ** 2, 0.0))
    half = math.degrees(math.acos(min((arc_x - hoop) / R, 1.0)))
    lane_top, lane_bot = W / 2 - spec.lane_width / 2, W / 2 + spec.lane_width / 2
    for side in (0, 1):
        def X(pts):  # mirror the left-half markings for the right half
            pts = np.asarray(pts, float)
            if side:
                pts = pts.copy()
                pts[:, 0] = L - pts[:, 0]
            return pts

        lines += [
            X([[0, lane_top], [spec.ft_line, lane_top], [spec.ft_line, lane_bot], [0, lane_bot]]),
            X(arc(spec.ft_line, W / 2, spec.circle_radius, 0, 360)),
            X([[0, c], [arc_x, c]]),
            X([[0, W - c], [arc_x, W - c]]),
            X(arc(hoop, W / 2, R, -half, half)),
        ]
    return lines


def draw_court_on_image(img: np.ndarray, H: np.ndarray, spec: CourtSpec,
                        color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    """Draw the court markings into a camera image using image -> court homography H.

    If the drawn lines sit on the real court lines, the court mapping is right.
    """
    import cv2

    Hinv = np.linalg.inv(H)
    h, w = img.shape[:2]
    for line in court_polylines(spec, step_deg=2.0):
        # densify straight segments so they bend correctly under perspective
        dense = []
        for a, b in zip(line[:-1], line[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.5))
            dense.append(np.linspace(a, b, n, endpoint=False))
        dense.append(line[-1:])
        pts = np.vstack(dense)
        p = np.column_stack([pts, np.ones(len(pts))]) @ Hinv.T
        ok = p[:, 2] > 1e-9   # in front of the camera
        xy = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
        ok &= (np.abs(xy[:, 0]) < 4 * w) & (np.abs(xy[:, 1]) < 4 * h)
        # draw each run of consecutive valid points
        start = None
        for i in range(len(xy) + 1):
            if i < len(xy) and ok[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= 2:
                    cv2.polylines(img, [xy[start:i].astype(np.int32)], False, color, thickness,
                                  cv2.LINE_AA)
                start = None
    return img


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

        img = np.full((self.height, self.width, 3), (60, 110, 165), np.uint8)  # wood, BGR
        for line in court_polylines(self.spec):
            pts = np.array([self.px(x, y) for x, y in line], np.int32)
            cv2.polylines(img, [pts], False, (245, 245, 245), 2, cv2.LINE_AA)
        for x in (self.spec.hoop_from_baseline, self.spec.length - self.spec.hoop_from_baseline):
            cv2.circle(img, self.px(x, self.spec.width / 2), max(2, int(0.75 * self.scale)),
                       (0, 90, 255), -1)
        return img
