"""Manual court calibration that follows camera pans.

1. Calibrate once per video: on one frame, the user clicks 4+ named court landmarks
   (calibration points). That gives the reference image -> court homography.
2. During processing, the camera (an XbotGo on a fixed mount) pans and zooms. A camera
   that rotates in place moves the image by a homography, so each frame is matched to
   the reference image with ORB features on the static background (walls, floor, lines)
   and RANSAC. Frames are chained to the previous analyzed frame, and periodically matched
   directly to the reference to limit drift.

Points are stored in normalized image coordinates (0-1), so a calibration made on the
4K original works on the 720p working copy too.

Files: <output_dir>/calibration/<video name>.json and .jpg (the reference frame).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .court import CourtSpec, homography_from_points

# Regions that move with the camera image, not the scene, as (x0, y0, x1, y1) fractions:
# the burned-in clock (bottom left) and the XbotGo logo (bottom right).
DEFAULT_MASK = [[0.0, 0.88, 0.22, 1.0], [0.78, 0.88, 1.0, 1.0]]


@dataclass
class CalibrationPoint:
    name: str
    u: float          # normalized image x (0-1)
    v: float          # normalized image y (0-1)
    x_ft: float
    y_ft: float


@dataclass
class Calibration:
    video: str                     # source video file name
    t: float                       # time (s) of the reference frame in the video
    standard: str                  # court preset, e.g. "high_school"
    spec: dict[str, float]         # court dimensions used for the points
    points: list[CalibrationPoint] = field(default_factory=list)

    @property
    def court_spec(self) -> CourtSpec:
        return CourtSpec(**self.spec)

    def homography(self) -> np.ndarray | None:
        """Normalized reference-image coordinates -> court feet."""
        if len(self.points) < 4:
            return None
        return homography_from_points([(p.u, p.v) for p in self.points],
                                      [(p.x_ft, p.y_ft) for p in self.points])

    def error_ft(self) -> float | None:
        """Mean distance (ft) between clicked points mapped to the court and the landmarks.

        Only meaningful with 5+ points (4 points always fit exactly).
        """
        H = self.homography()
        if H is None or len(self.points) < 5:
            return None
        from .court import to_court

        mapped = to_court(np.array([(p.u, p.v) for p in self.points]), H)
        target = np.array([(p.x_ft, p.y_ft) for p in self.points])
        return float(np.linalg.norm(mapped - target, axis=1).mean())

    # --- files ---------------------------------------------------------------

    @staticmethod
    def paths(output_dir: str | Path, video: str | Path) -> tuple[Path, Path]:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video).stem).strip("_")
        folder = Path(output_dir) / "calibration"
        return folder / f"{stem}.json", folder / f"{stem}.jpg"

    def save(self, output_dir: str | Path, reference_bgr: np.ndarray) -> Path:
        import cv2

        json_path, img_path = self.paths(output_dir, self.video)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), reference_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        json_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return json_path

    @classmethod
    def load(cls, output_dir: str | Path, video: str | Path) -> "Calibration | None":
        json_path, _ = cls.paths(output_dir, video)
        if not json_path.exists():
            return None
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["points"] = [CalibrationPoint(**p) for p in data["points"]]
        return cls(**data)

    @classmethod
    def reference_image(cls, output_dir: str | Path, video: str | Path) -> np.ndarray | None:
        import cv2

        _, img_path = cls.paths(output_dir, video)
        return cv2.imread(str(img_path)) if img_path.exists() else None


class CameraTracker:
    """Estimates the homography from each frame to the reference frame as the camera pans."""

    def __init__(self, reference_bgr: np.ndarray, work_width: int = 960,
                 mask: list[list[float]] | None = None, min_inliers: int = 40,
                 reanchor_every: int = 10, features: int = 3000):
        import cv2

        self.work_width = work_width
        self.mask_rects = DEFAULT_MASK if mask is None else mask
        self.min_inliers = min_inliers
        self.reanchor_every = reanchor_every
        self.orb = cv2.ORB_create(nfeatures=features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_size = reference_bgr.shape[1], reference_bgr.shape[0]
        self.ref = self._features(reference_bgr)
        self.prev = None
        self.M = None            # current small-frame -> small-reference homography
        self.steps = 0
        self.lost = 0            # frames where no mapping could be found

    def _small(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        scale = self.work_width / w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (self.work_width, int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _features(self, frame: np.ndarray):
        small = self._small(frame)
        h, w = small.shape
        mask = np.full((h, w), 255, np.uint8)
        for x0, y0, x1, y1 in self.mask_rects:
            mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        kps, desc = self.orb.detectAndCompute(small, mask)
        return small.shape, kps, desc

    def _match(self, a, b) -> np.ndarray | None:
        """Homography mapping image a -> image b, or None if it is not reliable."""
        import cv2

        _, kps_a, desc_a = a
        _, kps_b, desc_b = b
        if desc_a is None or desc_b is None or len(kps_a) < 8 or len(kps_b) < 8:
            return None
        pairs = self.matcher.knnMatch(desc_a, desc_b, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([kps_a[m.queryIdx].pt for m in good])
        dst = np.float32([kps_b[m.trainIdx].pt for m in good])
        M, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if M is None or int(inliers.sum()) < self.min_inliers:
            return None
        det = np.linalg.det(M[:2, :2])
        if not 0.33 < det < 3.0:   # implausible zoom: reject
            return None
        return M

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Small-frame -> small-reference homography for this frame (None if lost)."""
        cur = self._features(frame)
        M = None
        if self.prev is not None and self.M is not None:
            step = self._match(cur, self.prev)        # this frame -> previous frame
            if step is not None:
                M = self.M @ step
        if M is None or self.steps % self.reanchor_every == 0:
            direct = self._match(cur, self.ref)       # this frame -> reference
            if direct is not None:
                M = direct
        if M is None:
            self.lost += 1
        self.prev, self.M = cur, M
        self.steps += 1
        return M

    def frame_to_normalized(self, frame_shape: tuple[int, ...], M: np.ndarray) -> np.ndarray:
        """Full-frame pixels -> normalized reference coordinates, given the small-frame M."""
        h, w = frame_shape[:2]
        s = self.work_width / w
        S = np.diag([s, s, 1.0])
        ref_h = self.ref[0][0]
        N = np.diag([1.0 / self.work_width, 1.0 / ref_h, 1.0])
        return N @ M @ S


class CalibratedCourtMapper:
    """Court mapper for the pipeline: frame -> court homography from a calibration."""

    def __init__(self, calibration: Calibration, reference_bgr: np.ndarray,
                 court_cfg: dict[str, Any] | None = None):
        court_cfg = court_cfg or {}
        self.H_norm = calibration.homography()
        if self.H_norm is None:
            raise ValueError("Calibration needs at least 4 points")
        self.tracker = CameraTracker(
            reference_bgr,
            work_width=court_cfg.get("track_width", 960),
            mask=court_cfg.get("track_mask"),
            min_inliers=court_cfg.get("track_min_inliers", 40),
            reanchor_every=court_cfg.get("track_reanchor_every", 10),
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        M = self.tracker.update(frame)
        if M is None:
            return None
        return self.H_norm @ self.tracker.frame_to_normalized(frame.shape, M)


class CalibrationSession:
    """State behind the app's court setup panel (one video at a time)."""

    def __init__(self, video: str | Path, frame_bgr: np.ndarray, t: float, standard: str):
        self.video = Path(video)
        self.frame = frame_bgr
        self.t = float(t)
        self.standard = standard
        self.points: list[tuple[str, float, float]] = []   # (landmark, u, v)

    @classmethod
    def start(cls, video: str | Path, t: float, standard: str, output_dir: str | Path,
              width: int = 1280) -> tuple["CalibrationSession", bool]:
        """Resume the saved calibration for `video` if there is one, else load frame `t`.

        Returns (session, resumed).
        """
        import cv2

        saved = Calibration.load(output_dir, video)
        ref = Calibration.reference_image(output_dir, video)
        if saved is not None and ref is not None:
            session = cls(video, ref, saved.t, saved.standard)
            session.points = [(p.name, p.u, p.v) for p in saved.points]
            return session, True
        from .preview import read_frame

        frame = read_frame(str(video), t)
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        return cls(video, frame, t, standard), False

    @property
    def spec(self) -> CourtSpec:
        return CourtSpec.preset(self.standard)

    def landmark_names(self) -> list[str]:
        from .court import landmarks

        return list(landmarks(self.spec))

    def next_landmark(self) -> str | None:
        used = {name for name, _, _ in self.points}
        return next((n for n in self.landmark_names() if n not in used), None)

    def click(self, landmark: str, x_px: float, y_px: float) -> None:
        """Place `landmark` at a pixel of the displayed frame (replaces an earlier click)."""
        if landmark not in self.landmark_names():
            raise ValueError(f"Unknown landmark {landmark!r}")
        h, w = self.frame.shape[:2]
        self.points = [p for p in self.points if p[0] != landmark]
        self.points.append((landmark, x_px / w, y_px / h))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def calibration(self) -> Calibration:
        from .court import landmarks

        marks = landmarks(self.spec)
        return Calibration(
            video=self.video.name, t=self.t, standard=self.standard, spec=self.spec.to_dict(),
            points=[CalibrationPoint(n, u, v, *marks[n]) for n, u, v in self.points if n in marks],
        )

    def image_rgb(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(draw_calibration(self.frame, self.calibration()), cv2.COLOR_BGR2RGB)

    def status(self) -> str:
        n = len(self.points)
        if n < 4:
            return (f"**{n} of at least 4 points.** Pick a landmark, then click exactly where "
                    "it is on the floor. Spread the points out; don't use 4 on one line.")
        err = self.calibration().error_ft()
        msg = (f"**{n} points.** Check that the green court lines sit on the real court lines, "
               "then save.")
        if err is not None:
            msg += f" Fit: {err:.1f} ft off on average" + (
                " (good)." if err < 1.5 else " - a point may be on the wrong landmark.")
        else:
            msg += " Add a 5th point to get a fit check."
        return msg

    def save(self, output_dir: str | Path) -> Path:
        if len(self.points) < 4:
            raise ValueError("Click at least 4 landmarks before saving")
        return self.calibration().save(output_dir, self.frame)


def draw_calibration(frame_bgr: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Frame with the clicked points and, from 4 points on, the court lines in green."""
    import cv2

    from .court import draw_court_on_image

    img = frame_bgr.copy()
    h, w = img.shape[:2]
    H_norm = calibration.homography()
    if H_norm is not None:
        to_px = np.diag([1.0 / w, 1.0 / h, 1.0])   # pixels -> normalized
        draw_court_on_image(img, H_norm @ to_px, calibration.court_spec, (0, 255, 0),
                            max(2, w // 640))
    r = max(5, w // 200)
    for i, p in enumerate(calibration.points, start=1):
        x, y = int(p.u * w), int(p.v * h)
        cv2.circle(img, (x, y), r, (0, 255, 255), -1)
        cv2.circle(img, (x, y), r, (0, 0, 0), 1)
        cv2.putText(img, str(i), (x + r + 2, y - r), cv2.FONT_HERSHEY_SIMPLEX, r / 10,
                    (0, 255, 255), max(1, r // 4), cv2.LINE_AA)
    return img
