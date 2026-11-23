"""One-frame previews for setting up the models (used from the Colab notebook).

- `preview_detections`: what the detection model finds, and its class names, so the
  `detection.classes` config can be checked.
- `preview_keypoints`: the court keypoints the court model finds, numbered, and how well
  they fit `court.template` (index i -> court position of keypoint i).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def read_frame(video_path: str, t_s: float) -> np.ndarray:
    from .video import iter_frames

    for _, _, frame in iter_frames(video_path, 1, t_s, t_s + 1):
        return frame
    raise ValueError(f"No frame at {t_s}s in {video_path}")


def preview_detections(video_path: str, t_s: float, config: dict[str, Any]):
    """(RGB image with all detections, class names found in this frame, all model classes)."""
    import cv2
    import supervision as sv

    from .detection import Detector

    frame = read_frame(video_path, t_s)
    detector = Detector(config["detection"])
    result = detector.model.infer(frame, confidence=detector.confidence)[0]
    dets = sv.Detections.from_inference(result)
    names = dets.data.get("class_name", np.array([]))
    img = frame.copy()
    thick = max(2, frame.shape[1] // 800)
    for (x1, y1, x2, y2), name, conf in zip(dets.xyxy.astype(int), names, dets.confidence):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, max(30, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, thick * 0.5, (0, 255, 0), thick, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sorted(set(names.tolist())), detector.class_names()


def preview_keypoints(video_path: str, t_s: float, config: dict[str, Any]):
    """(RGB image, [(index, x, y, confidence)], fit error in feet).

    The image shows the numbered keypoints and, in green, the court lines drawn back
    into the frame from the estimated mapping. The mapping is right only if the green
    lines sit on the real court lines.

    The fit error is the mean distance (ft) between the confident keypoints mapped onto
    the court and their template positions. It only shows the keypoints agree with each
    other: a model that labels every point consistently wrong also gets a small error.
    """
    import cv2

    from .court import (CourtSpec, detect_keypoints, draw_court_on_image,
                        homography_from_keypoints, load_keypoint_model, to_court)

    court_cfg = config["court"]
    frame = read_frame(video_path, t_s)
    model = load_keypoint_model(court_cfg)
    xy, conf = detect_keypoints(model, frame, court_cfg.get("confidence", 0.3))
    img = frame.copy()
    points, fit_error = [], None
    min_conf = court_cfg.get("keypoint_confidence", 0.5)
    if xy is not None:
        thick = max(2, frame.shape[1] // 800)
        for i, ((x, y), c) in enumerate(zip(xy, conf)):
            points.append((i, float(x), float(y), float(c)))
            color = (0, 255, 255) if c >= min_conf else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 4 * thick, color, -1)
            cv2.putText(img, str(i), (int(x) + 6 * thick, int(y) - 6 * thick),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, color, thick, cv2.LINE_AA)
        template = court_cfg.get("template") or []
        H = homography_from_keypoints(xy, conf, template, min_conf)
        if H is not None:
            draw_court_on_image(img, H, CourtSpec.preset("nba"), (0, 255, 0), 2 * thick)  # template is NBA
            used = [i for i in range(min(len(xy), len(template)))
                    if template[i] is not None and conf[i] >= min_conf]
            mapped = to_court(xy[used], H)
            fit_error = float(np.mean(np.linalg.norm(mapped - np.asarray(
                [template[i] for i in used], dtype=float), axis=1)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), points, fit_error
