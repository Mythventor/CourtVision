"""One-frame previews for setting up the models (used from the Colab notebook).

- `preview_detections`: what the detection model finds, and its class names, so the
  `detection.classes` config can be checked.
- `preview_keypoints`: the court keypoints the court model finds, numbered, so the
  `court.template` config can be filled in (index i -> court position of keypoint i).
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
    """(RGB image with all detections, sorted class names found in this frame)."""
    import cv2
    import supervision as sv

    from .detection import Detector

    frame = read_frame(video_path, t_s)
    detector = Detector(config["detection"])
    result = detector.model.infer(frame, confidence=detector.confidence)[0]
    dets = sv.Detections.from_inference(result)
    names = dets.data.get("class_name", np.array([]))
    img = frame.copy()
    for (x1, y1, x2, y2), name, conf in zip(dets.xyxy.astype(int), names, dets.confidence):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sorted(set(names.tolist()))


def preview_keypoints(video_path: str, t_s: float, config: dict[str, Any]):
    """(RGB image with numbered court keypoints, list of (index, x, y, confidence))."""
    import cv2

    from .court import detect_keypoints, load_keypoint_model

    frame = read_frame(video_path, t_s)
    model = load_keypoint_model(config["court"])
    xy, conf = detect_keypoints(model, frame, config["court"].get("confidence", 0.3))
    img = frame.copy()
    points = []
    if xy is not None:
        min_conf = config["court"].get("keypoint_confidence", 0.5)
        for i, ((x, y), c) in enumerate(zip(xy, conf)):
            points.append((i, float(x), float(y), float(c)))
            color = (0, 255, 255) if c >= min_conf else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 6, color, -1)
            cv2.putText(img, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), points
