"""Read video frames with a stride."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def video_meta(path: str | Path) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, stride: int = 1, start_s: float = 0.0, end_s: float | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_seconds, BGR frame) for every `stride`-th frame."""
    import cv2

    meta = video_meta(path)
    cap = cv2.VideoCapture(str(path))
    start = int(start_s * meta.fps)
    end = int(end_s * meta.fps) if end_s is not None else meta.frame_count
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for idx in range(start, end):
            # grab() skips decoding, so skipped frames are cheap
            if not cap.grab():
                break
            if (idx - start) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield idx, idx / meta.fps, frame
    finally:
        cap.release()
