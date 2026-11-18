"""Render an annotated video: player boxes, ball, referees, and a live 2D court panel.

Detection runs on every `frame_stride`-th frame; the renderer draws every frame and
interpolates boxes and the ball in between, so playback is smooth.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .court import CourtCanvas, CourtSpec, to_court
from .possession import FrameRecord
from .teams import lab_to_bgr
from .video import iter_frames

BALL_COLOR = (0, 140, 255)      # orange, BGR
REF_COLOR = (170, 170, 170)
UNKNOWN_COLOR = (230, 230, 230)


@dataclass
class FrameState:
    """What to draw on one output frame (in source-video pixel coordinates)."""

    ids: list[int]
    boxes: np.ndarray            # (N, 4)
    ball: np.ndarray | None      # (2,) ball center
    referees: np.ndarray         # (M, 4)
    court_H: np.ndarray | None


def _center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def fill_ball_gaps(frames: list[FrameRecord], max_gap: int) -> list[np.ndarray | None]:
    """Ball center per analyzed frame, linearly filling gaps of up to `max_gap` frames."""
    centers = [None if f.ball_xyxy is None else _center(f.ball_xyxy) for f in frames]
    known = [i for i, c in enumerate(centers) if c is not None]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= max_gap + 1:
            for i in range(a + 1, b):
                alpha = (i - a) / (b - a)
                centers[i] = (1 - alpha) * centers[a] + alpha * centers[b]
    return centers


def interpolate(
    a: FrameRecord, b: FrameRecord | None,
    ball_a: np.ndarray | None, ball_b: np.ndarray | None, frame: int,
) -> FrameState:
    """State at `frame`, between analyzed frames `a` (at or before) and `b` (after)."""
    if b is None or b.frame == a.frame:
        alpha = 0.0
    else:
        alpha = min(max((frame - a.frame) / (b.frame - a.frame), 0.0), 1.0)
    near, far = (a, b) if alpha < 0.5 or b is None else (b, a)

    boxes_a = dict(zip(a.player_track_ids.tolist(), a.player_xyxy))
    boxes_b = dict(zip(b.player_track_ids.tolist(), b.player_xyxy)) if b is not None else {}
    ids, boxes = [], []
    for tid in dict.fromkeys([*boxes_a, *boxes_b]):
        if tid in boxes_a and tid in boxes_b:
            box = (1 - alpha) * boxes_a[tid] + alpha * boxes_b[tid]
        elif tid in boxes_a and alpha < 0.5:
            box = boxes_a[tid]
        elif tid in boxes_b and alpha >= 0.5:
            box = boxes_b[tid]
        else:
            continue
        ids.append(int(tid))
        boxes.append(box)

    if ball_a is not None and ball_b is not None:
        ball = (1 - alpha) * ball_a + alpha * ball_b
    else:
        ball = ball_a if alpha < 0.5 else ball_b

    H = near.court_H if near.court_H is not None else (far.court_H if far is not None else None)
    return FrameState(ids=ids, boxes=np.asarray(boxes).reshape(-1, 4), ball=ball,
                      referees=near.referee_xyxy, court_H=H)


class VideoSink:
    """Write BGR frames to a browser-playable MP4 (H.264 via ffmpeg; OpenCV fallback)."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        import cv2

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.proc = None
        self.writer = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE,
            )
        else:  # browsers may not play mp4v, but it keeps local runs working
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        else:
            self.writer.release()


def _even(n: float) -> int:
    return int(round(n / 2) * 2)


def _label(img, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    dark = (0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]) > 140
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0) if dark else (255, 255, 255), 1, cv2.LINE_AA)


def render_video(
    tracking,
    out_path: str | Path,
    config: dict[str, Any],
    progress: Callable[[float, str], None] | None = None,
    team_labels: dict[int, str] | None = None,
) -> Path:
    """Write the annotated video for a TrackingResult. `team_labels`: cluster -> team name."""
    import cv2

    rc = config["render"]
    meta = tracking.meta
    frames = tracking.frames
    if not frames:
        raise ValueError("No analyzed frames to render")

    scale = rc["width"] / meta.width
    vid_w, vid_h = _even(meta.width * scale), _even(meta.height * scale)
    canvas = CourtCanvas(CourtSpec.from_config(config["court"]), _even(rc["panel_width"]))
    court_img = canvas.draw()
    out_w, out_h = vid_w + canvas.width, _even(max(vid_h, canvas.height + 120))
    court_enabled = sum(p is not None for p in (config["court"].get("template") or [])) >= 4

    colors = {c: lab_to_bgr(lab) for c, lab in enumerate(tracking.cluster_colors)}
    team_labels = team_labels or {}
    balls = fill_ball_gaps(frames, rc.get("ball_max_gap", 5))
    trail: deque[tuple[int, int]] = deque(maxlen=rc.get("ball_trail", 12))

    sink = VideoSink(out_path, meta.fps, (out_w, out_h))
    k = 0
    first, last = frames[0].frame, frames[-1].frame
    try:
        for idx, t, frame in iter_frames(tracking.video, 1, frames[0].t, frames[-1].t + 1 / meta.fps):
            if idx > last:
                break
            while k + 1 < len(frames) and frames[k + 1].frame <= idx:
                k += 1
            b = frames[k + 1] if k + 1 < len(frames) else None
            state = interpolate(frames[k], b, balls[k], balls[k + 1] if b is not None else None, idx)

            out = np.zeros((out_h, out_w, 3), np.uint8)
            view = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)
            panel = court_img.copy()

            for box in state.referees:
                x1, y1, x2, y2 = (box * scale).astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), REF_COLOR, 1)

            feet = []
            for tid, box in zip(state.ids, state.boxes):
                cluster = tracking.clusters.get(tid)
                color = colors.get(cluster, UNKNOWN_COLOR)
                x1, y1, x2, y2 = (box * scale).astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
                number = tracking.numbers.get(tid)
                _label(view, f"#{number}" if number is not None else str(tid), x1, y1, color)
                feet.append(((box[0] + box[2]) / 2, box[3]))

            if state.ball is not None:
                bx, by = (state.ball * scale).astype(int)
                trail.append((bx, by))
                cv2.circle(view, (bx, by), 7, BALL_COLOR, 2)
            for i, (px, py) in enumerate(trail):
                cv2.circle(view, (px, py), 2 + i // 4, BALL_COLOR, -1)
            if state.ball is None and trail:
                trail.popleft()

            if state.court_H is not None:
                for (cx, cy), tid in zip(to_court(np.array(feet), state.court_H), state.ids):
                    if canvas.inside(cx, cy):
                        color = colors.get(tracking.clusters.get(tid), UNKNOWN_COLOR)
                        cv2.circle(panel, canvas.px(cx, cy), 7, color, -1)
                        cv2.circle(panel, canvas.px(cx, cy), 7, (20, 20, 20), 1)
                if state.ball is not None:
                    (cx, cy), = to_court(state.ball[None], state.court_H)
                    if canvas.inside(cx, cy):
                        cv2.circle(panel, canvas.px(cx, cy), 5, BALL_COLOR, -1)

            out[:vid_h, :vid_w] = view
            out[:canvas.height, vid_w:] = panel
            info = [
                f"t = {int(t // 60)}:{t % 60:04.1f}",
                f"players tracked: {len(state.ids)}",
                "2D panel: off (set court.template)" if not court_enabled
                else ("2D panel: court mapped" if state.court_H is not None
                      else "2D panel: court not found"),
            ]
            for i, line in enumerate(info):
                cv2.putText(out, line, (vid_w + 16, canvas.height + 28 + 24 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            for i, (c, color) in enumerate(colors.items()):  # team color legend
                y = canvas.height + 28 + 24 * (len(info) + i)
                cv2.rectangle(out, (vid_w + 16, y - 14), (vid_w + 32, y + 2), color, -1)
                cv2.putText(out, team_labels.get(c, f"Team color {c + 1}"), (vid_w + 40, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            sink.write(out)

            if progress and idx % 30 == 0:
                progress((idx - first) / max(last - first, 1), f"Rendering video ({t:.0f}s)")
    finally:
        sink.close()
    return Path(out_path)
