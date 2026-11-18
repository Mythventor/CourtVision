"""End-to-end pipeline: video -> tracks -> teams/numbers -> event log.

Stage 1 (`run_tracking`) is the expensive GPU pass over the video. Its result is saved
to disk (`TrackingResult.save`) so rendering and event detection can be re-run without it.
Stage 2 (`build_event_log`) is cheap and can be re-run after the user confirms which
color cluster is which team.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .events import EventLog
from .jersey import JerseyVoter, read_number
from .possession import FrameRecord, detect_events
from .teams import cluster_teams, color_feature, torso_crop
from .video import VideoMeta, iter_frames, video_meta

Progress = Callable[[float, str], None]
CACHE_VERSION = 1


@dataclass
class TrackingResult:
    meta: VideoMeta
    video: str = ""
    start_s: float = 0.0
    end_s: float | None = None
    frames: list[FrameRecord] = field(default_factory=list)
    clusters: dict[int, int] = field(default_factory=dict)  # track id -> color cluster 0/1
    cluster_colors: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))  # Lab
    numbers: dict[int, int | None] = field(default_factory=dict)  # track id -> jersey number
    stats: dict[str, Any] = field(default_factory=dict)  # run time, frames, court coverage

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CACHE_VERSION, "video": self.video, "meta": asdict(self.meta),
            "start_s": self.start_s, "end_s": self.end_s,
            "frames": [f.to_json() for f in self.frames],
            "clusters": {str(k): v for k, v in self.clusters.items()},
            "cluster_colors": np.asarray(self.cluster_colors).tolist(),
            "numbers": {str(k): v for k, v in self.numbers.items()},
            "stats": self.stats,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(path)  # never leave a half-written cache behind

    @classmethod
    def load(cls, path: str | Path) -> "TrackingResult":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            raise ValueError(f"Tracking cache {path} has an old format; reprocess the video")
        return cls(
            meta=VideoMeta(**data["meta"]), video=data["video"],
            start_s=data["start_s"], end_s=data["end_s"],
            frames=[FrameRecord.from_json(f) for f in data["frames"]],
            clusters={int(k): v for k, v in data["clusters"].items()},
            cluster_colors=np.asarray(data["cluster_colors"], dtype=float),
            numbers={int(k): v for k, v in data["numbers"].items()},
            stats=data.get("stats", {}),
        )


def run_tracking(
    video_path: str | Path,
    config: dict[str, Any],
    progress: Progress | None = None,
    read_jerseys: bool = False,
    start_s: float = 0.0,
    end_s: float | None = None,
    detector=None,
    court_mapper=None,
) -> TrackingResult:
    """Detect and track players/ball/hoop, map the court, collect jersey colors.

    `detector` and `court_mapper` default to the Roboflow models from the config; tests
    pass fakes. Jersey reading is off until `jersey.read_number` is implemented.
    """
    from .court import CourtMapper
    from .tracking import make_tracker

    stride = config["video"]["frame_stride"]
    every_n = max(1, config["court"].get("every_n", 1))
    meta = video_meta(video_path)
    if detector is None:
        from .detection import Detector

        detector = Detector(config["detection"])
    if court_mapper is None:
        court_mapper = CourtMapper(config["court"])
    tracker = make_tracker(config["tracking"], frame_rate=meta.fps / stride)

    end = min(end_s, meta.duration) if end_s else meta.duration
    result = TrackingResult(meta=meta, video=str(video_path), start_s=start_s, end_s=end)
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    jerseys = JerseyVoter(config["jersey"]["min_votes"])
    span = max(end - start_s, 1e-6)
    started = time.time()
    H = None

    for n, (idx, t, frame) in enumerate(iter_frames(video_path, stride, start_s, end)):
        dets = detector(frame)
        players = tracker(dets.players)
        track_ids = players.tracker_id if players.tracker_id is not None else np.zeros(0, int)
        if n % every_n == 0:
            H = court_mapper(frame)

        for xyxy, track_id in zip(players.xyxy, track_ids):
            crop = torso_crop(frame, xyxy)
            feature = color_feature(crop)
            if feature is not None:
                colors[int(track_id)].append(feature)
            if read_jerseys:
                jerseys.add(int(track_id), read_number(crop))

        ball = None
        if len(dets.ball):
            ball = dets.ball.xyxy[int(np.argmax(dets.ball.confidence))]
        result.frames.append(
            FrameRecord(
                frame=idx, t=t,
                player_xyxy=players.xyxy, player_track_ids=track_ids,
                ball_xyxy=ball, hoop_xyxy=dets.hoop.xyxy, referee_xyxy=dets.referees.xyxy,
                court_H=H,
            )
        )
        if progress and n % 10 == 0:
            fps = (n + 1) / max(time.time() - started, 1e-6)
            progress((t - start_s) / span,
                     f"Tracking {t - start_s:.0f}s / {span:.0f}s of video ({fps:.1f} frames/s)")

    result.clusters, result.cluster_colors = cluster_teams(colors)
    result.numbers = jerseys.assignments()
    elapsed = time.time() - started
    frames = len(result.frames)
    result.stats = {
        "frames": frames,
        "seconds": round(elapsed, 1),
        "frames_per_second": round(frames / elapsed, 2) if elapsed else None,
        "tracks": len(result.clusters),
        "court_mapped": round(sum(f.court_H is not None for f in result.frames) / frames, 3)
        if frames else 0.0,
        "ball_found": round(sum(f.ball_xyxy is not None for f in result.frames) / frames, 3)
        if frames else 0.0,
    }
    return result


def build_event_log(
    tracking: TrackingResult, cluster_to_team: dict[int, str]
) -> EventLog:
    """Turn tracking output into an event log, given which color cluster is which team."""
    track_team = {t: cluster_to_team[c] for t, c in tracking.clusters.items()}
    events = detect_events(tracking.frames, track_team, tracking.numbers)
    return EventLog(
        teams=list(cluster_to_team.values()),
        duration=tracking.meta.duration,
        events=events,
        source="cv",
    )
