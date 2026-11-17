"""End-to-end pipeline: video -> tracks -> teams/numbers -> event log.

Stage 1 (`run_tracking`) is the expensive GPU pass over the video.
Stage 2 (`build_event_log`) is cheap and can be re-run after the user confirms which
color cluster is which team.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .events import EventLog
from .jersey import JerseyVoter, read_number
from .possession import FrameRecord, detect_events
from .teams import cluster_teams, color_feature, torso_crop
from .video import VideoMeta, iter_frames, video_meta

Progress = Callable[[float, str], None]


@dataclass
class TrackingResult:
    meta: VideoMeta
    frames: list[FrameRecord] = field(default_factory=list)
    colors: dict[int, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    jerseys: JerseyVoter = field(default_factory=JerseyVoter)
    clusters: dict[int, int] = field(default_factory=dict)  # track id -> color cluster 0/1


def run_tracking(
    video_path: str | Path,
    config: dict[str, Any],
    progress: Progress | None = None,
    read_jerseys: bool = True,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> TrackingResult:
    """Detect and track players/ball/hoop, collect jersey colors and number readings."""
    from .detection import Detector
    from .tracking import make_tracker

    stride = config["video"]["frame_stride"]
    meta = video_meta(video_path)
    detector = Detector(config["detection"])
    tracker = make_tracker(config["tracking"], frame_rate=meta.fps / stride)
    result = TrackingResult(meta=meta, jerseys=JerseyVoter(config["jersey"]["min_votes"]))
    end = end_s if end_s is not None else meta.duration
    span = max(end - start_s, 1e-6)

    for idx, t, frame in iter_frames(video_path, stride, start_s, end_s):
        dets = detector(frame)
        players = tracker.update_with_detections(dets.players)

        for xyxy, track_id in zip(players.xyxy, players.tracker_id):
            crop = torso_crop(frame, xyxy)
            result.colors[int(track_id)].append(color_feature(crop))
            if read_jerseys:
                result.jerseys.add(int(track_id), read_number(crop))

        ball = None
        if len(dets.ball):
            ball = dets.ball.xyxy[int(np.argmax(dets.ball.confidence))]
        result.frames.append(
            FrameRecord(
                frame=idx,
                t=t,
                player_xyxy=players.xyxy,
                player_track_ids=players.tracker_id,
                ball_xyxy=ball,
                hoop_xyxy=dets.hoop.xyxy,
            )
        )
        if progress:
            progress((t - start_s) / span, f"Analyzing video ({t:.0f}s / {end:.0f}s)")

    result.clusters = cluster_teams(result.colors)
    return result


def build_event_log(
    tracking: TrackingResult, cluster_to_team: dict[int, str]
) -> EventLog:
    """Turn tracking output into an event log, given which color cluster is which team."""
    track_team = {t: cluster_to_team[c] for t, c in tracking.clusters.items()}
    track_number = tracking.jerseys.assignments()
    events = detect_events(tracking.frames, track_team, track_number)
    return EventLog(
        teams=list(cluster_to_team.values()),
        duration=tracking.meta.duration,
        events=events,
        source="cv",
    )
