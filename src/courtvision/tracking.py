"""Multi-object tracking of players across frames."""

from __future__ import annotations

from typing import Any


def make_tracker(track_config: dict[str, Any], frame_rate: float):
    """ByteTrack from supervision. `frame_rate` should be fps / frame_stride."""
    import supervision as sv

    return sv.ByteTrack(
        track_activation_threshold=track_config.get("track_activation_threshold", 0.25),
        lost_track_buffer=track_config.get("lost_track_buffer", 60),
        frame_rate=max(1, round(frame_rate)),
    )
