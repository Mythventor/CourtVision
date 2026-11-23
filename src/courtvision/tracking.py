"""Multi-object tracking of players across frames (ByteTrack)."""

from __future__ import annotations

from typing import Any


class PlayerTracker:
    """ByteTrack wrapper: `tracker(detections)` returns confirmed tracks with tracker_id.

    Uses Roboflow's `trackers` package (supervision >= 0.31 no longer ships ByteTrack),
    falling back to `supervision.ByteTrack` on older versions.
    """

    def __init__(self, track_config: dict[str, Any], frame_rate: float):
        params = dict(
            track_activation_threshold=track_config.get("track_activation_threshold", 0.25),
            lost_track_buffer=track_config.get("lost_track_buffer", 60),
            minimum_consecutive_frames=track_config.get("minimum_consecutive_frames", 1),
            frame_rate=max(1.0, float(frame_rate)),
        )
        try:
            from trackers import ByteTrackTracker

            self._tracker = ByteTrackTracker(**params)
            self._update = self._tracker.update
        except ImportError:
            import supervision as sv

            params["frame_rate"] = max(1, round(params["frame_rate"]))
            self._tracker = sv.ByteTrack(**params)
            self._update = self._tracker.update_with_detections

    def __call__(self, detections):
        tracked = self._update(detections)
        if tracked.tracker_id is None:
            return tracked[:0]
        return tracked[tracked.tracker_id >= 0]  # -1: not yet confirmed


def make_tracker(track_config: dict[str, Any], frame_rate: float) -> PlayerTracker:
    """`frame_rate` should be the analyzed rate, fps / frame_stride."""
    return PlayerTracker(track_config, frame_rate)
