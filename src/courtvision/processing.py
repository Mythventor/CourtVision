"""Process a video once and cache the results next to each other on disk (Drive on Colab).

    <output_dir>/<video name>_<start>-<end>/tracks.json.gz   tracking cache
    <output_dir>/<video name>_<start>-<end>/annotated.mp4    video with overlays + 2D panel
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pipeline import TrackingResult, run_tracking
from .render import render_video

Progress = Callable[[float, str], None]


@dataclass
class ProcessResult:
    folder: Path
    tracking: TrackingResult
    video: Path
    reused: bool          # True if the tracking cache was reused


def job_folder(config: dict[str, Any], video_path: str | Path,
               start_s: float = 0.0, end_s: float | None = None) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video_path).stem).strip("_")
    end = f"{int(end_s)}" if end_s else "end"
    return Path(config["output_dir"]) / f"{stem}_{int(start_s)}-{end}"


def process_video(
    video_path: str | Path,
    config: dict[str, Any],
    start_s: float = 0.0,
    end_s: float | None = None,
    progress: Progress | None = None,
    reprocess: bool = False,
    rerender: bool = False,
    **tracking_kwargs,
) -> ProcessResult:
    """Track (or load the cached tracks), then render the annotated video if needed."""
    folder = job_folder(config, video_path, start_s, end_s)
    tracks_path = folder / "tracks.json.gz"
    video_out = folder / "annotated.mp4"

    def stage(lo: float, hi: float) -> Progress | None:
        if progress is None:
            return None
        return lambda p, msg: progress(lo + (hi - lo) * min(max(p, 0.0), 1.0), msg)

    reused = tracks_path.exists() and not reprocess
    if reused:
        tracking = TrackingResult.load(tracks_path)
    else:
        tracking = run_tracking(video_path, config, progress=stage(0.0, 0.7),
                                start_s=start_s, end_s=end_s, **tracking_kwargs)
        tracking.save(tracks_path)

    if rerender or not reused or not video_out.exists():
        render_video(tracking, video_out, config, progress=stage(0.7, 1.0))
    return ProcessResult(folder=folder, tracking=tracking, video=video_out, reused=reused)
