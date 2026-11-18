"""Process a video once and cache the results next to each other on disk (Drive on Colab).

    <output_dir>/<video name>_<start>-<end>/source_720p.mp4  720p working copy of the range
    <output_dir>/<video name>_<start>-<end>/tracks.json.gz   tracking cache
    <output_dir>/<video name>_<start>-<end>/annotated.mp4    video with overlays + 2D panel
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pipeline import TrackingResult, run_tracking
from .proxy import make_proxy, proxy_enabled
from .render import render_video


def _meta(video_path: str | Path):
    try:
        from .video import video_meta

        return video_meta(video_path)
    except Exception:  # noqa: BLE001 - only used for progress and the fps cap
        return None

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
        vc = config["video"]
        if proxy_enabled(vc):
            # Work on a 720p copy of just this range; its time 0 is start_s in the original
            src = _meta(video_path)
            fps = vc.get("proxy_fps", 30)
            if src and src.fps:
                fps = min(fps, src.fps)  # never invent frames
            proxy = make_proxy(video_path, folder / "source_720p.mp4", start_s, end_s,
                               height=vc.get("proxy_height", 720), fps=fps,
                               progress=stage(0.0, 0.25), total_s=src.duration if src else None)
            tracking = run_tracking(proxy, config, progress=stage(0.25, 0.75), **tracking_kwargs)
            tracking.offset_s = start_s
            tracking.start_s = start_s
            tracking.end_s = start_s + tracking.meta.duration
        else:
            tracking = run_tracking(video_path, config, progress=stage(0.0, 0.75),
                                    start_s=start_s, end_s=end_s, **tracking_kwargs)
        tracking.source = str(video_path)
        tracking.save(tracks_path)

    if rerender or not reused or not video_out.exists():
        render_video(tracking, video_out, config, progress=stage(0.75, 1.0))
    return ProcessResult(folder=folder, tracking=tracking, video=video_out, reused=reused)
