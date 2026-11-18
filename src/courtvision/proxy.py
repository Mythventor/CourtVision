"""Make a small working copy ("proxy") of the part of a video being analyzed.

FAAM videos are 4K HEVC at 30 fps and up to ~20 GB. Decoding 4K HEVC frame by frame on
Colab's CPU is the slowest part of the pipeline, and the models resize frames to ~640 px
anyway. So the requested time range is converted once to a 720p H.264 copy with ffmpeg,
using the GPU's video decoder/encoder (NVDEC/NVENC) when available, and tracking and
rendering read that copy instead.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

Progress = Callable[[float, str], None]


@functools.lru_cache(maxsize=None)
def _ffmpeg_lists() -> tuple[str, str]:
    """(hwaccels, encoders) as printed by ffmpeg, or empty strings without ffmpeg."""
    if not shutil.which("ffmpeg"):
        return "", ""

    def run(*args: str) -> str:
        try:
            return subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True,
                                  text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    return run("-hwaccels"), run("-encoders")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def proxy_enabled(video_cfg: dict[str, Any]) -> bool:
    """`video.proxy`: true, false, or "auto" (use it when ffmpeg is installed)."""
    setting = video_cfg.get("proxy", "auto")
    if setting == "auto":
        return ffmpeg_available()
    return bool(setting)


def _commands(src: str, out: str, start_s: float, duration: float | None,
              height: int, fps: float) -> list[list[str]]:
    """ffmpeg commands to try in order: GPU decode+encode first, plain CPU last."""
    hwaccels, encoders = _ffmpeg_lists()
    cut = ["-ss", f"{start_s:.3f}"] + (["-t", f"{duration:.3f}"] if duration else [])
    video = ["-an", "-sn", "-vf", f"fps={fps},scale=-2:{height}", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    nvenc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-progress", "pipe:1"]

    cmds = []
    if "cuda" in hwaccels:
        enc = nvenc if "h264_nvenc" in encoders else x264
        cmds.append(base + ["-hwaccel", "cuda"] + cut + ["-i", src] + video + enc + [out])
    cmds.append(base + cut + ["-i", src] + video + x264 + [out])
    return cmds


def make_proxy(
    src: str | Path,
    out: str | Path,
    start_s: float = 0.0,
    end_s: float | None = None,
    height: int = 720,
    fps: float = 30.0,
    progress: Progress | None = None,
    total_s: float | None = None,
) -> Path:
    """Write the [start_s, end_s) part of `src` as a 720p H.264 file (reused if present).

    The proxy's time 0 is `start_s` in the original video.
    """
    out = Path(out)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.mp4")
    duration = (end_s - start_s) if end_s else None
    span = duration or (max(total_s - start_s, 1e-6) if total_s else None)

    errors = []
    for cmd in _commands(str(src), str(tmp), start_s, duration, height, fps):
        mode = "GPU" if "-hwaccel" in cmd else "CPU"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            # -progress writes key=value lines; out_time_us is the position written so far
            if progress and span and line.startswith("out_time_us="):
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                progress(min(done / span, 1.0),
                         f"Preparing 720p copy ({mode}): {done:.0f}s / {span:.0f}s")
        err = proc.stderr.read()
        if proc.wait() == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return out
        errors.append(f"{mode}: {err.strip()[-500:]}")
        tmp.unlink(missing_ok=True)
    raise RuntimeError("ffmpeg could not make the 720p copy:\n" + "\n".join(errors))
