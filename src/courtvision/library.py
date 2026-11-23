"""Find game videos in a folder (on Colab: the Drive folder with the instructor's videos).

Expected layout (any depth works):

    <video_root>/FAAM_HOOPS_2026_2027_VIDEOS/2026-09-26/<quarter or full-game videos>

Drive shortcuts show up in Colab as symbolic links, so the search follows links.
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}


def list_videos(root: str | Path, exclude: str | Path | None = None) -> dict[str, list[Path]]:
    """Videos under `root`, grouped by folder (relative to root), sorted by name.

    Follows symbolic links (Drive shortcuts). Hidden folders and anything inside
    `exclude` (e.g. the output folder with annotated videos) are skipped.
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    skip = os.path.realpath(exclude) if exclude else None
    seen: set[str] = set()   # real paths already walked, to avoid link loops
    games: dict[str, list[Path]] = {}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen or (skip and (real == skip or real.startswith(skip + os.sep))):
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))

        videos = sorted(f for f in filenames if Path(f).suffix.lower() in VIDEO_EXTS)
        if videos:
            folder = Path(dirpath).relative_to(root).as_posix()
            games[folder if folder != "." else "(top level)"] = [Path(dirpath) / f for f in videos]
    return dict(sorted(games.items()))


def describe_folder(root: str | Path, depth: int = 2) -> list[str]:
    """Indented listing of folders under `root` (marking links), for troubleshooting."""
    root = Path(root)
    lines = []

    def walk(path: Path, level: int) -> None:
        try:
            entries = sorted(path.iterdir(), key=lambda p: p.name)
        except OSError as e:
            lines.append("  " * level + f"(cannot read: {e})")
            return
        for p in entries:
            if p.name.startswith("."):
                continue
            if p.is_dir():
                tag = " -> (shortcut/link)" if p.is_symlink() else "/"
                lines.append("  " * level + p.name + tag)
                if level + 1 < depth:
                    walk(p, level + 1)
            elif p.suffix.lower() in VIDEO_EXTS:
                lines.append("  " * level + p.name)

    walk(root, 0)
    return lines
