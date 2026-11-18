"""Find game videos in a folder (on Colab: the Drive folder with the instructor's videos)."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}


def list_videos(root: str | Path, exclude: str | Path | None = None) -> dict[str, list[Path]]:
    """Videos under `root`, grouped by folder (relative to root), sorted by name.

    Files inside `exclude` (e.g. the output folder with annotated videos) are skipped.
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    skip = Path(exclude).resolve() if exclude else None
    games: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS or not path.is_file():
            continue
        if skip and skip in path.resolve().parents:
            continue
        folder = path.parent.relative_to(root).as_posix()
        games.setdefault(folder if folder != "." else "(top level)", []).append(path)
    return games
