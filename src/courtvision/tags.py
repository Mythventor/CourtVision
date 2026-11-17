"""Parse Coda-style tag CSV exports into CourtVision events.

Used as ground truth for evaluation, and optionally as an event source.
Column names and event names are configured in configs/default.yaml (`tags:`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .events import Event


def parse_time(value: str) -> float:
    """Parse "ss", "mm:ss" or "hh:mm:ss" (fractional seconds allowed) into seconds."""
    seconds = 0.0
    for part in value.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _optional(row: dict[str, Any], column: str | None) -> str:
    """Value of an optional column, or "" if the column is unset or absent."""
    if not column:
        return ""
    return (row.get(column) or "").strip()


def load_tags(path: str | Path, tag_config: dict[str, Any]) -> list[Event]:
    """Load a tag CSV. Rows whose event name is not in `event_map` are skipped."""
    cols = tag_config["columns"]
    event_map = tag_config["event_map"]
    events: list[Event] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {cols["time"], cols["event"]} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Tag CSV {path} is missing columns {sorted(missing)}; "
                f"found {reader.fieldnames}. Update `tags.columns` in the config."
            )
        for row in reader:
            spec = event_map.get(row[cols["event"]].strip())
            if spec is None:
                continue
            spec = dict(spec)
            team = _optional(row, cols.get("team"))
            player = _optional(row, cols.get("player"))
            events.append(
                Event(
                    t=parse_time(row[cols["time"]]),
                    type=spec.pop("type"),
                    team=team or None,
                    player=int(player) if player.isdigit() else None,
                    data=spec,
                )
            )
    return sorted(events, key=lambda e: e.t)
