"""Event log: the single output of the video pipeline.

Stats, the analytics page and the PDF report are all derived from the event log,
so stat logic can change without re-processing video.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EventType = Literal["possession", "shot", "rebound", "foul"]


@dataclass
class Event:
    t: float                      # seconds from start of video
    type: EventType
    team: str | None = None       # team name
    player: int | None = None     # jersey number (None if unknown)
    track_id: int | None = None   # tracker id, for debugging / manual correction
    frame: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    # shot:       {"made": bool, "points": 2 | 3}
    # rebound:    {"offensive": bool}
    # possession: `team` gains the ball at time t


@dataclass
class EventLog:
    teams: list[str]              # the two teams playing
    duration: float               # seconds of video analyzed
    events: list[Event] = field(default_factory=list)
    source: str = "cv"            # "cv" or "tags"

    def of_type(self, type_: EventType) -> list[Event]:
        return [e for e in self.events if e.type == type_]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EventLog":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        raw["events"] = [Event(**e) for e in raw["events"]]
        return cls(**raw)
