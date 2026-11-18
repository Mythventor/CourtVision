"""Turn per-frame tracks into game events: possession, shots, rebounds.

Possession drives most per-player stats: the shooter is whoever had the ball just
before a shot, and the rebounder is whoever gets it after a miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .events import Event


@dataclass
class FrameRecord:
    """What the pipeline keeps from each analyzed frame."""

    frame: int
    t: float
    player_xyxy: np.ndarray           # (N, 4)
    player_track_ids: np.ndarray      # (N,)
    ball_xyxy: np.ndarray | None      # (4,) best ball box, or None
    hoop_xyxy: np.ndarray             # (M, 4)
    referee_xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    court_H: np.ndarray | None = None  # (3, 3) image -> court feet, or None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        def boxes(a):
            return np.round(np.asarray(a, dtype=float), 1).tolist()

        return {
            "f": self.frame, "t": round(self.t, 3),
            "ids": [int(i) for i in self.player_track_ids], "players": boxes(self.player_xyxy),
            "ball": None if self.ball_xyxy is None else boxes(self.ball_xyxy),
            "hoops": boxes(self.hoop_xyxy), "refs": boxes(self.referee_xyxy),
            "H": None if self.court_H is None else np.asarray(self.court_H).ravel().tolist(),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "FrameRecord":
        def boxes(a):
            return np.asarray(a, dtype=float).reshape(-1, 4)

        return cls(
            frame=d["f"], t=d["t"],
            player_xyxy=boxes(d["players"]), player_track_ids=np.asarray(d["ids"], dtype=int),
            ball_xyxy=None if d["ball"] is None else np.asarray(d["ball"], dtype=float),
            hoop_xyxy=boxes(d["hoops"]), referee_xyxy=boxes(d["refs"]),
            court_H=None if d["H"] is None else np.asarray(d["H"], dtype=float).reshape(3, 3),
        )


def detect_events(
    frames: list[FrameRecord],
    track_team: dict[int, str],
    track_number: dict[int, int | None],
) -> list[Event]:
    """Derive possession / shot / rebound events from frame records.

    TODO (week 2):
      1. Ball holder per frame: nearest player to the ball, within a distance threshold.
      2. Possession events when the holding team changes (smooth over a few frames).
      3. Shot attempts: ball rises toward a hoop after leaving the holder's hands.
      4. Makes: ball passes through the hoop box from above.
      5. Rebounds: first holder after a missed shot.
    """
    raise NotImplementedError("Event detection is not implemented yet")
