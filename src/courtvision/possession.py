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
    extra: dict[str, Any] = field(default_factory=dict)


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
