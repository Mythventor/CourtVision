"""Compute team and player box-score stats from an event log."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .events import Event, EventLog

PlayerKey = tuple[str, int | str | None]  # (team, jersey number or name)

# Event type -> Line attribute, for stats that are simple counts
_COUNTS = {
    "rebound": "reb",
    "foul": "pf",
    "assist": "ast",
    "steal": "stl",
    "block": "blk",
    "turnover": "tov",
}


@dataclass
class Line:
    """Box-score line for a player or a team."""

    fgm: int = 0
    fga: int = 0
    tpm: int = 0  # three-pointers made (missed shots in tag logs have no point value)
    ftm: int = 0  # free throws made
    points: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0

    @property
    def missed(self) -> int:
        """Missed field goals (the "SA" stat in FAAM tag logs)."""
        return self.fga - self.fgm

    @property
    def fg_pct(self) -> float | None:
        return self.fgm / self.fga if self.fga else None

    def add(self, e: Event) -> None:
        if e.type == "shot":
            self.fga += 1
            if e.data.get("made"):
                points = e.data.get("points") or 2
                self.fgm += 1
                self.tpm += points == 3
                self.points += points
        elif e.type == "free_throw":
            self.ftm += 1
            self.points += 1
        elif e.type in _COUNTS:
            attr = _COUNTS[e.type]
            setattr(self, attr, getattr(self, attr) + 1)


@dataclass
class GameStats:
    teams: dict[str, Line]
    players: dict[PlayerKey, Line]
    possession_pct: dict[str, float]


def compute_stats(
    log: EventLog, where: Callable[[Event], bool] | None = None
) -> GameStats:
    """Box score from an event log. `where` filters events (e.g. one quarter)."""
    events: Iterable[Event] = log.events
    if where:
        events = [e for e in events if where(e)]

    teams = {t: Line() for t in log.teams}
    players: dict[PlayerKey, Line] = defaultdict(Line)
    for e in events:
        # Possession is per team (see possession_share); it gets no box-score line.
        if e.team not in teams or e.type == "possession":
            continue
        teams[e.team].add(e)
        players[(e.team, e.player)].add(e)

    return GameStats(teams=teams, players=dict(players), possession_pct=possession_share(log))


def possession_share(log: EventLog) -> dict[str, float]:
    """Fraction of possessed time per team, from "possession" events (ball gained at t)."""
    held = {t: 0.0 for t in log.teams}
    changes = sorted(log.of_type("possession"), key=lambda e: e.t)
    for i, cur in enumerate(changes):
        end = changes[i + 1].t if i + 1 < len(changes) else log.duration
        if cur.team in held:
            held[cur.team] += max(0.0, end - cur.t)
    total = sum(held.values())
    # No possession data (e.g. tags-only mode): leave it empty so the report shows "-".
    return {t: v / total for t, v in held.items()} if total else {}
