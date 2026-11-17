"""Compute team and player box-score stats from an event log."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .events import EventLog


@dataclass
class Line:
    """Box-score line for a player or a team."""

    fgm: int = 0
    fga: int = 0
    tpm: int = 0  # three-pointers made
    tpa: int = 0
    points: int = 0
    rebounds: int = 0
    fouls: int = 0

    @property
    def fg_pct(self) -> float | None:
        return self.fgm / self.fga if self.fga else None

    @property
    def tp_pct(self) -> float | None:
        return self.tpm / self.tpa if self.tpa else None


@dataclass
class GameStats:
    teams: dict[str, Line]
    players: dict[tuple[str, int | None], Line]  # (team, jersey number) -> line
    possession_pct: dict[str, float]


def compute_stats(log: EventLog) -> GameStats:
    teams = {t: Line() for t in log.teams}
    players: dict[tuple[str, int | None], Line] = defaultdict(Line)

    for e in log.events:
        # Possession is per team (see possession_share); it gets no player line.
        if e.team not in teams or e.type not in ("shot", "rebound", "foul"):
            continue
        for line in (teams[e.team], players[(e.team, e.player)]):
            if e.type == "shot":
                made = bool(e.data.get("made"))
                three = e.data.get("points") == 3
                line.fga += 1
                line.fgm += made
                line.tpa += three
                line.tpm += made and three
                if made:
                    line.points += 3 if three else 2
            elif e.type == "rebound":
                line.rebounds += 1
            elif e.type == "foul":
                line.fouls += 1

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
