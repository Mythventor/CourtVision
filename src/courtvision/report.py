"""Turn stats into tables for the app and the downloadable report."""

from __future__ import annotations

from pathlib import Path

from .rosters import Roster
from .stats import GameStats, Line

PLAYER_HEADERS = ["Team", "Player", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "REB", "FOULS"]
TEAM_HEADERS = ["Team", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "REB", "FOULS", "POSS%"]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _line(line: Line) -> list:
    return [line.points, line.fgm, line.fga, _pct(line.fg_pct), line.tpm, line.tpa,
            line.rebounds, line.fouls]


def team_rows(stats: GameStats) -> list[list]:
    return [
        [team, *_line(line), _pct(stats.possession_pct.get(team))]
        for team, line in stats.teams.items()
    ]


def player_rows(
    stats: GameStats,
    rosters: dict[str, Roster],
    labels: dict[str, str] | None = None,
    team: str | None = None,
) -> list[list]:
    """One row per player, sorted by team then points. `labels` maps team -> color label."""
    labels = labels or {}
    rows = []
    for (t, number), line in stats.players.items():
        if team and t != team:
            continue
        roster = rosters.get(t, Roster(t))
        rows.append([t, roster.name_for(number, labels.get(t)), *_line(line)])
    return sorted(rows, key=lambda r: (r[0], -r[2]))


def write_pdf(stats: GameStats, rosters: dict[str, Roster], path: str | Path,
              team: str | None = None) -> Path:
    """Write the scouting report PDF for one team (or both if `team` is None).

    TODO (week 3): pick a PDF library (e.g. fpdf2 or reportlab) and lay out the report.
    """
    raise NotImplementedError("PDF report is not implemented yet")
