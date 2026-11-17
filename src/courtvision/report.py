"""Turn stats into tables for the app and the downloadable report."""

from __future__ import annotations

from pathlib import Path

from .rosters import Roster
from .stats import GameStats, Line

_LINE_HEADERS = ["PTS", "FGM", "FGA", "FG%", "3PM", "FTM", "REB", "AST", "STL", "BLK", "TO", "PF"]
PLAYER_HEADERS = ["Team", "Player", *_LINE_HEADERS]
TEAM_HEADERS = ["Team", *_LINE_HEADERS, "POSS%"]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _line(line: Line) -> list:
    return [line.points, line.fgm, line.fga, _pct(line.fg_pct), line.tpm, line.ftm,
            line.reb, line.ast, line.stl, line.blk, line.tov, line.pf]


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
    for (t, player), line in stats.players.items():
        if team and t != team:
            continue
        roster = rosters.get(t, Roster(t))
        rows.append([t, roster.name_for(player, labels.get(t)), *_line(line)])
    return sorted(rows, key=lambda r: (r[0], -r[2]))


def write_pdf(stats: GameStats, rosters: dict[str, Roster], path: str | Path,
              team: str | None = None) -> Path:
    """Write the scouting report PDF for one team (or both if `team` is None).

    TODO (week 3): pick a PDF library (e.g. fpdf2 or reportlab) and lay out the report.
    """
    raise NotImplementedError("PDF report is not implemented yet")
