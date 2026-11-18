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


def highlights(stats: GameStats, rosters: dict[str, Roster], team: str) -> list[str]:
    """Short, positive notes on a team's standout players."""
    roster = rosters.get(team, Roster(team))
    players = [(roster.name_for(p), line) for (t, p), line in stats.players.items() if t == team]
    notes = []

    def best(key, label, fmt, min_fga=0):
        pool = [(n, l) for n, l in players if l.fga >= min_fga and key(l)]
        if pool:
            name, line = max(pool, key=lambda nl: key(nl[1]))
            notes.append(f"{label}: {name} ({fmt(line)})")

    best(lambda l: l.points, "Top scorer", lambda l: f"{l.points} pts")
    best(lambda l: l.fgm, "Most baskets", lambda l: f"{l.fgm} of {l.fga} shots")
    best(lambda l: l.fg_pct or 0, "Most efficient shooter (4+ shots)",
         lambda l: f"{l.fgm}/{l.fga}, {_pct(l.fg_pct)}", min_fga=4)
    best(lambda l: l.tpm, "Three-pointers", lambda l: f"{l.tpm} made")
    best(lambda l: l.ftm, "Free throws", lambda l: f"{l.ftm} made")
    best(lambda l: l.fga, "Most shots taken", lambda l: f"{l.fga} attempts")
    return notes


def _latin1(text: str) -> str:
    """The built-in PDF fonts only cover Latin-1."""
    return str(text).replace("–", "-").replace("—", "-").encode(
        "latin-1", "replace").decode("latin-1")


def write_pdf(
    stats: GameStats,
    rosters: dict[str, Roster],
    path: str | Path,
    team: str | None = None,
    source: str = "tags",
) -> Path:
    """Write the game report PDF for one team (or both teams if `team` is None)."""
    from fpdf import FPDF

    teams = list(stats.teams)
    pdf = FPDF(orientation="L", format="Letter")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "CourtVision game report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    score = "   ".join(f"{t} {stats.teams[t].points}" for t in teams)
    pdf.cell(0, 8, _latin1(score), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    origin = "the game's tag log" if source == "tags" else "video analysis"
    pdf.cell(0, 6, f"Stats from {origin}.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    def table(headers: list[str], rows: list[list], first_width: float) -> None:
        pdf.set_font("Helvetica", "", 9)
        rest = (pdf.epw - first_width) / (len(headers) - 1)
        with pdf.table(col_widths=(first_width, *[rest] * (len(headers) - 1)),
                       text_align=("LEFT", *["CENTER"] * (len(headers) - 1)),
                       line_height=6) as t:
            for row in [headers, *rows]:
                r = t.row()
                for value in row:
                    r.cell(_latin1(value))
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Team stats", new_x="LMARGIN", new_y="NEXT")
    table(TEAM_HEADERS, team_rows(stats), 40)

    for t in ([team] if team else teams):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _latin1(f"{t} players"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for note in highlights(stats, rosters, t):
            pdf.cell(0, 6, _latin1(f"- {note}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        rows = [r[1:] for r in player_rows(stats, rosters, team=t)]
        table(PLAYER_HEADERS[1:], rows, 55)

    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "FGA = made baskets + missed shots. Tagged misses do not say 2 or 3 "
                         "points, so three-point attempts are not shown. REB, AST, STL, BLK and "
                         "TO are only counted when they were tagged.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path
