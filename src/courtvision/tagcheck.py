"""Check that CourtVision reproduces a game's exported stats from its tag log.

A game folder holds `*_log.csv` (tags), `*_stats.csv` (per-player and per-team totals
by quarter) and `*_boxscore.csv` (points by quarter). Parsing the log and computing
stats must give the same numbers as the two exports.

    python -m courtvision.tagcheck "path/to/Games/Bulls vs Jazz"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .events import EventLog
from .stats import GameStats, Line, compute_stats
from .tags import load_tags

# Export column -> Line attribute
STAT_COLUMNS = {
    "PTS": "points", "REB": "reb", "AST": "ast", "STL": "stl",
    "BLK": "blk", "TO": "tov", "PF": "pf", "SA": "missed",
}
Key = tuple[str, str, str, str]  # (scope, team, period, player)


def _player_name(player) -> str:
    return f"Player {player}" if isinstance(player, int) else str(player)


def _line_counts(line: Line) -> Counter:
    return Counter({col: getattr(line, attr) for col, attr in STAT_COLUMNS.items()})


def log_from_tags(log_path: str | Path) -> EventLog:
    events = load_tags(log_path)  # no roster: players stay as logged names
    teams = list(dict.fromkeys(e.team for e in events if e.team))
    return EventLog(teams=teams, duration=max((e.t for e in events), default=0.0),
                    events=events, source="tags")


def computed_stats(log: EventLog) -> dict[Key, Counter]:
    periods = sorted({e.data["quarter"] for e in log.events})
    out: dict[Key, Counter] = {}
    for period in [*periods, "Total"]:
        where = None if period == "Total" else (lambda e, p=period: e.data["quarter"] == p)
        stats: GameStats = compute_stats(log, where=where)
        for team, line in stats.teams.items():
            out[("Team", team, period, team)] = _line_counts(line)
        for (team, player), line in stats.players.items():
            out[("Player", team, period, _player_name(player))] = _line_counts(line)
    return out


def exported_stats(stats_path: str | Path) -> dict[Key, Counter]:
    """Read *_stats.csv, summing duplicate rows (e.g. two players with one name)."""
    out: dict[Key, Counter] = defaultdict(Counter)
    with open(stats_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["Scope"], row["Team"], row["Period"], row["Player"].strip())
            out[key].update({col: int(row[col]) for col in STAT_COLUMNS})
    return out


def compare_stats(computed: dict[Key, Counter], exported: dict[Key, Counter]) -> list[str]:
    problems = []
    for key in sorted(set(computed) | set(exported)):
        ours, theirs = computed.get(key, Counter()), exported.get(key, Counter())
        for col in STAT_COLUMNS:
            if ours[col] != theirs[col]:
                scope, team, period, player = key
                problems.append(
                    f"{scope} {team} {period} {player}: {col} computed {ours[col]}, "
                    f"export {theirs[col]}"
                )
    return problems


def compare_boxscore(log: EventLog, boxscore_path: str | Path) -> list[str]:
    computed = computed_stats(log)
    problems = []
    with open(boxscore_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            team = row["Team"]
            for period, value in row.items():
                if period == "Team":
                    continue
                ours = computed.get(("Team", team, period, team), Counter())["PTS"]
                if ours != int(value):
                    problems.append(f"Box score {team} {period}: computed {ours}, export {value}")
    return problems


def check_game(folder: str | Path) -> list[str]:
    folder = Path(folder)

    def find(suffix: str) -> Path | None:
        hits = sorted(folder.glob(f"*_{suffix}.csv"))
        return hits[0] if hits else None

    log_path = find("log")
    if log_path is None:
        raise FileNotFoundError(f"No *_log.csv in {folder}")
    log = log_from_tags(log_path)
    problems = []
    if stats_path := find("stats"):
        problems += compare_stats(computed_stats(log), exported_stats(stats_path))
    if box_path := find("boxscore"):
        problems += compare_boxscore(log, box_path)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", help="Game folders")
    args = parser.parse_args()
    failed = False
    for folder in args.folders:
        problems = check_game(folder)
        print(f"{folder}: {'OK' if not problems else f'{len(problems)} mismatches'}")
        for p in problems:
            print(f"  {p}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
