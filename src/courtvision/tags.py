"""Parse FAAM tag logs (`*_log.csv`) into CourtVision events.

Log format, one row per tag:

    Timestamp,Quarter,Team,Player,Stat,Change
    "10/17/2026, 11:29:30 AM",Q1,Bulls,Jack Lentz,PTS,+2

- Timestamps are wall-clock times when the tag was pressed (1 s resolution, a few
  seconds after the play). Pass `video_start` (wall-clock time of the first video frame)
  to get event times in video seconds.
- PTS +2 / +3 is a made basket, PTS +1 a made free throw.
- SA is a shot attempt that did not score (made baskets are logged only as PTS), so
  field-goal attempts = SA + made baskets. Misses do not say 2 or 3 points.
- A negative Change undoes the latest matching tag (e.g. PF -1 cancels a PF +1).
- Players are logged by name; "Player 34" means jersey #34 with no name.

Used as ground truth for evaluation, and optionally as an event source.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from .events import Event
from .rosters import Roster

TIMESTAMP_FORMAT = "%m/%d/%Y, %I:%M:%S %p"
COLUMNS = {"Timestamp", "Quarter", "Team", "Player", "Stat", "Change"}
_UNNAMED = re.compile(r"^Player\s+(\d+)$", re.IGNORECASE)

# Stat code -> event type, for stats counted once per +1
COUNT_STATS = {
    "SA": "shot",        # missed shot attempt
    "PF": "foul",
    "REB": "rebound",
    "AST": "assist",
    "STL": "steal",
    "BLK": "block",
    "TO": "turnover",
}


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)


def resolve_player(name: str, roster: Roster | None) -> int | str | None:
    """Jersey number if known, else the logged name."""
    name = name.strip()
    if not name:
        return None
    m = _UNNAMED.match(name)
    if m:
        return int(m.group(1))
    if roster:
        number = roster.number_for(name)
        if number is not None:
            return number
    return name


def _events_for_row(stat: str, amount: int, t: float, team: str,
                    player: int | str | None, data: dict) -> list[Event]:
    if stat == "PTS":
        if amount == 1:
            return [Event(t=t, type="free_throw", team=team, player=player, data=data)]
        if amount in (2, 3):
            return [Event(t=t, type="shot", team=team, player=player,
                          data={**data, "made": True, "points": amount})]
        raise ValueError(f"Unexpected PTS change {amount:+d}")
    type_ = COUNT_STATS.get(stat)
    if type_ is None:
        return []
    extra = {"made": False, "points": None} if stat == "SA" else {}
    return [Event(t=t, type=type_, team=team, player=player, data={**data, **extra})
            for _ in range(amount)]


def load_tags(
    path: str | Path,
    rosters: dict[str, Roster] | None = None,
    video_start: datetime | None = None,
) -> list[Event]:
    """Load a FAAM tag log.

    Event times are seconds since `video_start`, or since the first tag if not given.
    Each event's `data` keeps the quarter, the original stat code and the wall-clock time.
    """
    rosters = rosters or {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Tag log {path} is missing columns {sorted(missing)}")
        rows = list(reader)

    if not rows:
        return []
    origin = video_start or parse_timestamp(rows[0]["Timestamp"])
    events: list[Event] = []

    for row in rows:
        wall = parse_timestamp(row["Timestamp"])
        team = row["Team"].strip()
        stat = row["Stat"].strip().upper()
        change = int(row["Change"])
        player = resolve_player(row["Player"], rosters.get(team))

        if change < 0:
            _undo(events, team, player, stat, -change)
            continue

        data = {"quarter": row["Quarter"].strip(), "stat": stat, "wall": wall.isoformat()}
        t = (wall - origin).total_seconds()
        events.extend(_events_for_row(stat, change, t, team, player, data))

    return sorted(events, key=lambda e: e.t)


def _points(e: Event) -> int:
    return 1 if e.type == "free_throw" else e.data["points"]


def _set_points(e: Event, points: int) -> None:
    if points == 1:
        e.type = "free_throw"
        e.data.pop("made", None)
        e.data.pop("points", None)
    else:
        e.type = "shot"
        e.data.update(made=True, points=points)


def _undo(events: list[Event], team: str, player, stat: str, amount: int) -> None:
    """Apply a negative correction to the latest matching tag(s).

    The tagging app removes points one at a time, so PTS -1 twice undoes a +2 basket:
    PTS -N takes N points off the player's latest scoring tags. Other stats -N remove
    the latest N tags.
    """
    for _ in range(amount):
        for i in range(len(events) - 1, -1, -1):
            e = events[i]
            if (e.team, e.player, e.data.get("stat")) == (team, player, stat):
                break
        else:
            raise ValueError(f"Correction {stat} -{amount} for {team} {player} has no tag to undo")
        if stat == "PTS" and _points(e) > 1:
            _set_points(e, _points(e) - 1)
        else:
            del events[i]
