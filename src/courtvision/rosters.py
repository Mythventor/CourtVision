"""Team rosters: map (team, jersey number) to player names.

Roster CSV format (one file, all teams):

    team,number,name
    Eagles,23,Jane Doe
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Roster:
    team: str
    players: dict[int, str] = field(default_factory=dict)  # jersey number -> name

    def name_for(self, player: int | str | None, label: str | None = None) -> str:
        """Display name for a player (jersey number, or a name from a tag log).

        Falls back to "<label> #<number>" (e.g. "Blue #23") when the number is not on
        the roster, and "<label> #?" when the number could not be read. `label`
        defaults to the team name; pass the jersey color to get "Blue #23".
        """
        label = label or self.team
        if player is None:
            return f"{label} #?"
        if isinstance(player, str):
            return player
        return self.players.get(player, f"{label} #{player}")

    def number_for(self, name: str) -> int | None:
        """Jersey number for a name, or None if absent or ambiguous.

        Matches the full name first, then the last name alone (tag logs sometimes use
        only last names). Case-insensitive.
        """
        key = name.strip().casefold()
        hits = [n for n, full in self.players.items() if full.casefold() == key]
        if not hits and " " not in key:
            hits = [n for n, full in self.players.items() if full.casefold().split()[-1:] == [key]]
        return hits[0] if len(hits) == 1 else None


HEADER = ["team", "number", "name"]


def load_rosters(path: str | Path) -> dict[str, Roster]:
    """Load all rosters from a CSV, keyed by team name.

    Columns are team, number, name. The header row is optional (the FAAM roster
    export has none).
    """
    rosters: dict[str, Roster] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if rows and [c.strip().lower() for c in rows[0]] == HEADER:
        rows = rows[1:]
    for line, row in enumerate(rows, start=1):
        if len(row) < 3:
            raise ValueError(f"Roster CSV {path} row {line}: expected team,number,name")
        team, number, name = (c.strip() for c in row[:3])
        roster = rosters.setdefault(team, Roster(team))
        # Real rosters have gaps and typos: skip bad rows with a warning, don't fail.
        if not number.isdigit():
            warnings.warn(f"Roster {team!r}: skipping {name!r}, jersey number {number!r} is not a number")
            continue
        if int(number) in roster.players:
            warnings.warn(
                f"Roster {team!r}: jersey #{number} is listed for both "
                f"{roster.players[int(number)]!r} and {name!r}; keeping the first"
            )
            continue
        roster.players[int(number)] = name
    return rosters
