"""Team rosters: map (team, jersey number) to player names.

Roster CSV format (one file, all teams):

    team,number,name
    Eagles,23,Jane Doe
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Roster:
    team: str
    players: dict[int, str] = field(default_factory=dict)  # jersey number -> name

    def name_for(self, number: int | None, label: str | None = None) -> str:
        """Display name for a jersey number.

        Falls back to "<label> #<number>" (e.g. "Blue #23") when the number is not on
        the roster, and "<label> #?" when the number could not be read. `label`
        defaults to the team name; pass the jersey color to get "Blue #23".
        """
        label = label or self.team
        if number is None:
            return f"{label} #?"
        return self.players.get(number, f"{label} #{number}")


def load_rosters(path: str | Path) -> dict[str, Roster]:
    """Load all rosters from a CSV, keyed by team name."""
    rosters: dict[str, Roster] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"team", "number", "name"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Roster CSV {path} is missing columns: {sorted(missing)}")
        for row in reader:
            team = row["team"].strip()
            number = int(row["number"])
            roster = rosters.setdefault(team, Roster(team))
            if number in roster.players:
                raise ValueError(f"Duplicate jersey #{number} on team {team!r}")
            roster.players[number] = row["name"].strip()
    return rosters
