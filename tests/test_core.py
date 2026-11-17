from pathlib import Path

import pytest

from courtvision.config import load_config
from courtvision.events import Event, EventLog
from courtvision.jersey import JerseyVoter
from courtvision.report import player_rows, team_rows
from courtvision.rosters import Roster, load_rosters
from courtvision.stats import compute_stats
from courtvision.tags import load_tags, parse_time

ROOT = Path(__file__).resolve().parents[1]
ROSTERS = ROOT / "examples" / "rosters" / "rosters.csv"
TAGS = ROOT / "examples" / "tags" / "sample_tags.csv"


def test_roster_lookup_and_fallbacks():
    rosters = load_rosters(ROSTERS)
    eagles = rosters["Eagles"]
    assert eagles.name_for(23) == "Sample Player C"
    assert eagles.name_for(99, label="Blue") == "Blue #99"
    assert eagles.name_for(None, label="Blue") == "Blue #?"


def test_roster_rejects_duplicate_numbers(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("team,number,name\nA,5,x\nA,5,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_rosters(p)


@pytest.mark.parametrize("raw,expected", [("12.5", 12.5), ("01:05.2", 65.2), ("1:00:00", 3600)])
def test_parse_time(raw, expected):
    assert parse_time(raw) == pytest.approx(expected)


def test_load_tags_from_example():
    events = load_tags(TAGS, load_config()["tags"])
    assert [e.type for e in events] == ["shot", "shot", "foul", "shot"]
    assert events[0].team == "Eagles" and events[0].player == 23
    assert events[0].data == {"made": True, "points": 2}


def test_stats_from_events():
    log = EventLog(
        teams=["A", "B"],
        duration=100.0,
        events=[
            Event(t=0, type="possession", team="A"),
            Event(t=10, type="shot", team="A", player=5, data={"made": True, "points": 3}),
            Event(t=20, type="shot", team="A", player=5, data={"made": False, "points": 2}),
            Event(t=25, type="possession", team="B"),
            Event(t=30, type="rebound", team="B", player=7),
            Event(t=40, type="foul", team="A", player=5),
        ],
    )
    stats = compute_stats(log)
    a5 = stats.players[("A", 5)]
    assert (a5.points, a5.fgm, a5.fga, a5.tpm, a5.tpa, a5.fouls) == (3, 1, 2, 1, 1, 1)
    assert stats.teams["B"].rebounds == 1
    assert stats.possession_pct == pytest.approx({"A": 0.25, "B": 0.75})

    rows = player_rows(stats, {"A": Roster("A", {5: "Five"})}, labels={"B": "Blue"})
    assert [r[1] for r in rows] == ["Five", "Blue #7"]
    assert team_rows(stats)[0][:2] == ["A", 3]


def test_event_log_round_trip(tmp_path):
    log = EventLog(teams=["A", "B"], duration=5.0,
                   events=[Event(t=1.0, type="shot", team="A", data={"made": True})])
    path = tmp_path / "log.json"
    log.save(path)
    assert EventLog.load(path) == log


def test_jersey_voter_needs_min_votes():
    voter = JerseyVoter(min_votes=2)
    for n in (23, 23, 28, None):
        voter.add(1, n)
    voter.add(2, 4)
    assert voter.assignments() == {1: 23, 2: None}
