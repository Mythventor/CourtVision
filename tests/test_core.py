from datetime import datetime
from pathlib import Path

import pytest

from courtvision.events import Event, EventLog
from courtvision.jersey import JerseyVoter
from courtvision.report import player_rows, team_rows
from courtvision.rosters import Roster, load_rosters
from courtvision.stats import compute_stats
from courtvision.tagcheck import check_game
from courtvision.tags import load_tags, parse_timestamp

ROOT = Path(__file__).resolve().parents[1]
ROSTERS = ROOT / "examples" / "rosters" / "rosters.csv"
GAME = ROOT / "examples" / "games" / "sample"
LOG = GAME / "sample_log.csv"

LOG_HEADER = "Timestamp,Quarter,Team,Player,Stat,Change\n"


def write_log(tmp_path, *rows: str) -> Path:
    p = tmp_path / "x_log.csv"
    p.write_text(LOG_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return p


# --- rosters ---------------------------------------------------------------

def test_roster_lookup_and_fallbacks():
    eagles = load_rosters(ROSTERS)["Eagles"]
    assert eagles.name_for(23) == "Sample Player C"
    assert eagles.name_for(99, label="Blue") == "Blue #99"
    assert eagles.name_for(None, label="Blue") == "Blue #?"
    assert eagles.name_for("Some Name") == "Some Name"


def test_roster_without_header_and_bad_rows(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("A,5,Ann Lee\nA,,No Number\nA,5,Dup Five\nB,7,Bo\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        rosters = load_rosters(p)
    assert rosters["A"].players == {5: "Ann Lee"}
    assert rosters["B"].players == {7: "Bo"}


def test_number_for_full_name_last_name_and_ambiguous():
    r = Roster("A", {1: "Ann Lee", 2: "Bo Kim", 3: "Cy Kim"})
    assert r.number_for("ann lee") == 1
    assert r.number_for("Lee") == 1
    assert r.number_for("Kim") is None  # two players with that last name
    assert r.number_for("Nobody") is None


# --- tag logs --------------------------------------------------------------

def test_parse_timestamp_across_noon():
    assert parse_timestamp("10/17/2026, 12:00:05 PM") == datetime(2026, 10, 17, 12, 0, 5)
    assert parse_timestamp("10/17/2026, 11:59:59 AM") == datetime(2026, 10, 17, 11, 59, 59)


def test_load_tags_sample():
    events = load_tags(LOG, rosters=load_rosters(ROSTERS))
    assert events[0].type == "shot" and events[0].player == 23  # name -> jersey number
    assert events[0].data["made"] and events[0].data["points"] == 2
    assert events[0].t == 0.0
    by_type = [(e.type, e.player) for e in events]
    assert ("free_throw", 15) in by_type
    assert ("shot", 34) in by_type          # "Player 34"
    assert all(e.player != 11 for e in events)  # +2 undone by two PTS -1


def test_video_start_offset(tmp_path):
    p = write_log(tmp_path, '"10/17/2026, 11:29:30 AM",Q1,A,X,PTS,+2')
    events = load_tags(p, video_start=datetime(2026, 10, 17, 11, 20, 0))
    assert events[0].t == 570.0


def test_pts_correction_partial(tmp_path):
    # A +3 with one PTS -1 becomes a +2
    p = write_log(
        tmp_path,
        '"10/17/2026, 11:00:00 AM",Q1,A,X,PTS,+3',
        '"10/17/2026, 11:00:05 AM",Q1,A,X,PTS,-1',
    )
    (e,) = load_tags(p)
    assert e.type == "shot" and e.data["points"] == 2


def test_correction_without_tag_fails(tmp_path):
    p = write_log(tmp_path, '"10/17/2026, 11:00:00 AM",Q1,A,X,PF,-1')
    with pytest.raises(ValueError, match="no tag to undo"):
        load_tags(p)


def test_sample_game_matches_exports():
    assert check_game(GAME) == []


def test_tagcheck_reports_mismatch(tmp_path):
    for name in ("sample_log.csv", "sample_boxscore.csv"):
        (tmp_path / name).write_bytes((GAME / name).read_bytes())
    (tmp_path / "sample_boxscore.csv").write_text("Team,Q1,Q2,Total\nEagles,2,0,3\n")
    assert check_game(tmp_path) == ["Box score Eagles Total: computed 2, export 3"]


# --- stats -----------------------------------------------------------------

def test_stats_from_events():
    log = EventLog(
        teams=["A", "B"],
        duration=100.0,
        events=[
            Event(t=0, type="possession", team="A"),
            Event(t=10, type="shot", team="A", player=5, data={"made": True, "points": 3}),
            Event(t=20, type="shot", team="A", player=5, data={"made": False, "points": None}),
            Event(t=22, type="free_throw", team="A", player=5),
            Event(t=25, type="possession", team="B"),
            Event(t=30, type="rebound", team="B", player=7),
            Event(t=40, type="foul", team="A", player=5),
        ],
    )
    stats = compute_stats(log)
    a5 = stats.players[("A", 5)]
    assert (a5.points, a5.fgm, a5.fga, a5.tpm, a5.ftm, a5.pf, a5.missed) == (4, 1, 2, 1, 1, 1, 1)
    assert stats.teams["B"].reb == 1
    assert stats.possession_pct == pytest.approx({"A": 0.25, "B": 0.75})

    rows = player_rows(stats, {"A": Roster("A", {5: "Five"})}, labels={"B": "Blue"})
    assert [r[1] for r in rows] == ["Five", "Blue #7"]
    assert team_rows(stats)[0][:2] == ["A", 4]


def test_no_possession_data_shows_dash():
    log = EventLog(teams=["A", "B"], duration=10.0,
                   events=[Event(t=1, type="shot", team="A", data={"made": True, "points": 2})])
    assert team_rows(compute_stats(log))[0][-1] == "-"


def test_event_log_round_trip(tmp_path):
    log = EventLog(teams=["A", "B"], duration=5.0,
                   events=[Event(t=1.0, type="shot", team="A", player="Some Name",
                                 data={"made": True, "points": 2})])
    path = tmp_path / "log.json"
    log.save(path)
    assert EventLog.load(path) == log


# --- report ----------------------------------------------------------------

def test_pdf_report_and_highlights(tmp_path):
    pytest.importorskip("fpdf")
    from courtvision.report import highlights, write_pdf

    rosters = load_rosters(ROSTERS)
    log = EventLog(teams=["Eagles", "Hawks"], duration=0.0,
                   events=load_tags(LOG, rosters=rosters), source="tags")
    stats = compute_stats(log)
    notes = highlights(stats, rosters, "Hawks")
    assert notes[0] == "Top scorer: Sample Player F (4 pts)"

    for team in (None, "Hawks"):
        pdf = write_pdf(stats, rosters, tmp_path / f"r_{team}.pdf", team=team)
        assert pdf.read_bytes().startswith(b"%PDF")


def test_config_override_merges(tmp_path):
    from courtvision.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("output_dir: /x\nrender:\n  width: 640\n", encoding="utf-8")
    config = load_config(p)
    assert config["output_dir"] == "/x"
    assert config["render"]["width"] == 640
    assert config["render"]["panel_width"] == 480  # untouched default kept


# --- jersey numbers --------------------------------------------------------

def test_jersey_voter_needs_min_votes():
    voter = JerseyVoter(min_votes=2)
    for n in (23, 23, 28, None):
        voter.add(1, n)
    voter.add(2, 4)
    assert voter.assignments() == {1: 23, 2: None}
