# CourtVision

Course project for CS 496 (Sport Video), Northwestern University.

CourtVision is a basketball video analytics tool for FAAM youth league games. Upload a
game video, pick the two teams, and get team and per-player stats plus a downloadable
report. Players are identified by jersey number and matched to team rosters; players
not on the roster show as e.g. "Blue #23".

## How it works

```
video ──► detect + track (GPU) ──► team colors, jersey numbers ──► event log (JSON) ──► stats / report
                                                                        ▲
                                         tag CSV (optional) ────────────┘  also used as ground truth
```

The pipeline's only output is an **event log** (possession, shot, rebound, foul events).
Stats, the app tables and the report are all computed from it, so stat logic can change
without re-processing video.

## Layout

| Path | Purpose |
|---|---|
| `src/courtvision/video.py` | Read frames with a stride |
| `src/courtvision/detection.py` | Roboflow detector (players, ball, hoop, referees) |
| `src/courtvision/tracking.py` | ByteTrack player tracking |
| `src/courtvision/teams.py` | Split tracks into two teams by jersey color |
| `src/courtvision/jersey.py` | Jersey number reading and per-track voting |
| `src/courtvision/possession.py` | Tracks → possession / shot / rebound events |
| `src/courtvision/pipeline.py` | Runs the stages above |
| `src/courtvision/events.py` | Event log format (save/load JSON) |
| `src/courtvision/tags.py` | Parse tag CSV exports |
| `src/courtvision/rosters.py` | Roster CSV and name lookup |
| `src/courtvision/stats.py` | Box-score stats from the event log |
| `src/courtvision/report.py` | Tables and PDF report |
| `src/courtvision/app.py` | Gradio web app |
| `configs/default.yaml` | Model id, thresholds, tag CSV column mapping |
| `notebooks/colab_app.ipynb` | Set up and run the app on Colab |
| `examples/` | Fake rosters and tags for testing |

## Status

- Working: rosters, tag CSV parsing, event log, stats, app in **tags-only mode** (upload a tag CSV, no video).
- TODO: pick a detection model (`detection.model_id`), jersey number reading, event detection
  (`possession.py`), team-color confirmation step in the app, PDF report.

## Run on Colab

Open `notebooks/colab_app.ipynb` in Colab with a T4 GPU runtime and run the cells. Add
`ROBOFLOW_API_KEY`, `COURTVISION_PASSWORD` and (for a private repo) `GITHUB_TOKEN` as
Colab secrets. The app prints a temporary share link protected by the password.

## Local development

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -e ".[dev,app]"
pytest
python -m courtvision.app     # run from the repo root
```

## Data and privacy

The videos show minors. Real videos, rosters and tag files stay on Google Drive and are
never committed: `data/`, `outputs/`, `rosters/`, `tags/` and video files are gitignored.
Only the fake files in `examples/` are tracked.

## Roster and tag formats

Rosters (one CSV for all teams):

```csv
team,number,name
Eagles,23,Jane Doe
```

Tags: column names and event names are mapped in `configs/default.yaml` under `tags:`.
Update that mapping to match the real export.
