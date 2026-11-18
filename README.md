# CourtVision

Course project for CS 496 (Sport Video), Northwestern University.

CourtVision is a basketball video analytics tool for FAAM youth league games. The web
app has two parts:

- **Video tracking:** pick a quarter video (from Google Drive on Colab), track players and
  the ball, and play it back with boxes around each player and a live 2D court panel.
- **Stats & report:** team and per-player stats from a game's tag log, with a
  downloadable PDF report.

Players are identified by jersey number and matched to team rosters; players not on the
roster show as e.g. "Blue #23".

## How it works

```
video ──► detect + track + court keypoints (GPU) ──► tracking cache ──► annotated video + 2D panel
                                                          │
                                                          └──► event log (JSON) ──► stats / report
tag log ──────────────────────────────────────────────────────────────▲   also used as ground truth
```

Each video is processed once; the tracking cache (`tracks.json.gz`) is saved next to the
annotated video, so playback, rendering and event detection never need the GPU pass again.
Stats and the report are computed from an **event log**, so stat logic can change without
re-processing video.

## Layout

| Path | Purpose |
|---|---|
| `src/courtvision/library.py` | Find game videos in the video folder |
| `src/courtvision/video.py` | Read frames with a stride |
| `src/courtvision/detection.py` | Roboflow detector (players, ball, hoop, referees) |
| `src/courtvision/tracking.py` | ByteTrack player tracking |
| `src/courtvision/court.py` | Court keypoints → homography, 2D court drawing |
| `src/courtvision/teams.py` | Split tracks into two teams by jersey color |
| `src/courtvision/jersey.py` | Jersey number reading and per-track voting |
| `src/courtvision/pipeline.py` | Tracking pass and its saved cache |
| `src/courtvision/render.py` | Annotated video with the 2D court panel |
| `src/courtvision/processing.py` | Process a video once, reuse saved results |
| `src/courtvision/preview.py` | One-frame model previews for setup |
| `src/courtvision/possession.py` | Tracks → possession / shot / rebound events |
| `src/courtvision/events.py` | Event log format (save/load JSON) |
| `src/courtvision/tags.py` | Parse tag logs |
| `src/courtvision/tagcheck.py` | Check parsed tags against a game's exported stats |
| `src/courtvision/rosters.py` | Roster CSV and name lookup |
| `src/courtvision/stats.py` | Box-score stats from the event log |
| `src/courtvision/report.py` | Tables and PDF report |
| `src/courtvision/app.py` | Gradio web app |
| `configs/default.yaml` | Models, court, rendering and path settings |
| `notebooks/colab_app.ipynb` | Set up, check the models, and run the app on Colab |
| `examples/` | Fake rosters and a fake game for testing |

## Status

- Working: rosters, tag log parsing (checked against the exported stats of all four games),
  stats, PDF report, video tracking with annotated playback and the 2D court panel
  (tested on a synthetic clip; not yet run on real footage).
- To set up on real footage: confirm the detection model's class names, and fill in
  `court.template` for the court keypoint model (both from the notebook previews).
- TODO: jersey number reading, event detection from video (`possession.py`), naming
  the two team colors in the app.

## Run on Colab

Open `notebooks/colab_app.ipynb` in Colab with a T4 GPU runtime and run the cells. Add
`ROBOFLOW_API_KEY` and `COURTVISION_PASSWORD` as Colab secrets. The notebook creates a
config at `MyDrive/CS496/courtvision/config.yaml` (the `DRIVE` folder set in the notebook) that points at your Drive folders; it only
needs the settings that differ from `configs/default.yaml`. The app prints a temporary
share link protected by the password.

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

Rosters: one CSV for all teams, columns `team,number,name`; the header row is optional.
Rows with a missing or duplicate jersey number are skipped with a warning.

Tag logs (`*_log.csv`, exported by the tagging app):

```csv
Timestamp,Quarter,Team,Player,Stat,Change
"10/17/2026, 11:29:30 AM",Q1,Bulls,Jack Lentz,PTS,+2
```

- `PTS` +2/+3 is a made basket, +1 a made free throw. `SA` is a shot that did not score,
  so field-goal attempts = SA + made baskets. Misses do not say 2 or 3 points.
- A negative `Change` is a correction. Points are removed one at a time, so two `PTS -1`
  rows undo a +2.
- Timestamps are wall-clock times. Set `tags.video_start` in the config to the time of the
  video's first frame to put tags on the video timeline.
- Players are matched to roster numbers by full name, then by last name; "Player 34"
  means jersey #34.

Check that a game's log reproduces its exported stats and box score:

```bash
python -m courtvision.tagcheck "path/to/Games/Bulls vs Jazz"
```
