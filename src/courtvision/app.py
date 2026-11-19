"""Gradio web app.

- Video tracking: pick a quarter video from the video folder (Drive on Colab), track
  players and the ball, and play it back with boxes and a 2D court panel.
- Stats & report: stats from a game's tag log, with a downloadable PDF report.

Run on Colab:  python -m courtvision.app --share --config <your config>
Set COURTVISION_PASSWORD to require a login (recommended with --share).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

from .config import load_config
from .events import EventLog
from .library import list_videos
from .report import PLAYER_HEADERS, TEAM_HEADERS, player_rows, team_rows, write_pdf
from .rosters import load_rosters
from .stats import compute_stats
from .tags import load_tags


def video_start(config) -> datetime | None:
    """`tags.video_start` from the config (YAML may already parse it as a datetime)."""
    value = config.get("tags", {}).get("video_start")
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def tag_stats(tags_csv, team_a, team_b, config, rosters):
    """Stats from a tag log for the two chosen teams."""
    import gradio as gr

    if not tags_csv:
        raise gr.Error("Upload the game's tag log (*_log.csv).")
    if not team_a or not team_b or team_a == team_b:
        raise gr.Error("Pick two different teams.")
    events = load_tags(tags_csv, rosters=rosters, video_start=video_start(config))
    logged = {e.team for e in events}
    if not {team_a, team_b} <= logged:
        raise gr.Error(f"The tag log has teams {sorted(logged)}; pick those.")
    log = EventLog(teams=[team_a, team_b], duration=max((e.t for e in events), default=0.0),
                   events=events, source="tags")
    return compute_stats(log)


def summary_markdown(result) -> str:
    s = result.tracking.stats
    lines = [
        f"**Saved to** `{result.folder}`" + (" (reused saved tracks)" if result.reused else ""),
        f"- Analyzed frames: {s.get('frames')} in {s.get('seconds')} s "
        f"({s.get('frames_per_second')} frames/s)",
        f"- Player tracks: {s.get('tracks')}",
        f"- Ball found in {s.get('ball_found', 0):.0%} of analyzed frames",
        f"- Court mapped in {s.get('court_mapped', 0):.0%} of analyzed frames"
        + ("" if s.get("court_mapped") else
           " (set up the court for this video under *Court setup* to get the 2D panel)"),
        f"- Off-court detections dropped (spectators, bench, walls): {s.get('off_court_dropped', 0)}",
    ]
    if result.tracking.video != result.tracking.source:
        lines.append("- Analyzed a 720p copy of the selected range (saved with the results)")
    return "\n".join(lines)


def build_app(config_path: str | None = None):
    import gradio as gr

    config = load_config(config_path)
    try:
        rosters = load_rosters(config["roster_csv"])
    except FileNotFoundError:
        warnings.warn(f"Roster file {config['roster_csv']} not found; names will not be shown")
        rosters = {}
    team_names = sorted(rosters)

    # Listing Drive folders is slow, so list once and again only on "Refresh list"
    cache: dict[str, list[Path]] = {}

    def library(refresh: bool = False):
        if refresh or not cache:
            cache.clear()
            cache.update(list_videos(config["video_root"], exclude=config["output_dir"]))
        return cache

    def files_in(game):
        """(label with size, file name) pairs for the video dropdown."""
        choices = []
        for p in library().get(game, []):
            try:
                size = f" ({p.stat().st_size / 1e9:.1f} GB)"
            except OSError:
                size = ""
            choices.append((p.name + size, p.name))
        return choices

    with gr.Blocks(title="CourtVision") as demo:
        gr.Markdown("# CourtVision")

        with gr.Tab("Video tracking"):
            games = sorted(library())
            first = files_in(games[0]) if games else []
            with gr.Row():
                game = gr.Dropdown(games, label="Day folder", value=games[0] if games else None)
                quarter = gr.Dropdown(first, label="Video (quarter or full game)",
                                      value=first[0][1] if first else None)
                refresh = gr.Button("Refresh list", size="sm")
            if not games:
                gr.Markdown(f"No videos found in `{config['video_root']}`. "
                            "Set `video_root` in your config.")
            clip = gr.Video(label="...or upload a short clip instead", sources=["upload"])

            with gr.Accordion("Court setup for the 2D panel (once per video)", open=False):
                gr.Markdown(
                    "1. Pick the court type and a frame time where the court lines are easy to "
                    "see, then **Load frame**.\n"
                    "2. Choose a landmark, then click exactly where it is on the floor. "
                    "Left/Right and far/near are as seen from the camera.\n"
                    "3. After 4+ points (spread out, not all on one line) the court is drawn in "
                    "green. When the green lines sit on the real lines, **Save**.\n\n"
                    "The video is then tracked with this setup; camera pans are followed "
                    "automatically.")
                with gr.Row():
                    cal_t = gr.Number(60, label="Frame time (seconds into the video)", minimum=0)
                    standard = gr.Radio(
                        [("High school (84 x 50 ft, 12 ft lane)", "high_school"),
                         ("NBA (94 x 50 ft, 16 ft lane)", "nba")],
                        value=config["court"].get("standard", "high_school"), label="Court type")
                    load_frame = gr.Button("Load frame")
                landmark = gr.Dropdown([], label="Landmark to click next")
                cal_image = gr.Image(label="Click the landmark on the frame", type="numpy",
                                     interactive=False)
                cal_status = gr.Markdown()
                with gr.Row():
                    undo_btn = gr.Button("Undo last point", size="sm")
                    clear_btn = gr.Button("Start over", size="sm")
                    save_btn = gr.Button("Save court setup", variant="primary", size="sm")
                cal_state = gr.State(None)

            with gr.Row():
                start_min = gr.Number(0, label="Start (minutes into the video)", minimum=0)
                end_min = gr.Number(0, label="End (minutes; 0 = end of video)", minimum=0)
                reprocess = gr.Checkbox(label="Reprocess (ignore saved tracks)")
            process = gr.Button("Track players and ball", variant="primary")
            output_video = gr.Video(label="Tracking + 2D court")
            summary = gr.Markdown()

            def on_game(g):
                choices = files_in(g)
                return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

            def on_refresh():
                gs = sorted(library(refresh=True))
                return gr.Dropdown(choices=gs, value=gs[0] if gs else None)

            def selected_video(g, q, clip_path) -> Path:
                if clip_path:
                    return Path(clip_path)
                match = [p for p in library().get(g, []) if p.name == q]
                if not match:
                    raise gr.Error("Pick a video (or upload a clip).")
                return match[0]

            # --- court setup ---------------------------------------------------
            def cal_view(session, note=""):
                names = session.landmark_names()
                return (session.image_rgb(), (note + "\n\n" if note else "") + session.status(),
                        session, gr.Dropdown(choices=names, value=session.next_landmark()))

            def on_load_frame(g, q, clip_path, t, std):
                from .calibration import CalibrationSession

                video = selected_video(g, q, clip_path)
                try:
                    session, resumed = CalibrationSession.start(video, t or 0, std,
                                                                config["output_dir"])
                except (ValueError, OSError) as e:
                    raise gr.Error(str(e))
                note = (f"Loaded the saved court setup for **{video.name}** (frame at "
                        f"{session.t:.0f} s). Click to adjust, or **Start over**."
                        if resumed else f"Frame at {session.t:.0f} s of **{video.name}**.")
                return cal_view(session, note)

            def require(session):
                if session is None:
                    raise gr.Error("Load a frame first.")
                return session

            def on_click(session, lm, evt: gr.SelectData):
                session = require(session)
                if not lm:
                    raise gr.Error("Choose the landmark you are clicking.")
                x, y = evt.index
                session.click(lm, x, y)
                return cal_view(session)

            # Gradio passes the click position only if it sees the gr.SelectData type hint.
            # `from __future__ import annotations` (top of this file) turns hints into
            # strings, which Gradio does not recognize, so set the real type here.
            on_click.__annotations__["evt"] = gr.SelectData

            def on_undo(session):
                session = require(session)
                session.undo()
                return cal_view(session)

            def on_clear(session):
                session = require(session)
                session.points = []
                return cal_view(session, "Cleared. A saved setup stays until you save again.")

            def on_standard(session, std):
                if session is None:
                    return gr.update(), gr.update(), session, gr.update()
                session.standard = std   # points keep their landmark names
                return cal_view(session)

            def on_save(session):
                session = require(session)
                try:
                    path = session.save(config["output_dir"])
                except ValueError as e:
                    raise gr.Error(str(e))
                return cal_view(session, f"Saved to `{path}`. Track the video now; it will use "
                                         "this setup (saved tracks are redone automatically).")

            cal_outputs = [cal_image, cal_status, cal_state, landmark]
            load_frame.click(on_load_frame, [game, quarter, clip, cal_t, standard], cal_outputs)
            cal_image.select(on_click, [cal_state, landmark], cal_outputs)
            undo_btn.click(on_undo, cal_state, cal_outputs)
            clear_btn.click(on_clear, cal_state, cal_outputs)
            standard.change(on_standard, [cal_state, standard], cal_outputs)
            save_btn.click(on_save, cal_state, cal_outputs)

            def on_process(g, q, clip_path, start, end, again, progress=gr.Progress()):
                from .processing import process_video

                path = selected_video(g, q, clip_path)
                end_s = (end or 0) * 60 or None
                if end_s is not None and end_s <= (start or 0) * 60:
                    raise gr.Error("End must be after start.")
                try:
                    result = process_video(
                        path, config, start_s=(start or 0) * 60, end_s=end_s, reprocess=again,
                        progress=lambda p, msg: progress(p, desc=msg),
                    )
                except (ValueError, FileNotFoundError, NotImplementedError) as e:
                    raise gr.Error(str(e))
                return str(result.video), summary_markdown(result)

            game.change(on_game, game, quarter)
            refresh.click(on_refresh, None, game)
            process.click(on_process, [game, quarter, clip, start_min, end_min, reprocess],
                          [output_video, summary])

        with gr.Tab("Stats & report"):
            gr.Markdown("Stats from a game's tag log (the `*_log.csv` export).")
            tags_csv = gr.File(label="Tag log CSV", file_types=[".csv"], type="filepath")
            with gr.Row():
                # Custom values allowed: some teams (e.g. Lakers, Suns) are not in the roster
                team_a = gr.Dropdown(team_names, label="Team A", allow_custom_value=True)
                team_b = gr.Dropdown(team_names, label="Team B", allow_custom_value=True)
                report_for = gr.Radio(["Both teams", "Team A", "Team B"], value="Both teams",
                                      label="Report for")
            run = gr.Button("Compute stats", variant="primary")
            team_table = gr.Dataframe(headers=TEAM_HEADERS, label="Team stats")
            player_table = gr.Dataframe(headers=PLAYER_HEADERS, label="Player stats")
            report_file = gr.File(label="Report (PDF)")

            def on_stats(tags, a, b, who):
                stats = tag_stats(tags, a, b, config, rosters)
                team = {"Team A": a, "Team B": b}.get(who)
                name = f"report_{a}_vs_{b}{'_' + team if team else ''}.pdf".replace(" ", "_")
                pdf = write_pdf(stats, rosters, Path(tempfile.mkdtemp()) / name, team=team)
                return team_rows(stats), player_rows(stats, rosters), str(pdf)

            run.click(on_stats, [tags_csv, team_a, team_b, report_for],
                      [team_table, player_table, report_file])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CourtVision web app")
    parser.add_argument("--config", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    config = load_config(args.config)
    password = os.environ.get("COURTVISION_PASSWORD")
    auth = ("courtvision", password) if password else None
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    build_app(args.config).queue().launch(
        share=args.share, auth=auth,
        show_error=True,  # show the actual error message in the app, not just "Error"
        allowed_paths=[str(Path(config["output_dir"]).resolve())],  # serve annotated videos
    )


if __name__ == "__main__":
    main()
