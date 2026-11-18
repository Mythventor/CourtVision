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
        f"- Court mapped in {s.get('court_mapped', 0):.0%} of analyzed frames",
    ]
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

    def library():
        return list_videos(config["video_root"], exclude=config["output_dir"])

    def files_in(game):
        return [p.name for p in library().get(game, [])]

    with gr.Blocks(title="CourtVision") as demo:
        gr.Markdown("# CourtVision")

        with gr.Tab("Video tracking"):
            games = sorted(library())
            with gr.Row():
                game = gr.Dropdown(games, label="Game folder", value=games[0] if games else None)
                quarter = gr.Dropdown(files_in(games[0]) if games else [], label="Video")
                refresh = gr.Button("Refresh list", size="sm")
            if not games:
                gr.Markdown(f"No videos found in `{config['video_root']}`. "
                            "Set `video_root` in your config.")
            clip = gr.Video(label="...or upload a short clip instead", sources=["upload"])
            with gr.Row():
                start_min = gr.Number(0, label="Start (minutes into the video)", minimum=0)
                end_min = gr.Number(0, label="End (minutes; 0 = end of video)", minimum=0)
                reprocess = gr.Checkbox(label="Reprocess (ignore saved tracks)")
            process = gr.Button("Track players and ball", variant="primary")
            output_video = gr.Video(label="Tracking + 2D court")
            summary = gr.Markdown()

            def on_game(g):
                names = files_in(g)
                return gr.Dropdown(choices=names, value=names[0] if names else None)

            def on_refresh():
                gs = sorted(library())
                return gr.Dropdown(choices=gs, value=gs[0] if gs else None)

            def on_process(g, q, clip_path, start, end, again, progress=gr.Progress()):
                from .processing import process_video

                if clip_path:
                    path = Path(clip_path)
                else:
                    match = [p for p in library().get(g, []) if p.name == q]
                    if not match:
                        raise gr.Error("Pick a video (or upload a clip).")
                    path = match[0]
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
        allowed_paths=[str(Path(config["output_dir"]).resolve())],  # serve annotated videos
    )


if __name__ == "__main__":
    main()
