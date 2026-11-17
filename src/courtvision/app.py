"""Gradio web app: upload a game, pick the teams, see stats.

Run on Colab:  python -m courtvision.app --share
Set COURTVISION_PASSWORD to require a login (recommended with --share).
"""

from __future__ import annotations

import argparse
import os

from .config import load_config
from .events import EventLog
from .report import PLAYER_HEADERS, TEAM_HEADERS, player_rows, team_rows
from .rosters import load_rosters
from .stats import compute_stats
from .tags import load_tags


def analyze(video, tags_csv, team_a, team_b, config, rosters, progress):
    import gradio as gr

    if not team_a or not team_b or team_a == team_b:
        raise gr.Error("Pick two different teams.")
    teams = [team_a, team_b]

    if tags_csv:
        # Tags-only mode: works before the CV pipeline is done; useful for checking stats.
        events = load_tags(tags_csv, config["tags"])
        duration = max((e.t for e in events), default=0.0)
        log = EventLog(teams=teams, duration=duration, events=events, source="tags")
    elif video:
        from .pipeline import build_event_log, run_tracking

        try:
            tracking = run_tracking(video, config, progress=lambda p, msg: progress(p, desc=msg))
            # TODO: ask the user which color cluster is which team before this step.
            log = build_event_log(tracking, {0: team_a, 1: team_b})
        except NotImplementedError as e:
            raise gr.Error(f"Video pipeline not finished yet: {e}")
    else:
        raise gr.Error("Upload a video or a tag CSV.")

    stats = compute_stats(log)
    return team_rows(stats), player_rows(stats, rosters)


def build_app(config_path: str | None = None):
    import gradio as gr

    config = load_config(config_path)
    rosters = load_rosters(config["roster_csv"])
    team_names = sorted(rosters)

    with gr.Blocks(title="CourtVision") as demo:
        gr.Markdown("# CourtVision\nUpload a game video and pick the two teams.")
        with gr.Row():
            video = gr.Video(label="Game video")
            tags_csv = gr.File(label="Tag CSV (optional)", file_types=[".csv"], type="filepath")
        with gr.Row():
            team_a = gr.Dropdown(team_names, label="Team A")
            team_b = gr.Dropdown(team_names, label="Team B")
        run = gr.Button("Analyze", variant="primary")
        team_table = gr.Dataframe(headers=TEAM_HEADERS, label="Team stats")
        player_table = gr.Dataframe(headers=PLAYER_HEADERS, label="Player stats")

        def on_run(video, tags_csv, team_a, team_b, progress=gr.Progress()):
            return analyze(video, tags_csv, team_a, team_b, config, rosters, progress)

        run.click(on_run, [video, tags_csv, team_a, team_b], [team_table, player_table])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CourtVision web app")
    parser.add_argument("--config", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    password = os.environ.get("COURTVISION_PASSWORD")
    auth = ("courtvision", password) if password else None
    build_app(args.config).queue().launch(share=args.share, auth=auth)


if __name__ == "__main__":
    main()
