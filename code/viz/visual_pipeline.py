#!/usr/bin/env python3
# viz/visual_pipeline.py
"""
High-level visualization pipeline for the Air Duel / ABI project.

This module is a single entry point that:
    - Builds the ABI hero radial play graphic.
    - Renders per-frame play graphics for a chosen highlight play.
    - Generates WR-level leaderboards (Air Battle Win, separation, catch vs expected).
    - Produces route / coverage scheme insights and team summary visuals.

Public API:
    - build_all_visuals(...)
"""

from pathlib import Path

import pandas as pd

# Individual visual functions from your modules
from viz.abi_hero_visual import (
    load_abi_data,
    get_example_play,
    plot_abi_radial_for_play,
)
# from viz.abi_metric_tracker import plot_abi_metrics_for_play  # (optional future use)
from viz.play_insights import build_play_graphics

from viz.summary_visuals import (
    load_wr_leaderboard,
    plot_wr_abw_leaderboard,
    plot_wr_separation_component,
    plot_wr_catch_over_expected_component,
    plot_route_type_profile,
    plot_coverage_type_profile,
    plot_route_coverage_heatmaps,
    plot_team_leaderboards,
)

# (later you could add)
# from viz.sep_visuals import plot_sep_distribution
# from viz.closing_visuals import plot_closing_leaderboard
# etc.


def build_all_visuals(
    out_enriched: str,
    abi_full_csv: str,
    wr_leaderboard_csv: str,
    db_leaderboard_csv: str,   # currently unused, kept for symmetry / future DB viz
    output_dir: str = "../visuals",
):
    """
    Run the full visualization stack for the project.

    Args:
        out_enriched:
            Path to enriched frame-level tracking data
            (e.g., ../data/processed/out_enriched.csv).
        abi_full_csv:
            Path to play-level ABI table
            (e.g., ../data/abi/results/abi_results_full.csv).
        wr_leaderboard_csv:
            Path to WR leaderboard with ABW and components
            (e.g., ../data/abi/results/abw_wr_leaderboard.csv).
        db_leaderboard_csv:
            Path to DB leaderboard (currently not used in this script, but
            kept as a parameter for future defensive visuals).
        output_dir:
            Root folder where all visual assets are written.

    Side effects:
        - Creates subfolders under `output_dir` and writes PNGs for:
            * ABI hero radial chart
            * A chosen highlight play (per-frame graphics)
            * WR leaderboards
            * Route / coverage scheme profiles
            * Route vs coverage heatmaps
            * Team-level summary leaderboards
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("📊 Building ABI visuals...")

    # ------------------------------------------------------------------
    # ① ABI HERO RADIAL CHART
    # ------------------------------------------------------------------
    df_abi = load_abi_data(abi_full_csv)
    top_play = get_example_play(df_abi, mode="top")  # "top", "low", or "median"

    fig = plot_abi_radial_for_play(top_play, df=df_abi)
    (outdir / "abi_hero").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outdir / "abi_hero/abi_hero_radial_top_play.png",
        dpi=200,
        bbox_inches="tight",
    )
    print("  ✓ Saved hero radial chart")

    # ------------------------------------------------------------------
    # ② Load frame-level enriched tracking data
    # ------------------------------------------------------------------
    out_df = pd.read_csv(out_enriched)

    # ------------------------------------------------------------------
    # ③ Choose a highlight play to render per-frame ABI metrics
    # ------------------------------------------------------------------
    # NOTE:
    #   These IDs are intentionally hard-coded for the submission to make
    #   the visuals fully reproducible. You can swap to another high-ABI
    #   play by changing (game_id, play_id) to a favorite example.
    game_id = 2024010711
    play_id = 3554

    print(f"📈 Rendering ABI metric progression for game={game_id}, play={play_id}...")

    build_play_graphics(
        out_enriched=out_df,
        abi_full=df_abi,
        game_id=game_id,
        play_id=play_id,
        fps=10.0,
        base_output_dir="../visuals/plays",
    )

    # ------------------------------------------------------------------
    # ④ WR summary visuals (Air Battle Win + components)
    # ------------------------------------------------------------------
    print("📊 Building WR summary visuals...")

    df_wr = load_wr_leaderboard(wr_leaderboard_csv)

    # Air Battle Win (ABW) top WRs
    (outdir / "receiver_leaderboard").mkdir(parents=True, exist_ok=True)
    plot_wr_abw_leaderboard(
        df_wr,
        out_path=outdir / "receiver_leaderboard/wr_top_abw.png",
        top_n=10,
    )

    # Separation creation component
    plot_wr_separation_component(
        df_wr,
        out_path=outdir / "receiver_leaderboard/wr_top_separation.png",
        top_n=10,
    )

    # Catch over expected component
    plot_wr_catch_over_expected_component(
        df_wr,
        out_path=outdir / "receiver_leaderboard/wr_top_catch_vs_expected.png",
        top_n=10,
    )
    print("  ✓ WR leaderboards saved")

    # ------------------------------------------------------------------
    # ⑤ Route and coverage scheme summaries
    # ------------------------------------------------------------------
    print("📊 Building route & coverage summaries...")

    (outdir / "scheme_insights").mkdir(parents=True, exist_ok=True)

    plot_route_type_profile(
        abi_full=df_abi,
        out_path=outdir / "scheme_insights/route_type_profile.png",
        min_targets=40,
    )

    plot_coverage_type_profile(
        abi_full=df_abi,
        out_path=outdir / "scheme_insights/coverage_type_profile.png",
        min_plays=80,
    )
    print("  ✓ Route & coverage profiles saved")

    # Route x Coverage heatmaps
    plot_route_coverage_heatmaps(
        abi_full=df_abi,
        output_dir="../visuals/scheme_insights/routeVScoverage_heatmaps",
        min_plays=40,   # tweak as needed
    )

    # ------------------------------------------------------------------
    # ⑥ Team-level offensive / defensive ABI leaderboards
    # ------------------------------------------------------------------
    plot_team_leaderboards(
        abi_full=df_abi,
        output_dir="../visuals/summary_teams",
        min_off_plays=80,   # tweak thresholds if needed
        min_def_plays=80,
        top_n=10,
    )

    print("✅ All visuals successfully generated!")
