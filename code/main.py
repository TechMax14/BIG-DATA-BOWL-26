#!/usr/bin/env python3
"""
AirDuel: Air Battle Index (ABI) pipeline entrypoint.

This script runs the full ABI workflow:

1. Load weekly tracking / play data from disk.
2. Apply basic cleaning and preprocessing.
3. Build an enriched tracking dataframe + qualifying play index.
4. Run all four ABI submetrics (separation, contested, closing, xCatch).
5. Aggregate submetrics into play-level ABI + player/team outputs.
6. Generate all visuals used in the writeup and competition video.

Run this file from the `code/` directory once the `data/` folder
has been populated with the competition inputs.
"""

from utils.data_loader import load_week_data
from utils.data_preprocessor import basic_clean, process_data
from metrics.metric_pipeline import run_metrics
from metrics.abi_aggregator import build_abi_outputs
from viz.visual_pipeline import build_all_visuals


def main():
    """Run the end-to-end AirDuel / ABI pipeline.

    Steps:
        1. Load raw tracking and play-level inputs.
        2. Clean and lightly standardize inputs.
        3. Build an enriched tracking dataframe and qualifying play index,
           restricted to downfield passes (air_yards >= PASS_LEN_MIN).
        4. Run all submetric pipelines (separation, contested, closing, xCatch).
        5. Aggregate submetrics into ABI and write play / player outputs to disk.
        6. Build and save all visuals (leaderboards, scheme insights, case studies).
    """
    PASS_LEN_MIN = 10.0  # minimum air yards to be considered a true downfield air battle
    FPS = 10.0           # tracking frames per second (used for time-based metrics)

    # 1. Load raw tracking + play data for the configured week(s)
    inp, out, supp = load_week_data()

    # 2. Apply basic cleaning and standardization across inputs
    inp, out, supp = basic_clean(inp, out, supp)

    # 3. Build enriched tracking dataframe + qualifying play index
    #    - Filters to passes with pass_length >= PASS_LEN_MIN
    #    - Optionally attaches play-level context fields
    df, qualifying_plays_index, _ = process_data(
        inp,
        out,
        supp,
        pass_length_min=PASS_LEN_MIN,        # keep consistent with metric filters
        include_play_context=True,
    )

    # 4. Run all submetrics (Separation + Contested + Closing + xCatch)
    metric_results = run_metrics(
        df,
        qualifying_plays_index,
        pass_length_min=PASS_LEN_MIN,  # shared threshold across metrics
        fps=FPS,

        # separation leaderboard prints (WR)
        sep_print_top_n=10,
        sep_min_targets=40,

        # closing leaderboard prints (DB)
        closing_print_top_n=10,
        closing_min_opp=40,
        closing_min_overlap_frames=6,
    )

    # 5. Aggregate submetrics into ABI outputs (play-level + player-level)
    abi_results = build_abi_outputs(
        sep_df=metric_results["sep"]["combined"],
        closing_df=metric_results["closing"]["play"],
        contested_df=metric_results["contested"]["scored"],
        xcatch_df=metric_results["xcatch"]["predictions"],
        output_dir="../data/abi/results",
        write_csv=True,
        min_wr_targets=40,
        min_db_plays=40,
        play_context_df=supp,
    )

    # 6. Build and save all visuals (leaderboards, team summaries, play packages)
    build_all_visuals(
        out_enriched="../data/processed/out_enriched.csv",
        abi_full_csv="../data/abi/results/abi_results_full.csv",
        wr_leaderboard_csv="../data/abi/results/abw_wr_leaderboard.csv",
        db_leaderboard_csv="../data/abi/results/abi_db_leaderboard.csv",
        hero_game_id=2023112602,
        hero_play_id=2848,
    )


if __name__ == "__main__":
    main()
