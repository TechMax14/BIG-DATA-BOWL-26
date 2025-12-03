#!/usr/bin/env python3
"""
Metric Pipeline Orchestrator

This module coordinates the execution of all four ABI submetric pipelines:
    1. Separation Creation (WR-focused)
    2. Defensive Closing Efficiency (DB-focused)
    3. Contested Arrival Severity (play-level)
    4. Expected Catch Surprise (xCatch model)

`run_metrics()` is the single entrypoint used by main.py. It dispatches inputs
to each submetric module, applies consistent filtering (e.g., pass_length_min),
and returns a structured dictionary with all intermediate metric outputs.

No ABI aggregation is performed here — that happens later in `abi_aggregator`.
"""

from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

from metrics.sep_creation_metric import run_sep_pipeline
from metrics.closing_eff_metric import run_closing_pipeline
from metrics.contested_catch_metric import run_contested_pipeline
from metrics.xCatch_prob_metric import run_xCatch_pipeline
from metrics.abi_aggregator import build_abi_outputs


def run_metrics(
    out_enriched: pd.DataFrame,
    plays_index: pd.DataFrame,
    *,
    # shared filtering
    pass_length_min: float = 10.0,
    fps: float = 10.0,

    # separation (WR)
    sep_print_top_n: Optional[int] = 10,
    sep_min_targets: int = 40,

    # closing (DB)
    closing_print_top_n: Optional[int] = 10,
    closing_min_opp: int = 40,
    closing_min_overlap_frames: int = 6,
) -> dict:
    """
    Run all four ABI submetrics in sequence and return their outputs.

    This function does not compute the final ABI score — it only produces
    the raw/combined tables needed by `build_abi_outputs()`.

    Args:
        out_enriched (pd.DataFrame):
            Enriched tracking table produced by `process_data()`. Contains
            WR/DB positions, ball tracking, air yards, and contextual fields.
        plays_index (pd.DataFrame):
            One-row-per-play index aligned to tracking data, ensuring consistent
            merging of play-level summaries.
        pass_length_min (float, optional):
            Minimum air_yards required for a play to be included in any metric.
            Should match the value used in main.py.
        fps (float, optional):
            Frames per second in tracking data. Used for all time-based metrics.
        sep_print_top_n (int | None):
            If provided, prints the top N WRs by separation gain.
        sep_min_targets (int):
            Minimum targets required for WR leaderboard appearance.
        closing_print_top_n (int | None):
            If provided, prints the top N defenders by closing efficiency.
        closing_min_opp (int):
            Minimum defensive "opportunities" required to appear in DB outputs.
        closing_min_overlap_frames (int):
            Minimum overlapping frames required to treat a WR/DB pair
            as a valid closing interaction.

    Returns:
        dict: Structured dictionary with outputs for each submetric:
            {
                "sep":       {...},
                "contested": {...},
                "closing":   {...},
                "xcatch":    {...},
            }
    """

    results: dict = {}

    # ------------------------------------------------------------------
    # 1) Separation Creation (WR)
    # ------------------------------------------------------------------
    sep_results = run_sep_pipeline(
        out_enriched,
        plays_index=plays_index,
        pass_length_min=pass_length_min,
        fps=fps,
        min_targets=sep_min_targets,
        print_top_n=sep_print_top_n,
    )
    results["sep"] = sep_results

    # ------------------------------------------------------------------
    # 2) Defensive Closing Efficiency (DB)
    # ------------------------------------------------------------------
    closing_results = run_closing_pipeline(
        out_enriched,
        plays_index=plays_index,
        pass_length_min=pass_length_min,
        fps=fps,
        min_opp=closing_min_opp,
        min_overlap_frames=closing_min_overlap_frames,
        print_top_n=closing_print_top_n,
    )
    results["closing"] = closing_results

    # ------------------------------------------------------------------
    # 3) Contested Arrival Severity (play-level)
    # ------------------------------------------------------------------
    contested_results = run_contested_pipeline(
        out_enriched,
        plays_index=plays_index,
        pass_length_min=pass_length_min,
        fps=fps,
    )
    results["contested"] = contested_results

    # ------------------------------------------------------------------
    # 4) Expected Catch Surprise (xCatch model)
    #    Uses outputs from separation, contested, and closing.
    # ------------------------------------------------------------------
    xcatch_results = run_xCatch_pipeline(
        out_enriched,
        sep_results=sep_results,
        contested_results=contested_results,
        closing_results=closing_results,
        plays_index=plays_index,
        fps=fps,
        train_on_all_plays=True,
        model_path="../data/abi/metrics/catch_probability/xCatch_model.joblib",
        training_table_path="../data/abi/metrics/catch_probability/xCatch_training.csv",
        predictions_path="../data/abi/metrics/catch_probability/xCatch_predictions.csv",
        train_model_now=True,       # currently retrains; can disable for production
        scale_mode="prob",
    )
    results["xcatch"] = xcatch_results

    # Returned dict is shaped explicitly; avoids accidental mismatch
    return {
        "sep": sep_results,
        "contested": contested_results,
        "closing": closing_results,
        "xcatch": xcatch_results,
    }
