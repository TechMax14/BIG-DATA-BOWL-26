#!/usr/bin/env python3
"""
utils/data_preprocessor.py

Clean, enrich, and index tracking data for the Air Duel / ABI project.

Responsibilities:
    - Normalize common dtypes across input / output / supplementary tables.
    - Attach player attributes and play-level context to the OUTPUT tracking.
    - Build a canonical targeted-play index (one row per qualifying pass).
    - Optionally filter out ultra-short plays with too few frames.
    - Optionally save enriched and index tables to disk for reuse.

Public API:
    - basic_clean(...)
    - build_output_enriched(...)
    - build_plays_index(...)
    - require_min_frames(...)
    - process_data(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Columns we expect to be string-like and want stripped / normalized
STRING_COLS = [
    "phase",
    "player_role",
    "player_side",
    "player_position",
    "player_name",
]

# Values used to identify the targeted receiver in the tracking data
TARGET_LABELS = {"Targeted Receiver", "TargetedReceiver"}


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce common numeric columns and strip whitespace from string columns.

    This function is intentionally light-touch and safe to apply to
    input, output, and supplementary tables.
    """
    d = df.copy()

    # Normalize strings
    for c in STRING_COLS:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()

    # Normalize numerics where present
    numeric_cols = [
        "game_id",
        "play_id",
        "nfl_id",
        "frame_id",
        "x",
        "y",
        "s",
        "a",
        "o",
        "dir",
        "pass_length",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def basic_clean(
    inp: pd.DataFrame,
    out: pd.DataFrame,
    supp: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply light dtype / string normalization to the three core tables.

    Returns:
        (inp_clean, out_clean, supp_clean)
    """
    return (
        _normalize_dtypes(inp),
        _normalize_dtypes(out),
        _normalize_dtypes(supp),
    )


def _mode_or_first(series: pd.Series) -> Any:
    """
    Aggregate helper: return the mode if it exists, otherwise the first non-null.

    Used for player attributes when collapsing from many frames → one row
    per (game_id, play_id, nfl_id).
    """
    if series.empty:
        return np.nan
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else series.dropna().iloc[0]


# ---------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------

def build_output_enriched(
    inp: pd.DataFrame,
    out: pd.DataFrame,
    supp: pd.DataFrame,
    *,
    include_play_context: bool = True,
    play_context_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build an enriched OUTPUT table:

        - Attaches player attributes from the INPUT table.
        - Optionally attaches play-level context from the supplementary table.

    Args:
        inp:  Raw input tracking table (contains player-level attributes).
        out:  Raw output tracking table (contains frames used for metrics).
        supp: Play-level supplementary table (routes, coverage, etc.).
        include_play_context: If True, merge in play context from `supp`.
        play_context_cols: Optional explicit list of context columns to attach.

    Returns:
        out_enriched: OUTPUT tracking with player attributes and context.
    """
    inp = _normalize_dtypes(inp)
    out = _normalize_dtypes(out)
    supp = _normalize_dtypes(supp)

    key_cols = ["game_id", "play_id", "nfl_id"]
    attr_cols = [
        c
        for c in [
            "player_name",
            "player_position",
            "player_side",
            "player_role",
        ]
        if c in inp.columns
    ]

    # ------------------------------
    # 1) Player attributes (per game/play/player)
    # ------------------------------
    if attr_cols:
        attrs = (
            inp[key_cols + attr_cols]
            .groupby(key_cols, as_index=False)
            .agg({c: _mode_or_first for c in attr_cols})
        )
    else:
        attrs = inp[key_cols].drop_duplicates()

    needed_out_cols = [
        c
        for c in ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]
        if c in out.columns
    ]

    enriched = (
        out[needed_out_cols]
        .dropna(subset=["game_id", "play_id", "nfl_id"])
        .merge(attrs, on=key_cols, how="left")
    )

    # ------------------------------
    # 2) Play context (routes, coverage, etc.)
    # ------------------------------
    if include_play_context and not supp.empty:
        if play_context_cols is None:
            play_context_cols = [
                "play_description",
                "pass_result",
                "pass_length",
                "route_of_targeted_receiver",
                "possession_team",
                "defensive_team",
                "team_coverage_type",
                "season",
                "week",
                "quarter",
                "down",
                "yards_to_go",
                "yardline_side",
                "yardline_number",
                "play_nullified_by_penalty",
            ]

        have = [c for c in play_context_cols if c in supp.columns]
        if have:
            ctx = supp[["game_id", "play_id"] + have].drop_duplicates()
            enriched = enriched.merge(ctx, on=["game_id", "play_id"], how="left")

    return enriched


# ---------------------------------------------------------------------
# Play index (canonical targeted plays)
# ---------------------------------------------------------------------

def build_plays_index(
    enriched: pd.DataFrame,
    *,
    pass_length_min: float = 10.0,
) -> pd.DataFrame:
    """
    Build a canonical targeted-play index: one row per qualifying pass play.

    Filters:
        - Must be tagged as the targeted receiver (TARGET_LABELS) on offense.
        - Must not be nullified by penalty, when that indicator exists.
        - Must have pass_length >= pass_length_min, when available.

    Returns:
        plays_index with columns:
            - game_id, play_id, tgt_nfl_id, tgt_name, tgt_pos
            - week, pass_length, pass_result
            - play_uid: stable "week_game_play" string key
    """
    if enriched.empty:
        return pd.DataFrame(columns=["game_id", "play_id", "tgt_nfl_id"])

    d = enriched.copy()

    pr = d.get("player_role").astype(str) if "player_role" in d.columns else None
    ps = d.get("player_side").astype(str) if "player_side" in d.columns else None

    tgt_mask = pr.isin(TARGET_LABELS) if pr is not None else False
    if ps is not None:
        tgt_mask &= ps.eq("Offense")

    # Optional filters
    if "play_nullified_by_penalty" in d.columns:
        d = d[d["play_nullified_by_penalty"] != "Y"]

    # NOTE: this is where we enforce pass_length >= pass_length_min for ABI.
    if "pass_length" in d.columns:
        d = d[d["pass_length"] >= pass_length_min]

    idx = (
        d.loc[
            tgt_mask,
            [
                "game_id",
                "play_id",
                "nfl_id",
                "player_name",
                "player_position",
                "week",
                "pass_length",
                "pass_result",
            ],
        ]
        .dropna(subset=["game_id", "play_id", "nfl_id"])
        .rename(
            columns={
                "nfl_id": "tgt_nfl_id",
                "player_name": "tgt_name",
                "player_position": "tgt_pos",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Stable unique key across the season
    idx["play_uid"] = (
        idx["week"].astype(int).astype(str)
        + "_"
        + idx["game_id"].astype(int).astype(str)
        + "_"
        + idx["play_id"].astype(int).astype(str)
    )

    return idx


def require_min_frames(
    out_enriched: pd.DataFrame,
    plays_index: pd.DataFrame,
    *,
    min_frames_per_play: int = 7,
) -> pd.DataFrame:
    """
    Keep only plays that have at least `min_frames_per_play` unique output frames.

    This screens out ultra-short windows that are too noisy for ABI metrics.

    Args:
        out_enriched: Enriched output tracking table.
        plays_index:  Current targeted-play index.
        min_frames_per_play: Minimum unique frame_id count per play.

    Returns:
        Filtered plays_index DataFrame.
    """
    if (
        out_enriched is None
        or out_enriched.empty
        or plays_index is None
        or plays_index.empty
    ):
        return plays_index

    fp = (
        out_enriched.groupby(["game_id", "play_id"])["frame_id"]
        .nunique()
        .rename("frames_per_play")
        .reset_index()
    )

    before = len(plays_index)
    refined = (
        plays_index.merge(fp, on=["game_id", "play_id"], how="left")
        .query("frames_per_play >= @min_frames_per_play")
        .drop(columns=["frames_per_play"])
    )
    after = len(refined)

    if after != before:
        print(
            f"🧭 plays_index refined by frames/play ≥ {min_frames_per_play}: "
            f"{before:,} → {after:,} plays"
        )

    return refined


# ---------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------

def process_data(
    inp: pd.DataFrame,
    out: pd.DataFrame,
    supp: pd.DataFrame,
    *,
    pass_length_min: float = 10.0,
    include_play_context: bool = True,
    save_to_disk: bool = True,
    processed_dir: str = "../data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full preprocessing pipeline for ABI:

        1) Build an enriched output tracking table (player attributes + context).
        2) Build a canonical targeted-play index with pass_length >= pass_length_min.
        3) Filter out plays with too few frames.
        4) Optionally persist enriched + index as CSVs.

    Args:
        inp: Raw INPUT tracking table.
        out: Raw OUTPUT tracking table.
        supp: Supplementary play-level table.
        pass_length_min: Minimum pass_length to keep in plays_index (e.g., 10+ yards).
        include_play_context: Whether to merge in play context from `supp`.
        save_to_disk: If True, write CSVs under `processed_dir`.
        processed_dir: Folder where processed CSVs are saved.

    Returns:
        (enriched_df, plays_index, summary_dict)
    """
    print("\n🔧 Enriching OUTPUT (attributes + context)...")
    enriched = build_output_enriched(
        inp,
        out,
        supp,
        include_play_context=include_play_context,
    )

    # ------------------------------
    # Build play index (downfield targeted plays)
    # ------------------------------
    plays_index = build_plays_index(
        enriched,
        pass_length_min=pass_length_min,
    )

    # Filter out ultrashort plays with too few frames to score reliably
    plays_index = require_min_frames(
        enriched,
        plays_index,
        min_frames_per_play=7,  # tweak this if you want stricter/looser filter
    )

    summary = {
        "rows_enriched": len(enriched),
        "rows_play_index": len(plays_index),
        "unique_plays": plays_index["play_uid"].nunique()
        if not plays_index.empty
        else 0,
    }

    print(f"  Enriched rows: {summary['rows_enriched']:,}")
    print(f"  Targeted plays (index): {summary['rows_play_index']:,}")

    # ------------------------------
    # Save to CSV (for reproducibility / downstream reuse)
    # ------------------------------
    if save_to_disk:
        processed_dir_path = Path(processed_dir)
        processed_dir_path.mkdir(parents=True, exist_ok=True)

        enriched_path = processed_dir_path / "out_enriched.csv"
        index_path = processed_dir_path / "plays_index.csv"

        enriched.to_csv(enriched_path, index=False)
        plays_index.to_csv(index_path, index=False)

        print(f"💾 Saved enriched CSV     → {enriched_path}")
        print(f"💾 Saved plays_index CSV → {index_path}")

    return enriched, plays_index, summary
