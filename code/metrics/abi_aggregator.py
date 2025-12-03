#!/usr/bin/env python3
"""
metrics/abi_aggregator.py

Air Duel: Air Battle Index (ABI)

Combine the four metric families (Separation, Closing, Contested, xCatch)
into a unified Air Battle Index (ABI) on a 0–100 scale:

    abi_100 = sep_delta_25
            + closing_eff_25
            + contested_severity_25
            + xcatch_surprise_25

Each submetric is already on a 0–25 scale, so ABI is a simple, interpretable
sum of four equally-weighted components:

    - Separation Creation (WR):    sep_delta_25
    - Defensive Closing (DB):      closing_eff_25
    - Contest Severity (catch pt): contested_severity_25
    - Expected Catch Surprise:     xcatch_surprise_25

This module produces:
    - Play-level ABI table (one row per targeted play)
    - Condensed ABI table for storytelling / visuals
    - WR leaderboard (Air Battle Win, ABW)
    - DB leaderboard (primary closer involvement)

Inputs are the *play-level* outputs from the metric pipelines:
    * sep_df:        separation pipeline (one row per targeted play)
    * closing_df:    closing pipeline play-level table (best closer per play)
    * contested_df:  contested pipeline (one row per targeted play)
    * xcatch_df:     xCatch predictions (one row per targeted play)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from metrics.abi_narratives import add_play_categories_and_sentences


# ----------------------------------------------------------------------
# Helpers for play blurbs
# ----------------------------------------------------------------------

def _fmt_down(down):
    """Format numeric down as '1st', '2nd', '3rd', '4th'."""
    try:
        d = int(down)
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(d, f"{d}th")
    except Exception:
        return "?"


def _make_play_blurb(ctx_row: dict) -> str:
    """
    Build a compact, human-readable play blurb from play context.

    Example shape:
        "MIN vs DET | W16 Q4 3rd&7 @ DET 28 | pass_len=22 | C | <play_description>"
    """
    wk = ctx_row.get("week")
    off = ctx_row.get("possession_team")
    deff = ctx_row.get("defensive_team")
    qtr = ctx_row.get("quarter")
    down = _fmt_down(ctx_row.get("down"))
    ytg = ctx_row.get("yards_to_go")
    yls = ctx_row.get("yardline_side")
    yln = ctx_row.get("yardline_number")
    pr  = ctx_row.get("pass_result")
    pl  = ctx_row.get("pass_length")
    desc= ctx_row.get("play_description")

    parts = []
    left = []
    if off or deff:
        left.append(f"{off or '?'} vs {deff or '?'}")
    if left:
        parts.append(" ".join(left))

    mid = []
    if wk is not None and pd.notna(wk):
        mid.append(f"W{int(wk)}")
    if qtr is not None and pd.notna(qtr):
        mid.append(f"Q{int(qtr)}")
    if (down and down != "?") or (ytg is not None and pd.notna(ytg)):
        mid.append(f"{down}&{int(ytg) if pd.notna(ytg) else '?'}")
    if yls or yln:
        mid.append(f"@ {yls or '?'} {int(yln) if pd.notna(yln) else '?'}")
    if mid:
        parts.append(" ".join(mid))

    right = []
    if pl is not None and pd.notna(pl):
        right.append(f"pass_len={int(pl)}")
    if pr and pd.notna(pr):
        right.append(str(pr))
    if right:
        parts.append(" | ".join(right))

    if desc and pd.notna(desc):
        parts.append(str(desc))

    return " | ".join(parts)


# ----------------------------------------------------------------------
# xCatch surprise scoring
# ----------------------------------------------------------------------

def add_xcatch_surprise_scores(xcatch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add offensive / defensive "surprise" scores from xCatch probabilities.

    Assumes:
        - xcatch_df has:
            * caught (0/1)
            * xcatch_prob in [0, 1]

    Adds:
        - xcatch_off_surprise  = y * (1 - p)
        - xcatch_def_surprise  = (1 - y) * p
        - xcatch_surprise      = max(off, def)
        - xcatch_surprise_25   = round(25 * xcatch_surprise)  (Int64 in 0–25)

    Interpretation:
        - Improbable catches   (low p, caught)      → high offensive surprise
        - Unexpected misses    (high p, not caught) → high defensive surprise
        - Routine outcomes                          → low surprise
    """
    df = xcatch_df.copy()

    if "caught" not in df.columns or "xcatch_prob" not in df.columns:
        raise ValueError("xcatch_df must contain 'caught' and 'xcatch_prob' to compute surprise scores.")

    p = pd.to_numeric(df["xcatch_prob"], errors="coerce").astype(float).clip(0.0, 1.0)
    y = pd.to_numeric(df["caught"], errors="coerce").astype(int).clip(0, 1)

    df["xcatch_off_surprise"] = y * (1.0 - p)
    df["xcatch_def_surprise"] = (1 - y) * p
    df["xcatch_surprise"] = df[["xcatch_off_surprise", "xcatch_def_surprise"]].max(axis=1)

    df["xcatch_surprise_25"] = (
        (df["xcatch_surprise"] * 25.0)
        .round()
        .astype("Int64")
        .clip(0, 25)
    )

    return df


# ----------------------------------------------------------------------
# Core ABI play-level builder
# ----------------------------------------------------------------------

def build_abi_plays_table(
    sep_df: pd.DataFrame,
    closing_df: pd.DataFrame,
    contested_df: pd.DataFrame,
    xcatch_df: pd.DataFrame,
    play_context_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build unified play-level ABI table from four metric tables.

    Inputs (expected columns):

        sep_df (separation, one row per targeted play):
            - game_id, play_id, tgt_nfl_id, tgt_name, tgt_pos
            - def_nfl_id, def_name, def_pos
            - week, pass_result, pass_length
            - route_of_targeted_receiver, team_coverage_type
            - first_sep, last_sep, delta, delta_per_s
            - sep_delta_25

        closing_df (play-level, best closer per play):
            - game_id, play_id
            - def_nfl_id, def_name, def_pos     (primary closer)
            - closing_eff_25, closing_eff_raw
            - closed_fraction
            - avg_closing_rate, peak_closing_rate
            - pct_time_closing, angle_efficiency_mean, path_efficiency
            - overlap_frames

        contested_df (contested arrival, play-level):
            - game_id, play_id, tgt_nfl_id
            - sep_at_arrival, n_defenders_r1, n_defenders_r2
            - closing_rate_last, pct_time_tight
            - contested_severity_25

        xcatch_df (xCatch predictions):
            - game_id, play_id, tgt_nfl_id
            - caught (0/1), xcatch_prob
            (this function adds xcatch_surprise + xcatch_surprise_25)

    Returns:
        DataFrame with one row per targeted play including:
            - IDs & context
            - separation raw + ABI score
            - closing raw + ABI score
            - contested raw + ABI score
            - xCatch prob + surprise + ABI score
            - abi_100 (0–100)
    """
    # Ensure we don't modify original inputs
    sep = sep_df.copy()
    closing = closing_df.copy()
    cont = contested_df.copy()
    xc = xcatch_df.copy()

    # Basic type hygiene for key columns
    for df in (sep, closing, cont, xc):
        for c in ("game_id", "play_id", "tgt_nfl_id"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add xCatch surprise & ABI-style score
    xc = add_xcatch_surprise_scores(xc)

    # --- Base: Separation table (WR-centric, one row per targeted play) ---
    base_cols = [
        "game_id", "play_id",
        "week", "pass_result", "pass_length",
        "route_of_targeted_receiver", "team_coverage_type",
        "tgt_nfl_id", "tgt_name", "tgt_pos",
        "def_nfl_id", "def_name", "def_pos",
        "first_sep", "last_sep", "delta", "delta_per_s",
        "sep_delta_25",
    ]
    base_cols = [c for c in base_cols if c in sep.columns]
    abi = sep[base_cols].copy()

    # --- Merge: Contested (tgt-level; 3-key join) ---
    cont_keys = ["game_id", "play_id", "tgt_nfl_id"]
    cont_cols = [
        "game_id", "play_id", "tgt_nfl_id",
        "sep_at_arrival", "n_defenders_r1", "n_defenders_r2",
        "closing_rate_last", "pct_time_tight",
        "contested_severity_25",
    ]
    cont_cols = [c for c in cont_cols if c in cont.columns]
    abi = abi.merge(
        cont[cont_cols].drop_duplicates(subset=cont_keys),
        on=cont_keys,
        how="left",
    )

    # --- Merge: Closing (play-level; 2-key join) ---
    close_keys = ["game_id", "play_id"]
    close_cols = [
        "game_id", "play_id",
        "def_nfl_id", "def_name", "def_pos",
        "closing_eff_25", "closing_eff_raw",
        "closed_fraction",
        "avg_closing_rate", "peak_closing_rate",
        "pct_time_closing", "angle_efficiency_mean", "path_efficiency",
        "overlap_frames",
    ]
    close_cols = [c for c in close_cols if c in closing.columns]
    closing_play = closing[close_cols].drop_duplicates(subset=close_keys)

    # Avoid clobbering WR-side def info with closing-side def info.
    # If both exist, rename the closing-side fields to *_closing.
    for c in ["def_nfl_id", "def_name", "def_pos"]:
        if c in closing_play.columns and c in abi.columns:
            closing_play = closing_play.rename(columns={c: f"{c}_closing"})

    abi = abi.merge(
        closing_play,
        on=close_keys,
        how="left",
    )

    # --- Merge: xCatch (tgt-level; 3-key join) ---
    xc_keys = ["game_id", "play_id", "tgt_nfl_id"]
    xc_cols = [
        "game_id", "play_id", "tgt_nfl_id",
        "caught",
        "xcatch_prob", "xcatch_25",
        "xcatch_off_surprise", "xcatch_def_surprise",
        "xcatch_surprise", "xcatch_surprise_25",
    ]
    xc_cols = [c for c in xc_cols if c in xc.columns]
    abi = abi.merge(
        xc[xc_cols].drop_duplicates(subset=xc_keys),
        on=xc_keys,
        how="left",
    )

    # --- Compute final ABI 0–100 ---
    for c in ("sep_delta_25", "closing_eff_25", "contested_severity_25", "xcatch_surprise_25"):
        if c in abi.columns:
            abi[c] = pd.to_numeric(abi[c], errors="coerce")

    abi["abi_100"] = (
        abi.get("sep_delta_25", 0).fillna(0).astype(float)
        + abi.get("closing_eff_25", 0).fillna(0).astype(float)
        + abi.get("contested_severity_25", 0).fillna(0).astype(float)
        + abi.get("xcatch_surprise_25", 0).fillna(0).astype(float)
    ).round(1)

    # --- Merge richer play context for blurbs (optional) ---
    if play_context_df is not None and not play_context_df.empty:
        ctx_cols = [
            "week",
            "possession_team",
            "defensive_team",
            "quarter",
            "down",
            "yards_to_go",
            "yardline_side",
            "yardline_number",
            "game_clock",
            "play_description",
        ]
        have_ctx = [c for c in ctx_cols if c in play_context_df.columns]
        if "week" in abi.columns and "week" in have_ctx:
            # avoid double week column in merge
            have_ctx.remove("week")
        if have_ctx:
            ctx = (
                play_context_df[["game_id", "play_id"] + have_ctx]
                .drop_duplicates(subset=["game_id", "play_id"])
            )
            abi = abi.merge(ctx, on=["game_id", "play_id"], how="left")

    # Optional nice ordering for key columns
    preferred_order = [
        # IDs & context
        "game_id", "play_id", "week",
        "pass_result", "pass_length",
        "route_of_targeted_receiver", "team_coverage_type",
        # WR
        "tgt_nfl_id", "tgt_name", "tgt_pos",
        # DB (from separation or closing)
        "def_nfl_id", "def_name", "def_pos",
        "def_nfl_id_closing", "def_name_closing", "def_pos_closing",
        # Separation
        "first_sep", "last_sep", "delta", "delta_per_s", "sep_delta_25",
        # Closing
        "closed_fraction", "avg_closing_rate", "peak_closing_rate",
        "pct_time_closing", "angle_efficiency_mean", "path_efficiency",
        "closing_eff_25",
        # Contested
        "sep_at_arrival", "n_defenders_r1", "n_defenders_r2",
        "closing_rate_last", "pct_time_tight", "contested_severity_25",
        # xCatch
        "caught", "xcatch_prob", "xcatch_25",
        "xcatch_off_surprise", "xcatch_def_surprise",
        "xcatch_surprise", "xcatch_surprise_25",
        # Final ABI
        "abi_100",
    ]
    ordered_cols = [c for c in preferred_order if c in abi.columns] + [
        c for c in abi.columns if c not in preferred_order
    ]
    abi = abi.reindex(columns=ordered_cols)

    return abi


def build_abi_condensed(abi_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Build a condensed ABI table ideal for visuals, dashboards, and narratives.

    Columns (if available):
        - game_id, play_id
        - play_blurb
        - tgt_name, route_of_targeted_receiver
        - sep_delta_25, closing_eff_25,
          contested_severity_25, xcatch_surprise_25,
        - abi_100
    """
    df = abi_plays.copy()

    # Build play_blurb from available context columns
    ctx_cols = [
        "week", "possession_team", "defensive_team", "quarter",
        "down", "yards_to_go", "yardline_side", "yardline_number",
        "pass_result", "pass_length", "route_of_targeted_receiver",
        "play_description",
    ]
    have = [c for c in ctx_cols if c in df.columns]

    df["play_blurb"] = df[have].apply(
        lambda r: _make_play_blurb(r.to_dict()),
        axis=1
    )

    keep = [
        "game_id", "play_id", "play_blurb",
        "tgt_name", "route_of_targeted_receiver",
        "sep_delta_25", "closing_eff_25",
        "contested_severity_25", "xcatch_surprise_25",
        "abi_100",
    ]
    keep = [c for c in keep if c in df.columns]

    return df[keep].copy()


# ----------------------------------------------------------------------
# Leaderboards
# ----------------------------------------------------------------------

def build_wr_leaderboard_with_abw(
    abi_full: pd.DataFrame,
    *,
    min_targets: int = 40,
) -> pd.DataFrame:
    """
    Build targeted-receiver leaderboard with Air Battle Win (ABW, 0–100).

    Uses:
        - Play-level ABI table including:
            game_id, play_id, pass_length,
            tgt_nfl_id, tgt_name, tgt_pos, possession_team,
            sep_delta_25, caught, xcatch_prob,
            xcatch_surprise_25, abi_100

    Notes:
        - Filters to downfield targets only (pass_length >= 10).
        - ABW combines:
            * Separation creation (sep_score_100)
            * Catch-over-expected (catch_score_100)

    Returns:
        One row per WR with:
            - n_targets
            - abi_avg_100
            - sep / catch components
            - abw_100
            - legacy columns expected by visuals (avg_* fields)
    """
    required = {
        "game_id",
        "play_id",
        "tgt_nfl_id",
        "tgt_name",
        "tgt_pos",
        "possession_team",
        "pass_length",
        "sep_delta_25",
        "caught",
        "xcatch_prob",
        "xcatch_surprise_25",
        "abi_100",
    }
    missing = required.difference(abi_full.columns)
    if missing:
        raise ValueError(f"abi_full missing required columns: {missing}")

    # Downfield targets only (pass_length >= 10)
    df = abi_full[abi_full["pass_length"] >= 10].copy()

    # Aggregate to WR level
    grp = (
        df.groupby(["tgt_nfl_id", "tgt_name", "tgt_pos", "possession_team"])
        .agg(
            n_targets=("play_id", "count"),
            abi_avg_100=("abi_100", "mean"),
            sep_delta_25_mean=("sep_delta_25", "mean"),
            catch_rate=("caught", "mean"),
            xcatch_prob_mean=("xcatch_prob", "mean"),
            xcatch_surprise_25_mean=("xcatch_surprise_25", "mean"),
        )
        .reset_index()
    )

    # Filter by volume
    grp = grp[grp["n_targets"] >= min_targets].copy()
    if grp.empty:
        raise ValueError(
            f"No receivers with at least {min_targets} downfield targets."
        )

    # Catch rate over expected
    grp["catch_over_expected"] = grp["catch_rate"] - grp["xcatch_prob_mean"]
    grp["catch_over_expected_pct"] = 100.0 * grp["catch_over_expected"]

    # ---- ABW component scores on 0–100 ----

    # Separation component: sep_delta_25 is already 0–25
    grp["sep_score_100"] = 100.0 * grp["sep_delta_25_mean"] / 25.0

    # Catch-over-expected component: league-normalized 0–100
    coe_min = grp["catch_over_expected"].min()
    coe_max = grp["catch_over_expected"].max()
    if coe_max > coe_min:
        grp["catch_score_100"] = 100.0 * (
            (grp["catch_over_expected"] - coe_min) / (coe_max - coe_min)
        )
    else:
        # degenerate case: everyone equal → flat 50
        grp["catch_score_100"] = 50.0

    # Final Air Battle Win score (WR-centric)
    grp["abw_100"] = 0.6 * grp["sep_score_100"] + 0.4 * grp["catch_score_100"]

    # ---- Legacy columns for existing visuals ----
    # (so summary_visuals.load_wr_leaderboard stops complaining)

    grp["targets"] = grp["n_targets"]
    grp["avg_abi_100"] = grp["abi_avg_100"].round(1)
    grp["avg_sep_delta_25"] = grp["sep_delta_25_mean"]
    grp["avg_sep_delta"] = grp["sep_delta_25_mean"]
    grp["avg_xcatch_prob"] = grp["xcatch_prob_mean"]
    grp["avg_xcatch_surprise_25"] = grp["xcatch_surprise_25_mean"]

    # Nice sorting for leaderboards
    grp = grp.sort_values("abw_100", ascending=False).reset_index(drop=True)

    return grp


def build_db_leaderboard(
    abi_plays: pd.DataFrame,
    *,
    min_plays: int = 40,
) -> pd.DataFrame:
    """
    Build DB leaderboard based on the primary closer per play.

    Uses def_nfl_id/def_name/def_pos as present in the play-level ABI table.

    Outputs per defender:
        - plays
        - catches_allowed
        - catch_rate_allowed
        - avg_closed_fraction
        - avg_closing_eff_25
        - avg_pct_time_tight        (time in tight windows vs this DB)
        - avg_contested_severity_25
        - avg_xcatch_prob           (expected catch vs this DB)
        - avg_xcatch_surprise_25
        - avg_abi_100               (air-battle intensity with this DB)
    """
    df = abi_plays.copy()
    if df.empty:
        return df

    # Prefer closing-side def fields if present
    for base, closing_name in [
        ("def_nfl_id", "def_nfl_id_closing"),
        ("def_name", "def_name_closing"),
        ("def_pos", "def_pos_closing"),
    ]:
        if closing_name in df.columns:
            df[base] = df[closing_name]

    # Drop plays without a known defender
    if "def_nfl_id" not in df.columns:
        return pd.DataFrame()

    df = df[~df["def_nfl_id"].isna()].copy()

    # Numeric hygiene
    num_cols = [
        "closed_fraction", "closing_eff_25",
        "pct_time_tight", "contested_severity_25",
        "xcatch_prob", "xcatch_surprise_25",
        "abi_100",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "caught" in df.columns:
        df["caught"] = pd.to_numeric(df["caught"], errors="coerce").fillna(0).astype(int)
    else:
        df["caught"] = 0

    group_keys = ["def_nfl_id"]
    if "def_name" in df.columns:
        group_keys.append("def_name")
    if "def_pos" in df.columns:
        group_keys.append("def_pos")

    grouped = df.groupby(group_keys, dropna=False)

    agg = grouped.agg(
        plays=("game_id", "count"),
        catches_allowed=("caught", "sum"),
        avg_closed_fraction=("closed_fraction", "mean"),
        avg_closing_eff_25=("closing_eff_25", "mean"),
        avg_pct_time_tight=("pct_time_tight", "mean"),
        avg_contested_severity_25=("contested_severity_25", "mean"),
        avg_xcatch_prob=("xcatch_prob", "mean"),
        avg_xcatch_surprise_25=("xcatch_surprise_25", "mean"),
        avg_abi_100=("abi_100", "mean"),
    ).reset_index()

    agg["catch_rate_allowed"] = agg["catches_allowed"] / agg["plays"]
    agg = agg[agg["plays"] >= min_plays].copy()

    for c in [
        "avg_closed_fraction",
        "avg_closing_eff_25",
        "avg_pct_time_tight",
        "avg_contested_severity_25",
        "avg_xcatch_prob",
        "avg_xcatch_surprise_25",
        "avg_abi_100",
        "catch_rate_allowed",
    ]:
        if c in agg.columns:
            agg[c] = agg[c].round(3)

    # For defenders, lower catch_rate_allowed + higher closing/ABI are good;
    # for a global "air-battle involvement" leaderboard, sort by avg_abi_100.
    agg = agg.sort_values(["avg_abi_100", "plays"], ascending=[False, False]).reset_index(drop=True)
    return agg


# ----------------------------------------------------------------------
# High-level entry point used by metric_pipeline (or standalone)
# ----------------------------------------------------------------------

def build_abi_outputs(
    sep_df: pd.DataFrame,
    closing_df: pd.DataFrame,
    contested_df: pd.DataFrame,
    xcatch_df: pd.DataFrame,
    *,
    output_dir: str = "../data/abi/metrics/abi",
    write_csv: bool = True,
    min_wr_targets: int = 40,
    min_db_plays: int = 40,
    play_context_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    High-level ABI orchestrator.

    Steps:
        1) Build play-level ABI table (abi_plays).
        2) Add natural-language play categories and sentences.
        3) Build a condensed ABI table for storytelling.
        4) Build WR leaderboard (ABW).
        5) Build DB leaderboard (primary closer).
        6) Optionally save all outputs as CSVs.

    Args:
        sep_df:        Separation play-level table.
        closing_df:    Closing play-level table (best closer per play).
        contested_df:  Contested play-level table.
        xcatch_df:     xCatch predictions table.
        output_dir:    Base directory for ABI outputs.
        write_csv:     If True, write CSVs to disk.
        min_wr_targets: Minimum downfield targets for WR leaderboard.
        min_db_plays:   Minimum plays for DB leaderboard.
        play_context_df: Optional raw play context for blurbs.

    Returns:
        dict:
            {
                "plays":          abi_plays_df,
                "wr_leaderboard": wr_lb_df,
                "db_leaderboard": db_lb_df,
            }
    """
    abi_plays = build_abi_plays_table(
        sep_df=sep_df,
        closing_df=closing_df,
        contested_df=contested_df,
        xcatch_df=xcatch_df,
        play_context_df=play_context_df,
    )
    abi_plays = add_play_categories_and_sentences(abi_plays)
    abi_condensed = build_abi_condensed(abi_plays)

    wr_lb = build_wr_leaderboard_with_abw(abi_plays, min_targets=min_wr_targets)
    db_lb = build_db_leaderboard(abi_plays, min_plays=min_db_plays)

    if write_csv:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        p_plays = out_dir / "abi_results_full.csv"
        p_condensed = out_dir / "abi_results_condensed.csv"
        p_wr = out_dir / "abw_wr_leaderboard.csv"
        p_db = out_dir / "abi_db_leaderboard.csv"

        abi_plays.to_csv(p_plays, index=False)
        abi_condensed.to_csv(p_condensed, index=False)
        wr_lb.to_csv(p_wr, index=False)
        db_lb.to_csv(p_db, index=False)

        print(f"💾 Saved qualifying play ABI scores table → {p_plays} ({len(abi_plays):,} plays)")
        print(f"💾 Saved ABI condensed table              → {p_condensed} ({len(abi_condensed):,} plays)")
        print(f"💾 Saved ABI WR leaderboard               → {p_wr} ({len(wr_lb):,} WRs)")
        print(f"💾 Saved ABI DB leaderboard               → {p_db} ({len(db_lb):,} DBs)")

    return {
        "plays": abi_plays,
        "wr_leaderboard": wr_lb,
        "db_leaderboard": db_lb,
    }
