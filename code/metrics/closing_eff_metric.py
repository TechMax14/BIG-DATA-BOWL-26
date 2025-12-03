#!/usr/bin/env python3
# metrics/closing_metric.py
"""
Defensive Closing Efficiency Metric

This module measures how effectively defenders close space on the targeted
receiver while the ball is in the air.

High-level flow:
    1. `_ball_window`:
        - Derive per-play [start_frame, end_frame] window for ball flight
          using pass_forward/pass_arrived/pass_outcome/interception events.
    2. `compute_closing_from_enriched`:
        - For each defender overlapping the targeted WR during ball flight:
          * track WR–DB separation
          * compute closing rates and pursuit angles
          * compute path length and path efficiency
    3. `score_closing_defender_play`:
        - Convert raw closing features into a 0–25 "closing_eff_25" score.
    4. `aggregate_closing_to_play`:
        - Reduce to the best closer per play.
    5. `defender_leaderboard`:
        - Build stability-filtered defender-level leaderboard.
    6. `run_closing_pipeline`:
        - Public entrypoint used by main/metric_pipeline to generate
          defender- and play-level CSVs plus leaderboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

TARGET_LABELS = {"Targeted Receiver", "TargetedReceiver"}


# ---------------------------------------------------------------------
# Ball-flight window helper
# ---------------------------------------------------------------------

def _ball_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-play [start_frame, end_frame] for ball flight.

    Prefer using tracking events:
        pass_forward -> (pass_arrived | pass_outcome | interception)

    If those events are not present or incomplete, falls back to the
    observed [min(frame_id), max(frame_id)] for each (game_id, play_id).

    Args:
        df: Full tracking slice including multiple plays.

    Returns:
        DataFrame with columns:
            game_id, play_id, start_frame, end_frame
    """
    frame_col = "frame_id"

    if "event" in df.columns:
        ev = (
            df[df["event"].isin(["pass_forward", "pass_arrived",
                                 "pass_outcome", "interception"])]
            [["game_id", "play_id", frame_col, "event"]]
            .sort_values(["game_id", "play_id", frame_col])
        )

        start = (
            ev[ev["event"].eq("pass_forward")]
            .drop_duplicates(["game_id", "play_id"])
            .rename(columns={frame_col: "start_frame"})[
                ["game_id", "play_id", "start_frame"]
            ]
        )

        terms = ev[ev["event"].isin(["pass_arrived", "pass_outcome", "interception"])]
        end = (
            terms.merge(start, on=["game_id", "play_id"], how="inner")
            .query(f"{frame_col} >= start_frame")
            .sort_values(["game_id", "play_id", frame_col])
            .drop_duplicates(["game_id", "play_id"])
            .rename(columns={frame_col: "end_frame"})[
                ["game_id", "play_id", "start_frame", "end_frame"]
            ]
        )
        if not end.empty:
            return end

    # Fallback: use observed min/max frame_id in play
    return (
        df.groupby(["game_id", "play_id"])["frame_id"]
        .agg(start_frame="min", end_frame="max")
        .reset_index()
    )


# ---------------------------------------------------------------------
# Core computation from enriched DF
# ---------------------------------------------------------------------

def compute_closing_from_enriched(
    out_enriched: pd.DataFrame,
    *,
    fps: float = 10.0,
    pass_length_min: Optional[float] = 10.0,
    lookback_full_flight: bool = True,  # True: entire ball flight; False: last ~0.5s only
    tail_window_s: float = 0.5,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Compute defender-centric closing features during ball flight.

    Produces one row per (game_id, play_id, def_nfl_id) with:

        - initial_sep, min_sep, last_sep
        - closed_distance, closed_fraction
        - avg_closing_rate, peak_closing_rate, pct_time_closing
        - angle_efficiency_mean
        - def_path_len, path_efficiency
        - overlap_frames (# of overlapping frames with WR during window)
        - def_name, def_pos
        - optional play context (week, pass_result, pass_length, coverage, etc.)

    Args:
        out_enriched: Enriched tracking data for all players and plays.
        fps: Frames per second of tracking (used to turn frame deltas into rates).
        pass_length_min: Minimum pass_length to include (air_yards filter).
        lookback_full_flight: If True, use the entire ball-flight window.
                              If False, restrict to last `tail_window_s` seconds.
        tail_window_s: Tail length (in seconds) when `lookback_full_flight=False`.
        debug: If True, prints helpful debug messages when no rows pass filters.

    Returns:
        DataFrame with defender-level closing features and play context.
    """
    if out_enriched is None or out_enriched.empty:
        return pd.DataFrame()

    d = out_enriched.copy()

    # numeric hygiene
    for c in ("game_id", "play_id", "frame_id", "nfl_id", "x", "y", "pass_length"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # optional pass-length filter
    if pass_length_min is not None and "pass_length" in d.columns:
        keep = (
            d.loc[d["pass_length"].ge(pass_length_min), ["game_id", "play_id"]]
            .dropna()
            .drop_duplicates()
        )
        if keep.empty:
            return pd.DataFrame()
        d = d.merge(keep, on=["game_id", "play_id"], how="inner")

    # ball flight window
    win = _ball_window(d)
    if win.empty:
        return pd.DataFrame()
    d = d.merge(win, on=["game_id", "play_id"], how="inner")

    # slice frames: full flight or last tail window
    if lookback_full_flight:
        d = d[(d["frame_id"] >= d["start_frame"]) & (d["frame_id"] <= d["end_frame"])]
    else:
        L = max(1, int(round(float(tail_window_s) * float(fps))))
        d = d[(d["frame_id"] <= d["end_frame"]) & (d["frame_id"] >= d["end_frame"] - L)]

    # targeted WR and defenders
    pr = d.get("player_role")
    ps = d.get("player_side")
    tgt_mask = pr.astype(str).isin(TARGET_LABELS) if pr is not None else pd.Series(False, index=d.index)
    if ps is not None:
        tgt_mask &= ps.eq("Offense")

    wr = d.loc[
        tgt_mask,
        ["game_id", "play_id", "frame_id", "nfl_id", "x", "y",
         "player_name", "player_position"],
    ].rename(
        columns={
            "nfl_id": "tgt_nfl_id",
            "x": "tgt_x",
            "y": "tgt_y",
            "player_name": "tgt_name",
            "player_position": "tgt_pos",
        }
    )
    if wr.empty:
        if debug:
            print("[closing] no targeted WR rows")
        return pd.DataFrame()

    def_mask = ps.eq("Defense") if ps is not None else pd.Series(False, index=d.index)
    defs = d.loc[
        def_mask,
        ["game_id", "play_id", "frame_id", "nfl_id", "x", "y",
         "player_name", "player_position"],
    ].rename(
        columns={
            "nfl_id": "def_nfl_id",
            "x": "def_x",
            "y": "def_y",
            "player_name": "def_name",
            "player_position": "def_pos",
        }
    )
    if defs.empty:
        if debug:
            print("[closing] no defender rows")
        return pd.DataFrame()

    # pair by frame: each defender with the targeted WR
    key = ["game_id", "play_id", "frame_id"]
    pairs = defs.merge(wr, on=key, how="inner", validate="many_to_many")
    if pairs.empty:
        if debug:
            print("[closing] no WR/DEF overlaps")
        return pd.DataFrame()

    # framewise separation
    dx = pairs["def_x"].to_numpy(np.float32) - pairs["tgt_x"].to_numpy(np.float32)
    dy = pairs["def_y"].to_numpy(np.float32) - pairs["tgt_y"].to_numpy(np.float32)
    pairs["sep"] = np.hypot(dx, dy, dtype=np.float32)

    # sort and compute defender step vectors
    pairs = pairs.sort_values(["game_id", "play_id", "def_nfl_id", "frame_id"])
    pairs["def_dx"] = pairs.groupby(["game_id", "play_id", "def_nfl_id"])["def_x"].diff().astype("float32")
    pairs["def_dy"] = pairs.groupby(["game_id", "play_id", "def_nfl_id"])["def_y"].diff().astype("float32")

    # pursuit unit vector DEF->WR (negative of sep vector)
    eps = 1e-6
    inv_norm = 1.0 / np.maximum(np.hypot(dx, dy, dtype=np.float32), eps)
    pairs["u_px"] = dx * (-inv_norm)
    pairs["u_py"] = dy * (-inv_norm)

    # defender movement unit
    step_norm = 1.0 / np.maximum(np.hypot(pairs["def_dx"], pairs["def_dy"]), eps)
    pairs["u_mx"] = pairs["def_dx"] * step_norm
    pairs["u_my"] = pairs["def_dy"] * step_norm

    # pursuit alignment (cos θ)
    pairs["angle_eff"] = (pairs["u_px"] * pairs["u_mx"] + pairs["u_py"] * pairs["u_my"]).clip(-1.0, 1.0)

    # build time series per defender
    def_ts = pairs[
        ["game_id", "play_id", "def_nfl_id", "def_name", "def_pos",
         "frame_id", "sep", "angle_eff"]
    ].copy()
    def_ts = def_ts.sort_values(["game_id", "play_id", "def_nfl_id", "frame_id"])

    # rates and flags
    def_ts["dsep"] = def_ts.groupby(["game_id", "play_id", "def_nfl_id"])["sep"].diff().astype("float32")
    def_ts["close_inst"] = (-def_ts["dsep"] * float(fps)).clip(lower=0.0)
    def_ts["is_closing"] = (def_ts["dsep"] < 0).astype("int8")

    # aggregate per defender-play
    agg_sep = (
        def_ts.groupby(["game_id", "play_id", "def_nfl_id"], as_index=False)
        .agg(
            initial_sep=("sep", "first"),
            min_sep=("sep", "min"),
            last_sep=("sep", "last"),
            avg_closing_rate=("close_inst", "mean"),
            peak_closing_rate=("close_inst", "max"),
            pct_time_closing=("is_closing", "mean"),
            angle_efficiency_mean=("angle_eff", "mean"),
        )
    )

    # defender identity (carry name/pos)
    id_map = (
        def_ts.groupby(["game_id", "play_id", "def_nfl_id"], as_index=False)
        .agg(def_name=("def_name", "first"), def_pos=("def_pos", "first"))
    )

    # path length
    def_steps = pairs[["game_id", "play_id", "def_nfl_id", "frame_id", "def_dx", "def_dy"]].copy()
    def_steps["step_len"] = np.hypot(def_steps["def_dx"], def_steps["def_dy"]).astype("float32")
    path = (
        def_steps.groupby(["game_id", "play_id", "def_nfl_id"], as_index=False)["step_len"]
        .sum()
        .rename(columns={"step_len": "def_path_len"})
    )

    # overlap frames per defender-play (for stability filtering)
    frames_per = (
        def_ts.groupby(["game_id", "play_id", "def_nfl_id"], as_index=False)
        .size()
        .rename(columns={"size": "overlap_frames"})
    )

    agg = (
        agg_sep
        .merge(id_map, on=["game_id", "play_id", "def_nfl_id"], how="left")
        .merge(path, on=["game_id", "play_id", "def_nfl_id"], how="left")
        .merge(frames_per, on=["game_id", "play_id", "def_nfl_id"], how="left")
    )

    # opportunity-normalized features
    agg["closed_distance"] = (agg["initial_sep"] - agg["min_sep"]).clip(lower=0.0)
    agg["closed_fraction"] = agg["closed_distance"] / np.maximum(agg["initial_sep"], 1e-6)

    straight_reduction = (agg["initial_sep"] - agg["last_sep"]).clip(lower=0.0)
    agg["path_efficiency"] = (
        straight_reduction / np.maximum(agg["def_path_len"], 1e-6)
    ).clip(0.0, 1.0)

    # play context (carry week, pass_result, pass_length, coverage if present)
    ctx_cols = [
        c
        for c in (
            "week",
            "pass_result",
            "pass_length",
            "route_of_targeted_receiver",
            "team_coverage_type",
        )
        if c in d.columns
    ]
    if ctx_cols:
        ctx_play = d[["game_id", "play_id"] + ctx_cols].drop_duplicates()
        agg = agg.merge(ctx_play, on=["game_id", "play_id"], how="left")

    return agg


# ---------------------------------------------------------------------
# Scoring (0–25)
# ---------------------------------------------------------------------

def _winsorize(s: pd.Series, p_low=0.05, p_high=0.95) -> pd.Series:
    """
    Clamp Series values to the [p_low, p_high] quantile range.

    Used to reduce the impact of outliers before computing z-scores.
    """
    if s.empty:
        return s
    lo, hi = s.quantile(p_low), s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)


def _robust_z(s: pd.Series) -> pd.Series:
    """
    Compute a robust z-score using median and IQR.

    If IQR is zero or not finite, fall back to standard deviation.
    If neither is usable, returns a Series of zeros.
    """
    if s.empty:
        return s
    med = s.median()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr and np.isfinite(iqr) and iqr > 0:
        return (s - med) / iqr
    std = s.std(ddof=0)
    return (s - med) / std if std and np.isfinite(std) and std > 0 else pd.Series(0.0, index=s.index)


def score_closing_defender_play(
    df_def_play: pd.DataFrame,
    *,
    weights: Dict[str, float] | None = None,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> pd.DataFrame:
    """
    Score each defender-play (game_id, play_id, def_nfl_id) with closing_eff_25 (0–25).

    The score is a weighted composite of:
        - closed_fraction
        - avg_closing_rate
        - peak_closing_rate
        - pct_time_closing
        - angle_efficiency_mean
        (optionally path_efficiency, if added to weights)

    Args:
        df_def_play: Defender-level closing feature dataframe.
        weights: Optional dict of feature_name -> weight. If None, uses:
            {
                "closed_fraction": 0.35,
                "avg_closing_rate": 0.30,
                "peak_closing_rate": 0.15,
                "pct_time_closing": 0.10,
                "angle_efficiency_mean": 0.10,
            }
        p_low: Lower quantile for winsorization.
        p_high: Upper quantile for winsorization.

    Returns:
        DataFrame with new columns:
            - closing_eff_raw
            - closing_eff_25 (Int64 in [0, 25])
    """
    if df_def_play is None or df_def_play.empty:
        return df_def_play

    w = weights or {
        "closed_fraction": 0.35,
        "avg_closing_rate": 0.30,
        "peak_closing_rate": 0.15,
        "pct_time_closing": 0.10,
        "angle_efficiency_mean": 0.10,
        # "path_efficiency": 0.00,  # optionally include later
    }

    df = df_def_play.copy()
    parts = []
    for f, wf in w.items():
        if f not in df.columns:
            continue
        s = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
        s = _winsorize(s, p_low, p_high)
        parts.append(_robust_z(s) * float(wf))

    df["closing_eff_raw"] = sum(parts) if parts else np.nan
    if df["closing_eff_raw"].notna().any():
        ranks = df["closing_eff_raw"].rank(method="average", pct=True)
        df["closing_eff_25"] = (ranks * 25).round().astype("Int64").clip(0, 25)
    else:
        df["closing_eff_25"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df


# ---------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------

def aggregate_closing_to_play(
    scored_def_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce defender-level closing to one row per play.

    Chooses the "best closer" per play, defined as the defender with the
    maximum closing_eff_25. This gives a single closing score per play,
    used later in ABI aggregation and visuals.

    Args:
        scored_def_df: Defender-level scores from `score_closing_defender_play`.

    Returns:
        Play-level DataFrame with the best closer and associated metrics.
    """
    if scored_def_df is None or scored_def_df.empty:
        return pd.DataFrame()

    df = scored_def_df.copy()
    if "closing_eff_25" not in df.columns:
        return pd.DataFrame()

    idx = df.groupby(["game_id", "play_id"])["closing_eff_25"].idxmax()
    best = df.loc[idx].reset_index(drop=True)

    keep = [
        c
        for c in [
            "game_id",
            "play_id",
            "week",
            "pass_result",
            "pass_length",
            "route_of_targeted_receiver",
            "team_coverage_type",
            "def_nfl_id",
            "def_name",
            "def_pos",
            "closing_eff_25",
            "closing_eff_raw",
            "closed_fraction",
            "avg_closing_rate",
            "peak_closing_rate",
            "pct_time_closing",
            "angle_efficiency_mean",
            "path_efficiency",
            "overlap_frames",
        ]
        if c in best.columns
    ]
    return best[keep].copy()


def defender_leaderboard(
    scored_def_df: pd.DataFrame,
    *,
    min_opp: int = 40,
    min_overlap_frames: int = 6,
) -> pd.DataFrame:
    """
    Build a defender-level closing leaderboard with stability filters.

    Filters:
        - Keeps only plays with overlap_frames >= min_overlap_frames
        - Keeps only defenders with at least min_opp qualifying plays

    Outputs per defender:
        - plays: Number of qualifying closing opportunities
        - catches_allowed: Completed passes in those opportunities (if available)
        - catch_rate_allowed: catches_allowed / plays
        - avg_closed_fraction
        - avg_avg_closing_rate
        - avg_peak_closing_rate
        - avg_pct_time_closing
        - avg_angle_efficiency_mean
        - avg_path_efficiency
        - avg_closing_eff_25
        - avg_closing_eff_raw
    """
    if scored_def_df is None or scored_def_df.empty:
        return pd.DataFrame()

    df = scored_def_df.copy()

    # stability filter on overlap frames
    if "overlap_frames" in df.columns:
        df = df[df["overlap_frames"] >= min_overlap_frames]

    if df.empty:
        return pd.DataFrame()

    # group keys: always include def_nfl_id; add name/pos if present
    group_keys = ["def_nfl_id"] + [c for c in ["def_name", "def_pos"] if c in df.columns]

    # ---- volume: plays + catches allowed ----
    agg_dict_counts = {"plays": ("game_id", "count")}
    if "pass_result" in df.columns:
        # assuming pass_result == "C" indicates a completed pass
        agg_dict_counts["catches_allowed"] = (
            "pass_result",
            lambda s: (s == "C").sum(),
        )

    counts = df.groupby(group_keys, dropna=False).agg(**agg_dict_counts).reset_index()

    # ---- means for core closing metrics ----
    metric_cols = [
        "closed_fraction",
        "avg_closing_rate",
        "peak_closing_rate",
        "pct_time_closing",
        "angle_efficiency_mean",
        "path_efficiency",
        "closing_eff_raw",
        "closing_eff_25",
    ]
    present_metrics = [c for c in metric_cols if c in df.columns]

    agg_means: dict = {}
    for c in present_metrics:
        agg_means[f"avg_{c}"] = (c, "mean")

    if agg_means:
        per_def = df.groupby(group_keys, dropna=False).agg(**agg_means).reset_index()
    else:
        per_def = counts[group_keys].copy()

    # ---- merge counts + means, apply min_opp filter ----
    lb = per_def.merge(counts, on=group_keys, how="inner")

    lb = lb.query("plays >= @min_opp").reset_index(drop=True)
    if lb.empty:
        return lb

    # catch rate allowed
    if "catches_allowed" in lb.columns:
        lb["catch_rate_allowed"] = lb["catches_allowed"] / lb["plays"]

    # rounding for nicer display
    if "avg_closing_eff_25" in lb.columns:
        lb["avg_closing_eff_25"] = lb["avg_closing_eff_25"].round(2)
    if "avg_closing_eff_raw" in lb.columns:
        lb["avg_closing_eff_raw"] = lb["avg_closing_eff_raw"].round(3)

    for c in [
        "avg_closed_fraction",
        "avg_avg_closing_rate",
        "avg_peak_closing_rate",
        "avg_pct_time_closing",
        "avg_angle_efficiency_mean",
        "avg_path_efficiency",
        "catch_rate_allowed",
    ]:
        if c in lb.columns:
            lb[c] = lb[c].round(3)

    return lb


# ---------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------

def run_closing_pipeline(
    out_enriched: pd.DataFrame,
    *,
    plays_index: Optional[pd.DataFrame] = None,
    pass_length_min: Optional[float] = 10.0,
    fps: float = 10.0,
    lookback_full_flight: bool = True,
    tail_window_s: float = 0.5,
    weights: Dict[str, float] | None = None,
    # IO
    write_intermediate: bool = False,
    path_all_defender_plays: str = "../data/abi/metrics/closing_efficiency/closing_defender_play.csv",
    path_play_scores: str = "../data/abi/metrics/closing_efficiency/closing_eff_plays_scored.csv",
    path_leaderboard: str = "../data/abi/metrics/closing_efficiency/closing_leaderboard.csv",
    min_opp: int = 40,
    min_overlap_frames: int = 6,
    print_top_n: int | None = 10,
) -> dict[str, pd.DataFrame]:
    """
    Full closing efficiency pipeline.

    Steps:
        1. Compute defender-level closing features from enriched tracking.
        2. Score each defender-play on a 0–25 scale (closing_eff_25).
        3. Aggregate to one "best closer" per play.
        4. (Optionally) align play-level scores to a global plays_index.
        5. Write:
            - defender-level CSV (optional)
            - play-level CSV (best closer)
            - defender leaderboard CSV
            - print top-N defenders by closing efficiency (optional)

    Args:
        out_enriched: Enriched tracking data.
        plays_index: Optional play index used to align play-level outputs.
        pass_length_min: Minimum pass length to include for closing metric.
        fps: Frames per second of tracking data.
        lookback_full_flight: Use full ball flight vs tail-only window.
        tail_window_s: Length of the tail window in seconds if not using full flight.
        weights: Optional feature weights passed to `score_closing_defender_play`.
        write_intermediate: If True, write defender-play CSV.
        path_all_defender_plays: Path for defender-play scores.
        path_play_scores: Path for play-level best-closer scores.
        path_leaderboard: Path for defender leaderboard.
        min_opp: Minimum plays required to keep a defender on leaderboard.
        min_overlap_frames: Minimum overlapping frames for a play to be counted.
        print_top_n: Number of defenders to print in top-N summary.

    Returns:
        dict of DataFrames:
            {
                "defender_play": scored defender-play table,
                "play":          play-level best-closer table,
                "leaderboard":   defender-level closing leaderboard,
            }
    """
    df_def = compute_closing_from_enriched(
        out_enriched,
        fps=fps,
        pass_length_min=pass_length_min,
        lookback_full_flight=lookback_full_flight,
        tail_window_s=tail_window_s,
    )
    if df_def.empty:
        print("[warn] closing: no rows produced.")
        return {"defender_play": df_def, "play": pd.DataFrame(), "leaderboard": pd.DataFrame()}

    scored = score_closing_defender_play(df_def, weights=weights)

    play_scores = aggregate_closing_to_play(scored)
    lb = defender_leaderboard(scored, min_opp=min_opp, min_overlap_frames=min_overlap_frames)

    # IO
    if write_intermediate:
        p_all = Path(path_all_defender_plays)
        p_all.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(p_all, index=False)
        print(f"💾 Saved closing (defender-play) → {p_all}  ({len(scored):,} rows)")

    # Align play-level scores to plays_index (game_id, play_id) before saving
    if plays_index is not None and not plays_index.empty and not play_scores.empty:
        keep = plays_index[["game_id", "play_id"]].drop_duplicates()
        before = len(play_scores)
        play_scores = play_scores.merge(keep, on=["game_id", "play_id"], how="inner")
        after = len(play_scores)
        if before != after:
            print(f"🔒 closing aligned to plays_index: {before:,} → {after:,} plays")

    p_play = Path(path_play_scores)
    p_play.parent.mkdir(parents=True, exist_ok=True)
    play_scores.to_csv(p_play, index=False)
    print(f"💾 Saved closing (play-level) → {p_play}  ({len(play_scores):,} plays)")

    p_lb = Path(path_leaderboard)
    p_lb.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(p_lb, index=False)
    print(f"💾 Saved closing leaderboard → {p_lb}  ({len(lb):,} defenders)")

    # Optional Top-N defenders print
    if print_top_n is not None and not lb.empty:

        # ensure sorting for the printed leaderboard
        sort_col = "avg_closing_eff_25"
        if sort_col in lb.columns:
            lb = lb.sort_values(sort_col, ascending=False)

        # base columns: identity + volume
        cols = ["def_nfl_id"]
        if "def_name" in lb.columns:
            cols.append("def_name")
        if "def_pos" in lb.columns:
            cols.append("def_pos")

        cols += ["plays"]
        if "catches_allowed" in lb.columns:
            cols.append("catches_allowed")
        if "catch_rate_allowed" in lb.columns:
            cols.append("catch_rate_allowed")

        # key story metrics
        for c in [
            "avg_closed_fraction",
            "avg_angle_efficiency_mean",
            "avg_path_efficiency",
            "avg_closing_eff_25",
        ]:
            if c in lb.columns:
                cols.append(c)

        top = lb.head(print_top_n)[cols]

        print(
            f"\n🏆 Top {print_top_n} Defenders — Closing Efficiency (0–25) | "
            f"min opps: {min_opp}, min overlap frames/play: {min_overlap_frames}"
        )
        print(top.to_string(index=False))

    return {"defender_play": scored, "play": play_scores, "leaderboard": lb}
