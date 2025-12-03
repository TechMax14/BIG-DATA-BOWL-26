#!/usr/bin/env python3
# metrics/contested_arrival.py
"""
Contested Arrival Severity Metric

This module measures how contested the catch point is at ball arrival for the
targeted receiver. It focuses on the final ~lookback_s seconds of ball flight.

High-level flow:
    1. `_ball_window`:
        - Determine [start_frame, end_frame] for ball flight using tracking events,
          with a fallback to min/max frame_id per play.
    2. `compute_contested_from_enriched`:
        - For each play, compute:
            * sep_at_arrival          : nearest defender–WR separation at arrival
            * n_defenders_r1 / r2     : defenders in 1 / 2 yards at arrival
            * closing_rate_last       : closing rate in the final lookback window
            * pct_time_tight          : fraction of window with tight coverage
          plus WR identity and play context.
    3. `score_contested`:
        - Combine tightness, crowding, and closing into a 0–25
          'contested_severity_25' score, with depth-aware binning.
    4. `run_contested_pipeline`:
        - Public entrypoint used by the overall metric pipeline. Computes,
          scores, optionally aligns to a plays_index, and writes a CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

TARGET_LABELS = {"Targeted Receiver", "TargetedReceiver"}


# ---------------------------------------------------------------------
# Ball-flight window helper (same idea as closing)
# ---------------------------------------------------------------------

def _ball_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find [start_frame, end_frame] per play for ball flight using events if available.

    Priority:
        - Use pass_forward -> (pass_arrived | pass_outcome | interception) events
        - Otherwise, fall back to observed [min(frame_id), max(frame_id)] within play.

    Args:
        df: Tracking data across one or more plays.

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

    # Fallback: use observed min/max frame_id per play
    grp = (
        df.groupby(["game_id", "play_id"])["frame_id"]
        .agg(start_frame="min", end_frame="max")
        .reset_index()
    )
    return grp


# ---------------------------------------------------------------------
# Core computation: contested features per play
# ---------------------------------------------------------------------

def compute_contested_from_enriched(
    out_enriched: pd.DataFrame,
    *,
    fps: float = 10.0,
    pass_length_min: Optional[float] = 10.0,
    tight_threshold_yd: float = 1.0,
    crowd_radii_yd: tuple[float, float] = (1.0, 2.0),
    lookback_s: float = 0.4,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Compute contest severity features per play from enriched tracking.

    Produces one row per (game_id, play_id) measuring contest at arrival:

        - sep_at_arrival:
            Nearest defender distance to the targeted WR at arrival frame.
        - n_defenders_r1 / n_defenders_r2:
            Number of defenders within r1 / r2 yards at arrival.
        - closing_rate_last:
            Average closing rate (yd/s) in the last ~lookback_s before arrival.
        - pct_time_tight:
            Fraction of frames in the lookback window with sep <= tight_threshold_yd.

    Also includes:
        - tgt_nfl_id, tgt_name, tgt_pos (if present)
        - Context fields (week, pass_result, pass_length, route, coverage, etc.)

    Args:
        out_enriched: Enriched tracking data (includes WR/DB positions and events).
        fps: Frames per second of tracking.
        pass_length_min: Filter to plays with minimum pass length (air_yards).
        tight_threshold_yd: Distance threshold (yards) for "tight" coverage.
        crowd_radii_yd: Radii (r1, r2) for counting nearby defenders at arrival.
        lookback_s: Length of the time window (seconds) before arrival to inspect.
        debug: If True, print debug messages when no data available.

    Returns:
        DataFrame with one row per play and contested features.
    """
    if out_enriched is None or out_enriched.empty:
        return pd.DataFrame()

    d = out_enriched.copy()
    frame_col = "frame_id"
    x_col, y_col = "x", "y"

    # numeric hygiene
    for c in ("game_id", "play_id", frame_col, "nfl_id", x_col, y_col, "pass_length"):
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

    # window frames for ball flight
    win = _ball_window(d)
    if win.empty:
        return pd.DataFrame()

    d = d.merge(win, on=["game_id", "play_id"], how="inner")
    d = d[(d[frame_col] >= d["start_frame"]) & (d[frame_col] <= d["end_frame"])]

    # targeted WR rows
    pr = d.get("player_role")
    ps = d.get("player_side")
    tgt_mask = pr.astype(str).isin(TARGET_LABELS) if pr is not None else pd.Series(False, index=d.index)
    if ps is not None:
        tgt_mask &= ps.eq("Offense")

    wr = d.loc[
        tgt_mask,
        ["game_id", "play_id", frame_col, "nfl_id", x_col, y_col,
         "player_name", "player_position"],
    ].rename(
        columns={
            "nfl_id": "tgt_nfl_id",
            x_col: "tgt_x",
            y_col: "tgt_y",
            "player_name": "tgt_name",
            "player_position": "tgt_pos",
        }
    )
    if wr.empty:
        if debug:
            print("[contested] No targeted WR rows.")
        return pd.DataFrame()

    # defenders
    def_mask = ps.eq("Defense") if ps is not None else pd.Series(False, index=d.index)
    defs = d.loc[
        def_mask,
        ["game_id", "play_id", frame_col, "nfl_id", x_col, y_col],
    ].rename(
        columns={
            "nfl_id": "def_nfl_id",
            x_col: "def_x",
            y_col: "def_y",
        }
    )
    if defs.empty:
        if debug:
            print("[contested] No defender rows.")
        return pd.DataFrame()

    # arrival frame per play
    arr = win[["game_id", "play_id", "end_frame"]].rename(columns={"end_frame": "arr_frame"})

    # lookback window length in frames
    L = max(1, int(round(float(lookback_s) * float(fps))))

    # slice WR & DEF to [arr_frame - L, arr_frame]
    wr = wr.merge(arr, on=["game_id", "play_id"], how="inner")
    wr_s = wr[(wr[frame_col] <= wr["arr_frame"]) & (wr[frame_col] >= wr["arr_frame"] - L)]

    defs = defs.merge(arr, on=["game_id", "play_id"], how="inner")
    def_s = defs[(defs[frame_col] <= defs["arr_frame"]) & (defs[frame_col] >= defs["arr_frame"] - L)]

    key = ["game_id", "play_id", frame_col]
    pairs = wr_s.merge(def_s, on=key, how="inner", validate="many_to_many")
    if pairs.empty:
        if debug:
            print("[contested] No WR/DEF pairs in window.")
        return pd.DataFrame()

    # per-frame nearest separation
    dx = pairs["def_x"].to_numpy(np.float32) - pairs["tgt_x"].to_numpy(np.float32)
    dy = pairs["def_y"].to_numpy(np.float32) - pairs["tgt_y"].to_numpy(np.float32)
    pairs["sep"] = np.hypot(dx, dy, dtype=np.float32)

    dmin = (
        pairs.groupby(key, as_index=False)["sep"]
        .min()
        .sort_values(key)
    )

    # sep at arrival (nearest)
    at_arr = dmin.merge(arr, on=["game_id", "play_id"], how="inner")
    at_arr = at_arr[at_arr[frame_col].eq(at_arr["arr_frame"])]
    nearest_at_arr = (
        at_arr.groupby(["game_id", "play_id"])["sep"]
        .min()
        .rename("sep_at_arrival")
        .reset_index()
    )

    # crowding counts at arrival
    r1, r2 = map(float, crowd_radii_yd)
    at_arr_pairs = pairs.merge(arr, on=["game_id", "play_id"], how="inner")
    at_arr_pairs = at_arr_pairs[at_arr_pairs[frame_col].eq(at_arr_pairs["arr_frame"])]

    crowd = (
        at_arr_pairs.assign(
            r1=(at_arr_pairs["sep"] <= r1).astype("int8"),
            r2=(at_arr_pairs["sep"] <= r2).astype("int8"),
        )
        .groupby(["game_id", "play_id"])
        .agg(
            n_defenders_r1=("r1", "sum"),
            n_defenders_r2=("r2", "sum"),
        )
        .reset_index()
    )

    # closing over last window (yd/s; only closing portions)
    dmin["dsep"] = dmin.groupby(["game_id", "play_id"])["sep"].diff().astype("float32")
    dmin["close_inst"] = (-dmin["dsep"] * float(fps)).clip(lower=0.0)
    closing = (
        dmin.groupby(["game_id", "play_id"])["close_inst"]
        .mean()
        .rename("closing_rate_last")
        .reset_index()
    )

    # pct time tight in window
    dmin["tight"] = (dmin["sep"] <= float(tight_threshold_yd)).astype("int8")
    tight_pct = (
        dmin.groupby(["game_id", "play_id"])["tight"]
        .mean()
        .rename("pct_time_tight")
        .reset_index()
    )

    # WR identity
    wr_id_cols = [c for c in ["tgt_nfl_id", "tgt_name", "tgt_pos"] if c in wr.columns]
    ctx_wr = wr[["game_id", "play_id"] + wr_id_cols].drop_duplicates()

    # Play-level context
    play_cols = [
        c
        for c in [
            "week",
            "pass_result",
            "pass_length",
            "route_of_targeted_receiver",
            "team_coverage_type",
        ]
        if c in d.columns
    ]
    if play_cols:
        ctx_play = d[["game_id", "play_id"] + play_cols].drop_duplicates()
        ctx = ctx_wr.merge(ctx_play, on=["game_id", "play_id"], how="left")
    else:
        ctx = ctx_wr.copy()

    out = (
        nearest_at_arr
        .merge(crowd,     on=["game_id", "play_id"], how="left")
        .merge(closing,   on=["game_id", "play_id"], how="left")
        .merge(tight_pct, on=["game_id", "play_id"], how="left")
        .merge(ctx,       on=["game_id", "play_id"], how="left")
    )

    # dtypes
    for c in ("sep_at_arrival", "closing_rate_last", "pct_time_tight"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in ("n_defenders_r1", "n_defenders_r2"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    return out


# ---------------------------------------------------------------------
# Scoring: contested_severity_25 (0–25, higher = more contested)
# ---------------------------------------------------------------------

def score_contested(
    contested_df: pd.DataFrame,
    *,
    weights: Dict[str, float] | None = None,
    winsor: tuple[float, float] = (0.05, 0.95),
    depth_col: str = "pass_length",
    bin_edges: Optional[List[float]] = None,
    min_bin_size: int = 50,
) -> pd.DataFrame:
    """
    Map contested components to a 0–25 'contested_severity_25' score (higher = more contested).

    The score combines:
        - tightness (inverse separation at arrival)
        - crowding (defenders within r1 and r2)
        - closing (closing_rate_last)

    Depth-bin ranking is used so that deep throws are scored relative to other
    deep throws (and not penalized for physics differences vs short throws).

    Args:
        contested_df: Play-level contested features from `compute_contested_from_enriched`.
        weights: Optional dict of component weights:
            default = {"tightness": 0.5, "crowding": 0.3, "closing": 0.2}.
        winsor: (p_low, p_high) quantiles for winsorization on each component.
        depth_col: Column name used for depth binning (e.g., "pass_length").
        bin_edges: Bin edges to use for depth bins. If None, uses [0, 10, 20, 30, 40, inf].
        min_bin_size: Minimum plays required per bin to use bin-specific ranking.

    Returns:
        DataFrame with new columns:
            - contested_raw
            - contested_severity_25 (Int64 in [0, 25])
    """
    if contested_df is None or contested_df.empty:
        return contested_df

    if bin_edges is None:
        bin_edges = [0, 10, 20, 30, 40, float("inf")]

    w = weights or {"tightness": 0.5, "crowding": 0.3, "closing": 0.2}
    p_low, p_high = winsor
    df = contested_df.copy()

    # components where larger = more contested
    tightness = -pd.to_numeric(df["sep_at_arrival"], errors="coerce")  # inverse sep
    crowding = (
        df["n_defenders_r1"].fillna(0).astype(float) * 1.0
        + df["n_defenders_r2"].fillna(0).astype(float) * 0.5
    )
    closing = pd.to_numeric(df["closing_rate_last"], errors="coerce").fillna(0.0)

    def _winsor(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(p_low), s.quantile(p_high)
        return s.clip(lower=lo, upper=hi)

    def _rz(s: pd.Series) -> pd.Series:
        med = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr and np.isfinite(iqr) and iqr > 0:
            return (s - med) / iqr
        std = s.std(ddof=0)
        return (s - med) / std if std and std > 0 else pd.Series(0.0, index=s.index)

    # composite "more contested" signal
    df["contested_raw"] = (
        _rz(_winsor(tightness)) * w["tightness"]
        + _rz(_winsor(crowding)) * w["crowding"]
        + _rz(_winsor(closing)) * w["closing"]
    )

    # depth-bin percentile ranking
    depth = pd.to_numeric(df.get(depth_col), errors="coerce").fillna(0.0).clip(lower=0.0)

    if bin_edges is not None and len(bin_edges) >= 2:
        df["_depth_bin"] = pd.cut(
            depth, bins=bin_edges, right=False, include_lowest=True, labels=False
        )
    else:
        df["_depth_bin"] = 0  # single bin

    # global rank as fallback
    global_rank_25 = df["contested_raw"].rank(method="average", pct=True) * 25.0

    # group rank per bin
    grp_rank_25 = df.groupby("_depth_bin", dropna=False)["contested_raw"].transform(
        lambda s: s.rank(method="average", pct=True) * 25.0
    )

    bin_sizes = df.groupby("_depth_bin")["contested_raw"].transform("size")
    use_grp = bin_sizes >= int(min_bin_size)
    rank_25 = grp_rank_25.where(use_grp, global_rank_25)

    df["contested_severity_25"] = (
        rank_25.round().astype("Int64").clip(0, 25)
    )

    df.drop(columns=["_depth_bin"], errors="ignore", inplace=True)
    return df


# ---------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------

def run_contested_pipeline(
    out_enriched: pd.DataFrame,
    *,
    plays_index: Optional[pd.DataFrame] = None,
    pass_length_min: Optional[float] = 10.0,
    fps: float = 10.0,
    tight_threshold_yd: float = 1.0,
    crowd_radii_yd: tuple[float, float] = (1.0, 2.0),
    lookback_s: float = 0.4,
    weights: Dict[str, float] | None = None,
    winsor: tuple[float, float] = (0.05, 0.95),
    depth_col: str = "pass_length",
    bin_edges: Optional[List[float]] = None,
    min_bin_size: int = 50,
    # IO
    path_contested_plays: str = "../data/abi/metrics/contested_catch/contested_catch_plays_scored.csv",
) -> dict[str, pd.DataFrame]:
    """
    Full contested arrival pipeline.

    Steps:
        1. Compute contest features for the final ~lookback_s of ball flight.
        2. Score each play on a 0–25 contested severity scale.
        3. (Optionally) align to a canonical plays_index.
        4. Write a play-level CSV of contested scores.

    Args:
        out_enriched: Enriched tracking DataFrame.
        plays_index: Optional play index to align outputs to.
        pass_length_min: Minimum pass length (air_yards) filter.
        fps: Frames per second in tracking data.
        tight_threshold_yd: Distance threshold defining "tight" coverage.
        crowd_radii_yd: Radii (r1, r2) used for counting nearby defenders.
        lookback_s: Time window (seconds) before arrival used for features.
        weights: Optional component weights for contested scoring.
        winsor: Quantiles for winsorization of each component.
        depth_col: Depth column for binning (e.g., "pass_length").
        bin_edges: Bin edges for depth bins.
        min_bin_size: Minimum plays required per depth bin before using bin ranks.
        path_contested_plays: Output CSV path for scored contested plays.

    Returns:
        dict of DataFrames:
            {
                "raw":    raw contested features,
                "scored": contested features + contested_severity_25,
            }
    """
    raw = compute_contested_from_enriched(
        out_enriched,
        fps=fps,
        pass_length_min=pass_length_min,
        tight_threshold_yd=tight_threshold_yd,
        crowd_radii_yd=crowd_radii_yd,
        lookback_s=lookback_s,
    )
    if raw.empty:
        print("[warn] contested: no rows produced.")
        return {"raw": raw, "scored": pd.DataFrame()}

    scored = score_contested(
        raw,
        weights=weights,
        winsor=winsor,
        depth_col=depth_col,
        bin_edges=bin_edges,
        min_bin_size=min_bin_size,
    )

    # Align to plays_index
    if plays_index is not None and not plays_index.empty and not scored.empty:
        if "tgt_nfl_id" in scored.columns and "tgt_nfl_id" in plays_index.columns:
            keys = ["game_id", "play_id", "tgt_nfl_id"]
        else:
            keys = ["game_id", "play_id"]

        keep = plays_index[keys].drop_duplicates()
        before = len(scored)
        scored = scored.merge(keep, on=keys, how="inner")
        after = len(scored)
        if before != after:
            print(f"🔒 contested aligned to plays_index: {before:,} → {after:,} plays")

    p_out = Path(path_contested_plays)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(p_out, index=False)
    print(f"💾 Saved contested plays → {p_out}  ({len(scored):,} plays)")

    return {"raw": raw, "scored": scored}
