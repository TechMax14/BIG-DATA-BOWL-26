# metrics/separation_created.py
"""
Separation Creation Metric

This module computes how much separation the targeted receiver creates (or loses)
relative to the nearest defender while the ball is in the air.

Pipeline overview:
    1. `compute_from_enriched`:
        - Takes enriched tracking data (out_enriched) and:
          * identifies targeted WR rows
          * identifies defensive rows
          * pairs WR–DB by (game, play, frame)
          * computes per-frame separation and summarizes per defender
    2. `reduce_to_nearest_per_play`:
        - Reduces to a single nearest defender per play (at throw or catch).
    3. `compute_sep_scores`:
        - Converts raw separation deltas into a 0–25 scaled score.
    4. `run_sep_pipeline`:
        - Orchestrates the whole separation metric flow and writes outputs:
          * play-level scored table
          * WR separation leaderboard (40+ targets by default)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

TARGET_LABELS = {"Targeted Receiver", "TargetedReceiver"}


# ---------- core computation from enriched ----------

def compute_from_enriched(
    out_enriched: pd.DataFrame,
    fps: float = 10.0,
    pass_length_min: Optional[float] = 20.0,   # set None to disable
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build defender–target separation summaries from enriched tracking data.

    For each (game_id, play_id, def_nfl_id), this computes:
        - first / last frame with overlapping WR–DB tracking
        - separation at first and last frame
        - delta in separation
        - duration in seconds
        - delta_per_s (change in separation per second)

    Args:
        out_enriched: Enriched tracking DataFrame containing at least:
            game_id, play_id, frame_id, nfl_id, x, y, player_role, player_side.
        fps: Frames per second in the tracking data.
        pass_length_min: If not None, restricts to plays where pass_length
            >= pass_length_min. Set to None to disable filtering.
        debug: If True, prints debug messages when no data is found.

    Returns:
        DataFrame with one row per (game_id, play_id, def_nfl_id) containing
        separation summary statistics and optional play context.
    """
    if out_enriched is None or out_enriched.empty:
        return pd.DataFrame()

    d = out_enriched.copy()

    # numeric hygiene
    for c in ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "pass_length"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # optional play filter on pass_length
    if pass_length_min is not None and "pass_length" in d.columns:
        keep_keys = (
            d.loc[d["pass_length"].ge(pass_length_min), ["game_id", "play_id"]]
             .dropna().drop_duplicates()
        )
        if keep_keys.empty:
            return pd.DataFrame()
        d = d.merge(keep_keys, on=["game_id", "play_id"], how="inner")

    pr = d.get("player_role")
    ps = d.get("player_side")

    # --- target rows ---
    tgt_mask = pr.astype(str).isin(TARGET_LABELS) if pr is not None else pd.Series(False, index=d.index)
    if ps is not None:
        tgt_mask &= ps.eq("Offense")

    tgt = d.loc[tgt_mask, [
        "game_id","play_id","frame_id","nfl_id","x","y",
        "player_name","player_position"
    ]].rename(columns={
        "nfl_id": "tgt_nfl_id",
        "x": "tgt_x",
        "y": "tgt_y",
        "player_name": "tgt_name",
        "player_position": "tgt_pos",
    })

    if tgt.empty:
        if debug:
            print("[debug] No target rows found in enriched output.")
        return pd.DataFrame()

    # --- defender rows ---
    def_mask = ps.eq("Defense") if ps is not None else pd.Series(False, index=d.index)
    defs = d.loc[def_mask, [
        "game_id","play_id","frame_id","nfl_id","x","y",
        "player_name","player_position"
    ]].rename(columns={
        "nfl_id": "def_nfl_id",
        "x": "def_x",
        "y": "def_y",
        "player_name": "def_name",
        "player_position": "def_pos",
    })

    if defs.empty:
        if debug:
            print("[debug] No defender rows found in enriched output.")
        return pd.DataFrame()

    # pair by frame
    key = ["game_id", "play_id", "frame_id"]
    pairs = defs.merge(tgt, on=key, how="inner")
    if pairs.empty:
        if debug:
            print("[debug] Defender/Target had no overlapping frames.")
        return pd.DataFrame()

    # per-frame separation
    pairs["sep"] = np.hypot(pairs["def_x"] - pairs["tgt_x"], pairs["def_y"] - pairs["tgt_y"])
    pairs = pairs.sort_values(["game_id", "play_id", "def_nfl_id", "frame_id"])

    # per-(game, play, defender)
    def _agg(g: pd.DataFrame) -> pd.Series:
        f0, f1 = int(g["frame_id"].min()), int(g["frame_id"].max())
        s0 = float(g.loc[g["frame_id"].idxmin(), "sep"])
        s1 = float(g.loc[g["frame_id"].idxmax(), "sep"])
        delta = s1 - s0
        dur = max((f1 - f0) / float(fps), 0.0)
        dps = (delta / dur) if dur > 0 else np.nan

        return pd.Series({
            "tgt_nfl_id": g["tgt_nfl_id"].iloc[0],
            "tgt_name": g["tgt_name"].dropna().iloc[0] if g["tgt_name"].notna().any() else None,
            "tgt_pos": g["tgt_pos"].dropna().iloc[0] if g["tgt_pos"].notna().any() else None,
            #"def_nfl_id": g["def_nfl_id"].iloc[0],
            "def_name": g["def_name"].dropna().iloc[0] if g["def_name"].notna().any() else None,
            "def_pos": g["def_pos"].dropna().iloc[0] if g["def_pos"].notna().any() else None,
            "first_frame": f0,
            "last_frame": f1,
            "first_sep": s0,
            "last_sep": s1,
            "delta": delta,
            "duration_s": dur,
            "delta_per_s": dps,
        })

    out = (
        pairs.groupby(["game_id", "play_id", "def_nfl_id"], sort=False)
             .apply(_agg)
             .reset_index()
    )

    # add play context (week + pass_result, etc.)
    add_cols = [
        c for c in [
            "week", "pass_result", "pass_length",
            "route_of_targeted_receiver", "play_description",
            "team_coverage_type",
        ]
        if c in d.columns
    ]
    if add_cols:
        ctx = d[["game_id", "play_id"] + add_cols].drop_duplicates()
        out = out.merge(ctx, on=["game_id", "play_id"], how="left")

    return out


# ---------- separation scoring helpers ----------

def _winsorize(s: pd.Series, p_low=0.05, p_high=0.95) -> pd.Series:
    """
    Clamp extreme values in a Series to the [p_low, p_high] quantile range.

    This makes the scoring robust to outliers before z-scoring.
    """
    if s.empty:
        return s
    lo, hi = s.quantile(p_low), s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)


def _robust_z(s: pd.Series) -> pd.Series:
    """
    Compute a robust z-score using median and IQR.

    Falls back to standard deviation if IQR is 0 or not finite.
    If neither is usable, returns zeros.
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


def reduce_to_nearest_per_play(sep_df: pd.DataFrame, mode: str = "throw") -> pd.DataFrame:
    """
    Keep exactly one defender per (game_id, play_id):

        - mode='throw' -> pick defender with min(first_sep)
        - mode='catch' -> pick defender with min(last_sep)

    This reduces the table down to a single "nearest defender" view per play.
    """
    if sep_df is None or sep_df.empty:
        return sep_df
    metric_col = {"throw": "first_sep", "catch": "last_sep"}.get(mode, "first_sep")
    if metric_col not in sep_df.columns:
        return sep_df
    idx = sep_df.groupby(["game_id", "play_id"], dropna=False)[metric_col].idxmin()
    out = sep_df.loc[idx].reset_index(drop=True)
    out["nearest_mode"] = mode
    return out


def compute_sep_scores(
    play_df: pd.DataFrame,
    weights: Dict[str, float] | None = None,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> pd.DataFrame:
    """Add sep_score_raw and sep_score_25 (0–25 integer) to a play-level DataFrame.

    Uses a weighted combination of features (default: delta_per_s and delta),
    winsorizes each feature, applies robust z-scoring, and then maps the
    resulting composite to a 0–25 percentile-based score.

    Args:
        play_df: Play-level separation summary dataframe.
        weights: Optional mapping of feature -> weight. If None, uses:
            {"delta_per_s": 0.7, "delta": 0.3}
        p_low: Lower quantile for winsorization.
        p_high: Upper quantile for winsorization.

    Returns:
        The same dataframe with:
            - sep_score_raw (float)
            - sep_score_25 (Int64 in [0, 25])
    """
    if play_df is None or play_df.empty:
        return play_df
    feats_default = {"delta_per_s": 0.7, "delta": 0.3}
    w = weights or feats_default
    use_feats = [f for f in w.keys() if f in play_df.columns]
    if not use_feats:
        out = play_df.copy()
        out["sep_score_raw"] = np.nan
        out["sep_score_25"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        return out

    comp = []
    for f in use_feats:
        s = pd.to_numeric(play_df[f], errors="coerce")
        s = _winsorize(s, p_low, p_high)
        comp.append(_robust_z(s) * float(w[f]))

    out = play_df.copy()
    out["sep_score_raw"] = sum(comp)
    if out["sep_score_raw"].notna().any():
        ranks = out["sep_score_raw"].rank(method="average", pct=True)
        out["sep_score_25"] = (ranks * 25).round().astype("Int64").clip(0, 25)
    else:
        out["sep_score_25"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    return out


def receiver_play_agg(sep_df: pd.DataFrame) -> pd.DataFrame:
    """Build a play-level table (one row per target) used for WR aggregation / visuals."""
    if sep_df is None or sep_df.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "game_id","play_id","week",
        "tgt_nfl_id","tgt_name","tgt_pos",
        "def_nfl_id","def_name","def_pos",
        "pass_result","pass_length",
        "route_of_targeted_receiver","team_coverage_type",
        "delta_per_s","delta","duration_s","first_sep","last_sep",
        "sep_score_25_throw","sep_score_25_catch",
        "sep_score_air_blend","sep_score_25","sep_delta_25",
    ] if c in sep_df.columns]
    return sep_df[cols].copy()


def receiver_counts(play_agg: pd.DataFrame) -> pd.DataFrame:
    """Compute targets, catches, and catch rate per receiver."""
    if play_agg is None or play_agg.empty:
        return pd.DataFrame()
    counts = (
        play_agg.groupby(["tgt_nfl_id", "tgt_name", "tgt_pos"], dropna=False)
                .agg(targets=("game_id", "count"),
                     catches=("pass_result", lambda s: (s == "C").sum()))
                .reset_index()
    )
    counts["catch_rate"] = counts["catches"] / counts["targets"]
    return counts


def print_top_receivers_by_sep(
    play_agg: pd.DataFrame,
    counts: pd.DataFrame,
    n: int = 10,
    min_targets: int = 40,
    pass_length_min: float | None = None,
    nearest_mode_label: str = "air",
    out_csv: str | None = "../data/abi/metrics/separation_delta/sep_delta_leaderboard.csv",
) -> pd.DataFrame:
    """
    Build and optionally print a WR separation leaderboard.

    The leaderboard aggregates per-play separation scores and keeps receivers
    with at least `min_targets` targets, reporting:

        - avg_sep_delta_25 (0–25 scale)
        - avg_delta_per_s
        - avg_delta
        - catch volume & catch rate

    Args:
        play_agg: Play-level separation table from `receiver_play_agg`.
        counts: Targets / catches per receiver from `receiver_counts`.
        n: Number of rows to print to stdout.
        min_targets: Minimum targets required to appear on the leaderboard.
        pass_length_min: Passed through for display only (context text).
        nearest_mode_label: Label for the "view" (e.g., 'throw', 'catch', 'air').
        out_csv: Path for saving the leaderboard CSV. If None, no file is written.

    Returns:
        Leaderboard DataFrame (may be empty if no eligible receivers).
    """
    if play_agg is None or play_agg.empty or counts is None or counts.empty:
        print("[warn] No data for leaderboard.")
        return pd.DataFrame()
    if "sep_delta_25" not in play_agg.columns:
        print("[warn] sep_delta_25 not found in play_agg.")
        return pd.DataFrame()

    # filter to receivers with enough volume
    eligible = counts[counts["targets"] >= min_targets]
    if eligible.empty:
        print(f"[warn] No receivers with at least {min_targets} targets.")
        return pd.DataFrame()

    group_cols = ["tgt_nfl_id","tgt_name","tgt_pos"]
    metric_cols = [c for c in [
        "sep_delta_25",
        "delta_per_s","delta",
        "duration_s","first_sep","last_sep",
    ] if c in play_agg.columns]

    if not metric_cols:
        print("[warn] No metric columns found for aggregation.")
        return pd.DataFrame()

    per_receiver = (
        play_agg.groupby(group_cols, dropna=False)[metric_cols]
                .mean()
                .reset_index()
    )

    # rename averages to "avg_*"
    for col in metric_cols:
        per_receiver[f"avg_{col}"] = per_receiver[col]
    per_receiver = per_receiver.drop(columns=metric_cols)

    # merge in volume stats
    lb = (
        per_receiver.merge(eligible, on=group_cols, how="inner")
                    .reset_index(drop=True)
    )

    # sort by ABI-style separation score
    sort_col = "avg_sep_delta_25"
    if sort_col in lb.columns:
        lb = lb.sort_values(sort_col, ascending=False)
    lb["avg_sep_delta_25"] = lb.get("avg_sep_delta_25", pd.Series(dtype=float)).round(2)
    for col in ["avg_delta_per_s","avg_delta","avg_duration_s","avg_first_sep","avg_last_sep"]:
        if col in lb.columns:
            lb[col] = lb[col].round(3)

    if out_csv:
        p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
        lb.to_csv(p, index=False)
        print(f"💾 Separation leaderboard exported → {p} ({len(lb):,} players)")

    # print top N
    cols_to_show = [c for c in [
        "tgt_nfl_id","tgt_name","tgt_pos",
        "targets","catches","catch_rate",
        "avg_sep_delta_25","avg_delta_per_s","avg_delta",
    ] if c in lb.columns]

    topn = lb.head(n)[cols_to_show]
    print(
        f"\n🏆 Top {n} Targeted Receivers — Separation Δ ABI (0–25) | "
        f"min targets: {min_targets} | min pass length: {pass_length_min} | view: {nearest_mode_label}"
    )
    print(topn.to_string(index=False))

    return lb


# ---------- public pipeline ----------

def run_sep_pipeline(
    out_enriched: pd.DataFrame,
    *,
    plays_index: Optional[pd.DataFrame] = None,
    pass_length_min: float = 10.0,
    fps: float = 10.0,
    weights: Dict[str, float] | None = None,
    write_intermediate: bool = False,
    # NEW default paths under data/abi/metrics/separation_delta
    path_all: str = "../data/abi/metrics/separation_delta/sep_delta_all.csv",
    path_combined: str = "../data/abi/metrics/separation_delta/sep_delta_plays_scored.csv",
    path_leaderboard: str = "../data/abi/metrics/separation_delta/separation_leaderboard.csv",
    min_targets: int = 40,
    print_top_n: int | None = 10,
    weight_throw: float = 0.8,
    weight_catch: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """
    Full separation metric pipeline.

    Steps:
        1. Compute defender–target separation summaries from enriched tracking.
        2. Reduce to nearest defender at throw and at catch.
        3. Score both views (0–25) using `compute_sep_scores`.
        4. Blend throw/catch scores into a single "air" view.
        5. Optionally align to `plays_index`.
        6. Write play-level and leaderboard CSVs.

    Args:
        out_enriched: Enriched tracking DataFrame used for all metrics.
        plays_index: Optional canonical play index used to align outputs
            to the global set of qualifying plays.
        pass_length_min: Minimum pass length to include in the metric.
        fps: Frames per second in tracking data.
        weights: Optional feature weights for scoring.
        write_intermediate: If True, writes full defender-level separation table.
        path_all: CSV path for all defender-level separation rows.
        path_combined: CSV path for the combined play-level separation scores.
        path_leaderboard: CSV path for WR separation leaderboard.
        min_targets: Minimum targets required to appear on WR leaderboard.
        print_top_n: Number of WRs to print in leaderboard summary.
        weight_throw: Weight assigned to throw-view score in air-blend.
        weight_catch: Weight assigned to catch-view score in air-blend.

    Returns:
        dict of DataFrames:
            {
                "sep_all":    all defender–target separation rows,
                "throw_play": nearest-defender (throw) play table with scores,
                "catch_play": nearest-defender (catch) play table with scores,
                "combined":   blended "air" separation table per play,
                "leaderboard": WR-level separation leaderboard,
            }
    """
    sep_all = compute_from_enriched(out_enriched, fps=fps, pass_length_min=pass_length_min)
    if sep_all is None or sep_all.empty:
        print("No separation rows returned (try lowering pass_length_min).")
        return {"sep_all": pd.DataFrame(), "throw_play": pd.DataFrame(),
                "catch_play": pd.DataFrame(), "combined": pd.DataFrame(),
                "leaderboard": pd.DataFrame()}

    if write_intermediate:
        p_all = Path(path_all); p_all.parent.mkdir(parents=True, exist_ok=True)
        sep_all.to_csv(p_all, index=False)
        print(f"💾 Saved ALL-defenders separation → {p_all}  ({len(sep_all):,} rows)")

    # reduce & score
    throw_play = compute_sep_scores(reduce_to_nearest_per_play(sep_all, mode="throw"), weights=weights)
    catch_play = compute_sep_scores(reduce_to_nearest_per_play(sep_all, mode="catch"), weights=weights)

    keys = ["game_id","play_id","tgt_nfl_id"]

    # keep a richer set of columns from the throw view:
    keep_cols_throw = [c for c in [
        "tgt_name","tgt_pos","week","pass_result","pass_length",
        "route_of_targeted_receiver","play_description","team_coverage_type",
        "def_nfl_id","def_name","def_pos",
        "first_frame","last_frame","first_sep","last_sep",
        "delta","duration_s","delta_per_s",
    ] if c in throw_play.columns]

    merged = throw_play[keys + keep_cols_throw + ["sep_score_25"]].merge(
        catch_play[keys + ["sep_score_25"]],
        on=keys,
        how="inner",
        suffixes=("_throw","_catch"),
        validate="one_to_one"
    )

    merged["sep_score_air_blend"] = (
        merged["sep_score_25_throw"].astype(float) * float(weight_throw) +
        merged["sep_score_25_catch"].astype(float) * float(weight_catch)
    )
    ranks = merged["sep_score_air_blend"].rank(method="average", pct=True)
    merged["sep_score_25"] = (ranks * 25).round().astype("Int64").clip(0, 25)

    # rename final ABI-style score
    merged["sep_delta_25"] = merged["sep_score_25"]

    out_cols = [c for c in [
        "game_id","play_id","week",
        "tgt_nfl_id","tgt_name","tgt_pos",
        "def_nfl_id","def_name","def_pos",
        "pass_result","pass_length",
        "route_of_targeted_receiver","team_coverage_type",
        "first_frame","last_frame","first_sep","last_sep",
        "delta","duration_s","delta_per_s",
        "sep_score_25_throw","sep_score_25_catch",
        "sep_score_air_blend","sep_delta_25",
    ] if c in merged.columns]
    combined = merged[out_cols].copy()

    # align to plays_index if provided (simple inner merge)
    if plays_index is not None and not plays_index.empty:
        keys_align = ["game_id", "play_id", "tgt_nfl_id"]
        keep = plays_index[keys_align].drop_duplicates()
        before = len(combined)
        combined = combined.merge(keep, on=keys_align, how="inner")
        after = len(combined)
        if before != after:
            print(f"🔒 separation aligned to plays_index: {before:,} → {after:,} plays")

    # --- PLAY-LEVEL CSV (for ABI visuals per play) ---
    p_out = Path(path_combined); p_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(p_out, index=False)
    print(f"💾 Saved separation Δ ABI plays → {p_out}  ({len(combined):,} plays)")

    # --- LEADERBOARD CSV (40+ targets) ---
    play_agg = receiver_play_agg(combined)
    counts   = receiver_counts(play_agg)
    lb = pd.DataFrame()
    if print_top_n is not None:
        lb = print_top_receivers_by_sep(
            play_agg=play_agg,
            counts=counts,
            n=print_top_n,
            min_targets=min_targets,
            pass_length_min=pass_length_min,
            nearest_mode_label=f"air({weight_throw:.1f}T/{weight_catch:.1f}C)",
            out_csv=path_leaderboard,
        )
    else:
        lb = print_top_receivers_by_sep(
            play_agg=play_agg,
            counts=counts,
            n=0,  # don't print
            min_targets=min_targets,
            pass_length_min=pass_length_min,
            nearest_mode_label=f"air({weight_throw:.1f}T/{weight_catch:.1f}C)",
            out_csv=path_leaderboard,
        )

    return {
        "sep_all": sep_all,
        "throw_play": throw_play,
        "catch_play": catch_play,
        "combined": combined,
        "leaderboard": lb,
    }
