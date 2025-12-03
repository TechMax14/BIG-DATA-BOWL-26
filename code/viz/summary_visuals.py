#!/usr/bin/env python3
"""
summary_visuals.py

Summary / leaderboard visuals for ABI + ABW metrics.

Key outputs:
  • WR leaderboards:
      - Air Battle Win score (ABW, 0–100)
      - Separation component
      - Catch-over-expected component
  • Scheme-level views:
      - Route type profile (per route_of_targeted_receiver)
      - Coverage type profile (per team_coverage_type)
      - Route × coverage heatmaps
  • Team-level leaderboards:
      - Offensive separation creation
      - Defensive closing efficiency
      - Defensive contest tightness
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_wr_leaderboard(path: PathLike) -> pd.DataFrame:
    """
    Load the WR / targeted-receiver leaderboard.

    Supports both the older abi_wr_leaderboard.csv and the newer
    abw_wr_leaderboard.csv produced by build_wr_leaderboard_with_abw.

    Normalizes columns so downstream plotting code can rely on:
      • targets
      • avg_abi_100
      • avg_sep_delta_25
      • avg_sep_delta
      • avg_xcatch_prob
      • avg_xcatch_surprise_25
      • abw_100
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Normalize column names across older/newer versions
    if "n_targets" in df.columns and "targets" not in df.columns:
        df["targets"] = df["n_targets"]

    if "abi_avg_100" in df.columns and "avg_abi_100" not in df.columns:
        df["avg_abi_100"] = df["abi_avg_100"]

    if "sep_delta_25_mean" in df.columns and "avg_sep_delta_25" not in df.columns:
        df["avg_sep_delta_25"] = df["sep_delta_25_mean"]

    if "sep_delta_25_mean" in df.columns and "avg_sep_delta" not in df.columns:
        df["avg_sep_delta"] = df["sep_delta_25_mean"]

    if "xcatch_prob_mean" in df.columns and "avg_xcatch_prob" not in df.columns:
        df["avg_xcatch_prob"] = df["xcatch_prob_mean"]

    if (
        "xcatch_surprise_25_mean" in df.columns
        and "avg_xcatch_surprise_25" not in df.columns
    ):
        df["avg_xcatch_surprise_25"] = df["xcatch_surprise_25_mean"]

    # If ABW not explicitly present, fall back to ABI average
    if "abw_100" not in df.columns and "avg_abi_100" in df.columns:
        df["abw_100"] = df["avg_abi_100"]

    # Basic hygiene
    for c in ("targets", "n_targets"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------
# 1) Top ABI / ABW WRs
# ---------------------------------------------------------------------

def plot_wr_abw_leaderboard(
    df: pd.DataFrame,
    out_path: PathLike,
    top_n: int = 10,
) -> None:
    """
    Top targeted receivers by Air Battle Win score (ABW, 0–100).

    Uses:
      • abw_100          : final WR-centric score (0–100)
      • targets/n_targets: volume of downfield targets
    """
    if "abw_100" not in df.columns:
        raise ValueError("WR leaderboard missing 'abw_100' column.")

    out_path = Path(out_path)

    df_plot = df.sort_values("abw_100", ascending=False).head(top_n).copy()
    df_plot = df_plot.iloc[::-1]  # best at top

    league_avg = df["abw_100"].mean()

    names = df_plot["tgt_name"]
    scores = df_plot["abw_100"]
    targets = df_plot["targets"] if "targets" in df_plot.columns else df_plot["n_targets"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(names, scores, color="#4a90e2")
    for i, (val, tgt) in enumerate(zip(scores, targets)):
        ax.text(
            val + 0.5,
            i,
            f"{val:.1f}  ({tgt} Tgt)",
            va="center",
            fontsize=10,
        )

    ax.axvline(
        league_avg,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"League avg ({league_avg:.1f})",
    )

    ax.set_xlabel("Air Battle Win score (0–100)")
    ax.set_title(
        "Top targeted receivers by Air Battle Win score\n"
        "(min 40 downfield targets, pass length ≥ 10 air yards)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 2) Top Separation Creators (component of ABW)
# ---------------------------------------------------------------------

def plot_wr_separation_component(
    df: pd.DataFrame,
    out_path: PathLike,
    top_n: int = 10,
) -> None:
    """
    Top targeted receivers by separation component of ABW.

    Uses:
      • sep_score_100     : separation contribution to ABW (0–100)
      • avg_sep_delta_25  : underlying separation score (0–25)
    """
    required = {"sep_score_100", "avg_sep_delta_25"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WR leaderboard missing columns: {missing}")

    out_path = Path(out_path)

    df_plot = df.sort_values("sep_score_100", ascending=False).head(top_n).copy()
    df_plot = df_plot.iloc[::-1]

    names = df_plot["tgt_name"]
    sep_score = df_plot["sep_score_100"]
    sep25 = df_plot["avg_sep_delta_25"]
    targets = df_plot["targets"] if "targets" in df_plot.columns else df_plot["n_targets"]

    league_avg = df["sep_score_100"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, sep_score, color="#2ecc71")

    for i, (val, sep_val, tgt) in enumerate(zip(sep_score, sep25, targets)):
        ax.text(
            val + 0.5,
            i,
            f"{sep_val:.2f} / 25  ({tgt} Tgt)",
            va="center",
            fontsize=10,
        )

    ax.axvline(
        league_avg,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"League avg ({league_avg:.1f})",
    )

    ax.set_xlabel("Separation contribution (0–100)")
    ax.set_title(
        "Top targeted receivers by separation creation\n"
        "(component of Air Battle Win score)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 3) Catch Rate vs Expected (hands / ball skills component)
# ---------------------------------------------------------------------

def plot_wr_catch_over_expected_component(
    df: pd.DataFrame,
    out_path: PathLike,
    top_n: int = 10,
) -> None:
    """
    Top targeted receivers by catch rate over expected (ball skills component).

    Uses:
      • catch_over_expected_pct : (actual - expected) in percentage points
      • catch_rate              : observed catch rate
      • xcatch_prob_mean        : expected catch probability
    """
    required = {"catch_over_expected_pct", "catch_rate", "xcatch_prob_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WR leaderboard missing columns: {missing}")

    out_path = Path(out_path)

    df_plot = df.sort_values("catch_over_expected_pct", ascending=False).head(top_n).copy()
    df_plot = df_plot.iloc[::-1]

    names = df_plot["tgt_name"]
    diff_pct = df_plot["catch_over_expected_pct"]
    catch_rate = df_plot["catch_rate"]
    exp_rate = df_plot["xcatch_prob_mean"]
    targets = df_plot["targets"] if "targets" in df_plot.columns else df_plot["n_targets"]

    league_avg = df["catch_over_expected_pct"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, diff_pct, color="#f4a742")

    for i, (d, act, exp, tgt) in enumerate(zip(diff_pct, catch_rate, exp_rate, targets)):
        ax.text(
            d + 0.5,
            i,
            f"+{d:.1f} pts  ({act:.1%} vs {exp:.1%}, {tgt} Tgt)",
            va="center",
            fontsize=9,
        )

    ax.axvline(
        league_avg,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"League avg ({league_avg:+.1f} pts)",
    )

    ax.set_xlabel("Catch rate over expected (percentage points)")
    ax.set_title(
        "Top targeted receivers by catch rate over expected\n"
        "(component of Air Battle Win score)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 4) Route type profile
# ---------------------------------------------------------------------

def plot_route_type_profile(
    abi_full: pd.DataFrame,
    out_path: PathLike,
    *,
    min_targets: int = 40,
) -> None:
    """
    For each route_of_targeted_receiver (with at least `min_targets`),
    show average:
      • separation (sep_delta_25),
      • closing (closing_eff_25),
      • contest (contested_severity_25),
      • catch difficulty (xcatch_surprise_25).

    Produces a 4-panel horizontal bar chart where all panels share the
    same route order (sorted by separation score).
    """
    out_path = Path(out_path).with_suffix(".png")

    df = abi_full.copy()

    needed = {
        "route_of_targeted_receiver",
        "sep_delta_25",
        "closing_eff_25",
        "contested_severity_25",
        "xcatch_surprise_25",
        "abi_100",
        "play_id",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"ABI full missing columns for route profile: {missing}")

    grp = df.groupby("route_of_targeted_receiver", dropna=True)
    agg = grp.agg(
        n_targets=("play_id", "count"),
        sep25=("sep_delta_25", "mean"),
        closing25=("closing_eff_25", "mean"),
        contest25=("contested_severity_25", "mean"),
        xcatch25=("xcatch_surprise_25", "mean"),
        abi100=("abi_100", "mean"),
    ).reset_index()

    agg = agg[agg["n_targets"] >= min_targets].copy()
    if agg.empty:
        raise ValueError("No routes pass min_targets filter for route profile.")

    agg = agg.sort_values("sep25", ascending=False)

    routes = agg["route_of_targeted_receiver"].tolist()
    y_pos = np.arange(len(routes))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7, 7),
        sharey=True,
    )

    def _style_axis(ax, xlabel: str):
        ax.set_yticks(y_pos)
        ax.set_yticklabels(routes, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_facecolor("#f7f7f7")
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_color("#cccccc")
            ax.spines[side].set_linewidth(1.0)

    # Separation
    ax0 = axes[0]
    ax0.barh(y_pos, agg["sep25"], color="#5cb85c", alpha=0.9)
    _style_axis(ax0, "Separation score (0–25)")
    ax0.set_title(
        f"Route type profile – separation & contest metrics\n"
        f"(routes with ≥ {min_targets} targets, pass length ≥ 10 air yards)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # Closing
    ax1 = axes[1]
    ax1.barh(y_pos, agg["closing25"], color="#4c8eda", alpha=0.9)
    _style_axis(ax1, "Closing score (0–25)")

    # Contest
    ax2 = axes[2]
    ax2.barh(y_pos, agg["contest25"], color="#f0ad4e", alpha=0.9)
    _style_axis(ax2, "Contest score (0–25)")

    # Catch difficulty
    ax3 = axes[3]
    ax3.barh(y_pos, agg["xcatch25"], color="#9467bd", alpha=0.9)
    _style_axis(ax3, "Catch difficulty score (0–25)")

    # Annotate target counts on last axis
    for y, n in zip(y_pos, agg["n_targets"]):
        ax3.text(
            ax3.get_xlim()[1] * 0.98,
            y,
            f"{n} tgt",
            va="center",
            ha="right",
            fontsize=8,
            color="#444",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 5) Coverage type profile
# ---------------------------------------------------------------------

def plot_coverage_type_profile(
    abi_full: pd.DataFrame,
    out_path: PathLike,
    *,
    min_plays: int = 80,
) -> None:
    """
    For each team_coverage_type (with at least `min_plays`),
    show average separation allowed, closing, contest, and ABI allowed.

    Produces a 4-panel horizontal bar chart ordered by separation allowed
    (higher sep25 = more space, i.e. worse in coverage).
    """
    out_path = Path(out_path).with_suffix(".png")

    df = abi_full.copy()

    needed = {
        "team_coverage_type",
        "sep_delta_25",
        "closing_eff_25",
        "contested_severity_25",
        "xcatch_surprise_25",
        "abi_100",
        "play_id",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"ABI full missing columns for coverage profile: {missing}")

    grp = df.groupby("team_coverage_type", dropna=True)
    agg = grp.agg(
        n_plays=("play_id", "count"),
        sep25=("sep_delta_25", "mean"),
        closing25=("closing_eff_25", "mean"),
        contest25=("contested_severity_25", "mean"),
        xcatch25=("xcatch_surprise_25", "mean"),
        abi100=("abi_100", "mean"),
    ).reset_index()

    agg = agg[agg["n_plays"] >= min_plays].copy()
    if agg.empty:
        raise ValueError("No coverage types pass min_plays filter for coverage profile.")

    # Higher separation = worse coverage, so sort descending by sep25
    agg = agg.sort_values("sep25", ascending=False)

    covs = agg["team_coverage_type"].tolist()
    y_pos = np.arange(len(covs))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7, 7),
        sharey=True,
    )

    def _style_axis(ax, xlabel: str):
        ax.set_yticks(y_pos)
        ax.set_yticklabels(covs, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_facecolor("#f7f7f7")
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_color("#cccccc")
            ax.spines[side].set_linewidth(1.0)

    # Separation allowed
    ax0 = axes[0]
    ax0.barh(y_pos, agg["sep25"], color="#d9534f", alpha=0.9)
    _style_axis(ax0, "Separation score allowed (0–25, higher = more space)")
    ax0.set_title(
        f"Coverage type profile – separation & contest allowed\n"
        f"(coverages with ≥ {min_plays} plays, pass length ≥ 10 air yards)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # Closing
    ax1 = axes[1]
    ax1.barh(y_pos, agg["closing25"], color="#4c8eda", alpha=0.9)
    _style_axis(ax1, "Closing score (0–25, higher = better closing)")

    # Contest
    ax2 = axes[2]
    ax2.barh(y_pos, agg["contest25"], color="#f0ad4e", alpha=0.9)
    _style_axis(ax2, "Contest score (0–25, higher = tighter at catch)")

    # ABI allowed
    ax3 = axes[3]
    ax3.barh(y_pos, agg["abi100"], color="#5e3c99", alpha=0.9)
    _style_axis(ax3, "Air Battle Index allowed (0–100, higher = more intense)")

    for y, n in zip(y_pos, agg["n_plays"]):
        ax3.text(
            ax3.get_xlim()[1] * 0.98,
            y,
            f"{n} plays",
            va="center",
            ha="right",
            fontsize=8,
            color="#444",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6) Route × coverage heatmaps
# ---------------------------------------------------------------------

def plot_route_coverage_heatmaps(
    abi_full: pd.DataFrame,
    output_dir: PathLike = "../data/abi/summary",
    *,
    min_plays: int = 40,
) -> None:
    """
    Build 3 heatmaps (route type × coverage type):

      1) Separation score (sep_delta_25, 0–25; higher = more space)
      2) Contest score (contested_severity_25, 0–25; higher = tighter)
      3) ABI (abi_100, 0–100; higher = more intense air battle)

    Only includes route–coverage combos with at least `min_plays`
    and pass_length ≥ 10 air yards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = abi_full.copy()

    needed = {
        "route_of_targeted_receiver",
        "team_coverage_type",
        "pass_length",
        "sep_delta_25",
        "contested_severity_25",
        "abi_100",
        "play_id",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(
            f"ABI full missing columns for route/coverage heatmaps: {missing}"
        )

    # Focus on downfield throws
    df = df[df["pass_length"] >= 10].copy()

    df["route"] = df["route_of_targeted_receiver"].astype(str).str.upper()
    df["coverage"] = df["team_coverage_type"].astype(str).str.upper()
    df = df.dropna(subset=["route", "coverage"])

    agg = (
        df.groupby(["route", "coverage"])
        .agg(
            n_plays=("play_id", "count"),
            sep25=("sep_delta_25", "mean"),
            contest25=("contested_severity_25", "mean"),
            abi100=("abi_100", "mean"),
        )
        .reset_index()
    )

    agg = agg[agg["n_plays"] >= min_plays].copy()
    if agg.empty:
        raise ValueError("No route×coverage combinations pass the min_plays filter.")

    routes = sorted(agg["route"].unique())
    coverages = sorted(agg["coverage"].unique())

    def _pivot_metric(col_name: str) -> pd.DataFrame:
        mat = (
            agg.pivot(index="route", columns="coverage", values=col_name)
            .reindex(index=routes, columns=coverages)
        )
        return mat

    sep_mat = _pivot_metric("sep25")
    con_mat = _pivot_metric("contest25")
    abi_mat = _pivot_metric("abi100")

    def _plot_heatmap(
        mat: pd.DataFrame,
        title: str,
        cmap: str,
        vmin=None,
        vmax=None,
        fname: str = "heatmap.png",
        fmt: str = ".1f",
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))

        data = mat.to_numpy()
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(np.arange(len(coverages)))
        ax.set_yticks(np.arange(len(routes)))
        ax.set_xticklabels(coverages, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(routes, fontsize=9)

        # grid-like look
        ax.set_xticks(np.arange(-0.5, len(coverages), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(routes), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotate cells
        for i in range(len(routes)):
            for j in range(len(coverages)):
                val = mat.iat[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)

    _plot_heatmap(
        sep_mat,
        title=(
            "Route × Coverage Heatmap — Separation score (0–25)\n"
            "higher = more space at catch point"
        ),
        cmap="YlGn",
        vmin=sep_mat.min().min(),
        vmax=sep_mat.max().max(),
        fname="heatmap_route_coverage_separation.png",
        fmt=".1f",
    )

    _plot_heatmap(
        con_mat,
        title=(
            "Route × Coverage Heatmap — Contest score (0–25)\n"
            "higher = tighter / more contested catch point"
        ),
        cmap="OrRd",
        vmin=con_mat.min().min(),
        vmax=con_mat.max().max(),
        fname="heatmap_route_coverage_contest.png",
        fmt=".1f",
    )

    _plot_heatmap(
        abi_mat,
        title=(
            "Route × Coverage Heatmap — Air Battle Index (0–100)\n"
            "higher = more intense air battle (separation, closing, contest, difficulty)"
        ),
        cmap="PuRd",
        vmin=abi_mat.min().min(),
        vmax=abi_mat.max().max(),
        fname="heatmap_route_coverage_abi.png",
        fmt=".1f",
    )

    print(f"✅ Saved route×coverage heatmaps → {output_dir}")


# ---------------------------------------------------------------------
# 7) Team-level leaderboards (offense + defense)
# ---------------------------------------------------------------------

def plot_team_leaderboards(
    abi_full: pd.DataFrame,
    output_dir: PathLike = "../visuals/summary_teams",
    *,
    min_off_plays: int = 80,
    min_def_plays: int = 80,
    top_n: int = 10,
) -> None:
    """
    Build three team-level leaderboards:

      C) Offensive separation:
           - Grouped by possession_team
           - Metric: avg sep_delta_25 (0–25; higher = more separation created)

      A) Defensive closing efficiency:
           - Grouped by defensive_team
           - Metric: avg closing_eff_25 (0–25; higher = better closing)

      B) Defensive contest tightness:
           - Grouped by defensive_team
           - Metric: avg contested_severity_25 (0–25; higher = tighter windows)

    Only uses plays with pass_length ≥ 10 air yards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = abi_full.copy()

    needed_cols = {
        "play_id",
        "pass_length",
        "sep_delta_25",
        "closing_eff_25",
        "contested_severity_25",
        "possession_team",
        "defensive_team",
    }
    missing = needed_cols.difference(df.columns)
    if missing:
        raise ValueError(f"ABI full missing required columns: {missing}")

    # Downfield-only filter
    df = df[df["pass_length"] >= 10].copy()

    def _horizontal_leaderboard(
        values: pd.Series,
        counts: pd.Series,
        title: str,
        xlabel: str,
        fname: str,
    ) -> None:
        """
        Generic horizontal bar leaderboard:
          • values: metric indexed by team
          • counts: play counts indexed by team
        """
        s = values.sort_values(ascending=False).head(top_n)
        teams = s.index.tolist()
        vals = s.values
        n = counts.loc[teams].values

        x_pos = np.arange(len(teams))

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(x_pos, vals, color="#4c9f50")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(teams, fontsize=10)
        ax.invert_yaxis()

        league_avg = values.mean()
        ax.axvline(
            league_avg,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"League avg ({league_avg:.1f})",
        )

        for i, (v, cnt) in enumerate(zip(vals, n)):
            ax.text(
                v + 0.2,
                i,
                f"{v:.1f}  ({int(cnt)} plays)",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
        ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.7)

        ax.legend(loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)

    # Offensive separation — by possession_team
    off_grp = (
        df.groupby("possession_team")
        .agg(
            n_plays=("play_id", "count"),
            sep25=("sep_delta_25", "mean"),
        )
        .reset_index()
    )
    off_grp = off_grp[off_grp["n_plays"] >= min_off_plays]

    if not off_grp.empty:
        sep_series = off_grp.set_index("possession_team")["sep25"]
        sep_counts = off_grp.set_index("possession_team")["n_plays"]
        _horizontal_leaderboard(
            values=sep_series,
            counts=sep_counts,
            title=(
                f"Top {top_n} offenses by separation creation\n"
                f"(min {min_off_plays} downfield targets, pass length ≥ 10 air yards)"
            ),
            xlabel="Average separation score (0–25; higher = more space)",
            fname="team_offensive_separation.png",
        )
    else:
        print("⚠️ No offensive teams passed min_off_plays filter.")

    # Defensive closing + contest — by defensive_team
    def_grp = (
        df.groupby("defensive_team")
        .agg(
            n_plays=("play_id", "count"),
            closing25=("closing_eff_25", "mean"),
            contest25=("contested_severity_25", "mean"),
        )
        .reset_index()
    )
    def_grp = def_grp[def_grp["n_plays"] >= min_def_plays]

    if def_grp.empty:
        print("⚠️ No defensive teams passed min_def_plays filter.")
        print(f"✅ Saved offensive team leaderboard(s) → {output_dir}")
        return

    closing_series = def_grp.set_index("defensive_team")["closing25"]
    def_counts = def_grp.set_index("defensive_team")["n_plays"]

    _horizontal_leaderboard(
        values=closing_series,
        counts=def_counts,
        title=(
            f"Top {top_n} defenses by closing efficiency\n"
            f"(min {min_def_plays} downfield targets defended, pass length ≥ 10 air yards)"
        ),
        xlabel="Average closing score (0–25; higher = faster/cleaner closing)",
        fname="team_defensive_closing.png",
    )

    contest_series = def_grp.set_index("defensive_team")["contest25"]

    _horizontal_leaderboard(
        values=contest_series,
        counts=def_counts,
        title=(
            f"Top {top_n} defenses by contest tightness\n"
            f"(min {min_def_plays} downfield targets defended, pass length ≥ 10 air yards)"
        ),
        xlabel="Average contest score (0–25; higher = tighter at catch point)",
        fname="team_defensive_contest.png",
    )

    print(f"✅ Saved team offensive/defensive leaderboards → {output_dir}")
