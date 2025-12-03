#!/usr/bin/env python3
"""
viz/abi_hero_visual.py

Hero visual for the Air Battle Index (ABI): a radial/radar plot showing all
four ABI components on a 0–25 scale for a single targeted pass play:

    - Separation Creation        (sep_delta_25)
    - Defensive Closing          (closing_eff_25)
    - Contested Arrival Severity (contested_severity_25)
    - Catch Difficulty Surprise  (xcatch_surprise_25)

This figure is designed to:
    • appear in the Kaggle writeup,
    • serve as a key panel in the broadcast-style competition video,
    • illustrate how the four components combine to form ABI_100.

Usage example:
    from viz.abi_hero_visual import load_abi_data, get_example_play, plot_abi_radial_for_play

    df = load_abi_data("abi_results_full.csv")
    row = get_example_play(df, mode="top")       # "top", "low", or "median"
    fig = plot_abi_radial_for_play(row, df=df)
    fig.savefig("abi_hero_example.png", dpi=200, bbox_inches="tight")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union, Iterable
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Mapping for radial chart labels → ABI column names
ABI_COMPONENT_MAP = {
    "Separation\n(0–25)": "sep_delta_25",
    "Closing\n(0–25)": "closing_eff_25",
    "Contest\n(0–25)": "contested_severity_25",
    "Catch Difficulty\n(0–25)": "xcatch_surprise_25",
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_abi_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the ABI full play-level table.

    Ensures all required ABI component columns + ABI_100 are present.

    Args:
        path: Path to abi_results_full.csv.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError if required columns are missing.
    """
    path = Path(path)
    df = pd.read_csv(path)

    required = list(ABI_COMPONENT_MAP.values()) + ["abi_100"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in ABI file: {missing}")

    return df


# ----------------------------------------------------------------------
# Representative play selection
# ----------------------------------------------------------------------

def get_example_play(
    df: pd.DataFrame,
    mode: Literal["top", "low", "median"] = "top",
    abi_col: str = "abi_100",
) -> pd.Series:
    """
    Select a representative play for visualization.

    Args:
        df: ABI play-level table.
        mode:
            "top"    → highest ABI play
            "low"    → lowest ABI play
            "median" → play nearest to median ABI
        abi_col: Column containing the final ABI score.

    Returns:
        A single row (pd.Series) for the chosen play.
    """
    if mode == "top":
        return df.sort_values(abi_col, ascending=False).iloc[0]
    elif mode == "low":
        return df.sort_values(abi_col, ascending=True).iloc[0]
    elif mode == "median":
        median_val = df[abi_col].median()
        idx = (df[abi_col] - median_val).abs().idxmin()
        return df.loc[idx]

    raise ValueError(f"Unknown mode: {mode!r}")


# ----------------------------------------------------------------------
# Radial chart helpers
# ----------------------------------------------------------------------

def _build_radial_values(row: pd.Series, max_score: float = 25.0):
    """
    Convert ABI component columns → radial geometry (values + angles).
    """
    labels = list(ABI_COMPONENT_MAP.keys())
    values = [float(row[col]) for col in ABI_COMPONENT_MAP.values()]

    # Close the shape for filled polygon
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(values))

    return labels, values, angles, max_score


def _compute_league_baseline(df: pd.DataFrame) -> Iterable[float]:
    """
    Return league-average values for each ABI component,
    in the ABI_COMPONENT_MAP order.
    """
    return [float(df[col].mean()) for col in ABI_COMPONENT_MAP.values()]


# ----------------------------------------------------------------------
# Plot function
# ----------------------------------------------------------------------

def plot_abi_radial_for_play(
    row: pd.Series,
    max_score: float = 25.0,
    show_title: bool = True,
    df: pd.DataFrame | None = None,
    baseline_values: Iterable[float] | None = None,
) -> plt.Figure:
    """
    Render the ABI radial chart for a single play.

    The chart optionally overlays a league-average polygon and includes
    an analytic-style highlight sentence if present.

    Args:
        row:
            A single ABI play row containing all 4 components + abi_100.
        max_score:
            Maximum per-component score (default 25).
        show_title:
            Whether to include target/team/game-level title information.
        df:
            Full ABI table (if provided and baseline_values=None,
            compute league-average baseline).
        baseline_values:
            Optional explicit baseline polygon values.

    Returns:
        The Matplotlib Figure object.
    """
    labels, values, angles, max_score = _build_radial_values(row, max_score=max_score)

    # ------------------------------------------------------------------
    # League baseline (optional overlay)
    # ------------------------------------------------------------------
    baseline = None
    if baseline_values is not None:
        baseline = list(baseline_values)
    elif df is not None:
        baseline = list(_compute_league_baseline(df))

    if baseline is not None:
        baseline.append(baseline[0])  # close loop

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, polar=True)

    # -- Draw baseline polygon
    if baseline is not None:
        ax.plot(angles, baseline, linewidth=1.8, linestyle="--", alpha=0.9)
        ax.fill(angles, baseline, alpha=0.08)

    # -- Draw play polygon
    ax.plot(angles, values, linewidth=3.0, marker="o")
    ax.fill(angles, values, alpha=0.30)

    # -- Radial limits & ring labels
    ax.set_ylim(0, max_score)
    ax.set_yticks([5, 15, 25])
    ax.set_yticklabels(["5", "15", "25"], fontsize=9)
    ax.tick_params(axis="y", pad=3)

    # Clean polar styling
    ax.spines["polar"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=1.0, alpha=0.8)

    # -- Component axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # -- Numeric labels at vertices
    for angle, val in zip(angles[:-1], values[:-1]):
        # Adjust label position inward/outward depending on angle
        r = max(val - max_score * 0.10, 0)  # nudge inward
        if 0 <= angle < np.pi/2 or 3*np.pi/2 < angle <= 2*np.pi:
            ha = "left"
        elif np.pi/2 < angle < 3*np.pi/2:
            ha = "right"
        else:
            ha = "center"
        va = "bottom" if 0 <= angle <= np.pi else "top"

        ax.text(angle, r, f"{val:.1f}", fontsize=8, ha=ha, va=va)

    # ------------------------------------------------------------------
    # Title block
    # ------------------------------------------------------------------
    if show_title:
        abi_val = float(row.get("abi_100", np.nan))
        title_lines = []

        tgt_name = row.get("tgt_name")
        defence = row.get("defensive_team")
        game_id = row.get("game_id")
        play_id = row.get("play_id")

        if tgt_name and defence:
            title_lines.append(f"{tgt_name} vs {defence}")
        elif tgt_name:
            title_lines.append(str(tgt_name))

        if game_id is not None and play_id is not None:
            title_lines.append(f"Game {game_id}, Play {play_id}")

        title_lines.append(f"Air Battle Index: {abi_val:.1f} / 100")

        ax.set_title("\n".join(title_lines), pad=30, fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------
    # Bottom caption: analytic narrative sentence
    # ------------------------------------------------------------------
    highlight = row.get("abi_highlight_sentence")
    if isinstance(highlight, str) and highlight.strip():
        wrapped = "\n".join(textwrap.wrap(highlight.strip(), width=80))
        fig.text(
            0.5,
            0.05,
            wrapped,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(left=0.12, right=0.88, top=0.78, bottom=0.18)
    return fig


# ----------------------------------------------------------------------
# CLI helper
# ----------------------------------------------------------------------

def main(
    path: Union[str, Path] = "abi_results_full.csv",
    outdir: Union[str, Path] = ".",
) -> None:
    """
    Simple CLI entry point:
        • Loads ABI data
        • Selects the top ABI play
        • Writes the radial chart PNG to the output directory
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_abi_data(path)
    row_top = get_example_play(df, mode="top")

    fig = plot_abi_radial_for_play(row_top, df=df)
    outfile = outdir / "abi_hero_radial_top_play.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved hero ABI radial chart → {outfile}")


if __name__ == "__main__":
    main()
