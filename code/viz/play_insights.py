#!/usr/bin/env python3
"""
viz/play_insights.py

Per-play graphics pack for the Air Battle Index (ABI).

For a given (game_id, play_id), this module builds a folder and saves:

    play_animation.gif       – Offense vs. defense tracking animation
    abi_score.gif            – Circular ABI score progress (synced to play length)
    abi_scorecard.png        – Static ABI + component bars vs league average
    metric_progression.png   – sep / closing / contest vs. time since throw
    catch_snapshot.png       – Ball-arrival snapshot with contest rings

These visuals are designed for:
  • broadcast-style explanations,
  • the Kaggle writeup,
  • and quick inspection of “what happened” on a single play.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Helper: safe folder naming
# ---------------------------------------------------------------------------

def _safe_slug(text: str) -> str:
    """
    Convert an arbitrary string into a filesystem-safe slug.

    Keeps only [A–Z, a–z, 0–9, '_', '-'] and replaces spaces with underscores.
    """
    text = (text or "").strip().replace(" ", "_")
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )
    return "".join(c for c in text if c in allowed) or "play"


def _make_play_folder(
    abi_row: pd.Series,
    base_dir: str | Path = "../data/abi/plays",
) -> Path:
    """
    Construct a human-readable folder name for this play and ensure it exists.

    Example folder name:
        Justin_Jefferson_2024010700_2660

    Args:
        abi_row: Single row from ABI play-level table for the chosen play.
        base_dir: Root directory under which to create the folder.

    Returns:
        Path to the created folder.
    """
    game_id = abi_row["game_id"]
    play_id = abi_row["play_id"]
    tgt_name = abi_row.get("tgt_name", "") or abi_row.get("player_name", "")
    slug = _safe_slug(str(tgt_name))

    folder = Path(base_dir) / f"{slug}_{game_id}_{play_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# Helper: slice tracking for one play
# ---------------------------------------------------------------------------

def _filter_play_frames(
    out_enriched: pd.DataFrame,
    game_id: int,
    play_id: int,
) -> pd.DataFrame:
    """
    Filter the frame-level tracking table down to a single (game_id, play_id).

    Args:
        out_enriched: Full enriched tracking data (from data_preprocessor).
        game_id: Desired game_id.
        play_id: Desired play_id.

    Returns:
        DataFrame containing only that play's frames.

    Raises:
        ValueError if there are no rows for the requested play.
    """
    mask = (out_enriched["game_id"] == game_id) & (out_enriched["play_id"] == play_id)
    df_play = out_enriched.loc[mask].copy()

    if df_play.empty:
        raise ValueError(
            f"No in-air tracking rows for game_id={game_id}, play_id={play_id}"
        )

    return df_play


# ---------------------------------------------------------------------------
# Helper: frame-level time series (sep / closing / contest)
# ---------------------------------------------------------------------------

def _compute_time_series(
    df_play: pd.DataFrame,
    tgt_nfl_id: int,
    fps: float,
) -> pd.DataFrame:
    """
    Build a simple per-frame time series relative to the targeted receiver.

    For each frame in the play, we compute:
      - time_since_throw (s)   : (frame_id - first_frame) / fps
      - sep_yards              : distance from target WR to nearest defender
      - closing                : - d(sep_yards)/dt (positive when defense closes)
      - contest                : 1 / (1 + sep_yards) in [0, 1) (higher = tighter)

    Args:
        df_play: Tracking data for a single play.
        tgt_nfl_id: Targeted receiver's nfl_id.
        fps: Frames per second for the tracking data.

    Returns:
        DataFrame sorted by time_since_throw.
    """
    required = {"frame_id", "x", "y", "nfl_id", "player_side"}
    missing = required.difference(df_play.columns)
    if missing:
        raise ValueError(f"df_play missing required columns: {missing}")

    df_play = df_play.sort_values("frame_id").copy()
    first_frame = df_play["frame_id"].min()
    df_play["time_since_throw"] = (df_play["frame_id"] - first_frame) / float(fps)

    rows: list[tuple[int, float, float]] = []

    # For each frame, compute WR ↔ nearest defender separation
    for frame_id, df_f in df_play.groupby("frame_id"):
        df_tgt = df_f[df_f["nfl_id"] == tgt_nfl_id]
        if df_tgt.empty:
            continue

        tgt = df_tgt.iloc[0]
        tx, ty = float(tgt["x"]), float(tgt["y"])

        df_def = df_f[df_f["player_side"] == "Defense"]
        if df_def.empty:
            sep_yards = np.nan
        else:
            dx = df_def["x"].to_numpy(float) - tx
            dy = df_def["y"].to_numpy(float) - ty
            dists = np.sqrt(dx * dx + dy * dy)
            sep_yards = float(np.nanmin(dists))

        t = float(df_f["time_since_throw"].iloc[0])
        rows.append((frame_id, t, sep_yards))

    ts = pd.DataFrame(rows, columns=["frame_id", "time_since_throw", "sep_yards"])
    ts = ts.sort_values("time_since_throw").reset_index(drop=True)

    # Closing = -d(separation)/dt (positive when DB is closing)
    t = ts["time_since_throw"].to_numpy(float)
    s = ts["sep_yards"].to_numpy(float)

    if len(t) > 1:
        ds_dt = np.gradient(s, t)
        closing = -ds_dt
    else:
        closing = np.zeros_like(s)

    # Contest heuristic: 1 / (1 + sep_yards)
    contest = 1.0 / (1.0 + np.maximum(s, 0.0))

    ts["closing"] = closing
    ts["contest"] = contest
    return ts


# ---------------------------------------------------------------------------
# 1) Play tracking animation → GIF (no ffmpeg)
# ---------------------------------------------------------------------------

def save_play_animation(
    df_play: pd.DataFrame,
    tgt_nfl_id: int,
    out_path: Path,
    fps: float = 10.0,
    pause_seconds: float = 2.0,
) -> int:
    """
    Create a GIF animation of the play:

      • Green field with NFL dimensions (x ∈ [0,120], y ∈ [0, 53.3])
      • End zones shaded differently (0–10, 110–120)
      • Yard lines and yard numbers (0–50–0)
      • Targeted WR in RED (with route tail)
      • All defenders in BLUE (with route tails)
      • Other offense in WHITE
      • Final frame held for `pause_seconds`

    Args:
        df_play: Frame-level tracking data for a single play.
        tgt_nfl_id: Targeted receiver's nfl_id.
        out_path: Where to save the GIF (suffix is forced to .gif).
        fps: Frames per second for the animation.
        pause_seconds: Time to hold on the final frame.

    Returns:
        Number of distinct tracking frames (used to sync ABI progress animation).
    """
    out_path = out_path.with_suffix(".gif")

    required = {"frame_id", "x", "y", "player_side", "nfl_id"}
    missing = required.difference(df_play.columns)
    if missing:
        raise ValueError(f"df_play missing required columns: {missing}")

    frame_ids = sorted(df_play["frame_id"].unique())
    n_frames = len(frame_ids)

    extra_frames = int(pause_seconds * fps)
    total_frames = n_frames + extra_frames

    # All defenders that appear in this play (for trails)
    def_ids = (
        df_play[df_play["player_side"] == "Defense"]["nfl_id"]
        .dropna()
        .astype(int)
        .unique()
    )
    def_ids = sorted(def_ids)

    # --- Set up field figure ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#3a8f3b")  # grass green

    # End zones (0–10 and 110–120)
    ax.axvspan(0, 10, color="#e0e0e0", alpha=0.8, zorder=0)
    ax.axvspan(110, 120, color="#e0e0e0", alpha=0.8, zorder=0)

    # Yard lines every 10 yards (10–110)
    for x in range(10, 120, 10):
        ax.axvline(x, color="white", linewidth=0.6, alpha=0.9, zorder=1)

    # End lines
    ax.axvline(0, color="white", linewidth=2.0, zorder=1)
    ax.axvline(120, color="white", linewidth=2.0, zorder=1)

    # Yard numbers: 0–50–0
    yard_numbers = ["0", "10", "20", "30", "40",
                    "50", "40", "30", "20", "10", "0"]
    yard_xs = list(range(10, 120, 10))  # 10..110

    for x, label in zip(yard_xs, yard_numbers):
        ax.text(
            x,
            2.0,
            label,
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # Live scatters
    off_scatter = ax.scatter([], [], s=60, c="white", edgecolors="black", zorder=3)
    def_scatter = ax.scatter([], [], s=60, c="#1f77b4", edgecolors="black", zorder=3)
    tgt_scatter = ax.scatter([], [], s=80, c="red", edgecolors="black", zorder=4)

    # Target route tail
    tgt_trail, = ax.plot(
        [],
        [],
        color="red",
        linewidth=2.0,
        alpha=0.9,
        zorder=2,
    )

    # Defender route tails
    def_trails: dict[int, any] = {}
    for did in def_ids:
        line, = ax.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
        )
        def_trails[did] = line

    # Title overlay
    title = ax.text(
        0.5,
        1.03,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
    )

    def _draw_frame(frame_idx: int):
        """Draw a given animation index (0..total_frames-1)."""
        if frame_idx < n_frames:
            frame_id = frame_ids[frame_idx]
        else:
            frame_id = frame_ids[-1]

        df_f = df_play[df_play["frame_id"] == frame_id]

        df_tgt = df_f[df_f["nfl_id"] == tgt_nfl_id]
        df_off = df_f[df_f["player_side"] == "Offense"]
        df_def = df_f[df_f["player_side"] == "Defense"]

        # Offense minus target
        if not df_tgt.empty:
            df_off_others = df_off[df_off["nfl_id"] != tgt_nfl_id]
        else:
            df_off_others = df_off

        # Scatter updates
        if not df_off_others.empty:
            off_scatter.set_offsets(df_off_others[["x", "y"]].to_numpy(float))
        else:
            off_scatter.set_offsets(np.empty((0, 2)))

        if not df_def.empty:
            def_scatter.set_offsets(df_def[["x", "y"]].to_numpy(float))
        else:
            def_scatter.set_offsets(np.empty((0, 2)))

        if not df_tgt.empty:
            tgt_scatter.set_offsets(df_tgt[["x", "y"]].to_numpy(float))
        else:
            tgt_scatter.set_offsets(np.empty((0, 2)))

        # Target trail
        df_tgt_trail = df_play[
            (df_play["nfl_id"] == tgt_nfl_id)
            & (df_play["frame_id"] <= frame_id)
        ].sort_values("frame_id")

        if not df_tgt_trail.empty:
            tgt_trail.set_data(
                df_tgt_trail["x"].to_numpy(float),
                df_tgt_trail["y"].to_numpy(float),
            )
        else:
            tgt_trail.set_data([], [])

        # Defender trails
        for did, line in def_trails.items():
            df_def_trail = df_play[
                (df_play["nfl_id"] == did)
                & (df_play["player_side"] == "Defense")
                & (df_play["frame_id"] <= frame_id)
            ].sort_values("frame_id")

            if not df_def_trail.empty:
                line.set_data(
                    df_def_trail["x"].to_numpy(float),
                    df_def_trail["y"].to_numpy(float),
                )
            else:
                line.set_data([], [])

        game_id = int(df_f["game_id"].iloc[0])
        play_id = int(df_f["play_id"].iloc[0])
        title.set_text(f"Game {game_id}, Play {play_id}   •   Frame {frame_id}")

        artists = [off_scatter, def_scatter, tgt_scatter, tgt_trail, title]
        artists.extend(def_trails.values())
        return artists

    def _init():
        off_scatter.set_offsets(np.empty((0, 2)))
        def_scatter.set_offsets(np.empty((0, 2)))
        tgt_scatter.set_offsets(np.empty((0, 2)))
        tgt_trail.set_data([], [])
        for line in def_trails.values():
            line.set_data([], [])

        artists = [off_scatter, def_scatter, tgt_scatter, tgt_trail, title]
        artists.extend(def_trails.values())
        return artists

    anim = animation.FuncAnimation(
        fig,
        lambda i: _draw_frame(i),
        init_func=_init,
        frames=total_frames,
        interval=1000.0 / fps,
        blit=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)

    return n_frames


# ---------------------------------------------------------------------------
# 2) ABI circular progress animation → GIF
# ---------------------------------------------------------------------------

def save_abi_progress_gif(
    abi_value: float,
    n_sync_frames: int,
    out_path: Path,
    fps: float = 10.0,
    pause_seconds: float = 2.0,
) -> None:
    """
    Create a circular "progress ring" GIF for the ABI score.

    • Hollow background ring
    • Foreground arc that sweeps from 0 → abi_value (0–100)
    • Center text that increments in sync with the arc
    • Animation length aligned to `n_sync_frames` for playback sync
    • Final frame held for `pause_seconds`

    Args:
        abi_value: Target ABI score (0–100).
        n_sync_frames: Number of frames to sync with the play animation.
        out_path: Where to save the GIF (suffix forced to .gif).
        fps: Frames per second.
        pause_seconds: Extra time to hold the final frame.
    """
    out_path = out_path.with_suffix(".gif")

    max_score = 100.0
    abi_value = float(abi_value)

    extra_frames = int(pause_seconds * fps) if pause_seconds > 0 else 0
    total_frames = n_sync_frames + extra_frames

    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Base track (full circle)
    theta_full = np.linspace(-np.pi / 2, 3 * np.pi / 2, 360)
    x_track = np.cos(theta_full)
    y_track = np.sin(theta_full)
    ax.plot(x_track, y_track, linewidth=12, color="#e6f4ff")

    # Foreground arc (animated)
    arc_line, = ax.plot([], [], linewidth=12, color="#0094ff")

    # Title + center text
    ax.text(
        0,
        1.25,
        "ABI Score:",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0070cc",
    )

    score_text = ax.text(
        0,
        0.05,
        "0",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0070cc",
    )

    ax.text(
        0,
        -0.25,
        "out of 100",
        ha="center",
        va="center",
        fontsize=9,
        color="#0070cc",
    )

    # Precomputed circle param
    theta_circle = np.linspace(-np.pi / 2, 3 * np.pi / 2, 360)

    def _update(frame_idx: int):
        if frame_idx < n_sync_frames:
            frac_time = frame_idx / max(n_sync_frames - 1, 1)
        else:
            frac_time = 1.0

        current_score = abi_value * frac_time
        frac_circle = current_score / max_score

        end_idx = int(len(theta_circle) * frac_circle)
        if end_idx <= 0:
            x_arc = []
            y_arc = []
        else:
            x_arc = np.cos(theta_circle[:end_idx])
            y_arc = np.sin(theta_circle[:end_idx])

        arc_line.set_data(x_arc, y_arc)
        score_text.set_text(f"{current_score:.0f}")

        return arc_line, score_text

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=total_frames,
        interval=1000.0 / fps,
        blit=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) ABI scorecard (static)
# ---------------------------------------------------------------------------

def save_abi_scorecard(
    abi_row: pd.Series,
    abi_full: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Save a static horizontal bar chart of ABI components vs league average.

    Components (all 0–25):
      - Separation
      - Closing
      - Contest
      - Catch Difficulty (xCatch surprise)

    League average for each component is drawn as a small red vertical line.

    Args:
        abi_row: Single ABI row for the specific play.
        abi_full: Full ABI play-level table (used for league averages).
        out_path: Where to save the PNG (suffix forced to .png).
    """
    out_path = out_path.with_suffix(".png")

    components = [
        ("Separation", "sep_delta_25"),
        ("Closing", "closing_eff_25"),
        ("Contest", "contested_severity_25"),
        ("Catch difficulty", "xcatch_surprise_25"),
    ]

    labels = [c[0] for c in components]
    vals = [float(abi_row[c[1]]) for c in components]
    league_means = [float(abi_full[c[1]].mean()) for c in components]

    # Reverse order so Separation appears at the top
    labels = labels[::-1]
    vals = vals[::-1]
    league_means = league_means[::-1]

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5, 3))

    bars = ax.barh(y_pos, vals, color="#4c8eda", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Component score (0–25)")

    # League averages and value labels
    for i, (mean, v) in enumerate(zip(league_means, vals)):
        ax.vlines(
            mean,
            i - 0.35,
            i + 0.35,
            colors="red",
            linewidth=2.0,
        )
        ax.text(
            mean,
            i + 0.42,
            "avg",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )
        ax.text(
            v + 0.4,
            i,
            f"{v:.1f}",
            va="center",
            fontsize=8,
            color="black",
        )

    abi_val = float(abi_row["abi_100"])
    ax.set_title(
        f"Air Battle Index: {abi_val:.1f} / 100",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Metric progression (static sep / closing / contest vs time)
# ---------------------------------------------------------------------------

def save_metric_progression(
    ts: pd.DataFrame,
    game_id: int,
    play_id: int,
    out_path: Path,
) -> None:
    """
    Save a 3-panel static plot of ABI metric progression over time:

      • Separation (yards)
      • Closing (positive = DB closing)
      • Contest (0–1)

    Args:
        ts: Time series table from _compute_time_series().
        game_id: Game id for title context.
        play_id: Play id for title context.
        out_path: Where to save the PNG (suffix forced to .png).
    """
    out_path = out_path.with_suffix(".png")

    t = ts["time_since_throw"].to_numpy(float)
    sep = ts["sep_yards"].to_numpy(float)
    closing = ts["closing"].to_numpy(float)
    contest = ts["contest"].to_numpy(float)

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, sep, linewidth=2)
    ax.set_ylabel("Separation (y)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

    ax = axes[1]
    ax.plot(t, closing, linewidth=2)
    ax.set_ylabel("Closing\n(+ = closing)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

    ax = axes[2]
    ax.plot(t, contest, linewidth=2)
    ax.set_ylabel("Contest\n(0–1)")
    ax.set_xlabel("Time since throw (s)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

    fig.suptitle(
        f"ABI metric progression – game {game_id}, play {play_id}",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Catch-arrival snapshot (field, WR, defenders, contest circle)
# ---------------------------------------------------------------------------

def save_catch_snapshot(
    df_play: pd.DataFrame,
    tgt_nfl_id: int,
    ts: pd.DataFrame,
    abi_row: pd.Series,
    out_path: Path,
) -> None:
    """
    Create a "catch-space" snapshot at the last frame of the play.

    In the snapshot:
      • Target WR is at (0, 0)
      • Defenders and other receivers are plotted relative to the target
      • Two contest rings (r1, r2) are drawn around the target
      • Text box displays:
          – Separation at arrival
          – Contest severity score (0–25)
          – Catch difficulty score (0–25)

    Args:
        df_play: Tracking data for a single play.
        tgt_nfl_id: Targeted receiver's nfl_id.
        ts: Time series returned by _compute_time_series().
        abi_row: ABI row for this play (for contest/xCatch scores).
        out_path: Where to save the PNG (suffix forced to .png).
    """
    out_path = out_path.with_suffix(".png")

    last_frame = df_play["frame_id"].max()
    df_last = df_play[df_play["frame_id"] == last_frame].copy()

    required = {"x", "y", "player_side", "nfl_id"}
    missing = required.difference(df_last.columns)
    if missing:
        raise ValueError(f"df_last missing required columns: {missing}")

    sep_arrival = float(ts["sep_yards"].iloc[-1]) if not ts.empty else np.nan
    contest_score = float(abi_row["contested_severity_25"])
    xcatch_score = float(abi_row["xcatch_surprise_25"])

    # Target absolute position at last frame
    df_tgt = df_last[df_last["nfl_id"] == tgt_nfl_id]
    if df_tgt.empty:
        raise ValueError("No target player found in last frame for snapshot.")
    tx = float(df_tgt["x"].iloc[0])
    ty = float(df_tgt["y"].iloc[0])

    # Helper: convert to coordinates relative to target
    def to_rel(df: pd.DataFrame) -> np.ndarray:
        arr = df[["x", "y"]].to_numpy(float)
        arr[:, 0] -= tx
        arr[:, 1] -= ty
        return arr

    df_def = df_last[df_last["player_side"] == "Defense"]
    df_off = df_last[df_last["player_side"] == "Offense"]
    df_off_other = df_off[df_off["nfl_id"] != tgt_nfl_id]

    rel_def = to_rel(df_def) if not df_def.empty else np.empty((0, 2))
    rel_off_other = to_rel(df_off_other) if not df_off_other.empty else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    # Contest rings
    r1 = 1.0  # "tight" window
    r2 = 3.0  # broader challenge zone

    circle1 = Circle(
        (0, 0),
        radius=r1,
        edgecolor="#ffcc00",
        facecolor="none",
        linestyle="-",
        linewidth=2,
    )
    circle2 = Circle(
        (0, 0),
        radius=r2,
        edgecolor="#999999",
        facecolor="none",
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Target WR at origin
    ax.scatter([0], [0], s=90, c="red", edgecolors="black", zorder=4, label="Target WR")

    # Other offense
    if rel_off_other.size:
        ax.scatter(
            rel_off_other[:, 0],
            rel_off_other[:, 1],
            s=50,
            c="#dddddd",
            edgecolors="black",
            zorder=3,
            label="Other offense",
        )

    # Defenders
    if rel_def.size:
        ax.scatter(
            rel_def[:, 0],
            rel_def[:, 1],
            s=60,
            c="#1f77b4",
            edgecolors="black",
            zorder=3,
            label="Defenders",
        )

    # Axis limits driven by data + margin
    if rel_def.size:
        max_def_r = np.nanmax(np.sqrt(rel_def[:, 0] ** 2 + rel_def[:, 1] ** 2))
    else:
        max_def_r = r2
    max_r = max(r2, max_def_r)
    margin = 0.5

    ax.set_xlim(-max_r - margin, max_r + margin)
    ax.set_ylim(-max_r - margin, max_r + margin)
    ax.set_xticks([])
    ax.set_yticks([])

    game_id = int(df_last["game_id"].iloc[0])
    play_id = int(df_last["play_id"].iloc[0])
    ax.set_title(
        f"Catch-space view – game {game_id}, play {play_id}",
        fontsize=11,
        fontweight="bold",
    )

    # Info box
    txt_lines = [
        (
            f"Separation at arrival: {sep_arrival:.2f} yards"
            if not np.isnan(sep_arrival)
            else "Separation at arrival: n/a"
        ),
        f"Contest score: {contest_score:.1f} / 25",
        f"Catch difficulty score: {xcatch_score:.1f} / 25",
        "",
        f"Inner ring (r1={r1:.1f}y): tight contest",
        f"Outer ring (r2={r2:.1f}y): challenge zone",
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def build_play_graphics(
    out_enriched: pd.DataFrame,
    abi_full: pd.DataFrame,
    game_id: int,
    play_id: int,
    *,

    fps: float = 10.0,
    base_output_dir: str | Path = "../visuals/plays",
) -> Path:
    """
    High-level entry point to build the full graphics pack for one play.

    Steps:
      1. Find ABI row and target WR for (game_id, play_id).
      2. Create a play-specific folder with a human-readable name.
      3. Slice `out_enriched` to just this play.
      4. Build frame-level metric time series (sep/closing/contest).
      5. Save:
           - play_animation.gif
           - abi_score.gif
           - abi_scorecard.png
           - metric_progression.png
           - catch_snapshot.png

    Args:
        out_enriched: Full frame-level tracking data.
        abi_full: Full ABI play-level table.
        game_id: Game id of the play to visualize.
        play_id: Play id of the play to visualize.
        fps: Frames per second for animations.
        base_output_dir: Root directory to hold play folders.

    Returns:
        Path to the folder containing all generated assets.
    """
    # ABI row (for naming + scoring)
    mask = (abi_full["game_id"] == game_id) & (abi_full["play_id"] == play_id)
    rows = abi_full.loc[mask]

    if rows.empty:
        raise ValueError(f"No ABI row found for game_id={game_id}, play_id={play_id}")

    abi_row = rows.iloc[0]
    abi_val = float(abi_row["abi_100"])
    tgt_nfl_id = int(abi_row["tgt_nfl_id"])

    play_folder = _make_play_folder(abi_row, base_dir=base_output_dir)

    # Frame-level data for this play
    df_play = _filter_play_frames(out_enriched, game_id, play_id)

    # Metric time series
    ts = _compute_time_series(df_play, tgt_nfl_id=tgt_nfl_id, fps=fps)

    # 1) tracking animation
    n_frames = save_play_animation(
        df_play=df_play,
        tgt_nfl_id=tgt_nfl_id,
        out_path=play_folder / "play_animation.gif",
        fps=fps,
    )

    # 2) ABI progress ring (synced to same number of frames)
    save_abi_progress_gif(
        abi_value=abi_val,
        n_sync_frames=n_frames,
        out_path=play_folder / "abi_score.gif",
        fps=fps,
    )

    # 3) ABI scorecard
    save_abi_scorecard(
        abi_row=abi_row,
        abi_full=abi_full,
        out_path=play_folder / "abi_scorecard.png",
    )

    # 4) Metric progression
    save_metric_progression(
        ts=ts,
        game_id=game_id,
        play_id=play_id,
        out_path=play_folder / "metric_progression.png",
    )

    # 5) Catch-space arrival snapshot
    save_catch_snapshot(
        df_play=df_play,
        tgt_nfl_id=tgt_nfl_id,
        ts=ts,
        abi_row=abi_row,
        out_path=play_folder / "catch_snapshot.png",
    )

    print(f"✅ Play graphics created for game {game_id}, play {play_id} → {play_folder}")
    return play_folder
