#!/usr/bin/env python3
# metrics/abi_narratives.py
"""
Narrative layer for Air Duel: Air Battle Index (ABI).

This module takes the analytic ABI outputs and adds:
    - A coarse play-level category label (e.g., "Highlight Catch",
      "Tight-Window Incompletion", "Burn / Wide Separation").
    - An analytic-style natural-language sentence summarizing the play.

It is intentionally opinionated but data-driven, using thresholds on:
    - xCatch probability and surprise
    - separation creation (sep_delta_25)
    - contested severity (contested_severity_25)
to describe how the air battle unfolded in a way that broadcasting /
coaching audiences can quickly understand.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _safe_float(val, default: float = np.nan) -> float:
    """Safely coerce a value to float, returning `default` on failure or NaN."""
    try:
        x = float(val)
        if np.isnan(x):
            return default
        return x
    except Exception:
        return default


def _fmt_pct(prob: Optional[float]) -> str:
    """Format a probability in [0,1] as 'XX%'; return 'n/a' if invalid."""
    if prob is None or not np.isfinite(prob):
        return "n/a"
    return f"{prob * 100:.0f}%"


def _fmt_down_and_distance(down, ytg) -> str:
    """
    Format down & distance as '3rd&7'.

    `down` is already like '3rd' from `_fmt_down` upstream; `ytg` is numeric.
    """
    if down in (None, "", "?"):
        return "unknown down & distance"
    try:
        y = int(ytg) if pd.notna(ytg) else None
    except Exception:
        y = None
    if y is None:
        return f"{down}&?"
    return f"{down}&{y}"


def _fmt_quarter(q) -> str:
    """Format quarter as 'Q1', 'Q2', ...; fall back to 'Q?'."""
    try:
        qi = int(q)
        return f"Q{qi}"
    except Exception:
        return "Q?"


def _fmt_clock(clock: Optional[str]) -> str:
    """
    Format game clock for narrative.

    Assumes Kaggles's 'game_clock' is already 'MM:SS'; falls back to
    "an unknown time" if missing.
    """
    if clock is None or (isinstance(clock, float) and np.isnan(clock)):
        return "an unknown time"
    return clock


def _first_last_name(full_name: str) -> str:
    """
    Return a shorter name for use in the sentence (typically last name).

    Falls back to 'the receiver' if the string is missing/blank.
    """
    if not isinstance(full_name, str) or not full_name.strip():
        return "the receiver"
    parts = full_name.split()
    if len(parts) >= 2:
        return parts[-1]  # last name
    return full_name


# ----------------------------------------------------------------------
# Category logic
# ----------------------------------------------------------------------

def classify_play(row: pd.Series) -> str:
    """
    Classify a play into a coarse ABI narrative category using analytic thresholds.

    Categories:
        - "Highlight Catch"
        - "Highlight Defense"
        - "Tight-Window Catch"
        - "Tight-Window Incompletion"
        - "Burn / Wide Separation"
        - "Routine Catch"
        - "Routine Incompletion"
        - "Notable Catch"
        - "Notable Incompletion"

    Rough intuition:
        - Surprise (xcatch_surprise_25) drives highlight vs non-highlight.
        - Contested severity (contested_severity_25) drives tight-window flags.
        - Separation creation (sep_delta_25) flags burn / wide-open wins.
        - Routine vs notable is the "middle" where nothing is extreme.
    """
    # outcome
    caught = int(_safe_float(row.get("caught", 0), 0))
    prob = _safe_float(row.get("xcatch_prob", np.nan), np.nan)
    surprise_25 = _safe_float(row.get("xcatch_surprise_25", np.nan), np.nan)
    sep_25 = _safe_float(row.get("sep_delta_25", np.nan), np.nan)
    cont_25 = _safe_float(row.get("contested_severity_25", np.nan), np.nan)

    # thresholds (tunable, but fixed for this submission)
    HIGH_SURPRISE = 18.0   # 0–25 scale
    MID_SURPRISE = 12.0
    HIGH_CONTEST = 15.0
    LOW_CONTEST = 8.0
    HIGH_SEP = 18.0
    LOW_SEP = 8.0
    HIGH_PROB = 0.75
    LOW_PROB = 0.25

    if caught == 1:
        # Highlight catch: improbable result (low prob or high surprise)
        if (np.isfinite(surprise_25) and surprise_25 >= HIGH_SURPRISE) or \
           (np.isfinite(prob) and prob <= LOW_PROB):
            return "Highlight Catch"

        # Tight-window catch: contested but not necessarily crazy surprise
        if np.isfinite(cont_25) and cont_25 >= HIGH_CONTEST:
            return "Tight-Window Catch"

        # Burn / wide-open: high separation, low contest
        if np.isfinite(sep_25) and sep_25 >= HIGH_SEP and \
           (np.isnan(cont_25) or cont_25 <= LOW_CONTEST):
            return "Burn / Wide Separation"

        # Routine catch: low surprise, low contest, modest separation
        if (np.isnan(surprise_25) or surprise_25 <= MID_SURPRISE) and \
           (np.isnan(cont_25) or cont_25 <= LOW_CONTEST) and \
           (np.isnan(sep_25) or sep_25 <= LOW_SEP):
            return "Routine Catch"

        return "Notable Catch"

    else:
        # Highlight defense: expected catch that fails
        if (np.isfinite(surprise_25) and surprise_25 >= HIGH_SURPRISE) or \
           (np.isfinite(prob) and prob >= HIGH_PROB):
            return "Highlight Defense"

        # Tight-window incompletion: highly contested, even if not insane surprise
        if np.isfinite(cont_25) and cont_25 >= HIGH_CONTEST:
            return "Tight-Window Incompletion"

        # Routine incompletion: low surprise, low contest
        if (np.isnan(surprise_25) or surprise_25 <= MID_SURPRISE) and \
           (np.isnan(cont_25) or cont_25 <= LOW_CONTEST):
            return "Routine Incompletion"

        return "Notable Incompletion"


# ----------------------------------------------------------------------
# Sentence generation (analytic tone)
# ----------------------------------------------------------------------

def generate_analytic_sentence(row: pd.Series) -> str:
    """
    Generate an analytic-style natural language summary for a single play.

    The text:
        - References down/distance, game state, and route when available.
        - Uses separation / closing / contested metrics where they exist.
        - Tailors language based on `abi_category` (or falls back to classify_play).

    This is designed to be broadcast-friendly but grounded in the ABI metrics.
    """
    category = row.get("abi_category") or classify_play(row)

    # Basic context
    tgt_name = row.get("tgt_name") or "the receiver"
    tgt_last = _first_last_name(tgt_name)
    route = row.get("route_of_targeted_receiver") or "route"
    down = row.get("down")
    ytg = row.get("yards_to_go")
    dd_str = _fmt_down_and_distance(row.get("down_display", row.get("down")), ytg)
    qtr_str = _fmt_quarter(row.get("quarter"))
    clock_str = _fmt_clock(row.get("game_clock"))
    off_team = row.get("possession_team") or "OFF"
    def_team = row.get("defensive_team") or "DEF"

    # Metrics
    sep_rate = _safe_float(row.get("delta_per_s", np.nan))
    close_rate = _safe_float(
        row.get("closing_rate_last", row.get("avg_closing_rate", np.nan))
    )
    cont_25 = _safe_float(row.get("contested_severity_25", np.nan))
    prob = _safe_float(row.get("xcatch_prob", np.nan))
    surprise_25 = _safe_float(row.get("xcatch_surprise_25", np.nan))
    abi = _safe_float(row.get("abi_100", np.nan))
    yards_gained = _safe_float(row.get("yards_gained", np.nan))
    pass_result = (row.get("pass_result") or "").upper()

    prob_str = _fmt_pct(prob)
    abi_str = f"{abi:.0f}" if np.isfinite(abi) else "n/a"

    # Defender text (we may only know primary closer)
    def_name = row.get("def_name") or row.get("def_name_closing") or None
    if def_name:
        def_phrase = def_name
    else:
        def_phrase = "the coverage"

    # Generic catch distance phrase
    if np.isfinite(yards_gained) and yards_gained >= 0:
        gain_phrase = f"for {int(yards_gained)} yards"
    else:
        gain_phrase = "for a gain"

    # ---- Category-specific templates ----

    # 1) Highlight Catch
    if category == "Highlight Catch":
        return (
            f"{tgt_name} runs a {route} concept for {off_team} on {dd_str} with {clock_str} left in {qtr_str}. "
            f"He creates {sep_rate:.1f} yards of separation per second, but with {def_phrase} "
            f"closing at {close_rate:.1f} yards/second, the target becomes highly contested. "
            f"Despite only a {prob_str} catch probability, "
            f"{tgt_last} completes the catch {gain_phrase} against {def_team}. [ABI={abi_str}]"
        )

    # 2) Highlight Defense
    if category == "Highlight Defense":
        return (
            f"{tgt_name} works a {route} route for {off_team} on {dd_str} with {clock_str} left in {qtr_str}. "
            f"The model viewed this as a high-probability completion ({prob_str}), but {def_phrase} "
            f"closes at {close_rate:.1f} yards/second and contests the catch. "
            f"The pass ultimately falls incomplete, making this one of the stronger defensive wins "
            f"for {def_team} on the play. [ABI={abi_str}]"
        )

    # 3) Tight-window Catch
    if category == "Tight-Window Catch":
        return (
            f"On {dd_str} in {qtr_str} with {clock_str} remaining, {tgt_name} runs a {route} route "
            f"against {def_team}. The separation and closing patterns produce a high contested-severity "
            f"score ({cont_25:.1f} on a 0–25 scale), but {tgt_last} still brings in the catch {gain_phrase}. "
            f"The play registers as a tight-window reception. [ABI={abi_str}]"
        )

    # 4) Tight-window Incompletion
    if category == "Tight-Window Incompletion":
        return (
            f"{off_team} targets {tgt_name} on a {route} concept on {dd_str} in {qtr_str} with {clock_str} on the clock. "
            f"Defensive leverage and closing create a high contested-severity environment "
            f"({cont_25:.1f} on a 0–25 scale), and the pass falls incomplete. "
            f"The metrics classify this as a tight-window defensive stand. [ABI={abi_str}]"
        )

    # 5) Burn / Wide Separation
    if category == "Burn / Wide Separation":
        return (
            f"{tgt_name} gains significant separation on a {route} route for {off_team} on {dd_str} "
            f"in {qtr_str} with {clock_str} left. His separation delta and separation rate "
            f"({sep_rate:.1f} yards/second) indicate a clear win vs coverage, leading to a routine completion "
            f"{gain_phrase}. The ABI profile is driven primarily by receiver separation. [ABI={abi_str}]"
        )

    # 6) Routine Catch
    if category == "Routine Catch":
        return (
            f"On {dd_str}, {off_team} completes a {route}-based throw to {tgt_name}. "
            f"Separation, closing, and contested metrics all fall in a typical range, and the xCatch "
            f"model assigns a moderate probability ({prob_str}) to the outcome. "
            f"The catch {gain_phrase} profiles as a routine completion. [ABI={abi_str}]"
        )

    # 7) Routine Incompletion
    if category == "Routine Incompletion":
        return (
            f"{off_team} targets {tgt_name} on a {route} route on {dd_str}, but the pass is incomplete. "
            f"Separation, closing, and contested metrics do not indicate an extreme advantage for either side, "
            f"and the xCatch probability ({prob_str}) aligns with a standard incompletion profile. [ABI={abi_str}]"
        )

    # 8) Notable Catch
    if category == "Notable Catch":
        return (
            f"{tgt_name} catches a {route} concept for {off_team} on {dd_str} in {qtr_str} "
            f"with {clock_str} remaining. The play shows a mixed ABI profile across separation "
            f"({sep_rate:.1f} y/s), closing ({close_rate:.1f} y/s), and contested metrics, resulting in a "
            f"mid-to-high air-battle intensity score. [ABI={abi_str}]"
        )

    # 9) Notable Incompletion
    if category == "Notable Incompletion":
        return (
            f"{off_team} looks to {tgt_name} on a {route} route on {dd_str}, but the pass is incomplete. "
            f"The interaction between separation ({sep_rate:.1f} y/s), defensive closing ({close_rate:.1f} y/s), "
            f"and contested metrics yields a non-routine air-battle profile even though the pass "
            f"is not completed. [ABI={abi_str}]"
        )

    # Fallback
    return (
        f"{off_team} targets {tgt_name} on a {route} route on {dd_str} in {qtr_str}. "
        f"ABI={abi_str} summarises the combined separation, closing, contested, and xCatch surprise "
        f"for the play."
    )


def add_play_categories_and_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append narrative columns to the full `abi_plays` table:

        - abi_category
        - abi_highlight_sentence

    Also ensures:
        - 'caught' exists (derived from pass_result == 'C' if needed).
        - 'down_display' exists (e.g., '3rd') for cleaner text.
    """
    out = df.copy()

    # Derive 'caught' if missing, using pass_result == 'C'
    if "caught" not in out.columns:
        pr = out.get("pass_result")
        if pr is not None:
            out["caught"] = (pr.astype(str).str.upper() == "C").astype(int)
        else:
            out["caught"] = 0

    # Derive a 'down_display' like "3rd" if you want nicer text.
    # If you already have formatted 'down' upstream, this is optional.
    if "down_display" not in out.columns and "down" in out.columns:
        def _down_disp(x):
            try:
                d = int(x)
                return {1: "1st", 2: "2nd", 3: "3rd"}.get(d, f"{d}th")
            except Exception:
                return "?"
        out["down_display"] = out["down"].apply(_down_disp)

    out["abi_category"] = out.apply(classify_play, axis=1)
    out["abi_highlight_sentence"] = out.apply(generate_analytic_sentence, axis=1)

    return out
