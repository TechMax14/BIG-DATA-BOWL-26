#!/usr/bin/env python3
# utils/data_loader.py
"""
Data loading utilities for the Air Duel / ABI project.

Responsibilities:
    - Define canonical data locations (train + supplementary).
    - Load per-week input/output tracking files for 2023 weeks 1–18.
    - Handle both comma- and tab-separated CSVs via a small "sniffer".
    - Provide a simple, in-terminal progress bar while reading lots of files.

Public API:
    - load_week_data() → (input_df, output_df, supp_df)
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# Hardcoded base paths (edit these to match your Kaggle / local layout)
# ----------------------------------------------------------------------

DATA_DIR = Path("../data")
TRAIN_DIR = DATA_DIR / "train"
SUPP_PATH = DATA_DIR / "supplementary_data.csv"

# Filenames follow: input_2023_w01.csv ... input_2023_w18.csv
# (and similarly for output_2023_w01.csv ... output_2023_w18.csv)
WEEKS = list(range(1, 18 + 1))


# ----------------------------------------------------------------------
# Small I/O helpers
# ----------------------------------------------------------------------

def _sniff_sep(path: Path, nbytes: int = 1024) -> str:
    """
    Detect whether a CSV is comma- or tab-separated by inspecting a small sample.

    Returns:
        "," or "\\t"
    """
    with open(path, "rb") as f:
        sample = f.read(nbytes)
    text = sample.decode("utf-8", errors="ignore")
    return "\t" if text.count("\t") > text.count(",") else ","


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Read a CSV using the detected delimiter, with a safe fallback engine.

    This helps with both standard NFL/Kaggle CSVs and any tab-delimited files.
    """
    sep = _sniff_sep(path)
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        # Fallback to python engine for "weird" CSVs
        return pd.read_csv(path, sep=sep, engine="python", low_memory=False)


def _existing_week_paths(kind: str) -> list[Path]:
    """
    Collect existing per-week CSV paths for the requested kind.

    Args:
        kind: "input" or "output"

    Returns:
        List of Path objects for weeks 1..18 where the file actually exists.
        Prints a warning for any missing weeks.
    """
    assert kind in {"input", "output"}
    paths: list[Path] = []
    for w in WEEKS:
        fname = f"{kind}_2023_w{w:02d}.csv"
        p = TRAIN_DIR / fname
        if p.exists():
            paths.append(p)
        else:
            print(f"[warn] missing {p}")
    return paths


# ----------------------------------------------------------------------
# Progress bar (simple in-place terminal bar)
# ----------------------------------------------------------------------

def _term_width(default: int = 80) -> int:
    """Best-effort terminal width detection (for nicer progress bars)."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def _progress_bar(idx: int, total: int, *, prefix: str = "", tail: str = "") -> None:
    """
    Print an in-place progress bar of the form:

        Loading inputs: [====......] 3/18 input_2023_w03.csv

    Call before and after each step to update.
    """
    width = max(10, _term_width() - len(prefix) - len(tail) - 12)  # padding
    frac = 0 if total <= 0 else idx / total
    fill = int(round(width * frac))
    bar = f"[{'=' * fill}{'.' * (width - fill)}]"
    msg = f"\r{prefix} {bar} {idx}/{total} {tail}"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if idx == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _read_many(paths: list[Path], label: str) -> list[pd.DataFrame]:
    """
    Read a list of CSVs with a progress bar, returning list of DataFrames.
    """
    dfs: list[pd.DataFrame] = []
    total = len(paths)
    for i, p in enumerate(paths, 1):
        _progress_bar(i - 1, total, prefix=label + ":")              # draw previous state
        df = _read_csv_auto(p)
        dfs.append(df)
        _progress_bar(i, total, prefix=label + ":", tail=p.name)     # update with filename
    return dfs


def _concat_csvs(paths: list[Path], label: str) -> pd.DataFrame:
    """
    Read and vertically concatenate multiple CSV files into a single DataFrame.
    """
    if not paths:
        return pd.DataFrame()
    dfs = _read_many(paths, label=label)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load_week_data():
    """
    Load all weekly input/output tracking data plus the season-level supplementary file.

    Returns:
        (input_df, output_df, supp_df)
            input_df : concatenated input_2023_wXX.csv across all existing weeks
            output_df: concatenated output_2023_wXX.csv across all existing weeks
            supp_df  : supplementary_data.csv (one-season context table; may be empty)

    Notes:
        - Missing week files are warned about but do not stop execution.
        - Supplementary file is optional; if missing, an empty DataFrame is returned.
    """
    input_paths = _existing_week_paths("input")
    output_paths = _existing_week_paths("output")

    input_df = _concat_csvs(input_paths, label="Loading inputs")
    output_df = _concat_csvs(output_paths, label="Loading outputs")

    # Supplementary is typically one file for the full season (routes, coverage, etc.)
    if not SUPP_PATH.exists():
        print(f"[warn] supplementary file missing: {SUPP_PATH}")
        supp_df = pd.DataFrame()
    else:
        print(f"Loading supplementary: {SUPP_PATH}")
        supp_df = _read_csv_auto(SUPP_PATH)

    print(
        f"✅ Files loaded:\n"
        f"  Input tracking:  {len(input_df):,} rows across {len(input_paths)} file(s)\n"
        f"  Output tracking: {len(output_df):,} rows across {len(output_paths)} file(s)\n"
        f"  Supplementary:   {len(supp_df):,} rows"
    )
    return input_df, output_df, supp_df
