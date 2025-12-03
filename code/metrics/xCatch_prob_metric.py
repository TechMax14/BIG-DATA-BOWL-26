#!/usr/bin/env python3
# metrics/xCatch_prob_metric.py
"""
Expected Catch Probability (xCatch) Model

This module builds an expected catch probability model (xCatch) using features
derived from the other ABI submetrics:

    - Separation creation (WR vs nearest DB)
    - Contested arrival severity (catch point tightness / crowding)
    - Defensive closing efficiency

High-level flow:
    1. `build_training_table`:
        - Merge separation, contested, and closing tables into a single
          play-level table with features + binary `caught` target.
    2. `train_expected_catch`:
        - Train a logistic regression model with balanced classes.
    3. `predict_expected_catch`:
        - Apply a saved model bundle to get xcatch_prob + xcatch_25.
    4. `add_xcatch_surprise_scores`:
        - Convert expected probabilities and outcomes into "surprise"
          scores useful for ABI (unexpected catches / breakups).
    5. `train_xCatch_model` / `run_xCatch_pipeline`:
        - Orchestrate the training, saving, prediction, and CSV outputs.

This is the "glue" model that turns the mechanics of separation/contested/closing
into an intuitive expected catch probability and surprise component.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import platform
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

# --- metric modules (existing ABI submetrics) ---
from metrics.sep_creation_metric import (
    compute_from_enriched,
    compute_sep_scores,
    reduce_to_nearest_per_play,
)
from metrics.contested_catch_metric import (
    compute_contested_from_enriched,
    score_contested,
)
from metrics.closing_eff_metric import (
    compute_closing_from_enriched,
    score_closing_defender_play,
    aggregate_closing_to_play,
)

# --------------------------------------------------------------------
# Constants / schema
# --------------------------------------------------------------------

MODEL_BUNDLE_SCHEMA_VERSION = "1.0.0"

# Prefer raw/mechanical features; fall back to scores if needed
FEATURE_CANDIDATES = [
    # Separation mechanics
    "first_sep", "last_sep", "delta", "delta_per_s",

    # Contested mechanics
    "sep_at_arrival",
    "n_defenders_r1", "n_defenders_r2",
    "closing_rate_last",

    # Closing mechanics (if ever added)
    "avg_closing_eff_25",

    # Context
    "pass_length",
    "down", "yards_to_go",
    "offense_formation",
    "receiver_alignment",
    "route_of_targeted_receiver",
    "pass_location_type",

    # Scores / fallbacks
    "sep_score_air_blend",
    "sep_score_25",
    "contested_severity_25",
]

KEYS = ["game_id", "play_id", "tgt_nfl_id"]


# --------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------

def _ensure_df(X: pd.DataFrame | np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    """
    Ensure X is a DataFrame with EXACTLY the expected feature order.

    Args:
        X: Input data as either a DataFrame or ndarray.
        feature_cols: Expected column order for the model.

    Returns:
        DataFrame aligned to `feature_cols` for consistent model input.
    """
    if isinstance(X, pd.DataFrame):
        return X.reindex(columns=feature_cols)
    return pd.DataFrame(X, columns=feature_cols)


def _env_meta() -> Dict[str, str]:
    """Capture lightweight environment metadata for the saved model bundle."""
    import sklearn as _sk
    return {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": _sk.__version__,
    }


# --------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------

def save_model(bundle: Dict[str, Any], path: str) -> None:
    """
    Save a self-contained model bundle.

    Expected keys in bundle:
        - model (fitted estimator)
        - feature_cols (list[str])
        - impute_medians (dict[str, float])
        - optional metrics: auc, brier
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "meta": _env_meta(),
        "model": bundle.get("model"),
        "feature_cols": list(bundle.get("feature_cols", [])),
        "impute_medians": dict(bundle.get("impute_medians", {})),
        "auc": float(bundle.get("auc")) if bundle.get("auc") is not None else None,
        "brier": float(bundle.get("brier")) if bundle.get("brier") is not None else None,
    }
    joblib.dump(serializable, p.as_posix(), compress=3)


def load_model(path: str) -> Dict[str, Any]:
    """
    Load and sanity-check a saved model bundle.

    Ensures required keys are present:
        - 'model'
        - 'feature_cols'
        - 'impute_medians' (added as empty dict if missing)
    """
    obj = joblib.load(Path(path).as_posix())
    if "model" not in obj or "feature_cols" not in obj:
        raise ValueError("xCatch model bundle missing required keys ('model', 'feature_cols').")
    if not isinstance(obj["feature_cols"], list) or not obj["feature_cols"]:
        raise ValueError("xCatch model bundle has empty or invalid 'feature_cols'.")
    if "impute_medians" not in obj:
        obj["impute_medians"] = {}
    return obj


# --------------------------------------------------------------------
# Table builder: merge SEP + CONTESTED + CLOSING into one table
# --------------------------------------------------------------------

def build_training_table(
    sep_df: pd.DataFrame,
    cont_df: pd.DataFrame,
    close_df: pd.DataFrame,
    *,
    extra_context_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge submetric outputs into one row per targeted play with features + `caught` target.

    Priority:
        - Use separation table as the base (contains pass_result and WR identity).
        - Merge contested table using 3-key join if tgt_nfl_id is present,
          otherwise fall back to (game_id, play_id).
        - Merge closing table similarly (often play-level).

    Args:
        sep_df: Separation "combined" table (per targeted play).
        cont_df: Contested "scored" table (per play, possibly with tgt_nfl_id).
        close_df: Closing "play" table (best closer per play).
        extra_context_cols: Optional additional cols to bring from sep_df.

    Returns:
        DataFrame with keys, 'caught', and feature columns ready for modeling.
    """
    # Base (separation) — keep keys + key context + mechanics
    sep_keep = list(set(
        KEYS + [
            "pass_result", "week", "tgt_name", "tgt_pos", "pass_length",
            "first_sep", "last_sep", "delta", "delta_per_s",
            "sep_score_air_blend", "sep_score_25",
            "down", "yards_to_go",
            "offense_formation", "receiver_alignment",
            "route_of_targeted_receiver", "pass_location_type",
        ]
    ))
    sep_keep = [c for c in sep_keep if c in sep_df.columns]
    base = sep_df[sep_keep].copy()

    if "pass_result" not in base.columns:
        raise ValueError("sep_df must contain 'pass_result' to construct 'caught' target.")

    # Binary target: was the pass completed to the targeted receiver?
    base["caught"] = (base["pass_result"] == "C").astype(int)

    # Contested table
    cont_keys = KEYS if all(k in cont_df.columns for k in KEYS) else ["game_id", "play_id"]
    cont_keep = list(set(
        cont_keys + [
            "contested_severity_25", "sep_at_arrival",
            "n_defenders_r1", "n_defenders_r2", "closing_rate_last",
        ]
    ))
    cont_keep = [c for c in cont_keep if c in cont_df.columns]

    merged = base.merge(
        cont_df[cont_keep].drop_duplicates(subset=cont_keys),
        on=cont_keys,
        how="inner",
        validate="one_to_one" if cont_keys == KEYS else "many_to_one",
    )

    # Closing table (often play-level)
    close_keys = KEYS if all(k in close_df.columns for k in KEYS) else ["game_id", "play_id"]
    close_keep = list(set(close_keys + ["avg_closing_eff_25"]))
    close_keep = [c for c in close_keep if c in close_df.columns]

    merged = merged.merge(
        close_df[close_keep].drop_duplicates(subset=close_keys),
        on=close_keys,
        how="inner",
        validate="one_to_one" if close_keys == KEYS else "many_to_one",
    )

    # Optional extra context from sep_df
    if extra_context_cols:
        have = [c for c in extra_context_cols if c in sep_df.columns]
        if have:
            merged = merged.merge(
                sep_df[KEYS + have].drop_duplicates(subset=KEYS),
                on=KEYS,
                how="left",
            )

    # One-hot for small categorical context
    cat_cols = [
        c
        for c in [
            "offense_formation",
            "receiver_alignment",
            "route_of_targeted_receiver",
            "pass_location_type",
        ]
        if c in merged.columns and merged[c].dtype == "O"
    ]
    if cat_cols:
        dummies = pd.get_dummies(merged[cat_cols], prefix=cat_cols, dummy_na=True)
        merged = pd.concat([merged.drop(columns=cat_cols), dummies], axis=1)

    # Final feature subset: any of FEATURE_CANDIDATES or dummies if created
    keep_feats = [c for c in FEATURE_CANDIDATES if c in merged.columns]
    # add any one-hot columns
    keep_feats += [c for c in merged.columns if any(c.startswith(f"{x}_") for x in cat_cols)]

    # Coerce numerics
    for c in keep_feats:
        if c in merged.columns and merged[c].dtype != "O":
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Reorder: keys + caught + everything else
    cols = KEYS + ["caught"] + [c for c in merged.columns if c not in KEYS + ["caught"]]
    merged = merged.reindex(columns=cols)

    return merged


# --------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------

def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a numeric feature matrix for modeling.

    - Drops non-feature columns (keys, identifiers, `caught`, etc.)
    - Fills NaNs with per-column medians
    - Drops all-NA columns

    Args:
        df: Training or prediction table from `build_training_table`.
        feature_cols: Optional explicit list of features to use.

    Returns:
        (X, cols):
            X    -> DataFrame of numeric features.
            cols -> List of feature names in the order used by the model.
    """
    if feature_cols:
        cols = [c for c in feature_cols if c in df.columns]
    else:
        cols = [
            c
            for c in df.columns
            if c not in KEYS + ["caught", "pass_result", "tgt_name", "tgt_pos", "week"]
        ]

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    non_all_na = [c for c in X.columns if X[c].notna().any()]
    X = X[non_all_na]

    return X, non_all_na


# --------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------

def train_expected_catch(
    train_df: pd.DataFrame,
    *,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a logistic regression baseline for expected catch probability.

    Uses:
        - Class balancing (class_weight='balanced') to handle skew.
        - Train/test split for evaluation (AUC + Brier).

    Args:
        train_df: Training table including 'caught' and feature columns.
        feature_cols: Optional explicit feature list; if None, infer.
        test_size: Held-out fraction for evaluation.
        random_state: Seed for reproducible splitting.

    Returns:
        Model bundle dict with keys:
            - model
            - feature_cols
            - auc
            - brier
            - impute_medians
    """
    if "caught" not in train_df.columns:
        raise ValueError("training table must include binary column 'caught'")

    X_all, cols = get_feature_matrix(train_df, feature_cols=feature_cols)
    y_all = train_df["caught"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(_ensure_df(X_tr, cols), y_tr)

    p_te = model.predict_proba(_ensure_df(X_te, cols))[:, 1]
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)

    return {
        "model": model,
        "feature_cols": cols,
        "auc": float(auc),
        "brier": float(brier),
        "impute_medians": X_all.median(numeric_only=True).to_dict(),
    }


def predict_expected_catch(
    df: pd.DataFrame,
    *,
    model_bundle: Dict[str, Any],
    scale_mode: str = "prob",
) -> pd.DataFrame:
    """
    Apply a saved xCatch model bundle and add xcatch_prob + xcatch_25.

    Args:
        df: Table produced by `build_training_table` (or compatible).
        model_bundle: Output from `train_expected_catch` / `load_model`.
        scale_mode:
            - "prob": map probability directly to 0–25 via 25 * p.
            - otherwise: rank-based 0–25 via probability percentiles.

    Returns:
        DataFrame with added columns:
            - xcatch_prob in [0,1]
            - xcatch_25 (Int64 in [0,25])
    """
    if not model_bundle or "model" not in model_bundle:
        raise ValueError("model_bundle missing 'model'")

    cols = model_bundle["feature_cols"]
    med = model_bundle.get("impute_medians", {})

    X = df.reindex(columns=cols).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        fill = med.get(c, np.nan)
        X[c] = X[c].fillna(fill)

    probs = model_bundle["model"].predict_proba(_ensure_df(X, cols))[:, 1]

    out = df.copy()
    out["xcatch_prob"] = probs

    if scale_mode == "prob":
        out["xcatch_25"] = (out["xcatch_prob"] * 25).round().astype("Int64").clip(0, 25)
    else:
        ranks = pd.Series(out["xcatch_prob"]).rank(method="average", pct=True)
        out["xcatch_25"] = (ranks * 25).round().astype("Int64").clip(0, 25)

    return out


# --------------------------------------------------------------------
# Training sources (broad, unfiltered)
# --------------------------------------------------------------------

def build_xcatch_training_sources(
    out_enriched: pd.DataFrame,
    *,
    fps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build broad training sources for xCatch from raw enriched tracking.

    Notes:
        - No pass_length filter applied.
        - No alignment to ABI plays_index.
        - Intended to capture as many catch / target events as possible.

    Returns:
        (sep_train, cont_train, close_train) DataFrames suitable for
        feeding into `build_training_table`.
    """
    # --- Separation ---
    sep_all = compute_from_enriched(out_enriched, fps=fps, pass_length_min=None)

    throw_play = compute_sep_scores(reduce_to_nearest_per_play(sep_all, mode="throw"))
    catch_play = compute_sep_scores(reduce_to_nearest_per_play(sep_all, mode="catch"))

    keys = ["game_id", "play_id", "tgt_nfl_id"]
    keep_cols_throw = [
        c
        for c in [
            "tgt_name",
            "tgt_pos",
            "week",
            "pass_result",
            "pass_length",
            "first_sep",
            "last_sep",
            "delta",
            "delta_per_s",
        ]
        if c in throw_play.columns
    ]

    sep_train = throw_play[keys + keep_cols_throw + ["sep_score_25"]].merge(
        catch_play[keys + ["sep_score_25"]],
        on=keys,
        how="inner",
        suffixes=("_throw", "_catch"),
        validate="one_to_one",
    )

    sep_train["sep_score_air_blend"] = (
        sep_train["sep_score_25_throw"].astype(float) * 0.8
        + sep_train["sep_score_25_catch"].astype(float) * 0.2
    )

    # --- Contested ---
    cont_train = compute_contested_from_enriched(
        out_enriched,
        fps=fps,
        pass_length_min=None,
    )
    cont_train = score_contested(cont_train)

    # --- Closing ---
    close_def = compute_closing_from_enriched(
        out_enriched,
        fps=fps,
        pass_length_min=None,
        lookback_full_flight=True,
        tail_window_s=0.5,
    )
    close_def = score_closing_defender_play(close_def)
    close_train = aggregate_closing_to_play(close_def)

    return sep_train, cont_train, close_train


# --------------------------------------------------------------------
# Expected Catch Surprise: scores higher for unexpected outcomes
# --------------------------------------------------------------------

def add_xcatch_surprise_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Add "surprise" scores given `caught` (0/1) and `xcatch_prob` in [0,1].

    Definitions:
        - xcatch_off_surprise = y * (1 - p)
            (high when a catch happens that the model thought was unlikely)
        - xcatch_def_surprise = (1 - y) * p
            (high when a non-catch happens that the model thought was likely)
        - xcatch_surprise = max(off, def)
        - xcatch_surprise_25_25 = round(25 * xcatch_surprise)

    Args:
        preds: DataFrame with 'caught' and 'xcatch_prob' already computed.

    Returns:
        DataFrame with additional surprise columns.
    """
    df = preds.copy()

    if "caught" not in df.columns or "xcatch_prob" not in df.columns:
        raise ValueError("Predictions table must contain 'caught' and 'xcatch_prob' to compute surprise scores.")

    p = df["xcatch_prob"].astype(float).clip(0.0, 1.0)
    y = df["caught"].astype(int).clip(0, 1)

    df["xcatch_off_surprise"] = y * (1.0 - p)
    df["xcatch_def_surprise"] = (1 - y) * p
    df["xcatch_surprise"] = df[["xcatch_off_surprise", "xcatch_def_surprise"]].max(axis=1)

    df["xcatch_surprise_25_25"] = (
        (df["xcatch_surprise"] * 25.0)
        .round()
        .astype("Int64")
        .clip(0, 25)
    )

    return df


# --------------------------------------------------------------------
# Core model pipeline: build → train → predict → save
# --------------------------------------------------------------------

def train_xCatch_model(
    sep_df: pd.DataFrame,
    cont_df: pd.DataFrame,
    closing_df: pd.DataFrame,
    *,
    train_sep_df: Optional[pd.DataFrame] = None,
    train_cont_df: Optional[pd.DataFrame] = None,
    train_closing_df: Optional[pd.DataFrame] = None,
    plays_index: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    model_path: str = "../data/abi/metrics/catch_probability/xCatch_model.joblib",
    training_table_path: str = "../data/abi/metrics/catch_probability/xCatch_training.csv",
    predictions_path: str = "../data/abi/metrics/catch_probability/xCatch_predictions_scored.csv",
    train_model_now: bool = True,
    scale_mode: str = "prob",
    qa_style: str = "compact",
) -> Dict[str, pd.DataFrame]:
    """
    Full xCatch training + prediction pipeline.

    Steps:
        1. Build prediction table from ABI subset (sep_df/cont_df/closing_df).
        2. Optionally align that table to `plays_index`.
        3. Build training table:
            - Prefer explicit broad sources (train_*_df) if provided.
            - Otherwise, reuse the ABI subset table.
        4. Train or load a logistic model.
        5. Predict xcatch_prob on ABI subset and compute surprise scores.
        6. Save training + prediction tables to CSV.

    Args:
        sep_df: ABI separation "combined" table.
        cont_df: ABI contested "scored" table.
        closing_df: ABI closing "play" table.
        train_sep_df / train_cont_df / train_closing_df:
            Optional broader training sources from `build_xcatch_training_sources`.
        plays_index: Optional canonical plays index to align prediction subset.
        feature_cols: Optional explicit feature list for model.
        model_path: Joblib path for saving/loading the model.
        training_table_path: CSV path for saving training table.
        predictions_path: CSV path for saving scored predictions.
        train_model_now: If True, train new model; otherwise load from model_path.
        scale_mode: Passed to `predict_expected_catch` ("prob" or rank-based).
        qa_style: "compact", "verbose", or "quiet" for logging.

    Returns:
        dict with:
            - "training": training table
            - "predictions": prediction table with xcatch columns
    """
    # Prediction table from ABI subset
    pred_table = build_training_table(sep_df, cont_df, closing_df)

    # Optional alignment to plays_index (should already match, but just in case)
    if plays_index is not None and not plays_index.empty:
        keys = [k for k in KEYS if k in pred_table.columns and k in plays_index.columns]
        if keys:
            before = len(pred_table)
            pred_table = pred_table.merge(
                plays_index[keys].drop_duplicates(),
                on=keys,
                how="inner",
            )
            after = len(pred_table)
            if qa_style != "quiet" and before != after:
                print(f"🔒 xCatch predict subset aligned to plays_index: {before:,} → {after:,}")

    # Training table: prefer explicit broad sources if provided
    if (
        train_sep_df is not None
        and train_cont_df is not None
        and train_closing_df is not None
        and not train_sep_df.empty
        and not train_cont_df.empty
        and not train_closing_df.empty
    ):
        train_table = build_training_table(train_sep_df, train_cont_df, train_closing_df)
    else:
        train_table = pred_table.copy()

    # Train or load model
    model_path_p = Path(model_path)
    if train_model_now or not model_path_p.exists():
        bundle = train_expected_catch(train_table, feature_cols=feature_cols)
        save_model(bundle, model_path)
        if qa_style != "quiet":
            print(f"🤖 Trained xCatch model — AUC={bundle['auc']:.3f}, Brier={bundle['brier']:.3f}")
            print(f"💾 Saved xCatch model → {model_path}")
    else:
        bundle = load_model(model_path)
        if qa_style == "verbose":
            print(f"📦 Loaded xCatch model from {model_path}")

    # Predict on ABI subset
    preds = predict_expected_catch(pred_table, model_bundle=bundle, scale_mode=scale_mode)

    # Add ABI-style surprise scores (for ABI aggregation later)
    preds = add_xcatch_surprise_scores(preds)

    # Persist training + predictions
    training_table_p = Path(training_table_path)
    training_table_p.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_csv(training_table_p, index=False)
    if qa_style != "quiet":
        print(f"💾 Saved xCatch training table → {training_table_p}  ({len(train_table):,} rows)")

    preds_p = Path(predictions_path)
    preds_p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(preds_p, index=False)
    if qa_style != "quiet":
        print(f"💾 Saved xCatch predictions → {preds_p}  ({len(preds):,} plays)")

    return {"training": train_table, "predictions": preds}


# --------------------------------------------------------------------
# Public adapter used by metric_pipeline.run_metrics
# --------------------------------------------------------------------

def run_xCatch_pipeline(
    out_enriched: pd.DataFrame,
    *,
    sep_results: Dict[str, Any],
    contested_results: Dict[str, Any],
    closing_results: Dict[str, Any],
    plays_index: Optional[pd.DataFrame] = None,
    fps: float = 10.0,
    train_on_all_plays: bool = True,
    model_path: str = "../data/abi/metrics/catch_probability/xCatch_model.joblib",
    training_table_path: str = "../data/abi/metrics/catch_probability/xCatch_training.csv",
    predictions_path: str = "../data/abi/metrics/catch_probability/xCatch_predictions.csv",
    train_model_now: bool = True,
    scale_mode: str = "prob",
) -> Dict[str, pd.DataFrame]:
    """
    High-level xCatch pipeline adapter for metric_pipeline.run_metrics.

    Uses:
        - ABI subset separation / contested / closing outputs for prediction.
        - Optionally builds broad, unfiltered training sources from out_enriched.
        - Trains or loads a logistic model and writes predictions.

    Args:
        out_enriched: Full enriched tracking table (for optional broad training).
        sep_results: Dict from separation pipeline (expects "combined" key).
        contested_results: Dict from contested pipeline (expects "scored" key).
        closing_results: Dict from closing pipeline (expects "play" key).
        plays_index: Optional canonical play index for alignment.
        fps: Frames per second (passed to training source builder).
        train_on_all_plays: If True, build broad training sources from out_enriched.
        model_path: Path for model joblib.
        training_table_path: Path for training CSV.
        predictions_path: Path for predictions CSV (ABI subset).
        train_model_now: If True, fit model; otherwise only load if exists.
        scale_mode: "prob" or rank-based scaling for xcatch_25.

    Returns:
        dict with:
            - "training": training table
            - "predictions": prediction table with xcatch fields
    """
    sep_df = sep_results.get("combined", pd.DataFrame())
    cont_df = contested_results.get("scored", pd.DataFrame())
    close_df = closing_results.get("play", pd.DataFrame())

    if sep_df.empty or cont_df.empty or close_df.empty:
        print("[warn] xCatch: one or more inputs are empty — skipping expected-catch.")
        return {"training": pd.DataFrame(), "predictions": pd.DataFrame()}

    # Ensure closing has tgt_nfl_id if possible
    if "tgt_nfl_id" not in close_df.columns and plays_index is not None and not plays_index.empty:
        keys = ["game_id", "play_id"]
        attach_cols = [
            c
            for c in ["game_id", "play_id", "tgt_nfl_id", "tgt_name", "tgt_pos"]
            if c in plays_index.columns
        ]
        close_df = close_df.merge(
            plays_index[attach_cols].drop_duplicates(keys),
            on=keys,
            how="left",
        )

    # Build broad training sources if requested
    train_sep_df = train_cont_df = train_closing_df = None
    if train_on_all_plays:
        train_sep_df, train_cont_df, train_closing_df = build_xcatch_training_sources(
            out_enriched,
            fps=fps,
        )

    return train_xCatch_model(
        sep_df=sep_df,
        cont_df=cont_df,
        closing_df=close_df,
        train_sep_df=train_sep_df,
        train_cont_df=train_cont_df,
        train_closing_df=train_closing_df,
        plays_index=plays_index,
        model_path=model_path,
        training_table_path=training_table_path,
        predictions_path=predictions_path,
        train_model_now=train_model_now,
        scale_mode=scale_mode,
        qa_style="compact",
    )
