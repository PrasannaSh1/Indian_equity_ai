"""Loads the persisted global model and runs inference on an arbitrary company's
feature row -- the "infer locally" half of train-globally-infer-locally (Phase 21).

Never retrains and never fabricates: if no model has been persisted yet, or the
registered model's feature schema doesn't match the code currently running, this
raises loudly rather than silently serving a stale or invalid prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.models.dataset import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.models.registry import DEFAULT_MODEL_DIR, load_latest, load_version

REPO_ROOT = Path(__file__).resolve().parents[2]


class NoTrainedModelError(RuntimeError):
    """Raised when no usable registered model exists yet."""


def load_model(model_version: str | None = None, model_dir: Path = DEFAULT_MODEL_DIR):
    """Returns (registry_entry, model_or_None, scaler). model is None exactly when
    the registered winner was the naive baseline (see train_global.py) -- predict_
    proba_up handles that case by returning the recorded base rate instead.
    """
    entry = load_version(model_version, model_dir) if model_version else load_latest(model_dir)
    if entry is None:
        raise NoTrainedModelError(
            "No trained global model is registered. Run `python -m src.models.train_global` first."
        )
    if entry["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise NoTrainedModelError(
            f"Registered model {entry['model_version']} was trained on feature schema "
            f"{entry['feature_schema_version']!r}, but the running code expects "
            f"{FEATURE_SCHEMA_VERSION!r}. Retrain before serving predictions."
        )

    model = joblib.load(REPO_ROOT / entry["model_path"]) if entry.get("model_path") else None
    scaler = joblib.load(REPO_ROOT / entry["scaler_path"]) if entry.get("scaler_path") else None
    return entry, model, scaler


def predict_proba_up(features: pd.DataFrame, entry: dict, model, scaler) -> pd.Series:
    """`features` must contain every column in FEATURE_COLUMNS -- the same schema
    training used (src.models.dataset.build_ml_dataset / build_latest_features both
    produce it), so training and inference can never silently diverge.
    """
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    X = features[FEATURE_COLUMNS]

    if model is None:
        return pd.Series(entry["base_rate"], index=X.index)

    X_input = scaler.transform(X) if entry.get("needs_scaling") else X
    return pd.Series(model.predict_proba(X_input)[:, 1], index=X.index)


def predict_one(feature_row: pd.Series, model_version: str | None = None) -> dict:
    """Convenience wrapper for a single company's latest feature row, e.g.
    `dataset.build_latest_features(indicator_df).iloc[0]`.
    """
    entry, model, scaler = load_model(model_version)
    proba = predict_proba_up(feature_row.to_frame().T, entry, model, scaler)
    return {
        "probability_up": float(proba.iloc[0]),
        "model_version": entry["model_version"],
        "model_name": entry["model_name"],
    }
