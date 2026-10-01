"""Loads persisted quantile-regression models and derives a price range for an
arbitrary company's latest feature row -- the "infer locally" half of the price-
quantile forecast (src.models.train_quantiles persists the "train globally" half).

Kept as a separate file from src.models.predict (the classifier's loader) rather
than merged into it: the two model types have different artifact shapes (one model
per quantile vs a single classifier + scaler) and are independently testable.
Never retrains and never fabricates: if no quantile model has been persisted yet,
or its feature schema doesn't match the running code, this raises loudly.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.models.dataset import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.models.registry import DEFAULT_MODEL_DIR, QUANTILE_REGISTRY_FILENAME, load_latest, load_version
from src.risk.quantiles import predict_quantiles, return_quantiles_to_price_range

REPO_ROOT = Path(__file__).resolve().parents[2]


class NoTrainedQuantileModelError(RuntimeError):
    """Raised when no usable registered quantile model exists yet."""


def load_quantile_models(model_version: str | None = None, model_dir: Path = DEFAULT_MODEL_DIR):
    """Returns (registry_entry, {quantile_float: fitted_model})."""
    entry = (
        load_version(model_version, model_dir, filename=QUANTILE_REGISTRY_FILENAME)
        if model_version
        else load_latest(model_dir, filename=QUANTILE_REGISTRY_FILENAME)
    )
    if entry is None:
        raise NoTrainedQuantileModelError(
            "No trained quantile model is registered. Run `python -m src.models.train_quantiles` first."
        )
    if entry["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise NoTrainedQuantileModelError(
            f"Registered quantile model {entry['model_version']} was trained on feature schema "
            f"{entry['feature_schema_version']!r}, but the running code expects "
            f"{FEATURE_SCHEMA_VERSION!r}. Retrain before serving predictions."
        )

    models = {float(key[1:]) / 100: joblib.load(REPO_ROOT / path) for key, path in entry["model_paths"].items()}
    return entry, models


def predict_price_quantiles(features: pd.DataFrame, close: pd.Series, models: dict[float, object]) -> pd.DataFrame:
    """`features` must contain every FEATURE_COLUMNS column; `close` must share
    `features`' index (the current price each quantile return is applied to).
    """
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    quantile_returns = predict_quantiles(models, features[FEATURE_COLUMNS])
    return return_quantiles_to_price_range(quantile_returns, close)
