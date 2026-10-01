"""Trains and persists quantile-regression models for a next-day price-range
forecast (website audit Sections 12-14: the forecast range must come from a real
fitted distribution, not an ad hoc combination of unrelated volatility metrics).

Reuses src.risk.quantiles' already-correct fit/predict machinery (one
LGBMRegressor per quantile, monotonicity-enforced) -- the only new thing here is
persistence + registry metadata, mirroring train_global.py's pattern exactly, so
the live orchestrator can load a quantile model the same way it loads the
classifier instead of recomputing a forecast ad hoc.

Reports the held-out interval coverage HONESTLY: this project's own documented
finding (SUMMARY.md) is ~64% empirical coverage against an ~80% target. This
number must never be hidden or silently improved-looking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.models.dataset import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, add_next_day_return, build_feature_table, chronological_split
from src.models.registry import QUANTILE_REGISTRY_FILENAME, register_model
from src.models.train_global import DEFAULT_DATA_PROCESSED, DEFAULT_MODEL_DIR, RANDOM_STATE
from src.risk.quantiles import DEFAULT_QUANTILES, fit_quantile_models, predict_quantiles

TARGET = "next_day_return"


def _lgbm_quantile_factory(q: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=q,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def build_quantile_dataset(technical_df: pd.DataFrame) -> pd.DataFrame:
    """Same feature transform as the classifier (build_feature_table), labeled
    with the continuous next_day_return target instead of the binary direction.
    """
    df = build_feature_table(technical_df)
    df = add_next_day_return(df)
    required = FEATURE_COLUMNS + [TARGET]
    return df.dropna(subset=required).reset_index(drop=True)


def train_quantile_models(
    data_processed: Path = DEFAULT_DATA_PROCESSED,
    model_dir: Path = DEFAULT_MODEL_DIR,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict:
    technical = pd.read_parquet(data_processed / "technical_features.parquet")
    dataset = build_quantile_dataset(technical)
    train, val, test = chronological_split(dataset, train_frac=train_frac, val_frac=val_frac)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    models = fit_quantile_models(X_train, y_train, _lgbm_quantile_factory, quantiles=quantiles)

    test_quantile_returns = predict_quantiles(models, X_test)
    low_col, high_col = f"q{int(min(quantiles) * 100)}", f"q{int(max(quantiles) * 100)}"
    interval_pct = int(round((max(quantiles) - min(quantiles)) * 100))
    within_interval = (y_test.values >= test_quantile_returns[low_col].values) & (
        y_test.values <= test_quantile_returns[high_col].values
    )
    coverage = float(within_interval.mean())

    model_version = f"quantile_v{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {}
    for q, model in models.items():
        key = f"q{int(q * 100)}"
        path = model_dir / f"{model_version}_{key}.joblib"
        joblib.dump(model, path)
        model_paths[key] = str(path.relative_to(model_dir.parents[1]))

    entry = {
        "model_version": model_version,
        "quantiles": list(quantiles),
        "model_paths": model_paths,
        "feature_columns": FEATURE_COLUMNS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_data_cutoff_date": str(pd.Timestamp(val["date"].max()).date()),
        "validation_method": "chronological_split (global calendar-date cutoffs; train 70% / val 15% / test 15%)",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "test_interval_pct": interval_pct,
        "test_coverage": coverage,
    }
    register_model(entry, model_dir=model_dir, filename=QUANTILE_REGISTRY_FILENAME)
    return entry


if __name__ == "__main__":
    result = train_quantile_models()
    print(f"Trained and persisted {result['model_version']}")
    print(
        f"Held-out {result['test_interval_pct']}% interval coverage: "
        f"{result['test_coverage']:.1%} (target: {result['test_interval_pct']}%)"
    )
