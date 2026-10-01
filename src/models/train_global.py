"""Trains and persists the global cross-company model (Phase 21).

Mirrors notebooks/05_baseline_ml.ipynb exactly: the same technical-only feature set
(FEATURE_COLUMNS), the same global chronological split, the same naive/logistic-
regression/random-forest/XGBoost/LightGBM comparison, selected by validation
ROC-AUC. The one thing the notebook never did is persist the winning model -- and
without a persisted model, nothing downstream (live inference for any company, seen
or unseen) can work. That's the only gap this module closes.

Deliberately does NOT use Phase 7's multimodal/ensemble model: SUMMARY.md documents
that it "performed well on one split but lost under walk-forward validation", so it
is not the properly-validated candidate for a live global model. The technical-only
baseline is what actually survived validation, honestly reported ROC-AUC ~= 0.50 and
all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import UNIVERSE
from src.models.baseline import naive_predict, naive_predict_proba
from src.models.dataset import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    build_ml_dataset,
    chronological_split,
)
from src.models.evaluate import classification_metrics
from src.models.registry import register_model

RANDOM_STATE = 42
DEFAULT_DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# Only logistic regression's inputs need standardizing; the tree-based models are
# scale-invariant and are trained/served on the raw feature matrix, exactly as
# notebook 05 does.
NEEDS_SCALING = {"logistic_regression"}


def _fit_candidates(X_train: pd.DataFrame, y_train: pd.Series, X_train_scaled) -> dict[str, object]:
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    return {"logistic_regression": logreg, "random_forest": rf, "xgboost": xgb, "lightgbm": lgbm}


def _predict(name: str, model, X, X_scaled):
    X_input = X_scaled if name in NEEDS_SCALING else X
    return model.predict(X_input), model.predict_proba(X_input)[:, 1]


def train_global_model(
    data_processed: Path = DEFAULT_DATA_PROCESSED,
    model_dir: Path = DEFAULT_MODEL_DIR,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict:
    """Fits every candidate, selects the winner by validation ROC-AUC (exactly as
    notebook 05 does), evaluates it once, honestly, on the held-out test split, then
    persists the winning model (+ scaler, + metadata) and returns the registry entry.

    If the naive baseline wins validation (a real possible outcome given this
    project's documented ~0.50 ROC-AUC finding), no model object is persisted --
    predict.py serves the recorded base rate instead of fabricating a fitted model.
    """
    technical = pd.read_parquet(data_processed / "technical_features.parquet")
    dataset = build_ml_dataset(technical)
    train, val, test = chronological_split(dataset, train_frac=train_frac, val_frac=val_frac)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = _fit_candidates(X_train, y_train, X_train_scaled)

    val_metrics = {
        "naive_baseline": classification_metrics(
            y_val.values,
            naive_predict(y_train, len(y_val)),
            naive_predict_proba(y_train, len(y_val)),
        )
    }
    for name, model in models.items():
        pred, proba = _predict(name, model, X_val, X_val_scaled)
        val_metrics[name] = classification_metrics(y_val.values, pred, proba)

    comparison = pd.DataFrame(val_metrics).T
    best_name = comparison["roc_auc"].idxmax()
    base_rate = float(y_train.mean())

    if best_name == "naive_baseline":
        best_model = None
        test_pred = naive_predict(y_train, len(y_test))
        test_proba = naive_predict_proba(y_train, len(y_test))
    else:
        best_model = models[best_name]
        test_pred, test_proba = _predict(best_name, best_model, X_test, X_test_scaled)

    test_metrics = classification_metrics(y_test.values, test_pred, test_proba)

    model_version = f"global_v{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = None
    if best_model is not None:
        model_path = model_dir / f"{model_version}.joblib"
        joblib.dump(best_model, model_path)
    scaler_path = model_dir / f"{model_version}_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    entry = {
        "model_version": model_version,
        "model_name": best_name,
        "needs_scaling": best_name in NEEDS_SCALING,
        "model_path": str(model_path.relative_to(model_dir.parents[1])) if model_path else None,
        "scaler_path": str(scaler_path.relative_to(model_dir.parents[1])),
        "base_rate": base_rate,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_universe_size": len(UNIVERSE),
        "training_universe": list(UNIVERSE),
        "feature_columns": FEATURE_COLUMNS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "validation_method": "chronological_split (global calendar-date cutoffs; train 70% / val 15% / test 15%)",
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        # TARGET_COLUMN (next_day_direction) is a strict 1-trading-day-ahead label --
        # recording that explicitly here lets callers (src.services.company_analysis)
        # disclose it honestly instead of silently relabeling this as a multi-day
        # forecast whenever a different horizon is requested (website audit: a
        # requested 60-day horizon must not silently return a 1-day prediction).
        "horizon_days": 1,
        "training_data_cutoff_date": str(pd.Timestamp(val["date"].max()).date()),
    }
    register_model(entry, model_dir=model_dir)
    return entry


if __name__ == "__main__":
    result = train_global_model()
    print(f"Trained and persisted {result['model_version']} (winner: {result['model_name']})")
    print("Validation ROC-AUC by model:")
    for name, metrics in result["validation_metrics"].items():
        print(f"  {name:20s} {metrics['roc_auc']:.4f}")
    print("Held-out test metrics (winning model):", result["test_metrics"])
