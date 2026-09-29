"""Phase 22: held-out-company generalization test.

Validates the central "train globally, infer locally" claim -- that a model trained
on one set of companies can generalize to companies it never saw during training --
by splitting src.config.UNIVERSE's *tickers* into a training subset and a held-out
subset before touching dates at all. This is a different split than
chronological_split (which train_global.py uses): that one holds out time for
companies the model has seen; this one holds out whole companies.

An additive research script, not a change to the production global model: it trains
its own throwaway model on the training-subset companies only, and never touches
the registry train_global.py writes to.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import UNIVERSE
from src.models.dataset import FEATURE_COLUMNS, TARGET_COLUMN, build_ml_dataset, chronological_split
from src.models.evaluate import classification_metrics
from src.models.train_global import DEFAULT_DATA_PROCESSED, RANDOM_STATE, _fit_candidates, _predict

HOLDOUT_FRACTION = 0.2


def split_universe(
    universe: list[str] = UNIVERSE, holdout_fraction: float = HOLDOUT_FRACTION, seed: int = RANDOM_STATE
) -> tuple[list[str], list[str]]:
    """Deterministically splits tickers (not dates) into a train subset and a held-
    out subset. Sorted before shuffling so the split is reproducible regardless of
    UNIVERSE's list order.
    """
    shuffled = sorted(universe)
    random.Random(seed).shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * holdout_fraction))
    return sorted(shuffled[n_holdout:]), sorted(shuffled[:n_holdout])


def evaluate_holdout_generalization(
    data_processed: Path = DEFAULT_DATA_PROCESSED,
    holdout_fraction: float = HOLDOUT_FRACTION,
    universe: list[str] = UNIVERSE,
) -> dict:
    technical = pd.read_parquet(data_processed / "technical_features.parquet")
    train_tickers, holdout_tickers = split_universe(universe, holdout_fraction)

    train_dataset = build_ml_dataset(technical[technical["symbol"].isin(train_tickers)])
    train_split, val_split, seen_test_split = chronological_split(train_dataset)

    X_train, y_train = train_split[FEATURE_COLUMNS], train_split[TARGET_COLUMN]
    X_val, y_val = val_split[FEATURE_COLUMNS], val_split[TARGET_COLUMN]

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = _fit_candidates(X_train, y_train, X_train_scaled)
    val_metrics = {}
    for name, model in models.items():
        pred, proba = _predict(name, model, X_val, X_val_scaled)
        val_metrics[name] = classification_metrics(y_val.values, pred, proba)
    best_name = pd.DataFrame(val_metrics).T["roc_auc"].idxmax()
    best_model = models[best_name]

    # "Seen companies, unseen dates" -- the same kind of evaluation train_global.py
    # does, on this training-subset-of-companies model, for a like-for-like baseline.
    X_seen_test, y_seen_test = seen_test_split[FEATURE_COLUMNS], seen_test_split[TARGET_COLUMN]
    X_seen_test_scaled = scaler.transform(X_seen_test)
    seen_pred, seen_proba = _predict(best_name, best_model, X_seen_test, X_seen_test_scaled)
    seen_companies_metrics = classification_metrics(y_seen_test.values, seen_pred, seen_proba)

    # "Unseen companies" -- the actual generalization test: these tickers
    # contributed zero rows to training, validation, or model selection.
    holdout_dataset = build_ml_dataset(technical[technical["symbol"].isin(holdout_tickers)])
    X_holdout, y_holdout = holdout_dataset[FEATURE_COLUMNS], holdout_dataset[TARGET_COLUMN]
    X_holdout_scaled = scaler.transform(X_holdout)
    holdout_pred, holdout_proba = _predict(best_name, best_model, X_holdout, X_holdout_scaled)
    unseen_companies_metrics = classification_metrics(y_holdout.values, holdout_pred, holdout_proba)

    comparison = pd.DataFrame(
        {"seen_companies_unseen_dates": seen_companies_metrics, "unseen_companies": unseen_companies_metrics}
    )
    comparison.to_csv(data_processed / "holdout_company_generalization_metrics.csv")

    return {
        "best_model": best_name,
        "train_tickers": train_tickers,
        "holdout_tickers": holdout_tickers,
        "seen_companies_unseen_dates_metrics": seen_companies_metrics,
        "unseen_companies_metrics": unseen_companies_metrics,
    }


if __name__ == "__main__":
    result = evaluate_holdout_generalization()
    print(f"Best model (selected on seen-company validation): {result['best_model']}")
    print(f"Held-out tickers ({len(result['holdout_tickers'])}): {result['holdout_tickers']}")
    print("\nSeen companies, unseen dates (comparable to train_global.py's test split):")
    print(result["seen_companies_unseen_dates_metrics"])
    print("\nUnseen companies (the actual generalization test):")
    print(result["unseen_companies_metrics"])
