"""Probabilistic quantile forecasts for next-day return (project plan Section 22):
"Expected next-day range: 10% quantile / median / 90% quantile" instead of a
single point prediction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_factory: Callable[[float], object],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[float, object]:
    """Fits one regressor per quantile. `model_factory(q)` must return an unfitted
    model configured for quantile regression at quantile level q (e.g. a
    LGBMRegressor(objective="quantile", alpha=q, ...)).
    """
    models = {}
    for q in quantiles:
        model = model_factory(q)
        model.fit(X_train, y_train)
        models[q] = model
    return models


def predict_quantiles(models: dict[float, object], X: pd.DataFrame) -> pd.DataFrame:
    """Predicts every fitted quantile, then enforces monotonicity (q10 <= q50 <= q90):
    independently trained quantile models have no built-in guarantee against
    "quantile crossing", so each row's predictions are sorted ascending before
    being reassigned back to their quantile labels -- a standard, simple fix.
    """
    sorted_quantiles = sorted(models)
    raw = np.column_stack([models[q].predict(X) for q in sorted_quantiles])
    sorted_raw = np.sort(raw, axis=1)
    return pd.DataFrame(sorted_raw, columns=[f"q{int(q * 100)}" for q in sorted_quantiles], index=X.index)


def return_quantiles_to_price_range(quantile_returns: pd.DataFrame, close_price: pd.Series) -> pd.DataFrame:
    """Converts predicted next-day return quantiles into a price range using today's
    close. `close_price` must share `quantile_returns`' index (aligned by index, not
    position, so a mismatch raises/produces NaN loudly rather than silently
    misaligning rows).
    """
    return quantile_returns.add(1.0).mul(close_price, axis=0)
