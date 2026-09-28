"""Stacking ensemble: one base model per modality, combined by a learned meta-model.

Per the project plan (Section 24): "Do not manually assume weights such as
25/25/25/25 initially. Learn ensemble weights using validation data." Base models
are fit on the training split; the meta-model is fit on the *validation* split's
predictions (out-of-sample for the base models), so it learns real relative
weights rather than memorizing in-sample overfit patterns.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_modality_models(
    X_train_by_modality: dict[str, pd.DataFrame],
    y_train: pd.Series,
    model_factory: Callable[[], object],
) -> dict[str, object]:
    """Fits one classifier (from model_factory(), a fresh instance per modality) per modality."""
    models = {}
    for modality, X in X_train_by_modality.items():
        model = model_factory()
        model.fit(X, y_train)
        models[modality] = model
    return models


def build_meta_features(models: dict[str, object], X_by_modality: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each modality model's predicted P(up) becomes one meta-feature column."""
    return pd.DataFrame(
        {modality: model.predict_proba(X_by_modality[modality])[:, 1] for modality, model in models.items()}
    )


def fit_meta_model(meta_features: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y)
    return meta_model


def ensemble_predict_proba(
    modality_models: dict[str, object], meta_model: LogisticRegression, X_by_modality: dict[str, pd.DataFrame]
) -> pd.Series:
    """End-to-end: modality models -> meta-features -> meta-model -> final P(up)."""
    meta_features = build_meta_features(modality_models, X_by_modality)
    return meta_model.predict_proba(meta_features)[:, 1]
