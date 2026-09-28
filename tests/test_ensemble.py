import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.models.ensemble import build_meta_features, ensemble_predict_proba, fit_meta_model, fit_modality_models


def _synthetic_split(n, rng):
    y = rng.integers(0, 2, size=n).astype(float)
    good = pd.DataFrame({"good_feature": y + rng.normal(0, 0.05, size=n)})
    bad = pd.DataFrame({"bad_feature": rng.normal(0, 1, size=n)})
    return {"good": good, "bad": bad}, pd.Series(y)


def test_fit_modality_models_fits_one_model_per_modality():
    rng = np.random.default_rng(0)
    X_train, y_train = _synthetic_split(100, rng)

    models = fit_modality_models(X_train, y_train, model_factory=LogisticRegression)

    assert set(models.keys()) == {"good", "bad"}
    assert all(hasattr(m, "predict_proba") for m in models.values())


def test_build_meta_features_has_one_column_per_modality_with_valid_probabilities():
    rng = np.random.default_rng(1)
    X_train, y_train = _synthetic_split(100, rng)
    models = fit_modality_models(X_train, y_train, model_factory=LogisticRegression)

    meta = build_meta_features(models, X_train)

    assert list(meta.columns) == ["good", "bad"]
    assert meta.shape[0] == 100
    assert meta.to_numpy().min() >= 0.0
    assert meta.to_numpy().max() <= 1.0


def test_ensemble_leverages_the_genuinely_predictive_modality():
    rng = np.random.default_rng(42)
    X_train, y_train = _synthetic_split(200, rng)
    X_val, y_val = _synthetic_split(100, rng)
    X_test, y_test = _synthetic_split(100, rng)

    models = fit_modality_models(X_train, y_train, model_factory=LogisticRegression)
    meta_features_val = build_meta_features(models, X_val)
    meta_model = fit_meta_model(meta_features_val, y_val)

    final_proba = ensemble_predict_proba(models, meta_model, X_test)

    assert roc_auc_score(y_test, final_proba) > 0.95
    # the meta-model should have learned to weight "good" far above "bad"
    coefs = dict(zip(meta_features_val.columns, meta_model.coef_[0]))
    assert coefs["good"] > coefs["bad"]
