from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from src.models.predict import NoTrainedModelError, load_model, predict_proba_up
from src.models.registry import register_model


def _feature_row():
    return pd.DataFrame([{col: 0.1 for col in FEATURE_COLUMNS}])


def test_load_model_raises_when_registry_empty(tmp_path):
    with pytest.raises(NoTrainedModelError):
        load_model(model_dir=tmp_path)


def test_load_model_raises_on_feature_schema_mismatch(tmp_path):
    register_model(
        {
            "model_version": "v1",
            "model_name": "naive_baseline",
            "training_date": "2026-01-01T00:00:00+00:00",
            "feature_schema_version": "0.9-stale",
            "base_rate": 0.5,
            "model_path": None,
            "scaler_path": None,
        },
        model_dir=tmp_path,
    )

    with pytest.raises(NoTrainedModelError):
        load_model(model_dir=tmp_path)


def test_predict_proba_up_uses_base_rate_when_model_is_none():
    entry = {"base_rate": 0.63, "needs_scaling": False}
    proba = predict_proba_up(_feature_row(), entry, model=None, scaler=None)

    assert proba.iloc[0] == pytest.approx(0.63)


def test_predict_proba_up_calls_model_predict_proba():
    entry = {"needs_scaling": False}
    stub_model = SimpleNamespace(predict_proba=lambda X: np.tile([0.3, 0.7], (len(X), 1)))

    proba = predict_proba_up(_feature_row(), entry, model=stub_model, scaler=None)

    assert proba.iloc[0] == pytest.approx(0.7)


def test_predict_proba_up_applies_scaler_when_needed():
    calls = {}

    class StubScaler:
        def transform(self, X):
            calls["called"] = True
            return X.values

    entry = {"needs_scaling": True}
    stub_model = SimpleNamespace(predict_proba=lambda X: np.tile([0.4, 0.6], (len(X), 1)))

    proba = predict_proba_up(_feature_row(), entry, model=stub_model, scaler=StubScaler())

    assert calls.get("called") is True
    assert proba.iloc[0] == pytest.approx(0.6)


def test_predict_proba_up_raises_on_missing_features():
    entry = {"needs_scaling": False}
    incomplete = pd.DataFrame([{FEATURE_COLUMNS[0]: 0.1}])

    with pytest.raises(ValueError):
        predict_proba_up(incomplete, entry, model=None, scaler=None)
