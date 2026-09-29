import numpy as np
import pandas as pd
import pytest

from src.risk.quantiles import fit_quantile_models, predict_quantiles, return_quantiles_to_price_range


class _FakeQuantileModel:
    def __init__(self, constant_prediction):
        self._constant_prediction = constant_prediction
        self.fit_called_with = None

    def fit(self, X, y):
        self.fit_called_with = (X, y)

    def predict(self, X):
        return np.full(len(X), self._constant_prediction)


def test_fit_quantile_models_fits_one_model_per_quantile():
    X_train = pd.DataFrame({"f": [1, 2, 3]})
    y_train = pd.Series([0.01, 0.02, -0.01])

    models = fit_quantile_models(
        X_train, y_train, model_factory=lambda q: _FakeQuantileModel(q), quantiles=(0.1, 0.5, 0.9)
    )

    assert set(models.keys()) == {0.1, 0.5, 0.9}
    for model in models.values():
        assert model.fit_called_with is not None


def test_predict_quantiles_enforces_monotonicity_when_models_cross():
    # Deliberately mis-specified so the "q10" model predicts higher than "q90"'s --
    # predict_quantiles must still return q10 <= q50 <= q90 per row.
    models = {0.1: _FakeQuantileModel(0.05), 0.5: _FakeQuantileModel(0.0), 0.9: _FakeQuantileModel(-0.05)}
    X = pd.DataFrame({"f": [1, 2]})

    out = predict_quantiles(models, X)

    assert list(out.columns) == ["q10", "q50", "q90"]
    assert (out["q10"] <= out["q50"]).all()
    assert (out["q50"] <= out["q90"]).all()
    # the underlying values are preserved, just reassigned to sorted-order columns
    assert out["q10"].iloc[0] == pytest.approx(-0.05)
    assert out["q90"].iloc[0] == pytest.approx(0.05)


def test_return_quantiles_to_price_range_matches_hand_computed_prices():
    quantile_returns = pd.DataFrame({"q10": [-0.05], "q50": [0.0], "q90": [0.05]})
    close_price = pd.Series([100.0])

    out = return_quantiles_to_price_range(quantile_returns, close_price)

    assert out["q10"].iloc[0] == pytest.approx(95.0)
    assert out["q50"].iloc[0] == pytest.approx(100.0)
    assert out["q90"].iloc[0] == pytest.approx(105.0)
