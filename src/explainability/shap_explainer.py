"""Per-prediction SHAP explanations for tree-based classifiers.

SHAP values from TreeExplainer are in the model's raw margin (log-odds) space,
not probability space: sigmoid(sum(shap_values) + expected_value) ==
predict_proba. Reported factor contributions are therefore log-odds units, not
probability points -- a larger positive value pushes the prediction toward
"up" but does not translate linearly into percentage points of probability.
"""

from __future__ import annotations

import pandas as pd
import shap


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def explain_predictions(explainer: shap.TreeExplainer, X: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of per-feature SHAP values (log-odds units), same
    shape/columns/index as X.
    """
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns, index=X.index)


def top_factors(shap_row: pd.Series, n: int = 5) -> dict[str, pd.Series]:
    """Splits one row's SHAP values into the top-n positive and top-n negative
    factors, each sorted by magnitude (largest absolute contribution first).
    """
    positive = shap_row[shap_row > 0].sort_values(ascending=False).head(n)
    negative = shap_row[shap_row < 0].sort_values().head(n)
    return {"positive": positive, "negative": negative}
