"""Historical (empirical) Value at Risk and Expected Shortfall.

Both are reported as return values (negative = a loss), not as positive
"amounts", so they compose naturally with other return-based features. E.g.
VaR_95 = -0.03 means: at 95% confidence, the loss will not exceed 3%.
"""

from __future__ import annotations

import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """The (1 - confidence) empirical percentile of historical returns."""
    return float(returns.quantile(1 - confidence))


def historical_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """The mean return in the tail at/below the VaR threshold (a more negative,
    more conservative measure of downside risk than VaR alone).
    """
    threshold = historical_var(returns, confidence)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float(threshold)
    return float(tail.mean())


def rolling_var_es(df: pd.DataFrame, window: int = 250, confidence: float = 0.95) -> pd.DataFrame:
    """Adds rolling, point-in-time-safe VaR/ES columns (each date's value uses only
    the trailing `window` days of returns up to and including that date), per symbol.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("symbol")["close"].pct_change(1)

    var_col = f"var_{int(confidence * 100)}"
    es_col = f"expected_shortfall_{int(confidence * 100)}"

    df[var_col] = df.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(window=window).apply(lambda w: historical_var(pd.Series(w), confidence), raw=False)
    )
    df[es_col] = df.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(window=window).apply(
            lambda w: historical_expected_shortfall(pd.Series(w), confidence), raw=False
        )
    )
    return df
