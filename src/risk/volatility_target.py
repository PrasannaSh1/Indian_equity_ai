"""Forward realized-volatility target for the volatility model.

Unlike Phase 1's rolling volatility (a backward-looking feature, computed
from *past* returns and safe to use as model input), this target looks
*forward*: it is only ever used as a training label, never as an input
feature -- using it as a feature would leak the future.
"""

from __future__ import annotations

import pandas as pd


def add_forward_volatility_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Adds `forward_volatility_{horizon}d`: the realized std of daily returns over
    the next `horizon` trading days (t+1 .. t+horizon), per symbol.

    The last `horizon` rows of each symbol have no full forward window and are
    left as NaN rather than computed on a partial (shorter) window, so every
    label reflects the same horizon.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    if "return_1d" not in df.columns:
        df["return_1d"] = df.groupby("symbol")["close"].pct_change(1)

    target_col = f"forward_volatility_{horizon}d"

    def _forward_realized_vol(returns: pd.Series) -> pd.Series:
        # Backward rolling std at position (t + horizon) covers returns
        # [t+1 .. t+horizon] -- exactly the forward window for position t -- so
        # shifting that result back by `horizon` aligns it onto row t.
        return returns.rolling(window=horizon).std().shift(-horizon)

    df[target_col] = df.groupby("symbol")["return_1d"].transform(_forward_realized_vol)
    return df
