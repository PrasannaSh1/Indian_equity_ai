"""Turns raw macro levels into stationary, model-ready features.

Raw levels (e.g. USD/INR trending from ~74 to ~88 over 5 years) are not
comparable across time and would let a model latch onto a spurious long-run
drift rather than genuine short-term signal, so every macro series is
converted to a 1-day percent change. India VIX is additionally kept as a
level, since its absolute level (not just its daily change) is itself a
meaningful, widely-used volatility-regime signal.
"""

from __future__ import annotations

import pandas as pd

from src.macro.ingestion import MACRO_TICKERS


def add_macro_returns(macro_df: pd.DataFrame) -> pd.DataFrame:
    df = macro_df.sort_values("date").copy()
    for name in MACRO_TICKERS:
        df[f"{name}_return_1d"] = df[name].pct_change(fill_method=None)
    return df
