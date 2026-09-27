"""Composite 0-100 Fundamental Score."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("roe", "debt_to_equity", "revenue_growth_yoy", "pe_ratio")


def fundamental_score(df: pd.DataFrame) -> pd.Series:
    """Averages four equally weighted 0-100 sub-scores. Requires columns:
    roe (fraction), debt_to_equity, revenue_growth_yoy (fraction), pe_ratio.

    - profitability: ROE scaled so 0% -> 0, 40%+ -> 100
    - leverage: Debt/Equity scaled so 0x -> 100, 2x+ -> 0
    - growth: revenue growth scaled so -20% -> 0, +40% -> 100
    - valuation: trailing P/E scaled so 0x -> 100 (cheap), 40x+ -> 0 (expensive)

    These caps are deliberately simple, documented heuristics (not sector-relative
    benchmarks) and are chosen so a "normal" healthy large-cap stock lands near the
    middle of the range.
    """
    profitability = (df["roe"].clip(lower=0, upper=0.40) / 0.40) * 100
    leverage = (1 - (df["debt_to_equity"].clip(lower=0, upper=2) / 2)) * 100
    growth = ((df["revenue_growth_yoy"].clip(lower=-0.20, upper=0.40) + 0.20) / 0.60) * 100
    valuation = (1 - (df["pe_ratio"].clip(lower=0, upper=40) / 40)) * 100

    missing_inputs = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)

    score = (profitability + leverage + growth + valuation) / 4
    score = score.clip(lower=0, upper=100)
    return score.where(~missing_inputs)
