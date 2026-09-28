"""Combines price/technical, fundamental, news, macro, and market-regime features
into one multimodal feature vector (project plan Section 23), respecting each
modality's real-world availability timestamp (Section 16: never let the model see
information that was not available at prediction time).
"""

from __future__ import annotations

import pandas as pd

from src.macro.ingestion import MACRO_TICKERS
from src.models.dataset import FEATURE_COLUMNS as TECHNICAL_FEATURE_COLUMNS

# Indian companies typically report annual results 45-60 days after fiscal
# year end; a fiscal year's fundamentals are treated as unknown to the market
# until this many days after fiscal_year_end, not on fiscal_year_end itself.
FUNDAMENTALS_REPORTING_LAG_DAYS = 60

FUNDAMENTAL_FEATURE_COLUMNS = [
    # roce/current_ratio/interest_coverage are deliberately excluded: bank income
    # statements (HDFCBANK) don't report EBIT or a current-assets/liabilities
    # split, so those columns are NaN for every HDFCBANK fiscal year -- including
    # them here would silently drop the entire symbol from the dataset (same
    # issue Phase 4's Fundamental Score already solved by excluding them).
    "roe",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_diluted_growth_yoy",
    "free_cash_flow_growth_yoy",
]

NEWS_FEATURE_COLUMNS = [
    "news_count",
    "sentiment_mean",
    "sentiment_std",
    "positive_news_ratio",
    "negative_news_ratio",
]

MACRO_FEATURE_COLUMNS = [f"{name}_return_1d" for name in MACRO_TICKERS] + ["india_vix"]

REGIME_FEATURE_COLUMNS = ["is_bull", "is_high_vol"]

MULTIMODAL_FEATURE_GROUPS = {
    "technical": TECHNICAL_FEATURE_COLUMNS,
    "fundamental": FUNDAMENTAL_FEATURE_COLUMNS,
    "news": NEWS_FEATURE_COLUMNS,
    "macro_regime": MACRO_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS,
}


def merge_fundamentals_pit(df: pd.DataFrame, fundamentals_annual: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time-safe fundamentals join: each fiscal year's ratios only become
    visible FUNDAMENTALS_REPORTING_LAG_DAYS after that fiscal year ends, using the
    latest fiscal year whose reporting lag has already elapsed as of `date`.
    """
    fa = fundamentals_annual.copy()
    fa["available_date"] = fa["fiscal_year_end"] + pd.Timedelta(days=FUNDAMENTALS_REPORTING_LAG_DAYS)
    fa = fa[["symbol", "available_date", *FUNDAMENTAL_FEATURE_COLUMNS]].sort_values("available_date")

    merged_parts = []
    for symbol, group in df.sort_values("date").groupby("symbol"):
        fa_symbol = fa[fa["symbol"] == symbol].drop(columns="symbol")
        merged = pd.merge_asof(
            group, fa_symbol, left_on="date", right_on="available_date", direction="backward"
        )
        merged_parts.append(merged)

    return pd.concat(merged_parts, ignore_index=True).drop(columns="available_date")


def merge_news_features(df: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """Left-joins daily news features onto (date, symbol); days with no news get zeros
    (no news is a real, informative state -- "nothing happened" -- not missing data).
    """
    merged = df.merge(news_daily, on=["date", "symbol"], how="left")
    merged[NEWS_FEATURE_COLUMNS] = merged[NEWS_FEATURE_COLUMNS].fillna(0.0)
    return merged


def merge_macro_and_regime(df: pd.DataFrame, macro_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    """Merges date-keyed (not symbol-keyed) macro and regime features onto every symbol's row."""
    macro_regime = macro_df.merge(regime_df, on="date", how="left")
    return df.merge(macro_regime, on="date", how="left")


def build_multimodal_dataset(
    technical_df: pd.DataFrame,
    fundamentals_annual: pd.DataFrame,
    news_daily: pd.DataFrame,
    macro_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full Phase 7 dataset: Phase 5's technical feature table + target, extended
    with point-in-time-safe fundamentals, news, macro, and regime features.
    """
    from src.models.dataset import add_target, build_feature_table

    df = build_feature_table(technical_df)
    df = add_target(df)
    df = merge_fundamentals_pit(df, fundamentals_annual)
    df = merge_news_features(df, news_daily)
    df = merge_macro_and_regime(df, macro_df, regime_df)

    all_feature_columns = [col for cols in MULTIMODAL_FEATURE_GROUPS.values() for col in cols]
    required = all_feature_columns + ["next_day_direction"]
    return df.dropna(subset=required).reset_index(drop=True)
