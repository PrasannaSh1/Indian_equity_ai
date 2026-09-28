"""Deduplication and company/entity resolution for news headlines."""

from __future__ import annotations

import pandas as pd

# Aliases used to confirm a headline/summary genuinely references the company,
# rather than merely appearing in that ticker's Yahoo Finance news feed (broad
# sector-roundup articles are common and often only tangentially relevant).
COMPANY_ALIASES = {
    "RELIANCE": ["reliance industries", "reliance", "ril"],
    "TCS": ["tata consultancy", "tcs"],
    "INFY": ["infosys"],
    "HDFCBANK": ["hdfc bank", "hdfc"],
    "ITC": ["itc limited", "itc ltd", "itc"],
}


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact-duplicate news items (same news_id and symbol)."""
    return df.drop_duplicates(subset=["news_id", "symbol"]).reset_index(drop=True)


def resolve_entity_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `mentions_company`: whether the headline/summary text actually
    references the company the article was fetched for.
    """
    df = df.copy()
    text = (df["headline"].fillna("") + " " + df["summary"].fillna("")).str.lower()

    def _mentions(row_text: str, symbol: str) -> bool:
        aliases = COMPANY_ALIASES.get(symbol, [symbol.lower()])
        return any(alias in row_text for alias in aliases)

    df["mentions_company"] = [
        _mentions(t, s) for t, s in zip(text, df["symbol"])
    ]
    return df
