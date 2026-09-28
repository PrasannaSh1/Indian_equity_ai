"""Downloads recent company news headlines from Yahoo Finance.

Note: Yahoo Finance's free news feed only returns a small, recent rolling
window per ticker (in practice, roughly the last ~10 items / few days) -- it
is not a historical news archive. A production system would need a licensed
news provider for multi-year backfill; see Indian_Equity_AI_Project_Plan.md
Section 5 on data-source licensing.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.ingestion.market_data import to_nse_ticker

REQUIRED_COLUMNS = [
    "news_id",
    "symbol",
    "headline",
    "summary",
    "source",
    "url",
    "published_timestamp",
]


def fetch_news(symbol: str) -> pd.DataFrame:
    """Fetches the current recent-news feed for a single NSE symbol."""
    ticker = yf.Ticker(to_nse_ticker(symbol))
    items = ticker.news or []

    rows = []
    for item in items:
        content = item.get("content", {})
        provider = content.get("provider") or {}
        canonical = content.get("canonicalUrl") or {}
        rows.append(
            {
                "news_id": item.get("id"),
                "symbol": symbol.strip().upper(),
                "headline": content.get("title", ""),
                "summary": content.get("summary", "") or content.get("description", ""),
                "source": provider.get("displayName", ""),
                "url": canonical.get("url", ""),
                "published_timestamp": content.get("pubDate"),
            }
        )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if not df.empty:
        df["published_timestamp"] = pd.to_datetime(df["published_timestamp"], utc=True)
    return df


def fetch_universe_news(symbols: list[str]) -> pd.DataFrame:
    """Fetches and concatenates the recent-news feed for multiple symbols."""
    frames = [fetch_news(symbol) for symbol in symbols]
    return pd.concat(frames, ignore_index=True)
