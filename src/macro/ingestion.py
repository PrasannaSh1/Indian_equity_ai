"""Downloads macroeconomic/global market series used as regime and multimodal features."""

from __future__ import annotations

import pandas as pd

from src.ingestion.market_data import download_index_ohlcv

MACRO_TICKERS = {
    "india_vix": "^INDIAVIX",
    "usdinr": "INR=X",
    "brent_crude": "BZ=F",
    "us10y_yield": "^TNX",
    "sp500": "^GSPC",
}


def download_macro_data(period: str = "5y") -> pd.DataFrame:
    """Downloads each macro series and returns a wide date-indexed table of close levels."""
    series = {}
    for name, ticker in MACRO_TICKERS.items():
        raw = download_index_ohlcv(ticker, period=period)
        series[name] = raw.set_index("date")["close"]
    wide = pd.DataFrame(series).sort_index()
    return wide.reset_index()
