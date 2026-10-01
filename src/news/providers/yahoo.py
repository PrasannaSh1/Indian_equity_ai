"""Yahoo Finance news provider -- wraps the existing src.news.ingestion.fetch_news
rather than duplicating its fetch logic. The only provider actually integrated
today; see src.news.providers.base for why a provider interface exists anyway.
"""

from __future__ import annotations

import pandas as pd

from src.news.ingestion import fetch_news
from src.news.providers.base import NewsProviderError


class YahooNewsProvider:
    name = "yahoo_finance"

    def fetch(self, symbol: str) -> pd.DataFrame:
        try:
            return fetch_news(symbol)
        except Exception as exc:  # noqa: BLE001 -- any underlying failure (network, parsing) is this provider's to report
            raise NewsProviderError(f"{self.name} failed for {symbol!r}: {exc}") from exc
