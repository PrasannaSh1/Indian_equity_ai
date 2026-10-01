"""News provider interface (website audit Section 28).

Only one real provider is integrated today (Yahoo Finance, via src.news.ingestion),
but the interface is defined now so a future provider can be added without
touching any call site -- fetch_with_fallback (chain.py) is written against this
interface, not against YahooNewsProvider specifically.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class NewsProviderError(RuntimeError):
    """Raised by a provider's fetch() when that specific provider fails."""


class NewsProvider(Protocol):
    name: str

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Returns a DataFrame shaped like src.news.ingestion.REQUIRED_COLUMNS, or
        raises NewsProviderError on failure. An empty-but-successful result (no
        news found) returns an empty DataFrame, not an exception.
        """
        ...
