import pandas as pd
import pytest

from src.news.ingestion import REQUIRED_COLUMNS
from src.news.providers.base import NewsProviderError
from src.news.providers.chain import fetch_with_fallback


class _StubProvider:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self._result = result
        self._error = error

    def fetch(self, symbol):
        if self._error:
            raise NewsProviderError(self._error)
        return self._result


def _articles_df(n=1):
    return pd.DataFrame(
        {
            "news_id": [f"id{i}" for i in range(n)],
            "symbol": ["TCS"] * n,
            "headline": [f"headline {i}" for i in range(n)],
            "summary": [""] * n,
            "source": ["Yahoo"] * n,
            "url": [""] * n,
            "published_timestamp": [pd.Timestamp("2026-09-30", tz="UTC")] * n,
        }
    )


def test_fetch_with_fallback_returns_first_providers_non_empty_result():
    primary = _StubProvider("primary", result=_articles_df(2))
    secondary = _StubProvider("secondary", result=_articles_df(5))

    articles, meta = fetch_with_fallback("TCS", providers=[primary, secondary])

    assert len(articles) == 2
    assert meta["provider"] == "primary"
    assert meta["provider_errors"] == []


def test_fetch_with_fallback_falls_back_when_primary_is_empty():
    primary = _StubProvider("primary", result=pd.DataFrame(columns=REQUIRED_COLUMNS))
    secondary = _StubProvider("secondary", result=_articles_df(1))

    articles, meta = fetch_with_fallback("TCS", providers=[primary, secondary])

    assert len(articles) == 1
    assert meta["provider"] == "secondary"


def test_fetch_with_fallback_falls_back_when_primary_errors():
    primary = _StubProvider("primary", error="boom")
    secondary = _StubProvider("secondary", result=_articles_df(1))

    articles, meta = fetch_with_fallback("TCS", providers=[primary, secondary])

    assert len(articles) == 1
    assert meta["provider"] == "secondary"
    assert meta["provider_errors"] == [{"provider": "primary", "error": "boom"}]


def test_fetch_with_fallback_returns_empty_df_and_records_errors_when_all_providers_fail():
    primary = _StubProvider("primary", error="boom1")
    secondary = _StubProvider("secondary", error="boom2")

    articles, meta = fetch_with_fallback("TCS", providers=[primary, secondary])

    assert articles.empty
    assert list(articles.columns) == REQUIRED_COLUMNS
    assert meta["provider"] is None
    assert len(meta["provider_errors"]) == 2


def test_fetch_with_fallback_returns_empty_df_when_all_providers_genuinely_have_no_news():
    primary = _StubProvider("primary", result=pd.DataFrame(columns=REQUIRED_COLUMNS))

    articles, meta = fetch_with_fallback("TCS", providers=[primary])

    assert articles.empty
    assert meta["provider"] is None
    assert meta["provider_errors"] == []  # no error -- genuinely no news, not a failure
