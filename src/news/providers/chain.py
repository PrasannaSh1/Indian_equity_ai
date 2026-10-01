"""Tries news providers in order, falling back on failure -- website audit
Section 28. Every-provider-failure and every-provider-empty both yield an empty
DataFrame + a meta dict recording what happened, never an exception and never a
fabricated fallback sentiment -- matches this project's existing "no fabricated
data" norm (see src.services.company_analysis._compute_news_block's empty-news
handling).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.news.ingestion import REQUIRED_COLUMNS
from src.news.providers.base import NewsProvider, NewsProviderError
from src.news.providers.yahoo import YahooNewsProvider

DEFAULT_PROVIDER_CHAIN: list[NewsProvider] = [YahooNewsProvider()]


def fetch_with_fallback(symbol: str, providers: list[NewsProvider] = DEFAULT_PROVIDER_CHAIN) -> tuple[pd.DataFrame, dict]:
    """Tries each provider in order, returning the first non-empty result.

    meta = {"provider": <name or None>, "retrieved_at": <utc iso timestamp>,
    "provider_errors": [{"provider": ..., "error": ...}, ...]}. If every provider
    fails or every provider returns empty, articles is an empty DataFrame and
    meta["provider"] is None -- the caller is responsible for distinguishing "no
    news exists" from "every provider errored" via provider_errors, and must not
    convert either into a fabricated sentiment.
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    provider_errors = []

    for provider in providers:
        try:
            articles = provider.fetch(symbol)
        except NewsProviderError as exc:
            provider_errors.append({"provider": provider.name, "error": str(exc)})
            continue
        if not articles.empty:
            return articles, {"provider": provider.name, "retrieved_at": retrieved_at, "provider_errors": provider_errors}

    return pd.DataFrame(columns=REQUIRED_COLUMNS), {
        "provider": None,
        "retrieved_at": retrieved_at,
        "provider_errors": provider_errors,
    }
