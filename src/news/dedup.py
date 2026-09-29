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
    # Phase 14 (50-stock scale-out): the raw ticker rarely appears verbatim in a
    # real headline (e.g. "HINDUNILVR" vs "Hindustan Unilever"), so without an
    # alias most non-original-5 stocks would fail entity resolution on every
    # genuine article. Not yet verified against live articles -- Yahoo's news
    # endpoint was returning zero results for every symbol (a confirmed external
    # 500 error, not specific to this project) at the time this was written.
    "ICICIBANK": ["icici bank", "icici"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "SBIN": ["state bank of india", "sbi"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "BAJFINANCE": ["bajaj finance"],
    "KOTAKBANK": ["kotak mahindra bank", "kotak bank", "kotak"],
    "LT": ["larsen & toubro", "larsen and toubro", "l&t"],
    "HCLTECH": ["hcl technologies", "hcltech", "hcl"],
    "AXISBANK": ["axis bank"],
    "ASIANPAINT": ["asian paints"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "SUNPHARMA": ["sun pharmaceutical", "sun pharma"],
    "TITAN": ["titan company", "titan"],
    "ULTRACEMCO": ["ultratech cement", "ultratech"],
    "WIPRO": ["wipro"],
    "NESTLEIND": ["nestle india", "nestle"],
    "ADANIENT": ["adani enterprises"],
    "ADANIPORTS": ["adani ports"],
    "BAJAJFINSV": ["bajaj finserv"],
    "NTPC": ["ntpc"],
    "POWERGRID": ["power grid corporation", "power grid"],
    "M&M": ["mahindra & mahindra", "mahindra and mahindra", "m&m"],
    "TATASTEEL": ["tata steel"],
    "TMPV": ["tata motors passenger vehicles", "tata motors"],
    "JSWSTEEL": ["jsw steel"],
    "HDFCLIFE": ["hdfc life"],
    "SBILIFE": ["sbi life"],
    "GRASIM": ["grasim industries", "grasim"],
    "TECHM": ["tech mahindra"],
    "INDUSINDBK": ["indusind bank"],
    "CIPLA": ["cipla"],
    "DRREDDY": ["dr reddy", "dr. reddy", "reddy's laboratories"],
    "EICHERMOT": ["eicher motors", "royal enfield"],
    "BRITANNIA": ["britannia industries", "britannia"],
    "DIVISLAB": ["divi's laboratories", "divis labs"],
    "COALINDIA": ["coal india"],
    "BPCL": ["bharat petroleum"],
    "HEROMOTOCO": ["hero motocorp"],
    "APOLLOHOSP": ["apollo hospitals"],
    "UPL": ["upl limited", "united phosphorus"],
    "BAJAJ-AUTO": ["bajaj auto"],
    "TATACONSUM": ["tata consumer products", "tata consumer"],
    "ONGC": ["oil and natural gas corporation", "ongc"],
    "HINDALCO": ["hindalco industries", "hindalco"],
    "DMART": ["avenue supermarts", "dmart", "d-mart"],
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

    # Cast to bool explicitly: an empty list (df has 0 rows, e.g. no news
    # currently available for any symbol) infers as float64, not bool, and
    # pandas silently drops every column when boolean-indexing a 0-row frame
    # with a non-bool mask -- discovered during the Phase 14 scale-out when
    # Yahoo's news endpoint returned zero articles for every symbol.
    df["mentions_company"] = pd.Series(
        [_mentions(t, s) for t, s in zip(text, df["symbol"])], index=df.index, dtype=bool
    )
    return df
