"""Resolves a user-supplied company identifier (ticker, .NS/.BO-suffixed ticker, or
company name) to a canonical symbol (Phase 15).

Aliases are only maintained for the training universe (src.config.UNIVERSE): this
project has no licensed NSE company-name master to draw a wider alias list from
(see SUMMARY.md's data-source limitations). Any identifier that doesn't match a
known alias is treated as a literal ticker candidate -- normalized and returned
unresolved-by-name, `in_training_universe=False` -- and left for the ingestion layer
(the only real source of truth for "does this actually trade on NSE") to confirm.
This is exactly Phase 16's distinction between the training universe and the wider
analysis universe: not being in the training universe is not a resolution failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import UNIVERSE

# NIFTY 50 constituent names for src.config.UNIVERSE's tickers (verified against the
# same corporate-action adjustments config.py documents: TMPV, DMART substitutions).
COMPANY_NAME_ALIASES: dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "BAJFINANCE": "Bajaj Finance",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "HCLTECH": "HCL Technologies",
    "AXISBANK": "Axis Bank",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharmaceutical Industries",
    "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro",
    "NESTLEIND": "Nestle India",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports and Special Economic Zone",
    "BAJAJFINSV": "Bajaj Finserv",
    "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corporation of India",
    "M&M": "Mahindra & Mahindra",
    "TATASTEEL": "Tata Steel",
    "TMPV": "Tata Motors Passenger Vehicles",
    "JSWSTEEL": "JSW Steel",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "GRASIM": "Grasim Industries",
    "TECHM": "Tech Mahindra",
    "INDUSINDBK": "IndusInd Bank",
    "CIPLA": "Cipla",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "BRITANNIA": "Britannia Industries",
    "DIVISLAB": "Divi's Laboratories",
    "COALINDIA": "Coal India",
    "BPCL": "Bharat Petroleum Corporation",
    "HEROMOTOCO": "Hero MotoCorp",
    "APOLLOHOSP": "Apollo Hospitals Enterprise",
    "UPL": "UPL Limited",
    "BAJAJ-AUTO": "Bajaj Auto",
    "TATACONSUM": "Tata Consumer Products",
    "ONGC": "Oil and Natural Gas Corporation",
    "HINDALCO": "Hindalco Industries",
    "DMART": "Avenue Supermarts",
}

_TRAINING_UNIVERSE = set(UNIVERSE)
_VALID_TICKER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&")


class CompanyNotResolvedError(ValueError):
    """Raised when an identifier is empty or clearly not a plausible ticker/name."""


class AmbiguousCompanyError(ValueError):
    """Raised when a name fragment matches more than one known company."""


@dataclass(frozen=True)
class ResolvedSecurity:
    symbol: str
    name: str | None
    in_training_universe: bool
    resolved_by: str  # "ticker" | "name_alias" | "unresolved_ticker_guess"


def _normalize_ticker(raw: str) -> str:
    symbol = raw.strip().upper()
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def resolve(identifier: str) -> ResolvedSecurity:
    if not identifier or not identifier.strip():
        raise CompanyNotResolvedError("Empty company identifier.")

    raw = identifier.strip()
    ticker_guess = _normalize_ticker(raw)

    if ticker_guess in _TRAINING_UNIVERSE:
        return ResolvedSecurity(
            symbol=ticker_guess,
            name=COMPANY_NAME_ALIASES.get(ticker_guess),
            in_training_universe=True,
            resolved_by="ticker",
        )

    name_matches = sorted(
        {
            symbol
            for symbol, name in COMPANY_NAME_ALIASES.items()
            if raw.lower() in name.lower() or name.lower() in raw.lower()
        }
    )
    if len(name_matches) == 1:
        symbol = name_matches[0]
        return ResolvedSecurity(
            symbol=symbol,
            name=COMPANY_NAME_ALIASES[symbol],
            in_training_universe=True,
            resolved_by="name_alias",
        )
    if len(name_matches) > 1:
        raise AmbiguousCompanyError(f"{raw!r} matches multiple companies: {name_matches}.")

    if not ticker_guess or not set(ticker_guess) <= _VALID_TICKER_CHARS:
        raise CompanyNotResolvedError(
            f"{raw!r} does not look like a valid NSE ticker and matches no known company name."
        )
    return ResolvedSecurity(
        symbol=ticker_guess,
        name=None,
        in_training_universe=False,
        resolved_by="unresolved_ticker_guess",
    )
