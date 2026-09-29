"""Central stock universe configuration (Phase 14: scale-out).

Single source of truth for which NSE symbols the pipeline processes -- every
notebook imports UNIVERSE from here instead of hardcoding its own copy, so
expanding the universe is a one-line change here, not an edit to N notebooks.

UNIVERSE is NIFTY 50's constituents (a natural, principled "~50 liquid
stocks" selection, not an arbitrary pick), each ticker verified to actually
resolve on Yahoo Finance as of 2026-09-30. Two adjustments from the textbook
NIFTY 50 list, both due to real corporate actions:
- Tata Motors demerged into passenger/commercial vehicle entities in 2025;
  "TATAMOTORS" no longer resolves. Using TMPV (Tata Motors Passenger
  Vehicles), the passenger-vehicle successor entity.
- "LTIM" (LTIMindtree) did not resolve via this data source at verification
  time; substituted with DMART (Avenue Supermarts), another well-known
  NSE large-cap, rather than leaving the universe short of 50.
"""

from __future__ import annotations

ORIGINAL_UNIVERSE = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ADANIENT",
    "ADANIPORTS", "BAJAJFINSV", "NTPC", "POWERGRID", "M&M", "TATASTEEL", "TMPV",
    "JSWSTEEL", "HDFCLIFE", "SBILIFE", "GRASIM", "TECHM", "INDUSINDBK", "CIPLA",
    "DRREDDY", "EICHERMOT", "BRITANNIA", "DIVISLAB", "COALINDIA", "BPCL", "HEROMOTOCO",
    "APOLLOHOSP", "UPL", "BAJAJ-AUTO", "TATACONSUM", "ONGC", "HINDALCO", "DMART",
]

assert len(UNIVERSE) == 50
assert len(set(UNIVERSE)) == len(UNIVERSE)  # no duplicates
