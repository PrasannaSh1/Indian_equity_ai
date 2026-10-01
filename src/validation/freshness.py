"""Trading-calendar-aware data-freshness policy (website audit, Sections 39-40).

Staleness must be measured in trading days, not calendar days, so a Friday's data
viewed on a Monday isn't wrongly flagged "3 days stale" -- weekends (and a short,
approximate NSE holiday list) don't count as failures.
"""

from __future__ import annotations

from datetime import date

import numpy as np

FRESH_MAX_TRADING_DAYS = 1
STALE_MAX_TRADING_DAYS = 3

# Illustrative/approximate NSE trading-holiday list -- manually maintained, not
# sourced from a live exchange-calendar library (none is a dependency of this
# project today; adding one is out of scope for this pass). Must be refreshed
# periodically; documented explicitly as an approximation, same honesty standard
# as this project's other stated assumptions (e.g. backtest transaction costs).
NSE_HOLIDAYS_APPROX: list[str] = [
    "2026-01-26",  # Republic Day
    "2026-03-06",  # Holi (approximate)
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Diwali (approximate)
    "2026-12-25",  # Christmas
]


def trading_days_stale(as_of: date, reference: date | None = None, holidays: list[str] = NSE_HOLIDAYS_APPROX) -> int:
    """Number of NSE trading days between as_of and reference (defaults to today),
    excluding weekends (numpy's default weekmask) and the given holiday list.
    """
    reference = reference or date.today()
    if reference <= as_of:
        return 0
    return int(np.busday_count(as_of, reference, holidays=holidays))


def staleness_label(trading_days: int) -> str:
    """'fresh' (<= 1 trading day), 'stale' (2-3), 'very_stale' (> 3)."""
    if trading_days <= FRESH_MAX_TRADING_DAYS:
        return "fresh"
    if trading_days <= STALE_MAX_TRADING_DAYS:
        return "stale"
    return "very_stale"
