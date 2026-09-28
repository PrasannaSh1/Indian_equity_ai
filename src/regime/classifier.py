"""Market-regime engine: classifies each day as Bull/Bear x High/Low Volatility
(project plan Section 17), using only information available up to that date.
"""

from __future__ import annotations

import pandas as pd

TREND_WINDOW = 60
MIN_VOL_HISTORY = 60


def classify_market_regime(
    nifty_close: pd.Series,
    india_vix: pd.Series,
    trend_window: int = TREND_WINDOW,
    min_vol_history: int = MIN_VOL_HISTORY,
) -> pd.DataFrame:
    """`nifty_close` and `india_vix` must be date-indexed (or share a `date` column
    already aligned) daily series, sorted ascending by date.

    - Trend: bull if the trailing `trend_window`-day NIFTY 50 return > 0, else bear.
    - Volatility: high if today's India VIX is above its own *expanding* historical
      median (i.e. the median of all VIX values up to and including today) -- using
      an expanding rather than full-sample median keeps the threshold point-in-time
      safe, since the full-sample median would leak knowledge of future VIX levels.
    """
    trend_return = nifty_close.pct_change(trend_window, fill_method=None)
    is_bull = trend_return > 0

    vix_expanding_median = india_vix.expanding(min_periods=min_vol_history).median()
    is_high_vol = india_vix > vix_expanding_median

    return pd.DataFrame(
        {
            "trend_return": trend_return,
            "is_bull": is_bull.astype(float).where(trend_return.notna()),
            "vix_level": india_vix,
            "is_high_vol": is_high_vol.astype(float).where(vix_expanding_median.notna()),
        }
    )
