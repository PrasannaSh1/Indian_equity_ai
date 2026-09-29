"""Entry/exit decision-support layer (project plan Section 27).

Deliberately a rule-based calculator, not another ML model: it combines
already-computed forecasts (probability_up) and volatility (ATR) into
hypothetical price zones. Long-only, consistent with Phase 9's backtest
convention -- no position is suggested when the signal isn't bullish.

These are model-generated hypothetical levels, not guaranteed prices.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLD = 0.55
DEFAULT_ENTRY_BUFFER_PCT = 0.005  # 0.5% band above today's close
DEFAULT_STOP_ATR_MULTIPLE = 1.5
DEFAULT_REWARD_RISK_RATIO = 2.0


def compute_entry_exit(
    close: pd.Series,
    atr: pd.Series,
    probability_up: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    entry_buffer_pct: float = DEFAULT_ENTRY_BUFFER_PCT,
    stop_atr_multiple: float = DEFAULT_STOP_ATR_MULTIPLE,
    reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
) -> pd.DataFrame:
    """For rows where probability_up > threshold: an entry zone [close, close *
    (1 + entry_buffer_pct)], a stop stop_atr_multiple*ATR below close, and a
    target reward_risk_ratio times the risk (close-to-stop distance) above
    close. Rows without a bullish signal get no suggested position (NaN
    zones), not a fabricated trade.
    """
    has_signal = probability_up > threshold

    risk_per_share = stop_atr_multiple * atr
    entry_low = close.where(has_signal)
    entry_high = (close * (1 + entry_buffer_pct)).where(has_signal)
    stop = (close - risk_per_share).where(has_signal)
    target = (close + reward_risk_ratio * risk_per_share).where(has_signal)

    return pd.DataFrame(
        {
            "has_signal": has_signal,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop,
            "target": target,
        }
    )
