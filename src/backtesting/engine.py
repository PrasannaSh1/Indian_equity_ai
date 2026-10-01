"""Signal-driven backtest simulation with transaction costs and slippage.

Trading convention (deliberately simple, documented rather than hidden): a
signal generated from day t's features is "entered" at day t's close and
"exited" at day t+1's close -- i.e. the trade's realized return is exactly
next_day_return, the same quantity the model predicts. This is a
simplification (a real order can't fill at the exact close price used to
generate the signal); it is chosen because it maps directly onto what the
Phase 5/7/8 models forecast, and the cost/slippage assumptions below are
intended to compensate for some of that optimism.

Cost assumptions are itemized (website audit Section 26) via
compute_itemized_transaction_costs(), covering brokerage, STT, exchange
transaction charges, SEBI turnover fees, stamp duty, GST on the applicable
components, and slippage -- see that function's docstring for exact rates and
the explicit caveat that these are illustrative research-grade approximations,
not a live broker rate card. simulate_trades still accepts the older
transaction_cost_bps/slippage_bps flat-sum override for backward compatibility
(and because this project's own hand-computed tests pin exact values against
it) -- passing either explicitly preserves the old combined-bps behavior
exactly; omitting both uses the itemized total instead.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TRANSACTION_COST_BPS = 10
DEFAULT_SLIPPAGE_BPS = 5


def compute_itemized_transaction_costs(
    brokerage_bps: float = 0.0,
    stt_bps: float = 10.0,
    exchange_txn_bps: float = 0.35,
    sebi_turnover_bps: float = 0.01,
    stamp_duty_bps: float = 1.5,
    gst_rate: float = 0.18,
    slippage_bps: float = 5.0,
) -> dict:
    """Itemized round-trip transaction cost for a typical Indian retail equity
    DELIVERY trade (not intraday, which has a different, lower STT rate and
    additional charges this project does not model). Rates are illustrative
    research-grade approximations as of this project's documentation date, not a
    live broker rate card -- verify current rates before relying on this for a
    real trading decision:

    - brokerage_bps=0: many Indian discount brokers charge zero brokerage on
      equity delivery trades; 0 is the common case, not a simplification that
      ignores a real cost.
    - stt_bps=10 (0.1%): Securities Transaction Tax on the SELL leg of a delivery
      trade (charged once per round trip in this model, not doubled, since STT is
      sell-only for delivery).
    - exchange_txn_bps=0.35 + sebi_turnover_bps=0.01: NSE transaction charges and
      SEBI's turnover fee, both ad valorem on traded value.
    - stamp_duty_bps=1.5 (0.015%): stamp duty on the BUY leg only, per Indian
      stamp-duty rules for delivery trades.
    - gst_rate=0.18 (18%): GST applies to brokerage + exchange transaction
      charges (not to STT or stamp duty, which are themselves taxes).
    - slippage_bps=5: unchanged from this project's original flat estimate --
      price impact/spread cost for liquid large-cap NSE stocks, not a government
      levy, so it isn't part of the GST base.
    """
    exchange_and_sebi_bps = exchange_txn_bps + sebi_turnover_bps
    gst_bps = gst_rate * (brokerage_bps + exchange_and_sebi_bps)
    total_bps = brokerage_bps + stt_bps + exchange_and_sebi_bps + stamp_duty_bps + gst_bps + slippage_bps
    return {
        "brokerage_bps": brokerage_bps,
        "stt_bps": stt_bps,
        "exchange_txn_bps": exchange_txn_bps,
        "sebi_turnover_bps": sebi_turnover_bps,
        "stamp_duty_bps": stamp_duty_bps,
        "gst_bps": gst_bps,
        "slippage_bps": slippage_bps,
        "total_bps": total_bps,
    }


def generate_signals(probability_up: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Long-only signal: 1 (take the trade) if probability_up > threshold, else 0 (stay in cash)."""
    return (probability_up > threshold).astype(int)


def simulate_trades(
    df: pd.DataFrame,
    threshold: float = 0.55,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
) -> pd.DataFrame:
    """Requires columns: date, symbol, next_day_return, probability_up.

    Adds: signal, gross_return (next_day_return if signal==1, else 0),
    cost (round-trip friction, only charged when a trade is actually taken),
    net_return (gross_return - cost).

    If either transaction_cost_bps or slippage_bps is passed explicitly, uses the
    old flat combined-bps model exactly (the omitted one falls back to its
    DEFAULT_* constant) -- this is what this module's existing hand-computed
    tests pin. If both are omitted, uses compute_itemized_transaction_costs()'s
    total instead, since that's a more transparent (if still approximate)
    breakdown of the same round-trip cost.
    """
    df = df.copy()
    df["signal"] = generate_signals(df["probability_up"], threshold)
    df["gross_return"] = df["next_day_return"] * df["signal"]

    if transaction_cost_bps is None and slippage_bps is None:
        round_trip_cost = compute_itemized_transaction_costs()["total_bps"] / 10_000
    else:
        transaction_cost_bps = DEFAULT_TRANSACTION_COST_BPS if transaction_cost_bps is None else transaction_cost_bps
        slippage_bps = DEFAULT_SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        round_trip_cost = (transaction_cost_bps + slippage_bps) / 10_000

    df["cost"] = round_trip_cost * df["signal"]
    df["net_return"] = df["gross_return"] - df["cost"]
    return df


def aggregate_portfolio_returns(trades: pd.DataFrame) -> pd.Series:
    """Equal-weights net_return across every symbol with an active signal that day;
    a day with no active signals anywhere returns 0 (fully in cash).
    """
    active = trades[trades["signal"] == 1]
    daily_return = active.groupby("date")["net_return"].mean()

    all_dates = pd.Series(trades["date"].unique()).sort_values()
    return daily_return.reindex(all_dates, fill_value=0.0).sort_index()
