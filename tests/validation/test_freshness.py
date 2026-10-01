from datetime import date

from src.validation.freshness import staleness_label, trading_days_stale


def test_staleness_label_thresholds():
    assert staleness_label(0) == "fresh"
    assert staleness_label(1) == "fresh"
    assert staleness_label(2) == "stale"
    assert staleness_label(3) == "stale"
    assert staleness_label(4) == "very_stale"


def test_trading_days_stale_does_not_count_weekend_as_stale():
    # Friday data viewed on the following Monday is 1 trading day stale, not 3 calendar days.
    # holidays=[] isolates weekend-exclusion behavior from the module's holiday list.
    friday = date(2026, 9, 25)
    monday = date(2026, 9, 28)

    assert trading_days_stale(friday, monday, holidays=[]) == 1
    assert staleness_label(trading_days_stale(friday, monday, holidays=[])) == "fresh"


def test_trading_days_stale_excludes_configured_holiday():
    before_holiday = date(2026, 10, 1)
    after_holiday = date(2026, 10, 5)  # spans the Oct 2 Gandhi Jayanti holiday + weekend

    days = trading_days_stale(before_holiday, after_holiday, holidays=["2026-10-02"])

    assert days == 1


def test_trading_days_stale_reproduces_the_real_repo_gap():
    # The actual confirmed drift: model_predictions (09-21) vs prices_daily (09-28).
    stale_date = date(2026, 9, 21)
    fresh_date = date(2026, 9, 28)

    days = trading_days_stale(stale_date, fresh_date)

    assert days == 5  # one week minus the weekend
    assert staleness_label(days) == "very_stale"


def test_trading_days_stale_returns_zero_when_reference_not_after_as_of():
    same_day = date(2026, 9, 28)

    assert trading_days_stale(same_day, same_day) == 0
