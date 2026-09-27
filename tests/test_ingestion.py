from src.ingestion.market_data import to_nse_ticker


def test_to_nse_ticker_appends_suffix():
    assert to_nse_ticker("reliance") == "RELIANCE.NS"


def test_to_nse_ticker_is_idempotent():
    assert to_nse_ticker("RELIANCE.NS") == "RELIANCE.NS"


def test_to_nse_ticker_passes_through_index_tickers():
    assert to_nse_ticker("^nsei") == "^NSEI"
