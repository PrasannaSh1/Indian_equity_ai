from unittest.mock import MagicMock, patch

import pytest
import requests

from app.api_client import ApiClient


def _mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status.side_effect = (
        None if status_code < 400 else requests.HTTPError(f"{status_code} error")
    )
    return mock


def test_overview_requests_the_correct_url_and_returns_json():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response({"close": 100.0})) as mock_get:
        result = client.overview("RELIANCE")

    mock_get.assert_called_once_with("http://testserver/stocks/RELIANCE/overview", params=None, timeout=15)
    assert result == {"close": 100.0}


def test_ask_posts_the_symbol_and_query_as_json_body():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.post", return_value=_mock_response({"answer": "..."})) as mock_post:
        result = client.ask("TCS", "why is this bullish")

    mock_post.assert_called_once_with(
        "http://testserver/ai-analyst/ask",
        json={"symbol": "TCS", "query": "why is this bullish"},
        timeout=15,
    )
    assert result == {"answer": "..."}


def test_base_url_trailing_slash_is_stripped():
    client = ApiClient(base_url="http://testserver/")
    with patch("app.api_client.requests.get", return_value=_mock_response([])) as mock_get:
        client.list_stocks()

    mock_get.assert_called_once_with("http://testserver/stocks", params=None, timeout=15)


def test_http_error_status_propagates():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response({}, status_code=404)):
        with pytest.raises(requests.HTTPError):
            client.overview("UNKNOWN")


def test_history_passes_days_as_a_query_param():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response([{"close": 1.0}])) as mock_get:
        result = client.history("RELIANCE", days=90)

    mock_get.assert_called_once_with(
        "http://testserver/stocks/RELIANCE/history", params={"days": 90}, timeout=15
    )
    assert result == [{"close": 1.0}]


def test_analyze_posts_company_horizon_and_analysis_type():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.post", return_value=_mock_response({"company": {}})) as mock_post:
        result = client.analyze("DIXON", horizon="60d", analysis_type="quick")

    mock_post.assert_called_once_with(
        "http://testserver/analysis",
        json={"company": "DIXON", "horizon": "60d", "analysis_type": "quick"},
        timeout=15,
    )
    assert result == {"company": {}}


def test_resolve_company_requests_the_correct_url():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response({"symbol": "TCS"})) as mock_get:
        result = client.resolve_company("TCS")

    mock_get.assert_called_once_with("http://testserver/company/TCS/resolve", params=None, timeout=15)
    assert result == {"symbol": "TCS"}


def test_snapshot_requests_the_correct_url():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response({"as_of": "2026-09-28"})) as mock_get:
        result = client.snapshot("TCS")

    mock_get.assert_called_once_with("http://testserver/stocks/TCS/snapshot", params=None, timeout=15)
    assert result == {"as_of": "2026-09-28"}


def test_lineage_requests_the_correct_url():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response([{"category": "market_data"}])) as mock_get:
        result = client.lineage("TCS")

    mock_get.assert_called_once_with("http://testserver/stocks/TCS/lineage", params=None, timeout=15)
    assert result == [{"category": "market_data"}]
