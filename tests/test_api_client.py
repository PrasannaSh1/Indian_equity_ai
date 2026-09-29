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

    mock_get.assert_called_once_with("http://testserver/stocks/RELIANCE/overview", timeout=15)
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

    mock_get.assert_called_once_with("http://testserver/stocks", timeout=15)


def test_http_error_status_propagates():
    client = ApiClient(base_url="http://testserver")
    with patch("app.api_client.requests.get", return_value=_mock_response({}, status_code=404)):
        with pytest.raises(requests.HTTPError):
            client.overview("UNKNOWN")
