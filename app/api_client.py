"""Thin HTTP client wrapping the FastAPI backend (src/api/main.py), used by the
Streamlit dashboard. Kept separate from dashboard.py's rendering code so the
request/response logic is unit-testable without needing Streamlit or a live server.
"""

from __future__ import annotations

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 15


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def list_stocks(self) -> list[str]:
        return self._get("/stocks")

    def overview(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/overview")

    def technical(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/technical")

    def history(self, symbol: str, days: int = 180) -> list[dict]:
        return self._get(f"/stocks/{symbol}/history", params={"days": days})

    def fundamentals(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/fundamentals")

    def news(self, symbol: str) -> list[dict]:
        return self._get(f"/stocks/{symbol}/news")

    def forecast(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/forecast")

    def risk(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/risk")

    def explanation(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/explanation")

    def entry_exit(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/entry-exit")

    def backtest(self) -> list[dict]:
        return self._get("/backtest")

    def ask(self, symbol: str, query: str) -> dict:
        return self._post("/ai-analyst/ask", {"symbol": symbol, "query": query})

    def resolve_company(self, identifier: str) -> dict:
        return self._get(f"/company/{identifier}/resolve")

    def snapshot(self, symbol: str) -> dict:
        return self._get(f"/stocks/{symbol}/snapshot")

    def lineage(self, symbol: str) -> list[dict]:
        return self._get(f"/stocks/{symbol}/lineage")

    def analyze(self, company: str, horizon: str = "5d", analysis_type: str = "full") -> dict:
        """Phase 26: the live pipeline for ANY supported company, not just the
        precomputed 50-stock universe the other client methods above read.
        """
        return self._post(
            "/analysis",
            {"company": company, "horizon": horizon, "analysis_type": analysis_type},
        )

    def _get(self, path: str, params: dict | None = None):
        response = requests.get(f"{self.base_url}{path}", params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict):
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
