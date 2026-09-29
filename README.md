# Indian Equity AI Intelligence Platform

AI/ML-powered research and decision-support platform for the Indian stock market. Combines market data, technical indicators, fundamentals, financial news/sentiment, corporate events, and macroeconomic data to produce probabilistic, explainable equity insights.

This is a **research and decision-support tool**, not a guaranteed prediction or autonomous trading system.

See:

- `Indian_Equity_AI_Project_Plan.md` — full project vision, architecture, and rationale
- `PROJECT_PLAN.md` — phased execution plan (current progress tracker)

## Setup

```bash
# Activate the existing venv (Windows)
venv\Scripts\activate

pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Data licensing & commercial-use notes

This project is currently for **educational/research use only**. Before any commercial or
production deployment, review:

- **Price/fundamentals/news data (`yfinance` → Yahoo Finance):** relayed, not licensed for
  redistribution. Yahoo's terms generally prohibit commercial redistribution of their data
  without a separate license. None of this project's price/fundamentals/news documents are
  tagged as an official Tier 1 source (see `src/rag/documents.py`) for exactly this reason —
  everything is Yahoo-relayed (Tier 3) or this project's own derived model output.
- **Direct NSE/BSE/SEBI data:** not currently used. Official Bhavcopy/filings/regulatory data
  would need separate data-sharing/licensing agreements with the exchange or regulator before
  commercial use (see `Indian_Equity_AI_Project_Plan.md` Section 5).
- **ML models used (FinBERT, sentence-transformers/all-MiniLM-L6-v2):** open-weight models with
  permissive licenses at the time of writing; verify current license terms before commercial
  redistribution or deployment.
- **Rate limits:** the ingestion code (`src/ingestion`, `src/fundamentals`, `src/news`) makes
  sequential, unauthenticated requests to Yahoo Finance. It has no built-in backoff/rate-limit
  handling — scaling well beyond ~50 stocks or running frequently would need that added.

## Platform positioning

Per `Indian_Equity_AI_Project_Plan.md` Section 42, this platform should be described as an
**AI-powered Indian Equity Research and Decision-Support Platform**, not an autonomous trading
system. Every model output in this codebase (probability, price range, risk label,
entry/exit zone, AI Analyst answer) is a model-generated estimate, not a guaranteed prediction
or investment advice — see the disclaimer surfaced in the dashboard (`app/dashboard.py`,
`DISCLAIMER`) and the AI Analyst's answer composer (`src/rag/analyst.py`, `DISCLAIMER`).

If this system is ever extended to give personalized investment advice or operate in a
SEBI-regulated context, obtain qualified legal/regulatory advice first — this project does not
constitute that advice.
