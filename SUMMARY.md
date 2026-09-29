# Indian Equity AI Platform — Project Summary

> **This is the single living summary of the whole project.** Update it whenever a phase
> changes, a new finding emerges, or a bug is found/fixed — it should always reflect current
> reality, not a point-in-time snapshot. For full phase-by-phase detail (checklists, DoD text,
> exact file lists), see `PROJECT_PLAN.md`. For the original vision/spec, see
> `Indian_Equity_AI_Project_Plan.md`.

**What this is:** an AI/ML research and decision-support platform for Indian equities —
price/technical/fundamental/news/macro data feeding probabilistic, explainable predictions,
served through a FastAPI backend and Streamlit dashboard with an AI Analyst.
**What this is not:** a guaranteed-returns system or investment advice.

**Status:** All 14 planned phases complete. Universe: 50 stocks (NIFTY 50), configured in
`src/config.py`. Test suite: 155/155 passing.

---

## The one finding that matters most

Every phase that tested for predictive edge (Phases 5, 7, 8, 9, 12) found next-day/short-term
direction from price, technical, fundamental, news, and macro features sits at **~0.50 ROC-AUC
— essentially no edge over a coin flip.** Phase 9's realistic backtest (transaction costs +
slippage) showed a strategy built on this signal **loses money outright** (‑64.9% net of cost
over the test window, vs ‑7.8% for simply holding NIFTY 50). This is reported consistently and
plainly throughout, not reframed as a partial success — it is itself a legitimate research
result (consistent with weak-form market efficiency), and the honest baseline this platform's
outputs should be read against.

One partial exception: Phase 12's 60-day long-term horizon model scored ROC-AUC 0.570 on the
50-stock universe (vs 0.453 on 5) — a single train/val/test split, **not yet walk-forward
validated**. Worth re-testing properly (Phase 9-style) before treating it as a real signal.

---

## Phase-by-phase

| # | Phase | Built | Key result |
|---|---|---|---|
| 0 | Environment & Repo Setup | Folder structure, `requirements.txt`, git repo, smoke test | Working env, FastAPI+Streamlit stack chosen |
| 1 | Historical Market Data | `src/ingestion/market_data.py` | 5→50 stocks, 5yr daily OHLCV, zero data-quality issues |
| 2 | Exploratory Data Analysis | `src/features/risk.py` | Returns, volatility, drawdown, beta, correlation vs NIFTY 50 |
| 3 | Technical Analysis Engine | `src/technical/*` | SMA/EMA/RSI/MACD/ATR/Bollinger/ADX/OBV/VWAP + 0–100 Technical Score, each hand-verified |
| 4 | Fundamental Analysis Engine | `src/fundamentals/*` | ROE/ROCE/Debt-Equity/growth/valuation + 0–100 Fundamental Score |
| 5 | Baseline ML Models | `src/models/{dataset,baseline,evaluate}.py` | LogReg/RF/XGBoost/LightGBM vs naive; ROC-AUC ~0.50 (no leakage, no edge) |
| 6 | News Intelligence & Sentiment | `src/news/*` | FinBERT sentiment + event classification on real Yahoo news |
| 7 | Multimodal ML Model | `src/macro/*`, `src/regime/*`, stacking ensemble | Ensemble "wins" on 1 split, **loses** under walk-forward — reported honestly |
| 8 | Volatility/Risk Engine | `src/risk/{volatility_target,quantiles,var_es,engine}.py` | Every prediction ships probability + price range + risk label |
| 9 | Backtesting & Validation | `src/backtesting/{engine,metrics}.py` | H6: no edge survives realistic costs (net ‑64.9% vs benchmark ‑7.8%) |
| 10 | Explainable AI | `src/explainability/{shap_explainer,confidence}.py` | SHAP factor breakdown + confidence per prediction |
| 11 | LLM + RAG Layer | `src/rag/{documents,retrieval,analyst}.py` | Grounded, cited AI Analyst (template composer, not generative — see below) |
| 12 | Entry/Exit & Horizons | `src/models/horizons.py`, `src/entryexit/engine.py` | Swing (5d) + long-term (60d) horizons; rule-based entry/stop/target zones |
| 13 | Application & Dashboard | `src/db/*`, `src/api/main.py`, `app/*` | FastAPI + SQLite + Streamlit (9 tabs), verified live in browser |
| 14 | Scale-Out | `src/config.py` | 5→50 stocks (NIFTY 50), full pipeline re-run, zero code drift |

---

## Key architectural decisions (asked of the user, not assumed)

- **LLM (Phase 11):** no API key configured → the "AI Analyst" is a **deterministic template
  composer**, not a generative model call. It can only restate retrieved, cited evidence, so it
  structurally cannot invent predictions. `answer_query()`'s interface is decoupled from *how*
  answers are composed, so a real LLM can be swapped in later behind a strict "cite only
  retrieved evidence" prompt.
- **Database (Phase 13):** PostgreSQL was installed but not running as a service → persistence
  is **SQLite via SQLAlchemy Core** instead. Same schema; portable to Postgres later by changing
  the connection URL only.
- **Scale-out scope (Phase 14):** the user chose a **full re-run of every phase's notebook**
  on the 50-stock universe, not just a config-only proof.

---

## Known data-source limitations (honest, not hidden)

- **Nothing is Tier-1 sourced.** All price/fundamentals/news data is Yahoo-Finance-relayed
  (Tier 3), not fetched directly from NSE/BSE/SEBI. This project's own model outputs are a
  distinct "internal model output" category, never conflated with an independent source. See
  `src/rag/documents.py` and `README.md`'s licensing section.
- **News coverage is thin.** Yahoo's free news feed is a small recent rolling window
  (~10 items/ticker), not a historical archive. During the Phase 14 scale-out, Yahoo's news
  endpoint had a **confirmed external outage** (HTTP 500 for every symbol, including AAPL/MSFT)
  — handled gracefully (0 articles, no crash), not faked.
- **Sector-index correlation is incomplete.** `^CNXFMCG`/`^CNXENERGY` (ITC/RELIANCE's sectors)
  return near-empty history on Yahoo; skipped with a note (Phase 2).
- **Fundamentals coverage varies.** Bank statements (HDFCBANK) don't report EBIT, so
  ROCE/current-ratio/interest-coverage are `NaN` for banks by design (Phase 4). At 50-stock
  scale, 47/50 got a full Fundamental Score; 3 didn't (reported via warning, not a crash).
- **Quantile forecast is under-calibrated.** Phase 8's 10–90 price range showed ~64% empirical
  coverage against an ~80% target — read as relative-uncertainty guidance, not a literal
  confidence interval, until recalibrated.
- **Intraday is out of scope.** This project only has daily EOD data; a genuine intraday model
  (Section 26) would need a different, likely paid, real-time data source (Phase 12).

---

## Real bugs found and fixed along the way

Each of these was caught by actually running the code/app (not just writing it), consistent with
this project's practice of verifying before declaring a phase done.

1. **`yfinance==0.2.50` auth handshake broken** → upgraded to `1.7.0` (Phase 1).
2. **`beta()` duplicate-column bug**: two Series with the same name collapsed into a DataFrame
   under `pd.concat`, breaking indexing (Phase 2).
3. **`technical_score` NaN-comparison bug**: `sma_50 > NaN` evaluates `False` in pandas, not
   `NaN` — silently faked a "downtrend" score during warm-up instead of `NaN` (Phase 3).
4. **Windows torch/pyarrow DLL conflict**: `import torch` must happen before
   pandas/pyarrow anywhere in the process, or native DLL loading fails (`WinError 1114`).
   Originally fixed per-file; later found to be **process-wide**, not per-file — pytest runs all
   test modules in one process, so a root `tests/conftest.py` was needed (Phases 6, 11, 13).
5. **`FUNDAMENTAL_FEATURE_COLUMNS` silently dropped HDFCBANK** (100%) via a `dropna` on
   bank-incompatible columns (Phase 7).
6. **Risk label was 100% "Medium"**: absolute scoring caps were calibrated for a more volatile
   universe than 5 blue-chips. Fixed with relative (train-tertile) labeling (Phase 8).
7. **`explainability_predictions.parquet` / `explainability_shap_values.parquet` index
   mismatch**: saved with different (unaligned) indices, would silently misalign every row on
   reload (found while building Phase 11).
8. **`event_types or "none detected"` printed literal "nan"**: pandas `NaN` is truthy in Python,
   so `or` never falls back (Phase 11).
9. **SQLite in-memory test isolation**: FastAPI's threadpool + plain `sqlite:///:memory:` gave
   each connection its own empty DB. Fixed with `StaticPool` (Phase 13).
10. **NaN breaks JSON serialization**: Starlette disallows `NaN`; first fix attempt
    (`df.where(cond, None)`) silently failed because `None` in a `float64` column coerces back
    to `NaN`. Real fix sanitizes on plain dicts after `to_dict()` (Phase 13).
11. **`resolve_entity_mentions` column-loss bug**: an empty `mentions_company` column inferred
    as `float64`, and boolean-indexing with a non-bool empty mask dropped every column (Phase 14).
12. **F-string `\n` escaping bug**: a literal newline arrived via a bash heredoc where an escape
    sequence was intended, causing a `SyntaxError` (Phase 14).
13. **Phase 4's hard assert too brittle for 50 stocks**: would abort the whole run over one
    stock's data gap; changed to per-stock `try/except` + visible warning (Phase 14).

---

## Tech stack

Python, pandas/numpy/pyarrow, scikit-learn/XGBoost/LightGBM, SHAP, PyTorch + Transformers
(FinBERT) + sentence-transformers (RAG embeddings), FastAPI + SQLAlchemy (SQLite) + Streamlit +
Plotly, pytest. Full list in `requirements.txt`.

## Where things live

- **Research notebooks:** `notebooks/01` through `notebooks/12`, one per phase — each is the
  authoritative, executed record of that phase's results.
- **Reusable pipeline code:** `src/` (mirrors the phase structure: `ingestion`, `features`,
  `technical`, `fundamentals`, `models`, `news`, `macro`, `regime`, `risk`, `backtesting`,
  `explainability`, `rag`, `entryexit`, `db`, `api`).
- **App:** `src/api/main.py` (FastAPI) + `app/dashboard.py` (Streamlit) + `app/{api_client,charts}.py`.
- **Tests:** `tests/`, 155 passing — hand-computed or analytically-derived reference values
  throughout, not just "does it run" checks.
- **Data:** `data/processed/*` (parquet/CSV, regenerated by the notebooks) and `data/app.db`
  (SQLite, regenerated by `python -m src.db.load_data`) — both gitignored, not committed.

## Running it

```bash
venv\Scripts\activate
pip install -r requirements.txt
pytest                                    # 155 tests
python -m src.db.load_data                # populate data/app.db from data/processed/
uvicorn src.api.main:app --reload         # API on :8000
streamlit run app/dashboard.py            # dashboard on :8501
```
