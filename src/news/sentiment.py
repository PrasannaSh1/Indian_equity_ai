"""FinBERT-based financial sentiment scoring.

Outputs full positive/neutral/negative probabilities rather than a single
label (project plan Section 13: "Do not reduce everything to a simple
positive/negative label").

IMPORTANT (Windows): `import torch` must happen before pandas/pyarrow are
imported anywhere in the process, or torch's native DLL load fails with
`OSError: [WinError 1114] ... c10.dll` -- pyarrow's bundled Arrow C++ runtime
conflicts with torch's native libraries if pyarrow loads first, and pandas
imports pyarrow eagerly. Any notebook/script calling load_finbert_pipeline()
must `import torch` before its first `import pandas`.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

MODEL_NAME = "ProsusAI/finbert"


def load_finbert_pipeline() -> Callable[[list[str]], list]:
    """Loads the FinBERT text-classification pipeline (downloads weights on first use).

    Imports torch directly before transformers: on this Windows environment,
    letting transformers' lazy-import machinery trigger torch's DLL loading
    indirectly is flaky (intermittent WinError 1114 from torch's c10.dll),
    while a direct top-level `import torch` is consistently reliable.
    """
    import torch  # noqa: F401
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)


def score_sentiment(texts: list[str], classify_fn: Callable[[list[str]], list]) -> pd.DataFrame:
    """Scores each text, returning positive/neutral/negative probabilities per row.

    `classify_fn` is injectable (defaults to a loaded FinBERT pipeline in
    production use) so unit tests can substitute a stub instead of loading the
    actual ~400MB model.
    """
    if not texts:
        return pd.DataFrame(columns=["positive_probability", "neutral_probability", "negative_probability"])

    raw_results = classify_fn(list(texts))
    rows = []
    for result in raw_results:
        scores = {item["label"].lower(): item["score"] for item in result}
        rows.append(
            {
                "positive_probability": scores.get("positive", 0.0),
                "neutral_probability": scores.get("neutral", 0.0),
                "negative_probability": scores.get("negative", 0.0),
            }
        )
    return pd.DataFrame(rows)


def add_sentiment(df: pd.DataFrame, classify_fn: Callable[[list[str]], list]) -> pd.DataFrame:
    """Adds positive/neutral/negative probability columns, scored on headline + summary."""
    df = df.copy()
    text = (df["headline"].fillna("") + ". " + df["summary"].fillna("")).tolist()
    sentiment = score_sentiment(text, classify_fn)
    return pd.concat([df.reset_index(drop=True), sentiment], axis=1)
