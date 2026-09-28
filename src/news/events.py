"""Rule-based financial event classification.

A transparent keyword classifier is used instead of a trained ML classifier
because there is no labeled event dataset for Indian equity news available in
this project -- consistent with the project plan's "start simple, add
complexity later" principle. A headline can match zero, one, or several event
types; each matched type is a separate row in event-feature tables downstream.
"""

from __future__ import annotations

import pandas as pd

EVENT_KEYWORDS = {
    "earnings": ["earnings", "quarterly results", "net profit", "q1 results", "q2 results",
                 "q3 results", "q4 results", "profit rises", "profit falls", "results announced"],
    "dividend": ["dividend"],
    "acquisition": ["acquire", "acquisition", "acquires", "acquired"],
    "merger": ["merger", "merges with", "merge with"],
    "debt": ["debt", "bond issue", "loan", "borrowing"],
    "management_change": ["ceo", "resigns", "resignation", "appoints", "appointed",
                           "steps down", "new chairman", "new md"],
    "regulatory_issue": ["sebi", "regulatory", "probe", "penalty", "fine", "compliance"],
    "product_launch": ["launch", "launches", "unveils", "unveiled"],
    "contract_win": ["contract", "order win", "wins order", "bags order", "bags contract"],
    "legal_issue": ["lawsuit", "litigation", "court", "legal action"],
    "expansion": ["expansion", "expand", "new plant", "new facility", "capacity expansion"],
    "credit_rating": ["credit rating", "downgrade", "upgrade", "rating action"],
}


def classify_event(text: str) -> list[str]:
    """Returns the list of event types whose keywords appear in `text` (case-insensitive)."""
    lowered = text.lower()
    return [event_type for event_type, keywords in EVENT_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)]


def add_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an `event_types` column (list[str], possibly empty) per news row."""
    df = df.copy()
    text = df["headline"].fillna("") + " " + df["summary"].fillna("")
    df["event_types"] = text.map(classify_event)
    return df
