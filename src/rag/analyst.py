"""The AI Analyst: composes grounded, sourced answers from retrieved evidence.

Deliberately template-based, not a generative LLM call (see PROJECT_PLAN.md
Phase 11 notes for why): the composer can only restate retrieved document text
plus its source/tier metadata, so it cannot invent facts or predictions that
aren't already in the retrieved evidence -- "explain quantitative outputs and
retrieved evidence rather than invent predictions" (project plan Section 32)
is satisfied by construction, not by prompting a model to behave.
"""

from __future__ import annotations

from src.rag.retrieval import VectorIndex

DISCLAIMER = (
    "This is a research/decision-support summary generated from this project's own "
    "data and models, not investment advice or a guaranteed prediction."
)


def answer_query(query: str, symbol: str, index: VectorIndex, k: int = 5) -> dict:
    """Retrieves the top-k pieces of evidence for `symbol` relevant to `query`,
    and composes a cited, plain-text answer restating only that evidence.
    """
    evidence = index.search(query, k=k, symbol=symbol)

    if not evidence:
        return {
            "answer": f"No evidence is available for {symbol} yet to answer this question.",
            "sources": [],
        }

    lines = [f"Based on the retrieved evidence for {symbol}:"]
    sources = []
    for i, doc in enumerate(evidence, start=1):
        lines.append(f"{i}. {doc['text']} (Source: {doc['source']}, {doc['tier_label']})")
        sources.append(
            {
                "doc_id": doc["doc_id"],
                "category": doc["category"],
                "source": doc["source"],
                "tier_label": doc["tier_label"],
                "date": doc.get("date"),
            }
        )
    lines.append("")
    lines.append(DISCLAIMER)

    return {"answer": "\n".join(lines), "sources": sources}
