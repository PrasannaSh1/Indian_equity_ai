import numpy as np

from src.rag.analyst import DISCLAIMER, answer_query
from src.rag.retrieval import VectorIndex

TEXT_TO_VEC = {"why is X bullish": np.array([1.0, 0.0]), "doc_text": np.array([1.0, 0.0])}


def fake_embed(texts):
    return np.array([TEXT_TO_VEC[t] for t in texts])


def _doc(doc_id, symbol, text, tier, tier_label, source="Test Source"):
    return {
        "doc_id": doc_id,
        "symbol": symbol,
        "category": "test",
        "text": text,
        "source": source,
        "tier": tier,
        "tier_label": tier_label,
    }


def test_answer_query_only_restates_retrieved_evidence_text():
    index = VectorIndex(fake_embed)
    index.build([_doc("d1", "RELIANCE", "doc_text", tier=3, tier_label="Tier 3 (financial website)")])

    result = answer_query("why is X bullish", "RELIANCE", index, k=5)

    assert "doc_text" in result["answer"]  # the evidence text appears verbatim
    assert "Test Source" in result["answer"]  # its source is cited
    assert "Tier 3" in result["answer"]  # its tier is cited
    assert DISCLAIMER in result["answer"]
    assert result["sources"] == [
        {"doc_id": "d1", "category": "test", "source": "Test Source", "tier_label": "Tier 3 (financial website)", "date": None}
    ]


def test_answer_query_handles_no_evidence_gracefully():
    index = VectorIndex(fake_embed)
    index.build([_doc("d1", "OTHER_SYMBOL", "doc_text", tier=3, tier_label="Tier 3")])

    result = answer_query("why is X bullish", "RELIANCE", index, k=5)

    assert result["sources"] == []
    assert "No evidence" in result["answer"]


def test_answer_query_cites_every_returned_source():
    index = VectorIndex(fake_embed)
    index.build(
        [
            _doc("d1", "RELIANCE", "doc_text", tier=3, tier_label="Tier 3", source="Source One"),
            _doc("d2", "RELIANCE", "doc_text", tier=1, tier_label="Tier 1", source="Source Two"),
        ]
    )

    result = answer_query("why is X bullish", "RELIANCE", index, k=2)

    assert len(result["sources"]) == 2
    assert "Source One" in result["answer"]
    assert "Source Two" in result["answer"]
