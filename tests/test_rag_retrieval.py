import numpy as np
import pytest

from src.rag.retrieval import VectorIndex

# A small controllable fake embedding space: each known text maps to a fixed
# 2D unit vector, so similarity is fully deterministic and hand-verifiable.
TEXT_TO_VEC = {
    "query": np.array([1.0, 0.0]),
    "doc_same_direction": np.array([1.0, 0.0]),  # identical to query -> similarity 1.0
    "doc_orthogonal": np.array([0.0, 1.0]),  # orthogonal to query -> similarity 0.0
    "doc_opposite": np.array([-1.0, 0.0]),  # opposite -> similarity -1.0
}


def fake_embed(texts):
    return np.array([TEXT_TO_VEC[t] for t in texts])


def _doc(doc_id, symbol, text, tier):
    return {"doc_id": doc_id, "symbol": symbol, "category": "test", "text": text, "source": "test", "tier": tier}


def test_search_ranks_by_similarity_when_tiers_are_equal():
    index = VectorIndex(fake_embed)
    index.build(
        [
            _doc("a", "X", "doc_orthogonal", tier=3),
            _doc("b", "X", "doc_same_direction", tier=3),
            _doc("c", "X", "doc_opposite", tier=3),
        ]
    )
    results = index.search("query", k=3)

    assert [r["doc_id"] for r in results] == ["b", "a", "c"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_applies_tier_trust_weighting_when_similarity_ties():
    index = VectorIndex(fake_embed)
    index.build(
        [
            _doc("low_tier", "X", "doc_same_direction", tier=4),  # trust weight 0.6
            _doc("high_tier", "X", "doc_same_direction", tier=1),  # trust weight 1.0
        ]
    )
    results = index.search("query", k=2)

    # identical similarity (both "doc_same_direction") -> ranking must come from tier trust
    assert results[0]["doc_id"] == "high_tier"
    assert results[0]["score"] > results[1]["score"]


def test_search_filters_by_symbol():
    index = VectorIndex(fake_embed)
    index.build(
        [
            _doc("a", "X", "doc_same_direction", tier=3),
            _doc("b", "Y", "doc_same_direction", tier=3),
        ]
    )
    results = index.search("query", k=5, symbol="Y")

    assert [r["doc_id"] for r in results] == ["b"]


def test_search_returns_empty_list_when_no_documents_match_symbol():
    index = VectorIndex(fake_embed)
    index.build([_doc("a", "X", "doc_same_direction", tier=3)])

    assert index.search("query", k=5, symbol="NONEXISTENT") == []
