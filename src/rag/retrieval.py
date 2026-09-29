"""In-memory vector index for the RAG document corpus.

A full deployment would use pgvector (see project plan Section 36's tech
stack) once the Postgres-backed application exists (Phase 13); this in-memory
index is a deliberate, documented simplification so Phase 11 doesn't need to
stand up a database first, and is straightforward to swap out later.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TIER_TRUST_WEIGHTS = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.6, "model_output": 0.85}

EmbedFn = Callable[[list[str]], np.ndarray]


def load_sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> EmbedFn:
    """Loads a sentence-transformers model as an embed_fn.

    IMPORTANT (Windows): import torch before pandas/pyarrow anywhere in the
    process, or torch's native DLL load can fail -- see src/news/sentiment.py
    for the full explanation of this project's earlier finding.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


class VectorIndex:
    def __init__(self, embed_fn: EmbedFn):
        self._embed_fn = embed_fn
        self.documents: list[dict] = []
        self._embeddings: np.ndarray | None = None

    def build(self, documents: list[dict]) -> None:
        self.documents = documents
        if not documents:
            self._embeddings = np.zeros((0, 0))
            return
        raw = self._embed_fn([d["text"] for d in documents])
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = raw / norms  # normalize once here so callers' embed_fn need not

    def search(self, query: str, k: int = 5, symbol: str | None = None) -> list[dict]:
        """Returns the top-k documents by tier-weighted cosine similarity.
        If `symbol` is given, only documents tagged with that symbol are considered.
        """
        candidate_idx = [
            i for i, d in enumerate(self.documents) if symbol is None or d["symbol"] == symbol
        ]
        if not candidate_idx:
            return []

        query_emb = self._embed_fn([query])[0]
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1.0)

        candidate_embeddings = self._embeddings[candidate_idx]
        similarity = candidate_embeddings @ query_emb

        results = []
        for local_i, doc_i in enumerate(candidate_idx):
            doc = self.documents[doc_i]
            trust = TIER_TRUST_WEIGHTS.get(doc["tier"], 0.7)
            results.append({**doc, "similarity": float(similarity[local_i]), "score": float(similarity[local_i] * trust)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]
