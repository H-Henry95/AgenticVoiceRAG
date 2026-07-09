"""FAISS-backed VectorStore — the free/local default.

Uses inner-product search on L2-normalized vectors, which is equivalent to
cosine similarity. Documents are stored alongside the index in a sidecar pickle
so provenance metadata survives restarts.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .base import Document, ScoredDocument, VectorStore


class FaissStore(VectorStore):
    def __init__(self, index_path: str, dim: int) -> None:
        self.index_path = Path(index_path)
        self.dim = dim
        self._index = None  # lazily created faiss index
        self._docs: list[Document] = []

    # -- internals ----------------------------------------------------------
    def _ensure_index(self):
        if self._index is None:
            import faiss  # imported lazily so the module loads without the dep

            self._index = faiss.IndexFlatIP(self.dim)
        return self._index

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # -- interface ----------------------------------------------------------
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        index = self._ensure_index()
        vecs = self._normalize(np.asarray(embeddings, dtype="float32"))
        index.add(vecs)
        self._docs.extend(documents)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredDocument]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = self._normalize(np.asarray([query_embedding], dtype="float32"))
        scores, idxs = self._index.search(q, min(top_k, self._index.ntotal))
        results: list[ScoredDocument] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(ScoredDocument(document=self._docs[idx], score=float(score)))
        return results

    def persist(self) -> None:
        import faiss

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(self.index_path / "faiss.index"))
        with open(self.index_path / "docs.pkl", "wb") as fh:
            pickle.dump(self._docs, fh)
        meta = {"dim": self.dim, "count": len(self._docs)}
        (self.index_path / "meta.json").write_text(json.dumps(meta, indent=2))

    def load(self) -> None:
        import faiss

        index_file = self.index_path / "faiss.index"
        docs_file = self.index_path / "docs.pkl"
        if not index_file.exists():
            return
        self._index = faiss.read_index(str(index_file))
        with open(docs_file, "rb") as fh:
            self._docs = pickle.load(fh)

    def count(self) -> int:
        return len(self._docs)
