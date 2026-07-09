"""Pinecone-backed VectorStore.

Intentionally left as a thin stub: the point of the abstraction is that this file
is the *only* thing that changes when you move from local FAISS to a managed
cloud index. Fill in the four methods when you're ready to demo the hosted path;
nothing else in the codebase needs to change.
"""
from __future__ import annotations

from .base import Document, ScoredDocument, VectorStore


class PineconeStore(VectorStore):
    def __init__(self, api_key: str, index_name: str, dim: int) -> None:
        self.index_name = index_name
        self.dim = dim
        self._client = None
        self._api_key = api_key
        # from pinecone import Pinecone
        # self._client = Pinecone(api_key=api_key)
        # self._index = self._client.Index(index_name)

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        raise NotImplementedError(
            "Pinecone path is a stub. Upsert (id, vector, metadata) tuples here. "
            "Because everything depends only on VectorStore, this is the ONLY file "
            "you touch to switch backends."
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredDocument]:
        raise NotImplementedError("Query the index and map matches -> ScoredDocument.")

    def persist(self) -> None:
        # Managed service persists server-side; nothing to do locally.
        pass

    def load(self) -> None:
        pass

    def count(self) -> int:
        raise NotImplementedError("Return index.describe_index_stats() total.")
