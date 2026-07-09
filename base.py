"""Vector-store abstraction.

The whole point of this interface is that the rest of the codebase never imports
FAISS, Chroma, or Pinecone directly — it depends only on `VectorStore`. Swapping
backends (e.g. FAISS locally -> Pinecone in the cloud) is a one-line change in
`factory.py` plus credentials, with zero changes to ingestion, retrieval, or the
agent. Being able to explain *why* that decoupling matters is a senior signal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A chunk plus its provenance. Provenance travels with the chunk end-to-end
    so every answer can be traced back to a source (see docs/DESIGN.md)."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDocument:
    document: Document
    score: float


class VectorStore(ABC):
    """Minimal surface every backend must implement."""

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredDocument]:
        ...

    @abstractmethod
    def persist(self) -> None:
        """Flush to disk / remote so the index survives a restart."""

    @abstractmethod
    def load(self) -> None:
        """Restore a previously persisted index. No-op for managed services."""

    @abstractmethod
    def count(self) -> int:
        ...
