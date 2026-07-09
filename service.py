"""Retrieval as a reusable service.

Deliberately framed as its own component (not buried in the agent) so it can be
exposed as a standalone microservice and reused by the eval harness. This is the
"retrieval microservice" boundary from the JD.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.common.config import Settings
from src.ingestion.embedder import Embedder
from src.vectorstore.base import ScoredDocument
from src.vectorstore.factory import make_vector_store


@dataclass
class RetrievalResult:
    chunks: list[ScoredDocument]

    def as_context(self) -> str:
        """Format retrieved chunks into a prompt-ready, citable context block."""
        blocks = []
        for i, sc in enumerate(self.chunks, start=1):
            src = sc.document.metadata.get("source", "unknown")
            blocks.append(f"[{i}] (source: {src})\n{sc.document.text}")
        return "\n\n".join(blocks)

    def citations(self) -> list[dict]:
        return [
            {"marker": i, "source": sc.document.metadata.get("source"), "score": sc.score}
            for i, sc in enumerate(self.chunks, start=1)
        ]


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model)
        self.store = make_vector_store(settings)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        q_emb = self.embedder.embed_query(query)
        hits = self.store.search(q_emb, top_k or self.settings.top_k)
        return RetrievalResult(chunks=hits)
