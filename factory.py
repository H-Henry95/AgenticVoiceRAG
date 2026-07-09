"""Backend selection lives in exactly one place."""
from __future__ import annotations

from src.common.config import Settings

from .base import VectorStore


def make_vector_store(settings: Settings) -> VectorStore:
    backend = settings.vector_backend.lower()

    if backend == "faiss":
        from .faiss_store import FaissStore

        store = FaissStore(settings.vector_index_path, settings.embedding_dim)
        store.load()
        return store

    if backend == "pinecone":
        from .pinecone_store import PineconeStore

        return PineconeStore(
            settings.pinecone_api_key, settings.pinecone_index, settings.embedding_dim
        )

    raise ValueError(f"Unknown vector_backend: {settings.vector_backend!r}")
