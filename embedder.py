"""Local embeddings via sentence-transformers — free, offline, no API key.

The model is loaded once and cached. BGE models expect a short instruction prefix
on the *query* side for best retrieval quality; we apply it in `embed_query`.
"""
from __future__ import annotations

from functools import lru_cache

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _load_model(self.model_name)
        return model.encode(texts, normalize_embeddings=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _load_model(self.model_name)
        prefixed = _QUERY_INSTRUCTION + text if "bge" in self.model_name.lower() else text
        return model.encode([prefixed], normalize_embeddings=False)[0].tolist()
