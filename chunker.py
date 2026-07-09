"""Simple, dependency-free character chunker with overlap.

Start here. When you want better boundaries, swap in a sentence- or token-aware
splitter (LangChain's RecursiveCharacterTextSplitter is a drop-in). Chunking
strategy is one of the biggest levers on RAG quality — track it as an experiment
parameter in MLflow so you can compare configs (see src/eval/harness.py).
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks
