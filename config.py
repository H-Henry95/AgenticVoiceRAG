"""Central configuration.

Everything is driven by environment variables (see .env.example) so the exact
same code runs fully-local for development and against hosted APIs for a
shareable demo. That local-vs-hosted switch is a deliberate design decision —
see docs/DESIGN.md, "Cost & deployment".
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    # "ollama" (local) | "openai_compatible" (Groq, local vLLM, etc.)
    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="llama3.1:8b")
    # Base URL for an OpenAI-compatible endpoint. Ollama serves one at :11434/v1.
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_api_key: str = Field(default="not-needed-for-local")

    # --- Embeddings (local, free) -----------------------------------------
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384)

    # --- Vector store ------------------------------------------------------
    # "faiss" | "chroma" | "pinecone" — see vectorstore/factory.py
    vector_backend: str = Field(default="faiss")
    vector_index_path: str = Field(default=str(DATA_DIR / "index"))
    pinecone_api_key: str = Field(default="")
    pinecone_index: str = Field(default="agentic-voice-rag")

    # --- Retrieval ---------------------------------------------------------
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)
    top_k: int = Field(default=5)

    # --- Observability -----------------------------------------------------
    mlflow_tracking_uri: str = Field(default=str(REPO_ROOT / "mlruns"))
    langfuse_host: str = Field(default="")  # empty = tracing disabled
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")


@lru_cache
def get_settings() -> Settings:
    return Settings()
