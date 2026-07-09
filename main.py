"""The orchestrator microservice: HTTP front door to the agent.

Endpoints:
    GET  /health        liveness + index size
    POST /query         run the agent graph on a question
    POST /ingest        (dev convenience) re-run ingestion

Run:  uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.graph import run as run_agent
from src.common.config import get_settings
from src.ingestion.pipeline import ingest
from src.vectorstore.factory import make_vector_store

app = FastAPI(title="Agentic Voice RAG", version="0.1.0")


class QueryIn(BaseModel):
    query: str


class QueryOut(BaseModel):
    answer: str
    citations: list[dict] = []
    verified: bool = False
    trace: list[str] = []


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    store = make_vector_store(settings)
    return {"status": "ok", "indexed_chunks": store.count(), "backend": settings.vector_backend}


@app.post("/query", response_model=QueryOut)
def query(body: QueryIn) -> QueryOut:
    settings = get_settings()
    state = run_agent(settings, body.query)
    return QueryOut(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        verified=state.get("verified", False),
        trace=state.get("trace", []),
    )


@app.post("/ingest")
def ingest_endpoint() -> dict:
    settings = get_settings()
    n = ingest(settings)
    return {"ingested_chunks": n}
