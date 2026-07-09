"""Offline evaluation harness with MLflow experiment tracking.

Answers the JD's "evaluation" and "document experiments" lines. Runs a small
golden set through the agent, scores answers, and logs params + metrics to MLflow
so you can compare configs (chunk size, top_k, model) as tracked experiments.

Metrics here are simple stand-ins; swap in RAGAS (faithfulness, answer_relevancy,
context_precision) or DeepEval once the pipeline is stable. Keeping the interface
now means the wiring doesn't change when you upgrade the metrics.

Run:  python -m src.eval.harness
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.graph import run as run_agent
from src.common.config import get_settings

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.json"


def _load_golden() -> list[dict]:
    if GOLDEN_SET.exists():
        return json.loads(GOLDEN_SET.read_text())
    # Seed example; replace with questions grounded in YOUR documents.
    return [
        {"query": "What is the refund window?", "must_include": ["30"]},
        {"query": "Which plans include SSO?", "must_include": ["enterprise"]},
    ]


def _score(answer: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    hits = sum(1 for kw in must_include if kw.lower() in answer.lower())
    return hits / len(must_include)


def evaluate() -> dict:
    settings = get_settings()
    golden = _load_golden()

    scores, grounded = [], 0
    for item in golden:
        state = run_agent(settings, item["query"])
        answer = state.get("answer", "")
        scores.append(_score(answer, item.get("must_include", [])))
        grounded += int(state.get("verified", False))

    metrics = {
        "keyword_recall": sum(scores) / len(scores) if scores else 0.0,
        "verified_rate": grounded / len(golden) if golden else 0.0,
        "n_examples": len(golden),
    }

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("agentic-voice-rag")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "llm_model": settings.llm_model,
                    "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "top_k": settings.top_k,
                    "vector_backend": settings.vector_backend,
                }
            )
            mlflow.log_metrics(metrics)
    except Exception as exc:  # MLflow optional for a first run
        print(f"[eval] MLflow logging skipped: {exc}")

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
