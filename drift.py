"""Production monitoring hooks — the "drift, bias, fairness, reliability" line.

Two lightweight, dependency-free checks you can run on live traffic:

1. Embedding drift: compare the mean embedding of recent queries against a
   reference (training/launch) distribution via cosine distance. A rising score
   means users are asking about things your index may not cover well.

2. Output safety: a fast keyword/heuristic gate for unsafe or biased output,
   as a pre-return guardrail. Upgrade to a proper classifier (e.g. a small
   toxicity model, or Detoxify) when you're ready — the call site won't change.

For richer, dashboarded drift and data-quality reports, Evidently is the natural
free upgrade; this module is the always-on, cheap first line.
"""
from __future__ import annotations

import math


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def embedding_drift(reference_mean: list[float], recent_mean: list[float]) -> float:
    """Return drift score in [0, 2]; alert above a threshold you calibrate (~0.15)."""
    return cosine_distance(reference_mean, recent_mean)


_UNSAFE_MARKERS = {"kill", "bomb", "slur_placeholder"}  # replace with a real list/model


def output_is_safe(text: str) -> bool:
    lowered = text.lower()
    return not any(marker in lowered for marker in _UNSAFE_MARKERS)


def check_response(text: str) -> dict:
    """Guardrail to call just before returning an answer to the user."""
    safe = output_is_safe(text)
    return {"safe": safe, "action": "allow" if safe else "block_and_log"}
