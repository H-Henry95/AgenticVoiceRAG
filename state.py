"""Shared state that flows through the LangGraph nodes."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str                 # user's question
    route: str                 # "retrieve" | "tool" | "direct"
    context: str               # retrieved, citable context
    citations: list[dict]      # provenance for the answer
    tool_result: Any           # output from an MCP tool, if used
    draft_answer: str          # answer before verification
    verified: bool             # did the verification node approve it?
    answer: str                # final answer returned to the user
    trace: list[str]           # human-readable step log for debugging/demo
