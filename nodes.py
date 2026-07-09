"""The individual reasoning steps of the agent.

Each node takes state and returns a partial state update. Keeping them small and
pure makes them independently testable and easy to trace in Langfuse. The
verification node is what turns "a RAG call" into "an agent": it checks the draft
answer against retrieved context before anything is returned to the user.
"""
from __future__ import annotations

from src.common.config import Settings
from src.common.llm import LLMClient
from src.retrieval.service import RetrievalService

from .state import AgentState

_ROUTER_SYSTEM = (
    "You are a router. Classify the user request into exactly one word: "
    "'retrieve' if it needs knowledge-base lookup, 'tool' if it needs an action "
    "or live calculation, or 'direct' if it is chit-chat. Reply with only the word."
)

_ANSWER_SYSTEM = (
    "You answer strictly from the provided context. Cite sources with the bracket "
    "markers shown, e.g. [1]. If the context does not contain the answer, say you "
    "don't know. Do not invent facts."
)

_VERIFY_SYSTEM = (
    "You are a strict fact-checker. Given CONTEXT and an ANSWER, reply 'PASS' if "
    "every claim in the ANSWER is supported by the CONTEXT, otherwise reply 'FAIL' "
    "followed by a one-line reason."
)


class Nodes:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.retrieval = RetrievalService(settings)

    def route(self, state: AgentState) -> AgentState:
        decision = self.llm.complete(_ROUTER_SYSTEM, state["query"]).strip().lower()
        route = next((r for r in ("retrieve", "tool", "direct") if r in decision), "retrieve")
        return {"route": route, "trace": state.get("trace", []) + [f"route={route}"]}

    def retrieve(self, state: AgentState) -> AgentState:
        result = self.retrieval.retrieve(state["query"])
        return {
            "context": result.as_context(),
            "citations": result.citations(),
            "trace": state.get("trace", []) + [f"retrieved {len(result.chunks)} chunks"],
        }

    def call_tool(self, state: AgentState) -> AgentState:
        # Placeholder: wire this to the MCP client in src/agent/mcp_client.py.
        # Kept explicit so the graph is complete and runnable end to end.
        return {
            "tool_result": None,
            "trace": state.get("trace", []) + ["tool node (connect MCP client here)"],
        }

    def generate(self, state: AgentState) -> AgentState:
        context = state.get("context", "") or "(no context retrieved)"
        user = f"CONTEXT:\n{context}\n\nQUESTION: {state['query']}"
        draft = self.llm.complete(_ANSWER_SYSTEM, user)
        return {"draft_answer": draft, "trace": state.get("trace", []) + ["generated draft"]}

    def verify(self, state: AgentState) -> AgentState:
        context = state.get("context", "")
        if not context:
            # Nothing to verify against (e.g. direct chit-chat): pass through.
            return {"answer": state.get("draft_answer", ""), "verified": True}
        user = f"CONTEXT:\n{context}\n\nANSWER:\n{state.get('draft_answer', '')}"
        verdict = self.llm.complete(_VERIFY_SYSTEM, user).strip()
        passed = verdict.upper().startswith("PASS")
        answer = state.get("draft_answer", "") if passed else (
            "I couldn't confirm an answer grounded in the sources, so I'd rather not guess."
        )
        return {
            "verified": passed,
            "answer": answer,
            "trace": state.get("trace", []) + [f"verify={'PASS' if passed else 'FAIL'}"],
        }
