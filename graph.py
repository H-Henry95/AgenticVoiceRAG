"""Assemble the nodes into a LangGraph state machine.

Flow:
    route ─┬─(retrieve)→ retrieve → generate → verify → END
           ├─(tool)────→ call_tool → generate → verify → END
           └─(direct)──→ generate → verify → END

The conditional edge after `route` is what makes this agentic rather than a fixed
pipeline: the graph decides at runtime which path to take.
"""
from __future__ import annotations

from src.common.config import Settings

from .nodes import Nodes
from .state import AgentState


def build_graph(settings: Settings):
    from langgraph.graph import END, START, StateGraph

    nodes = Nodes(settings)
    g = StateGraph(AgentState)

    g.add_node("route", nodes.route)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("call_tool", nodes.call_tool)
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {"retrieve": "retrieve", "tool": "call_tool", "direct": "generate"},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("call_tool", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)

    return g.compile()


def run(settings: Settings, query: str) -> AgentState:
    graph = build_graph(settings)
    return graph.invoke({"query": query, "trace": []})
