"""A real MCP server — the rarest, most differentiating skill in the JD.

Exposes three tools over the Model Context Protocol using the official Python SDK
(FastMCP). Any MCP-compatible client (your agent, Claude Desktop, etc.) can
discover and call them. Backed by a local SQLite DB so it runs fully offline.

Run it:  python -m src.mcp_server.server
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"

mcp = FastMCP("agentic-voice-rag-tools")


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tickets ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, body TEXT, status TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS accounts ("
        "id TEXT PRIMARY KEY, name TEXT, plan TEXT, seats INTEGER)"
    )
    conn.commit()
    return conn


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3200 * 12 * 0.85'.

    Uses a restricted evaluator (digits and operators only) — never eval user text
    directly in production; this keeps the surface tiny and safe.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - guarded above
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"


@mcp.tool()
def lookup_account(account_id: str) -> dict:
    """Look up an account by id. Returns plan and seat count for support answers."""
    conn = _db()
    row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    if row is None:
        return {"found": False, "account_id": account_id}
    return {"found": True, **dict(row)}


@mcp.tool()
def create_ticket(title: str, body: str) -> dict:
    """Create a support ticket. This is a *write* action — exactly the kind of
    side-effecting tool an agent should call explicitly rather than hallucinate."""
    conn = _db()
    cur = conn.execute(
        "INSERT INTO tickets (title, body, status) VALUES (?, ?, 'open')", (title, body)
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return {"ticket_id": ticket_id, "status": "open"}


if __name__ == "__main__":
    # Default transport is stdio, which is what MCP clients expect.
    mcp.run()
