"""Client the agent uses to discover and call tools from the MCP server.

This closes the loop: your LangGraph `call_tool` node uses this to invoke tools
over MCP, so the agent is a genuine MCP *client* talking to your MCP *server* —
not a hardcoded function call. Phase 3 work; kept minimal so the graph stays
runnable before it's wired in.
"""
from __future__ import annotations

import sys
from typing import Any


async def list_and_call(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Spawn the local MCP server over stdio, call one tool, return its result."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp_server.server"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # tools = await session.list_tools()  # discovery for the router
            result = await session.call_tool(tool_name, arguments)
            return result.content
