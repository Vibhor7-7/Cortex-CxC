#!/usr/bin/env python3
"""
MCP stdio client test for Cortex MCP server.

This script spins up the MCP server via stdio, initializes a session,
lists tools, and exercises search_memory + fetch_chat.
"""

import asyncio
import os
import re
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_conversation_id(text: str) -> str | None:
    match = re.search(r"Conversation ID\*\*:\s*([a-f0-9\-]{8,})", text, re.I)
    if match:
        return match.group(1)
    return None


async def main() -> None:
    query = os.getenv("MCP_TEST_QUERY", "Python")
    limit = int(os.getenv("MCP_TEST_LIMIT", "3"))

    server = StdioServerParameters(
        command=os.getenv("MCP_TEST_COMMAND", "python"),
        args=os.getenv("MCP_TEST_ARGS", "-m backend.cortex_mcp.server").split(),
        cwd=os.getenv("MCP_TEST_CWD", str(_repo_root())),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        session = ClientSession(
            read_stream,
            write_stream,
            client_info=Implementation(
                name="cortex-mcp-test-client",
                version="1.0.0",
            ),
        )

        await session.initialize()

        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        print("[OK] Tools:", ", ".join(tool_names))

        search_result = await session.call_tool(
            "search_memory",
            {"query": query, "limit": limit},
        )

        content_text = "".join(
            c.text for c in (search_result.content or []) if getattr(c, "text", None)
        )

        print("\n[search_memory]\n")
        print(content_text or "(no content)")

        conversation_id = _extract_conversation_id(content_text)
        if not conversation_id:
            print("\n[WARNING] No conversation_id found in search results. Skipping fetch_chat.")
            return

        fetch_result = await session.call_tool(
            "fetch_chat",
            {"conversation_id": conversation_id},
        )

        fetch_text = "".join(
            c.text for c in (fetch_result.content or []) if getattr(c, "text", None)
        )

        print("\n[fetch_chat]\n")
        print(fetch_text or "(no content)")


if __name__ == "__main__":
    asyncio.run(main())
