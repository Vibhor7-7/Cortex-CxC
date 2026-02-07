#!/usr/bin/env python3
"""
Test script for CORTEX MCP Server.

This script simulates MCP tool calls to verify both tools work correctly.
Run this with the backend API running.
"""

import asyncio
import httpx

try:
    from backend.cortex_mcp.config import config
except ImportError:
    from config import config


async def test_search_memory():
    """Test the search_memory endpoint via backend API."""
    print("")
    print("=" * 80)
    print("Testing search_memory tool")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{config.backend_api_url}/api/search/",
                json={"query": "Python", "limit": 3}
            )
            response.raise_for_status()
            data = response.json()

            print("")
            print("[SUCCESS] Search successful!")
            print(f"   Found {len(data.get('results', []))} results")
            print(f"   Search time: {data.get('search_time_ms', 0):.0f}ms")

            results = data.get('results', [])
            if results:
                print("")
                print("Sample result:")
                r = results[0]
                print(f"   - Title: {r.get('title', 'N/A')}")
                print(f"   - ID: {r.get('conversation_id', 'N/A')}")
                print(f"   - Score: {r.get('score', 0.0):.2f}")
                return r.get('conversation_id')
            else:
                print("")
                print("[WARNING] No results found (database might be empty)")
                return None

        except Exception as e:
            print("")
            print(f"[ERROR] Search failed: {e}")
            return None


async def test_fetch_chat(conversation_id: str):
    """Test the fetch_chat endpoint via backend API."""
    print("")
    print("=" * 80)
    print("Testing fetch_chat tool")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{config.backend_api_url}/api/chats/{conversation_id}"
            )
            response.raise_for_status()
            chat = response.json()

            print("")
            print("[SUCCESS] Fetch successful!")
            print(f"   Title: {chat.get('title', 'N/A')}")
            print(f"   Messages: {chat.get('message_count', 0)}")
            print(f"   Created: {chat.get('created_at', 'N/A')}")

            messages = chat.get('messages', [])
            if messages:
                print("")
                print("First message preview:")
                msg = messages[0]
                content = msg.get('content', '')
                preview = content[:100] + "[truncated]" if len(content) > 100 else content
                print(f"   {msg.get('role', 'unknown')}: {preview}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print("")
                print(f"[WARNING] Conversation not found (ID: {conversation_id})")
            else:
                print("")
                print(f"[ERROR] Fetch failed: {e}")
        except Exception as e:
            print("")
            print(f"[ERROR] Fetch failed: {e}")


async def test_backend_health():
    """Test backend health endpoint."""
    print("")
    print("=" * 80)
    print("Testing backend connectivity")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{config.backend_api_url}/health")
            response.raise_for_status()
            health = response.json()

            print("")
            print("[SUCCESS] Backend is healthy!")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Database: {'[OK]' if health.get('database_connected') else '[FAIL]'}")
            print(f"   Ollama: {'[OK]' if health.get('ollama_connected') else '[FAIL]'}")
            print(f"   Vector Store: {'[OK]' if health.get('chroma_ready') else '[FAIL]'}")

            return health.get('database_connected', False)

        except Exception as e:
            print("")
            print(f"[ERROR] Backend is not reachable: {e}")
            print(f"   Make sure the backend is running at {config.backend_api_url}")
            return False


async def main():
    """Run all MCP tool tests."""
    print("")
    print("=" * 80)
    print("CORTEX MCP Server Tool Tests")
    print("=" * 80)
    print("")
    print(f"Backend API URL: {config.backend_api_url}")

    backend_ok = await test_backend_health()
    if not backend_ok:
        print("")
        print("[STOP] Cannot proceed - backend is not available")
        return

    conversation_id = await test_search_memory()

    if conversation_id:
        await test_fetch_chat(conversation_id)
    else:
        print("")
        print("[WARNING] Skipping fetch_chat test (no conversation ID)")

    print("")
    print("=" * 80)
    print("[SUCCESS] MCP Tool Tests Complete!")
    print("=" * 80)
    print("")
    print("Next steps:")
    print("1. Start the MCP server: python -m backend.cortex_mcp.server")
    print("2. Configure Claude Desktop with the MCP server")
    print("3. Test tools in Claude Desktop")
    print("")


if __name__ == "__main__":
    asyncio.run(main())
