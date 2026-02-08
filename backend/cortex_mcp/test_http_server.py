#!/usr/bin/env python3
"""
Test script for the HTTP/SSE MCP Server (ChatGPT integration).

This script tests the MCP endpoints that ChatGPT will use.
"""

import asyncio
import httpx
import json


MCP_SERVER_URL = "http://localhost:8001"


async def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MCP_SERVER_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200


async def test_mcp_initialize():
    """Test MCP initialize method."""
    print("\n" + "=" * 60)
    print("Testing MCP initialize")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MCP_SERVER_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {}
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200


async def test_mcp_list_tools():
    """Test MCP tools/list method."""
    print("\n" + "=" * 60)
    print("Testing MCP tools/list")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MCP_SERVER_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tools/list",
                "params": {}
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        tools = data.get("result", {}).get("tools", [])
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description'][:60]}...")
        return response.status_code == 200


async def test_mcp_search_memory():
    """Test MCP tools/call for search_memory."""
    print("\n" + "=" * 60)
    print("Testing MCP tools/call (search_memory)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{MCP_SERVER_URL}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "3",
                "method": "tools/call",
                "params": {
                    "name": "search_memory",
                    "arguments": {
                        "query": "Python programming",
                        "limit": 3
                    }
                }
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        if "result" in data:
            content = data["result"].get("content", [])
            if content:
                text = content[0].get("text", "")
                print(f"Response preview:\n{text[:500]}...")
        else:
            print(f"Error: {data.get('error')}")
        return response.status_code == 200


async def test_sse_endpoint():
    """Test SSE endpoint."""
    print("\n" + "=" * 60)
    print("Testing SSE POST endpoint")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{MCP_SERVER_URL}/sse",
            json={
                "jsonrpc": "2.0",
                "id": "4",
                "method": "tools/list",
                "params": {}
            }
        )
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        # Parse SSE response
        text = response.text
        if text.startswith("data: "):
            data = json.loads(text[6:].strip())
            tools = data.get("result", {}).get("tools", [])
            print(f"Found {len(tools)} tools via SSE")
        return response.status_code == 200


async def main():
    print("=" * 60)
    print("Cortex MCP HTTP/SSE Server Tests")
    print("=" * 60)
    print(f"Server URL: {MCP_SERVER_URL}")

    results = []

    try:
        results.append(("Health", await test_health()))
        results.append(("Initialize", await test_mcp_initialize()))
        results.append(("List Tools", await test_mcp_list_tools()))
        results.append(("Search Memory", await test_mcp_search_memory()))
        results.append(("SSE Endpoint", await test_sse_endpoint()))
    except httpx.ConnectError:
        print("\n[ERROR] Cannot connect to MCP server!")
        print(f"Make sure the server is running: uvicorn backend.cortex_mcp.http_server:app --port 8001")
        return

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))


if __name__ == "__main__":
    asyncio.run(main())
