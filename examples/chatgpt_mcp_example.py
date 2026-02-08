#!/usr/bin/env python3
"""
Example script showing how to use Cortex Memory with ChatGPT via MCP.

Prerequisites:
1. Cortex backend running on localhost:8000
2. MCP HTTP server running on localhost:8001
3. Public URL (ngrok) pointing to the MCP server
4. OpenAI API key set in OPENAI_API_KEY environment variable

Usage:
    # Start ngrok first
    ngrok http 8001
    
    # Then run this script with your ngrok URL
    python chatgpt_example.py "https://abc123.ngrok.io"
"""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)


def search_memory_with_chatgpt(query: str, mcp_server_url: str) -> str:
    """
    Search your Cortex memory using ChatGPT with MCP.
    
    Args:
        query: What to search for
        mcp_server_url: Public URL of your MCP server (e.g., ngrok URL)
    
    Returns:
        ChatGPT's response with search results
    """
    client = OpenAI()
    
    # Ensure URL ends with /sse for SSE transport
    if not mcp_server_url.endswith("/sse"):
        mcp_server_url = mcp_server_url.rstrip("/") + "/sse"
    
    print(f"Connecting to MCP server: {mcp_server_url}")
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        resp = client.responses.create(
            model="gpt-4o",  # or gpt-4, gpt-4-turbo
            tools=[
                {
                    "type": "mcp",
                    "server_label": "cortex-memory",
                    "server_description": "Search and retrieve past AI chat conversations from your memory.",
                    "server_url": mcp_server_url,
                    "require_approval": "never",
                    "allowed_tools": ["search_memory", "fetch_chat"]
                }
            ],
            input=query
        )
        
        return resp.output_text
        
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Export your key: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    # Get MCP server URL from command line or prompt
    if len(sys.argv) > 1:
        mcp_url = sys.argv[1]
    else:
        print("Cortex Memory + ChatGPT MCP Example")
        print("=" * 50)
        print("\nFirst, start ngrok to expose your MCP server:")
        print("  ngrok http 8001")
        print("\nThen enter the ngrok URL below.")
        print()
        mcp_url = input("Enter your MCP server URL (e.g., https://abc123.ngrok.io): ").strip()
        
        if not mcp_url:
            print("Error: MCP server URL required")
            sys.exit(1)
    
    # Example queries
    queries = [
        "Search my memory for conversations about Python programming",
        "What did I discuss about machine learning?",
        "Find conversations where I talked about APIs"
    ]
    
    print("\n" + "=" * 50)
    print("Running example queries...")
    print("=" * 50)
    
    for query in queries[:1]:  # Just run the first query as demo
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        result = search_memory_with_chatgpt(query, mcp_url)
        print(f"\n🤖 Response:\n{result}")
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
