# MCP Guide

This guide explains how to run and validate the Cortex MCP (Model Context Protocol) server and integrate it with Claude Desktop.

## Prerequisites

1. Start the Cortex backend API on `http://localhost:8000`.
2. Ensure Ollama is running for embeddings/summaries.

## Start MCP Server

From the repo root:

```bash
./start_mcp.sh
```

Or directly:

```bash
python -m backend.cortex_mcp.server
```

## Claude Desktop Config

Add this to your Claude Desktop config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "cortex-memory": {
      "command": "python",
      "args": ["-m", "backend.cortex_mcp.server"],
      "cwd": "/Users/tanayj/Cortex-CxC"
    }
  }
}
```

Update the `cwd` path to your local repo path.

## Tools

- `search_memory`: semantic search over ingested conversations
- `fetch_chat`: fetches a full conversation by `conversation_id`

## Tests

### Backend API Test (no MCP transport)

```bash
python backend/cortex_mcp/test_mcp.py
```

### MCP stdio Client Test (MCP transport)

```bash
python backend/cortex_mcp/test_client.py
```

Optional env overrides:

- `MCP_TEST_QUERY` (default `Python`)
- `MCP_TEST_LIMIT` (default `3`)
- `MCP_TEST_COMMAND` (default `python`)
- `MCP_TEST_ARGS` (default `-m backend.cortex_mcp.server`)
- `MCP_TEST_CWD` (default repo root)

## Troubleshooting

- Backend connection errors: verify `MCP_BACKEND_API_URL` in `backend/.env`
- No results: ensure chats are ingested and vector store is populated
- Claude not finding tools: verify `cwd` in Claude config and restart Claude Desktop

## Reference

Full integration details: `docs/MCP_INTEGRATION.md`
