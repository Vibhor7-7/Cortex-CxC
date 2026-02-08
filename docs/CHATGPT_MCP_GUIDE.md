# ChatGPT MCP Integration Guide

This guide explains how to integrate Cortex Memory with ChatGPT using the OpenAI Responses API and MCP (Model Context Protocol).

## Overview

ChatGPT supports remote MCP servers via the Responses API. Unlike Claude Desktop which uses stdio transport, **ChatGPT requires an HTTP/SSE server accessible on the public internet**.

```
ChatGPT API <--> Your Public MCP Server (HTTP/SSE) <--> Cortex Backend API <--> Vector Store + SQLite
```

## Prerequisites

1. **Cortex Backend running** on `http://localhost:8000`
2. **MCP HTTP Server running** on port `8001`
3. **Public URL** for the MCP server (via ngrok, Cloudflare Tunnel, or deployed server)
4. **OpenAI API key** with access to the Responses API

## Quick Start

### Step 1: Start the Cortex Backend

```bash
cd /Users/sharvibhor/Desktop/Projects/Cortex-CxC
source .venv/bin/activate
python -m backend.main
```

### Step 2: Start the MCP HTTP Server

```bash
# In a new terminal
cd /Users/sharvibhor/Desktop/Projects/Cortex-CxC
source .venv/bin/activate
uvicorn backend.cortex_mcp.http_server:app --host 0.0.0.0 --port 8001
```

### Step 3: Expose to Public Internet (for testing)

Using ngrok:
```bash
ngrok http 8001
```

This gives you a public URL like `https://abc123.ngrok.io`

### Step 4: Use with ChatGPT Responses API

```python
from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o",  # or gpt-4, gpt-4-turbo, etc.
    tools=[
        {
            "type": "mcp",
            "server_label": "cortex-memory",
            "server_description": "Search and retrieve past AI chat conversations from your memory.",
            "server_url": "https://YOUR-NGROK-URL.ngrok.io/sse",  # Your public MCP server URL
            "require_approval": "never"
        }
    ],
    input="Search my memory for conversations about Python async programming"
)

print(resp.output_text)
```

## Available Tools

### search_memory

Searches your chat history using semantic search.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional): Max results (default: 5)

**Example usage via API:**
```python
resp = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "cortex-memory",
        "server_url": "https://your-server.com/sse",
        "require_approval": "never",
        "allowed_tools": ["search_memory"]  # Optional: filter to specific tools
    }],
    input="What conversations did I have about React hooks?"
)
```

### fetch_chat

Retrieves a complete conversation by ID.

**Parameters:**
- `conversation_id` (string, required): UUID of the conversation

**Example:**
```python
resp = client.responses.create(
    model="gpt-4o",
    tools=[{
        "type": "mcp",
        "server_label": "cortex-memory",
        "server_url": "https://your-server.com/sse",
        "require_approval": "never"
    }],
    input="Fetch the full conversation with ID abc-123-def"
)
```

## MCP Server Endpoints

The HTTP server provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/mcp` | POST | MCP JSON-RPC (Streamable HTTP) |
| `/sse` | GET | SSE connection |
| `/sse` | POST | MCP via SSE transport |

## Testing

### Test the MCP Server Locally

```bash
# Run the test script
python backend/cortex_mcp/test_http_server.py
```

### Manual curl Tests

```bash
# Health check
curl http://localhost:8001/health

# List tools
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}'

# Search memory
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"search_memory","arguments":{"query":"Python","limit":3}}}'
```

## Deployment Options

### Option 1: ngrok (Development/Testing)

```bash
ngrok http 8001
```

### Option 2: Cloudflare Tunnel (Free, Production-ready)

```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:8001
```

### Option 3: Deploy to Cloud

Deploy the MCP server to any cloud provider:

**Railway/Render/Fly.io:**
```bash
# Dockerfile approach
docker build -t cortex-mcp .
docker run -p 8001:8001 cortex-mcp
```

**Important:** If deploying separately from the backend, update `MCP_BACKEND_API_URL` in your environment to point to your deployed backend API.

## Security Considerations

⚠️ **Important Security Notes:**

1. **Authentication**: The current server has no authentication. For production:
   - Add API key authentication
   - Use the `authorization` parameter in the MCP tool config

2. **Data Privacy**: The MCP server can access your chat history. Only expose to trusted services.

3. **CORS**: The server allows all origins by default. Restrict for production.

4. **Rate Limiting**: Add rate limiting for production deployments.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_NAME` | `cortex-memory` | Server identifier |
| `MCP_SERVER_VERSION` | `1.0.0` | Server version |
| `MCP_BACKEND_API_URL` | `http://localhost:8000` | Cortex backend URL |
| `MCP_LOG_LEVEL` | `INFO` | Logging level |

## Troubleshooting

### "Cannot connect to MCP server"
- Ensure the HTTP server is running on port 8001
- Check that your public URL is accessible

### "Error communicating with backend"
- Verify the Cortex backend is running on port 8000
- Check `MCP_BACKEND_API_URL` configuration

### "No results found"
- Ensure you have ingested chat data
- Try broader search terms

### ChatGPT not calling tools
- Verify the `server_url` is publicly accessible
- Check the MCP server logs for errors
- Ensure the server responds to `/sse` endpoint

## Example: Full Integration

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your public MCP server URL
MCP_SERVER_URL = "https://your-server.ngrok.io/sse"

def search_memory(query: str) -> str:
    """Search your chat memory via ChatGPT."""
    resp = client.responses.create(
        model="gpt-4o",
        tools=[{
            "type": "mcp",
            "server_label": "cortex-memory",
            "server_description": "Search past AI conversations",
            "server_url": MCP_SERVER_URL,
            "require_approval": "never"
        }],
        input=f"Search my memory for: {query}"
    )
    return resp.output_text


# Usage
result = search_memory("Python async programming")
print(result)
```

## Comparison: Claude Desktop vs ChatGPT

| Feature | Claude Desktop | ChatGPT |
|---------|---------------|---------|
| Transport | stdio | HTTP/SSE |
| Server Location | Local | Public Internet |
| Configuration | JSON config file | API parameters |
| Authentication | None needed | Optional OAuth |
