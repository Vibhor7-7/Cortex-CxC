# MCP Integration Guide

## Overview

Cortex now includes a Model Context Protocol (MCP) server that enables Claude Desktop and other MCP-compatible AI assistants to search and retrieve your chat history memory directly.

## Architecture

The MCP server acts as a bridge between Claude Desktop and the Cortex backend:

```
Claude Desktop <-> MCP Server (stdio) <-> Cortex Backend API <-> Vector Store + SQLite
```

## Components

### 1. MCP Server (`backend/cortex_mcp/server.py`)
- Implements Anthropic's Model Context Protocol
- Uses stdio transport for communication with Claude Desktop
- Provides two tools: `search_memory` and `fetch_chat`
- Handles errors and logging for debugging

### 2. Configuration (`backend/cortex_mcp/config.py`)
- Server metadata (name, version)
- Backend API URL (default: http://localhost:8000)
- Logging level configuration
- Environment variable support with `MCP_` prefix

### 3. Available Tools

#### search_memory
Searches through your chat history using hybrid semantic + keyword search.

**Parameters:**
- `query` (string, required): Search query to find relevant conversations
- `limit` (integer, optional): Maximum number of results to return (default: 5)

**Returns:**
- List of relevant conversations with:
  - Chat ID
  - Relevance score
  - Summary

**Example:**
```json
{
  "query": "python async programming",
  "limit": 3
}
```

#### fetch_chat
Retrieves a complete chat conversation by ID.

**Parameters:**
- `chat_id` (string, required): The ID of the chat to fetch

**Returns:**
- Complete chat details including:
  - Chat ID
  - Creation timestamp
  - All messages with roles and content

**Example:**
```json
{
  "chat_id": "conv_abc123"
}
```

## Setup Instructions

### 1. Install Dependencies

The MCP SDK is already included in `backend/requirements.txt`:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Add to `backend/.env` if you want to customize:

```bash
MCP_SERVER_NAME=cortex-memory
MCP_SERVER_VERSION=1.0.0
MCP_BACKEND_API_URL=http://localhost:8000
MCP_LOG_LEVEL=INFO
```

### 3. Start the Cortex Backend

```bash
cd backend
python main.py
```

The backend must be running on port 8000 for the MCP server to communicate with it.

### 4. Configure Claude Desktop

Edit your Claude Desktop config file at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following configuration:

```json
{
  "mcpServers": {
    "cortex-memory": {
      "command": "python",
      "args": ["-m", "cortex_mcp.server"],
      "cwd": "/Users/tanayj/Cortex-CxC/backend"
    }
  }
}
```

**Important:** Update the `cwd` path to match your actual Cortex installation directory.

### 5. Restart Claude Desktop

After updating the config, restart Claude Desktop completely to load the MCP server.

## Usage

Once configured, you can ask Claude to search your memory:

**Example prompts:**
- "Search my memory for conversations about React hooks"
- "Find chats where I discussed machine learning optimization"
- "What did I learn about database indexing?"

Claude will automatically use the `search_memory` tool to find relevant conversations from your Cortex memory.

## Troubleshooting

### MCP Server Not Found

**Issue:** Claude says the tool is unavailable.

**Solution:**
1. Check that the `cwd` path in `claude_desktop_config.json` is correct
2. Verify Python is in your PATH
3. Restart Claude Desktop completely

### Connection Errors

**Issue:** Tools return "Error communicating with backend"

**Solution:**
1. Ensure Cortex backend is running on `http://localhost:8000`
2. Test the backend with: `curl http://localhost:8000/health`
3. Check `MCP_BACKEND_API_URL` in your config

### No Results Returned

**Issue:** Searches return empty results

**Solution:**
1. Verify you have uploaded and ingested chat data
2. Check that vector store is populated: `GET /api/chats`
3. Try broader search queries

### Viewing Logs

Enable debug logging by setting in `backend/.env`:

```bash
MCP_LOG_LEVEL=DEBUG
```

Logs will appear in Claude Desktop's developer console or terminal output.

## Testing the MCP Server

### Manual Test (Without Claude Desktop)

```bash
cd backend
python -m cortex_mcp.server
```

The server will start and wait for stdio input. You can test it by sending MCP protocol messages, but this is primarily for debugging.

### Integration Test

1. Start the Cortex backend
2. Upload some chat HTML files
3. Open Claude Desktop
4. Ask: "Search my memory for [topic you uploaded]"
5. Verify Claude returns relevant results

## API Endpoints Used

The MCP server communicates with these Cortex backend endpoints:

- `POST /api/search` - For the `search_memory` tool
- `GET /api/chats/{chat_id}` - For the `fetch_chat` tool

Make sure these endpoints are working correctly.

## Security Notes

- The MCP server runs locally and only communicates via stdio
- No external network connections are made
- API keys are never exposed to the MCP server
- All data stays on your local machine

## Future Enhancements

Potential improvements for future versions:

- [ ] Add `list_topics` tool to browse available conversation topics
- [ ] Add `list_clusters` tool to explore conversation clusters
- [ ] Support filtering by date ranges
- [ ] Add conversation summarization tool
- [ ] Support batch operations for multiple chat retrieval
- [ ] Add conversation statistics tool

## Related Documentation

- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Cortex Backend API](../backend/README.md)
- [Implementation Status](./IMPLEMENTATION_STATUS.md)
