# Cortex MCP Server

MCP (Model Context Protocol) server for Cortex memory system.

## Usage

### Running the server

```bash
cd /Users/tanayj/Cortex-CxC/backend
python -m cortex_mcp.server
```

### Configuration

Set environment variables in `.env` (optional, using MCP_ prefix):

```
MCP_SERVER_NAME=cortex-memory
MCP_SERVER_VERSION=1.0.0
MCP_BACKEND_API_URL=http://localhost:8000
MCP_LOG_LEVEL=INFO
```

## Available Tools

### search_memory
Search through chat history and context to find relevant past conversations.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional): Maximum results (default: 5)

### fetch_chat
Fetch a specific chat conversation by ID.

**Parameters:**
- `chat_id` (string, required): The chat ID to fetch

## Integration with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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
