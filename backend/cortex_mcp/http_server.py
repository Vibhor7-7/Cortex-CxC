"""
HTTP/SSE MCP Server for ChatGPT Integration.

This server implements the MCP protocol over HTTP with Server-Sent Events (SSE),
which is required for ChatGPT's Responses API MCP tool integration.

Run with: python -m backend.cortex_mcp.http_server
Or: uvicorn backend.cortex_mcp.http_server:app --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import config, backboard_config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# MCP Protocol Models
# ============================================================================

class MCPRequest(BaseModel):
    """Base MCP JSON-RPC request."""
    jsonrpc: str = "2.0"
    id: Optional[str | int] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """Base MCP JSON-RPC response."""
    jsonrpc: str = "2.0"
    id: Optional[str | int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = [
    {
        "name": "search_memory",
        "description": "Search past AI chat conversations using semantic search. Returns conversation metadata including title, summary, topics, and message previews. Use this to find relevant past conversations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant conversations"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },
    {
        "name": "fetch_chat",
        "description": "Retrieve full content and messages from a specific conversation by ID. Returns complete conversation history with all messages in chronological order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation to fetch"
                }
            },
            "required": ["conversation_id"],
            "additionalProperties": False
        }
    }
]

# ============================================================================
# Tool Execution
# ============================================================================

async def execute_search_memory(query: str, limit: int = 5) -> str:
    """Execute search_memory tool."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{config.backend_api_url}/api/search/",
                json={"query": query, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])

            if not results:
                return f"No conversations found matching '{query}'. Try using different keywords or broader terms."

            formatted_results = []
            for idx, r in enumerate(results, 1):
                result_text = f"**Result {idx}** (Relevance: {r.get('score', 0.0):.2f})\n"
                result_text += f"- Conversation ID: {r.get('conversation_id', 'N/A')}\n"
                result_text += f"- Title: {r.get('title', 'Untitled')}\n"
                result_text += f"- Summary: {r.get('summary', 'No summary available')}\n"

                topics = r.get('topics', [])
                if topics:
                    result_text += f"- Topics: {', '.join(topics)}\n"

                preview = r.get('message_preview')
                if preview:
                    truncated = preview[:300] + "..." if len(preview) > 300 else preview
                    result_text += f"- Preview: {truncated}\n"

                formatted_results.append(result_text)

            search_time = data.get('search_time_ms', 0)
            header = f"Found {len(results)} relevant conversation(s) for '{query}' (searched in {search_time:.0f}ms)\n\n"
            return header + "\n".join(formatted_results)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error in search_memory: {e}")
            return f"Error searching memory: {str(e)}"
        except Exception as e:
            logger.error(f"Error in search_memory: {e}", exc_info=True)
            return f"Error executing search: {str(e)}"


async def execute_fetch_chat(conversation_id: str) -> str:
    """Execute fetch_chat tool."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{config.backend_api_url}/api/chats/{conversation_id}"
            )
            response.raise_for_status()
            data = response.json()

            result_text = f"**Conversation Details**\n\n"
            result_text += f"- ID: {data.get('id', 'N/A')}\n"
            result_text += f"- Title: {data.get('title', 'Untitled')}\n"

            summary = data.get('summary')
            if summary:
                result_text += f"- Summary: {summary}\n"

            topics = data.get('topics', [])
            if topics:
                result_text += f"- Topics: {', '.join(topics)}\n"

            message_count = data.get('message_count', 0)
            result_text += f"- Message Count: {message_count}\n"

            created_at = data.get('created_at')
            if created_at:
                result_text += f"- Created: {created_at}\n"

            messages = data.get('messages', [])
            if messages:
                result_text += f"\n**Conversation Transcript** ({len(messages)} messages):\n\n"

                for msg in messages:
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')
                    timestamp = msg.get('created_at', '')

                    result_text += f"**{role}**"
                    if timestamp:
                        result_text += f" ({timestamp})"
                    result_text += f":\n{content}\n\n"
            else:
                result_text += "\nNo messages found in this conversation.\n"

            return result_text

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Conversation not found: {conversation_id}"
            logger.error(f"HTTP error in fetch_chat: {e}")
            return f"Error fetching chat: {str(e)}"
        except Exception as e:
            logger.error(f"Error in fetch_chat: {e}", exc_info=True)
            return f"Error fetching chat: {str(e)}"


async def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool by name."""
    if name == "search_memory":
        return await execute_search_memory(
            query=arguments.get("query", ""),
            limit=arguments.get("limit", 5)
        )
    elif name == "fetch_chat":
        return await execute_fetch_chat(
            conversation_id=arguments.get("conversation_id", "")
        )
    else:
        return f"Unknown tool: {name}"


# ============================================================================
# MCP JSON-RPC Handler
# ============================================================================

async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle an MCP JSON-RPC request."""
    method = request.method
    params = request.params or {}
    request_id = request.id

    logger.info(f"MCP request: method={method}, id={request_id}")

    try:
        if method == "initialize":
            # Return server capabilities
            return MCPResponse(
                id=request_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": config.server_name,
                        "version": config.server_version
                    }
                }
            )

        elif method == "tools/list":
            # Return available tools
            return MCPResponse(
                id=request_id,
                result={"tools": TOOLS}
            )

        elif method == "tools/call":
            # Execute a tool
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            logger.info(f"Tool call: {tool_name} with args: {tool_args}")

            result = await execute_tool(tool_name, tool_args)

            return MCPResponse(
                id=request_id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            )

        elif method == "ping":
            return MCPResponse(id=request_id, result={})

        else:
            return MCPResponse(
                id=request_id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            )

    except Exception as e:
        logger.error(f"Error handling MCP request: {e}", exc_info=True)
        return MCPResponse(
            id=request_id,
            error={
                "code": -32603,
                "message": str(e)
            }
        )


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info(f"Starting {config.server_name} HTTP/SSE MCP Server v{config.server_version}")
    logger.info(f"Backend API URL: {config.backend_api_url}")
    yield
    logger.info("Shutting down MCP HTTP Server")


app = FastAPI(
    title="Cortex Memory MCP Server",
    description="MCP server for searching and retrieving AI chat history",
    version=config.server_version,
    lifespan=lifespan
)

# Add CORS middleware for ChatGPT access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OpenAI needs to access this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": config.server_name,
        "version": config.server_version,
        "protocol": "MCP",
        "transport": "HTTP/SSE"
    }


@app.get("/health")
async def health():
    """Health check for the MCP server."""
    # Check backend connectivity
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.backend_api_url}/health")
            backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False

    return {
        "status": "healthy" if backend_healthy else "degraded",
        "backend_connected": backend_healthy,
        "server": config.server_name,
        "version": config.server_version
    }


@app.post("/mcp")
async def mcp_post(request: Request):
    """
    Handle MCP requests via HTTP POST (Streamable HTTP transport).
    This is the primary endpoint for ChatGPT MCP integration.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
        response = await handle_mcp_request(mcp_request)
        return JSONResponse(content=response.model_dump(exclude_none=True))
    except Exception as e:
        logger.error(f"Error processing MCP POST request: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)}
            }
        )


@app.get("/sse")
async def mcp_sse(request: Request):
    """
    SSE endpoint for MCP communication.
    ChatGPT can connect to this for Server-Sent Events transport.
    """
    async def event_generator():
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'server': config.server_name})}\n\n"

        # Keep connection alive
        while True:
            if await request.is_disconnected():
                break
            # Send keepalive
            yield f": keepalive\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/sse")
async def mcp_sse_post(request: Request):
    """
    Handle MCP requests via SSE POST.
    This supports the HTTP/SSE transport that ChatGPT uses.
    """
    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)
        response = await handle_mcp_request(mcp_request)

        async def event_generator():
            yield f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"Error processing MCP SSE request: {e}", exc_info=True)

        async def error_generator():
            yield f"data: {json.dumps({'jsonrpc': '2.0', 'error': {'code': -32603, 'message': str(e)}})}\n\n"

        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.cortex_mcp.http_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
