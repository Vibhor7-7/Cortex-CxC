import asyncio
import logging
from typing import Any
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from .config import config

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


server = Server(config.server_name)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_memory",
            description="Search through chat history and context to find relevant past conversations",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant chat history"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="fetch_chat",
            description="Fetch a specific chat conversation by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "The ID of the chat to fetch"
                    }
                },
                "required": ["chat_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        logger.info(f"Tool invoked: {name} with arguments: {arguments}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            if name == "search_memory":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)

                response = await client.post(
                    f"{config.backend_api_url}/search",
                    json={"query": query, "limit": limit}
                )
                response.raise_for_status()
                results = response.json()

                return [TextContent(
                    type="text",
                    text=f"Found {len(results)} relevant conversations:\n\n" +
                         "\n\n".join([
                             f"Chat ID: {r['chat_id']}\n"
                             f"Relevance: {r.get('score', 'N/A')}\n"
                             f"Summary: {r.get('summary', 'No summary available')}"
                             for r in results
                         ])
                )]

            elif name == "fetch_chat":
                chat_id = arguments.get("chat_id")

                response = await client.get(
                    f"{config.backend_api_url}/chats/{chat_id}"
                )
                response.raise_for_status()
                chat = response.json()

                return [TextContent(
                    type="text",
                    text=f"Chat ID: {chat['id']}\n"
                         f"Created: {chat.get('created_at', 'Unknown')}\n\n"
                         f"Messages:\n" +
                         "\n\n".join([
                             f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                             for msg in chat.get('messages', [])
                         ])
                )]

            else:
                raise ValueError(f"Unknown tool: {name}")

    except httpx.HTTPError as e:
        logger.error(f"HTTP error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error communicating with backend: {str(e)}"
        )]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing tool: {str(e)}"
        )]


async def main():
    logger.info(f"Starting {config.server_name} v{config.server_version}")
    logger.info(f"Backend API URL: {config.backend_api_url}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
