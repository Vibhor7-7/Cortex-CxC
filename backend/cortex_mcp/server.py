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
            description="Search past AI chat conversations using OpenAI Vector Store (hybrid semantic + keyword retrieval). Returns conversation metadata including title, summary, topics, and message previews.",
            inputSchema={
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
                    f"{config.backend_api_url}/api/search",
                    json={"query": query, "limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])

                if not results:
                    return [TextContent(
                        type="text",
                        text=f"No conversations found matching '{query}'. Try using different keywords or broader terms."
                    )]

                formatted_results = []
                for idx, r in enumerate(results, 1):
                    result_text = f"**Result {idx}** (Relevance: {r.get('score', 0.0):.2f})\n"
                    result_text += f"- **Conversation ID**: {r.get('conversation_id', 'N/A')}\n"
                    result_text += f"- **Title**: {r.get('title', 'Untitled')}\n"
                    result_text += f"- **Summary**: {r.get('summary', 'No summary available')}\n"

                    topics = r.get('topics', [])
                    if topics:
                        result_text += f"- **Topics**: {', '.join(topics)}\n"

                    cluster = r.get('cluster_name')
                    if cluster:
                        result_text += f"- **Cluster**: {cluster}\n"

                    preview = r.get('message_preview')
                    if preview:
                        truncated = preview[:300] + "..." if len(preview) > 300 else preview
                        result_text += f"- **Preview**: {truncated}\n"

                    message_count = r.get('message_count')
                    if message_count:
                        result_text += f"- **Messages**: {message_count}\n"

                    formatted_results.append(result_text)

                search_time = data.get('search_time_ms', 0)
                header = f"Found {len(results)} relevant conversation(s) for '{query}' (searched in {search_time:.0f}ms)\n\n"

                return [TextContent(
                    type="text",
                    text=header + "\n\n".join(formatted_results)
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
