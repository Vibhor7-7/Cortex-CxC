import asyncio
import argparse
import logging
import sys
from typing import Any, List
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from .config import config, backboard_config

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for guard disable (set via CLI)
_guard_disabled = False


server = Server(config.server_name)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_memory",
            description="Search past AI chat conversations using local semantic search (nomic-embed-text embeddings via Ollama). Returns conversation metadata including title, summary, topics, and message previews.",
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
            description="Retrieve full content and messages from a specific conversation by ID. Returns complete conversation history with all messages in chronological order, including timestamps and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "UUID of the conversation to fetch"
                    }
                },
                "required": ["conversation_id"]
            }
        )
    ]


async def _apply_guard_filter(query: str, results: List[dict]) -> tuple[List[dict], int]:
    """
    Apply Backboard guard filtering to search results.

    Returns:
        Tuple of (filtered_results, blocked_count)
    """
    global _guard_disabled

    if _guard_disabled or not backboard_config.guard_active:
        return results, 0

    try:
        from backend.services.backboard_guard import get_relevance_guard
        guard = get_relevance_guard()

        if not guard.is_available:
            return results, 0

        filtered = await guard.filter_relevant_memories(query, results)
        blocked_count = len(results) - len(filtered)
        return filtered, blocked_count
    except Exception as e:
        logger.warning(f"Guard filtering failed: {e}")
        return results, 0


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        logger.info(f"Tool invoked: {name} with arguments: {arguments}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            if name == "search_memory":
                query = arguments.get("query")
                limit = arguments.get("limit", 5)

                response = await client.post(
                    f"{config.backend_api_url}/api/search/",
                    json={"query": query, "limit": limit}
                )
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])

                # Apply Backboard guard filtering
                results, blocked_count = await _apply_guard_filter(query, results)

                if not results:
                    guard_note = ""
                    if blocked_count > 0:
                        guard_note = f"\n[GUARD] {blocked_count} low-confidence result(s) were filtered."
                    return [TextContent(
                        type="text",
                        text=f"No conversations found matching '{query}'. Try using different keywords or broader terms.{guard_note}"
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
                header = f"Found {len(results)} relevant conversation(s) for '{query}' (searched in {search_time:.0f}ms)\n"

                # Add guard status if results were filtered
                guard_status = ""
                if blocked_count > 0:
                    guard_status = f"[GUARD] {blocked_count} low-confidence result(s) were filtered.\n"

                return [TextContent(
                    type="text",
                    text=header + guard_status + "\n" + "\n\n".join(formatted_results)
                )]

            elif name == "fetch_chat":
                conversation_id = arguments.get("conversation_id")

                if not conversation_id:
                    return [TextContent(
                        type="text",
                        text="Error: conversation_id parameter is required"
                    )]

                response = await client.get(
                    f"{config.backend_api_url}/api/chats/{conversation_id}"
                )
                response.raise_for_status()
                data = response.json()

                result_text = f"**Conversation Details**\n\n"
                result_text += f"- **ID**: {data.get('id', 'N/A')}\n"
                result_text += f"- **Title**: {data.get('title', 'Untitled')}\n"

                summary = data.get('summary')
                if summary:
                    result_text += f"- **Summary**: {summary}\n"

                topics = data.get('topics', [])
                if topics:
                    result_text += f"- **Topics**: {', '.join(topics)}\n"

                cluster = data.get('cluster_name')
                if cluster:
                    result_text += f"- **Cluster**: {cluster}\n"

                message_count = data.get('message_count', 0)
                result_text += f"- **Message Count**: {message_count}\n"

                created_at = data.get('created_at')
                if created_at:
                    result_text += f"- **Created**: {created_at}\n"

                messages = data.get('messages', [])
                if messages:
                    result_text += f"\n**Conversation Transcript** ({len(messages)} messages):\n\n"
                    result_text += "=" * 80 + "\n\n"

                    for msg in messages:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')
                        timestamp = msg.get('created_at', '')

                        result_text += f"**{role}**"
                        if timestamp:
                            result_text += f" (at {timestamp})"
                        result_text += ":\n"
                        result_text += f"{content}\n\n"
                        result_text += "-" * 80 + "\n\n"
                else:
                    result_text += "\nNo messages found in this conversation.\n"

                return [TextContent(
                    type="text",
                    text=result_text
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CORTEX Memory MCP Server")
    parser.add_argument(
        "--disable-guard",
        action="store_true",
        help="Disable Backboard relevance guard filtering"
    )
    return parser.parse_args()


async def main():
    global _guard_disabled

    # Parse CLI arguments
    args = parse_args()
    _guard_disabled = args.disable_guard

    logger.info(f"Starting {config.server_name} v{config.server_version}")
    logger.info(f"Backend API URL: {config.backend_api_url}")

    # Log guard status
    if _guard_disabled:
        logger.info("Backboard guard: DISABLED (via --disable-guard flag)")
    elif backboard_config.guard_active:
        logger.info(f"Backboard guard: ENABLED (threshold: {backboard_config.guard_threshold})")
    else:
        logger.info("Backboard guard: DISABLED (API key not configured)")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
