"""
LLM-based conversation summarization service using OpenAI GPT-4o-mini.

This module handles:
- Generating 2-3 sentence summaries of conversations
- Extracting 3-5 main topics from conversations
- Caching summaries to avoid regeneration
- Retry logic with exponential backoff
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


# Initialize OpenAI client (lazily to allow tests without API key)
_client = None

def get_client():
    """Get or initialize the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = AsyncOpenAI(api_key=api_key)
    return _client

# Cache directory for summaries
CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "summaries"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def summarize_conversation(
    messages: List[Dict[str, Any]],
    conversation_id: str | None = None,
    use_cache: bool = True
) -> Tuple[str, List[str]]:
    """
    Generate a summary and extract topics from a conversation using GPT-4o-mini.
    
    Args:
        messages: List of message dictionaries with role and content
        conversation_id: Optional conversation ID for caching
        use_cache: Whether to use cached summaries if available
    
    Returns:
        Tuple of (summary, topics):
        - summary: 2-3 sentence summary string
        - topics: List of 3-5 topic strings
    
    Raises:
        ValueError: If messages list is empty
        openai.APIError: If API call fails after retries
    """
    if not messages:
        raise ValueError("Cannot summarize: messages list is empty")
    
    # Check cache first
    if use_cache and conversation_id:
        cached = _load_from_cache(conversation_id)
        if cached:
            return cached["summary"], cached["topics"]
    
    # Format conversation for the LLM
    conversation_text = _format_conversation(messages)
    
    # Generate summary and topics using GPT-4o-mini
    summary, topics = await _call_gpt4o_mini(conversation_text)
    
    # Cache the result
    if conversation_id:
        _save_to_cache(conversation_id, summary, topics)
    
    return summary, topics


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_gpt4o_mini(conversation_text: str) -> Tuple[str, List[str]]:
    """
    Call GPT-4o-mini API with structured JSON output.
    
    Args:
        conversation_text: Formatted conversation text
    
    Returns:
        Tuple of (summary, topics)
    
    Raises:
        Exception: If API call fails or response is invalid
    """
    system_prompt = """You are a helpful assistant that analyzes AI chat conversations.
Your task is to:
1. Generate a concise 2-3 sentence summary of the conversation
2. Extract 3-5 main topics or themes discussed

Return your response as JSON with this structure:
{
  "summary": "2-3 sentence summary here",
  "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
}

Keep topics short (1-3 words each) and specific."""

    user_prompt = f"""Analyze this conversation and provide a summary and topics:

{conversation_text}

Remember to return only valid JSON with "summary" and "topics" fields."""

    try:
        response = await get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Validate response structure
        if "summary" not in result or "topics" not in result:
            raise ValueError("Invalid response structure from GPT-4o-mini")
        
        summary = result["summary"].strip()
        topics = result["topics"]
        
        # Ensure topics is a list
        if not isinstance(topics, list):
            topics = [str(topics)]
        
        # Limit to 5 topics and ensure they're strings
        topics = [str(t).strip() for t in topics[:5]]
        
        # Filter out empty topics
        topics = [t for t in topics if t]
        
        # Ensure at least 1 topic
        if not topics:
            topics = ["General Discussion"]
        
        return summary, topics
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise Exception(f"GPT-4o-mini API call failed: {e}")


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """
    Format messages into a readable conversation transcript.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted conversation string
    """
    lines = []
    
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        
        # Truncate very long messages
        if len(content) > 1000:
            content = content[:997] + "..."
        
        lines.append(f"{role}: {content}")
    
    return "\n\n".join(lines)


def _load_from_cache(conversation_id: str) -> Dict[str, Any] | None:
    """
    Load cached summary and topics from disk.
    
    Args:
        conversation_id: Conversation ID
    
    Returns:
        Dictionary with summary and topics, or None if not cached
    """
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_to_cache(conversation_id: str, summary: str, topics: List[str]) -> None:
    """
    Save summary and topics to cache.
    
    Args:
        conversation_id: Conversation ID
        summary: Summary text
        topics: List of topics
    """
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    
    data = {
        "summary": summary,
        "topics": topics
    }
    
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError:
        pass  # Silently fail if cache write fails


def generate_cache_key(messages: List[Dict[str, Any]]) -> str:
    """
    Generate a deterministic cache key from messages.
    
    Useful when conversation_id is not yet available.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        SHA256 hash of the conversation content
    """
    # Create a stable string representation
    content = json.dumps(
        [(m.get("role"), m.get("content")) for m in messages],
        sort_keys=True
    )
    
    # Generate hash
    return hashlib.sha256(content.encode()).hexdigest()


def clear_cache() -> int:
    """
    Clear all cached summaries.
    
    Returns:
        Number of cache files deleted
    """
    count = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except IOError:
            pass
    
    return count
