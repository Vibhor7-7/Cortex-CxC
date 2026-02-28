"""
LLM-based conversation summarization service supporting Ollama (local) and Groq (cloud).

This module handles:
- Generating 2-3 sentence summaries of conversations
- Extracting 3-5 main topics from conversations
- Provider routing (Ollama Qwen 2.5 or Groq llama-3.1-8b-instant)
- Caching summaries to avoid regeneration
- Retry logic with exponential backoff
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from backend.services.provider import get_chat_provider


# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Cache directory for summaries (configurable via env for Docker)
_cache_base = Path(os.getenv("CACHE_DIR", str(Path(__file__).parent.parent.parent / ".cache")))
CACHE_DIR = _cache_base / "summaries"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Shared prompts
SUMMARIZE_SYSTEM_PROMPT = """You are a helpful assistant that analyzes AI chat conversations.
Your task is to:
1. Generate a concise 2-3 sentence summary of the conversation
2. Extract 3-5 main topics or themes discussed

Return your response as JSON with this exact structure:
{
  "summary": "2-3 sentence summary here",
  "topics": ["topic1", "topic2", "topic3"]
}

Keep topics short (1-3 words each) and specific.
Return ONLY valid JSON, no other text."""


def _build_summarize_user_prompt(conversation_text: str) -> str:
    return (
        f"Analyze this conversation and provide a summary and topics:\n\n"
        f"{conversation_text}\n\n"
        f'Return ONLY valid JSON with "summary" and "topics" fields.'
    )


def _parse_summary_response(content: str) -> Tuple[str, List[str]]:
    """Parse and validate the JSON summary response from any LLM provider."""
    result = json.loads(content)

    if "summary" not in result or "topics" not in result:
        raise ValueError("Invalid response structure: missing summary or topics")

    summary = result["summary"].strip()
    topics = result["topics"]

    if not isinstance(topics, list):
        topics = [str(topics)]

    # Limit to 5 topics and ensure they're strings
    topics = [str(t).strip() for t in topics[:5]]
    topics = [t for t in topics if t]

    if not topics:
        topics = ["General Discussion"]

    return summary, topics


async def summarize_conversation(
    messages: List[Dict[str, Any]],
    conversation_id: str | None = None,
    use_cache: bool = True
) -> Tuple[str, List[str]]:
    """
    Generate a summary and extract topics from a conversation.

    Routes to Groq or Ollama based on provider config.

    Args:
        messages: List of message dictionaries with role and content
        conversation_id: Optional conversation ID for caching
        use_cache: Whether to use cached summaries if available

    Returns:
        Tuple of (summary, topics)

    Raises:
        ValueError: If messages list is empty
        httpx.HTTPError: If API call fails after retries
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

    # Generate summary using the configured provider
    provider = get_chat_provider()
    if provider == "groq":
        summary, topics = await _call_groq(conversation_text)
    else:
        summary, topics = await _call_ollama(conversation_text)

    # Cache the result
    if conversation_id:
        _save_to_cache(conversation_id, summary, topics)

    return summary, topics


# ---------------------------------------------------------------------------
# Groq API (cloud)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_groq(conversation_text: str) -> Tuple[str, List[str]]:
    """
    Call Groq API (llama-3.1-8b-instant) with structured JSON output.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_summarize_user_prompt(conversation_text)}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.3,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("Empty response from Groq")

        return _parse_summary_response(content)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response from Groq: {e}")
    except httpx.HTTPError as e:
        raise Exception(f"Groq API call failed: {e}")
    except Exception as e:
        if "Groq" in str(e) or "Failed to parse" in str(e):
            raise
        raise Exception(f"Groq summarization failed: {e}")


# ---------------------------------------------------------------------------
# Ollama API (local)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_ollama(conversation_text: str) -> Tuple[str, List[str]]:
    """
    Call Qwen 2.5 via Ollama's local HTTP API with structured JSON output.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_summarize_user_prompt(conversation_text)}
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500
                    }
                }
            )
            response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from Ollama")

        return _parse_summary_response(content)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response from Qwen 2.5: {e}")
    except httpx.HTTPError as e:
        raise Exception(f"Ollama API call failed (is 'ollama serve' running?): {e}")
    except Exception as e:
        if "Failed to parse" in str(e) or "Ollama API" in str(e):
            raise
        raise Exception(f"Qwen 2.5 summarization failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format messages into a readable conversation transcript."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if len(content) > 1000:
            content = content[:997] + "..."
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _load_from_cache(conversation_id: str) -> Dict[str, Any] | None:
    """Load cached summary and topics from disk."""
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_to_cache(conversation_id: str, summary: str, topics: List[str]) -> None:
    """Save summary and topics to cache."""
    cache_file = CACHE_DIR / f"{conversation_id}.json"
    data = {"summary": summary, "topics": topics}
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def generate_cache_key(messages: List[Dict[str, Any]]) -> str:
    """Generate a deterministic cache key from messages."""
    content = json.dumps(
        [(m.get("role"), m.get("content")) for m in messages],
        sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()


def clear_cache() -> int:
    """Clear all cached summaries."""
    count = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except IOError:
            pass
    return count
