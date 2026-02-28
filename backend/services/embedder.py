"""
Embedding generation service supporting Ollama (local) and HuggingFace Inference API (cloud).

This module handles:
- Generating 768-dimensional embeddings from text
- Provider routing (Ollama nomic-embed-text or HuggingFace nomic-embed-text-v1.5)
- Retry logic with exponential backoff
- Caching embeddings by conversation ID
- Batch embedding generation for multiple texts
"""

import os
import json
import hashlib
from typing import List, Dict, Any
from pathlib import Path

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from backend.services.provider import get_embedding_provider


# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSION = 768  # nomic-embed-text produces 768D embeddings

# HuggingFace configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/nomic-ai/nomic-embed-text-v1.5"

# Cache directory for embeddings (configurable via env for Docker)
_cache_base = Path(os.getenv("CACHE_DIR", str(Path(__file__).parent.parent.parent / ".cache")))
CACHE_DIR = _cache_base / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def generate_embedding(
    text: str,
    conversation_id: str | None = None,
    use_cache: bool = True
) -> List[float]:
    """
    Generate a 768-dimensional embedding vector from text.

    Routes to HuggingFace or Ollama based on provider config.

    Args:
        text: Text content to embed
        conversation_id: Optional conversation ID for caching
        use_cache: Whether to use cached embeddings if available

    Returns:
        List of 768 floats representing the embedding vector

    Raises:
        ValueError: If text is empty
        httpx.HTTPError: If API call fails after retries
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding: text is empty")

    # Check cache first
    if use_cache and conversation_id:
        cached = _load_from_cache(conversation_id)
        if cached:
            return cached

    # Generate embedding using the configured provider
    provider = get_embedding_provider()
    if provider == "huggingface":
        embedding = await _call_huggingface_embedding_api(text)
    else:
        embedding = await _call_embedding_api(text)

    # Cache the result
    if conversation_id:
        _save_to_cache(conversation_id, embedding)

    return embedding


async def generate_embeddings_batch(
    texts: List[str],
    conversation_ids: List[str] | None = None,
    use_cache: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single batch.

    More efficient than calling generate_embedding multiple times.

    Args:
        texts: List of text strings to embed
        conversation_ids: Optional list of conversation IDs for caching
        use_cache: Whether to use cached embeddings if available

    Returns:
        List of embedding vectors (one per input text)

    Raises:
        ValueError: If texts list is empty or lengths don't match
        httpx.HTTPError: If API call fails after retries
    """
    if not texts:
        raise ValueError("Cannot generate embeddings: texts list is empty")

    if conversation_ids and len(texts) != len(conversation_ids):
        raise ValueError("texts and conversation_ids must have same length")

    # Check cache for each text
    embeddings = []
    texts_to_generate = []
    indices_to_generate = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Text at index {i} is empty")

        # Check cache if conversation_id provided
        cached = None
        if use_cache and conversation_ids and conversation_ids[i]:
            cached = _load_from_cache(conversation_ids[i])

        if cached:
            embeddings.append(cached)
        else:
            embeddings.append(None)  # Placeholder
            texts_to_generate.append(text)
            indices_to_generate.append(i)

    # Generate embeddings for non-cached texts
    if texts_to_generate:
        provider = get_embedding_provider()
        if provider == "huggingface":
            generated = await _call_huggingface_embedding_api_batch(texts_to_generate)
        else:
            generated = await _call_embedding_api_batch(texts_to_generate)

        # Fill in the generated embeddings
        for idx, embedding in zip(indices_to_generate, generated):
            embeddings[idx] = embedding

            # Cache if conversation_id provided
            if conversation_ids and conversation_ids[idx]:
                _save_to_cache(conversation_ids[idx], embedding)

    return embeddings


# ---------------------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_huggingface_embedding_api(text: str) -> List[float]:
    """
    Call HuggingFace Inference API for a single text embedding.
    Uses nomic-embed-text-v1.5 (same 768D as local Ollama model).
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                HF_MODEL_URL,
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                json={"inputs": text, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()

        result = response.json()
        # HF returns [[768 floats]] for single input
        embedding = result[0] if isinstance(result[0], list) else result

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}D embedding, got {len(embedding)}D"
            )
        return embedding

    except httpx.HTTPError as e:
        raise Exception(f"HuggingFace Embedding API call failed: {e}")
    except Exception as e:
        if "HuggingFace" in str(e):
            raise
        raise Exception(f"HuggingFace embedding generation failed: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_huggingface_embedding_api_batch(texts: List[str]) -> List[List[float]]:
    """
    Call HuggingFace Inference API for multiple texts in a single HTTP call.
    HF supports native batching — send a list of strings, get a list of embeddings.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                HF_MODEL_URL,
                headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()

        result = response.json()  # [[768 floats], [768 floats], ...]

        for i, embedding in enumerate(result):
            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION}D embedding at index {i}, "
                    f"got {len(embedding)}D"
                )
        return result

    except httpx.HTTPError as e:
        raise Exception(f"HuggingFace Embedding API batch call failed: {e}")
    except Exception as e:
        if "HuggingFace" in str(e):
            raise
        raise Exception(f"HuggingFace batch embedding generation failed: {e}")


# ---------------------------------------------------------------------------
# Ollama API (local)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_embedding_api(text: str) -> List[float]:
    """
    Call Ollama Embeddings API (nomic-embed-text) for a single text.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()

        result = response.json()
        embedding = result.get("embedding", [])

        # Validate dimension
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}D embedding, "
                f"got {len(embedding)}D"
            )

        return embedding

    except httpx.HTTPError as e:
        raise Exception(f"Ollama Embeddings API call failed (is 'ollama serve' running?): {e}")
    except Exception as e:
        if "Ollama" in str(e):
            raise
        raise Exception(f"Embedding generation failed: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def _call_embedding_api_batch(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama Embeddings API for multiple texts (sequentially, as Ollama
    does not support batch embedding in a single call).
    """
    embeddings = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={
                        "model": EMBEDDING_MODEL,
                        "prompt": text
                    }
                )
                response.raise_for_status()

                result = response.json()
                embedding = result.get("embedding", [])

                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSION}D embedding, "
                        f"got {len(embedding)}D"
                    )

                embeddings.append(embedding)

            except Exception as e:
                raise Exception(f"Ollama Embeddings API batch call failed: {e}")

    return embeddings


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_from_cache(conversation_id: str) -> List[float] | None:
    """Load cached embedding from disk."""
    cache_file = CACHE_DIR / f"{conversation_id}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("embedding")
    except (json.JSONDecodeError, IOError):
        return None


def _save_to_cache(conversation_id: str, embedding: List[float]) -> None:
    """Save embedding to cache."""
    cache_file = CACHE_DIR / f"{conversation_id}.json"

    data = {
        "embedding": embedding,
        "dimension": len(embedding)
    }

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except IOError:
        pass  # Silently fail if cache write fails


def prepare_text_for_embedding(
    title: str,
    summary: str,
    topics: List[str],
    messages: List[Dict[str, Any]] | None = None,
    max_message_length: int = 2000
) -> str:
    """
    Prepare a text representation of a conversation for embedding.

    Combines title, summary, topics, and optionally message content
    into a single text string optimized for embedding generation.
    """
    parts = []

    if title:
        parts.append(f"Title: {title}")

    if topics:
        topics_str = ", ".join(topics)
        parts.append(f"Topics: {topics_str}")

    if summary:
        parts.append(f"Summary: {summary}")

    if messages:
        message_content = []
        current_length = 0

        for msg in messages:
            content = msg.get("content", "")
            if current_length + len(content) > max_message_length:
                remaining = max_message_length - current_length
                if remaining > 100:
                    message_content.append(content[:remaining] + "...")
                break

            message_content.append(content)
            current_length += len(content)

        if message_content:
            parts.append("Content: " + " ".join(message_content))

    return "\n\n".join(parts)


def clear_cache() -> int:
    """Clear all cached embeddings."""
    count = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except IOError:
            pass

    return count
