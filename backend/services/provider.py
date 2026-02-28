"""
Provider detection for embedding and chat/summarization backends.

Routes to HuggingFace / Groq (cloud) or Ollama (local) based on env vars.
"""

import os


def get_embedding_provider() -> str:
    """Return 'huggingface' or 'ollama' based on env config."""
    explicit = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if explicit in ("huggingface", "ollama"):
        return explicit
    # Auto-detect: if HF token is set, use HuggingFace
    return "huggingface" if os.getenv("HF_API_TOKEN") else "ollama"


def get_chat_provider() -> str:
    """Return 'groq' or 'ollama' based on env config."""
    explicit = os.getenv("CHAT_PROVIDER", "").lower()
    if explicit in ("groq", "ollama"):
        return explicit
    # Auto-detect: if Groq key is set, use Groq
    return "groq" if os.getenv("GROQ_API_KEY") else "ollama"
