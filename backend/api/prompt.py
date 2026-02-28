"""
Prompt generation API endpoint.

Takes a set of selected conversation IDs, gathers their summaries and topics,
and sends them to the active chat provider (Groq or Ollama/Qwen) to produce
a reusable system prompt that the user can paste into a new ChatGPT session.
"""

import os
import json
import time

import httpx
from fastapi import APIRouter, HTTPException

from backend.database import get_db_context
from backend.models import Conversation
from backend.schemas import GeneratePromptRequest, GeneratePromptResponse
from backend.services.provider import get_chat_provider

router = APIRouter(prefix="/api/prompt", tags=["prompt"])

# Ollama configuration (same as summarizer)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Shared prompt text
PROMPT_SYSTEM_MSG = (
    "You are an expert prompt engineer. The user will give you summaries of "
    "their previous AI conversations. Your job is to write a single, clear "
    "system prompt that a user can paste at the start of a **new** ChatGPT "
    "session so the assistant has all the relevant background.\n\n"
    "Guidelines:\n"
    '- Speak in second person ("You are an assistant that\u2026").\n'
    "- Weave the key facts, decisions, and preferences from the summaries "
    "into the prompt naturally.\n"
    "- Keep it between 150-400 words \u2014 concise but thorough.\n"
    "- Do NOT include JSON, code fences, or markdown headers.\n"
    "- Output ONLY the system prompt text, nothing else."
)


@router.post("/generate", response_model=GeneratePromptResponse)
async def generate_system_prompt(body: GeneratePromptRequest):
    """
    Generate a system prompt from selected conversations.

    Fetches the summaries and topics of the requested conversations, then
    asks the active chat provider to synthesise a concise system prompt.
    """
    start = time.time()

    # --- 1. Fetch conversation data from DB ---
    conv_data: list[dict] = []
    with get_db_context() as db:
        conversations = (
            db.query(Conversation)
            .filter(Conversation.id.in_(body.conversation_ids))
            .all()
        )
        for c in conversations:
            conv_data.append({
                "title": c.title,
                "summary": c.summary,
                "topics": list(c.topics) if c.topics else [],
            })

    if not conv_data:
        raise HTTPException(status_code=404, detail="None of the requested conversations were found")

    # --- 2. Build context block for the LLM ---
    context_parts: list[str] = []
    for c in conv_data:
        topics_str = ", ".join(c["topics"]) if c["topics"] else "general"
        context_parts.append(
            f"Title: {c['title']}\n"
            f"Topics: {topics_str}\n"
            f"Summary: {c['summary'] or 'No summary available'}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    user_msg = (
        "Here are summaries of conversations the user wants to carry forward:\n\n"
        f"{context_block}\n\n"
        "Write the system prompt now."
    )

    # --- 3. Call the active chat provider ---
    provider = get_chat_provider()
    try:
        if provider == "groq":
            content = await _call_groq_prompt(user_msg)
        else:
            content = await _call_ollama_prompt(user_msg)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed ({provider}): {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt generation failed ({provider}): {exc}",
        )

    elapsed_ms = (time.time() - start) * 1000

    return GeneratePromptResponse(
        prompt=content,
        conversations_used=len(conv_data),
        processing_time_ms=round(elapsed_ms, 1),
    )


async def _call_groq_prompt(user_msg: str) -> str:
    """Call Groq API for prompt generation (plain text, not JSON)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.5,
                "max_tokens": 800,
            },
        )
        resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"].strip()
    if not content:
        raise ValueError("Empty response from Groq")
    return content


async def _call_ollama_prompt(user_msg: str) -> str:
    """Call Ollama/Qwen for prompt generation."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 800,
                },
            },
        )
        resp.raise_for_status()

    content = resp.json().get("message", {}).get("content", "").strip()
    if not content:
        raise ValueError("Empty response from Ollama")
    return content
