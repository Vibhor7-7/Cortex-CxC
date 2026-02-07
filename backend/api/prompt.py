"""
Prompt generation API endpoint.

Takes a set of selected conversation IDs, gathers their summaries and topics,
and sends them to Qwen 2.5 via Ollama to produce a reusable system prompt that
the user can paste into a new ChatGPT (or any LLM) session.
"""

import os
import json
import time

import httpx
from fastapi import APIRouter, HTTPException

from backend.database import get_db_context
from backend.models import Conversation
from backend.schemas import GeneratePromptRequest, GeneratePromptResponse

router = APIRouter(prefix="/api/prompt", tags=["prompt"])

# Ollama configuration (same as summarizer)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


@router.post("/generate", response_model=GeneratePromptResponse)
async def generate_system_prompt(body: GeneratePromptRequest):
    """
    Generate a system prompt from selected conversations.

    Fetches the summaries and topics of the requested conversations, then
    asks Qwen 2.5 to synthesise a concise system prompt that carries all
    the relevant context forward into a fresh chat session.
    """
    start = time.time()

    # --- 1. Fetch conversation data from DB ---
    # Extract all needed fields while session is open to avoid DetachedInstanceError
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

    # --- 3. Call Qwen via Ollama ---
    system_msg = (
        "You are an expert prompt engineer. The user will give you summaries of "
        "their previous AI conversations. Your job is to write a single, clear "
        "system prompt that a user can paste at the start of a **new** ChatGPT "
        "session so the assistant has all the relevant background.\n\n"
        "Guidelines:\n"
        "- Speak in second person (\"You are an assistant that…\").\n"
        "- Weave the key facts, decisions, and preferences from the summaries "
        "into the prompt naturally.\n"
        "- Keep it between 150-400 words — concise but thorough.\n"
        "- Do NOT include JSON, code fences, or markdown headers.\n"
        "- Output ONLY the system prompt text, nothing else."
    )

    user_msg = (
        "Here are summaries of conversations the user wants to carry forward:\n\n"
        f"{context_block}\n\n"
        "Write the system prompt now."
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_msg},
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

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt generation failed: {exc}",
        )

    elapsed_ms = (time.time() - start) * 1000

    return GeneratePromptResponse(
        prompt=content,
        conversations_used=len(conv_data),
        processing_time_ms=round(elapsed_ms, 1),
    )
