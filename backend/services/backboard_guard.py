"""
Backboard.io Relevance Guard Service for CORTEX MCP.

Provides safety filtering to block low-confidence memories before context injection.
Uses Backboard.io's LLM API for fast relevance checking.
"""

import json
import logging
from typing import List, Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.schemas import GuardResult

logger = logging.getLogger(__name__)


# ============================================================================
# Guard Prompt (Optimized for Speed - Single-Hop Reasoning)
# ============================================================================

GUARD_PROMPT = """Is this memory relevant to the user's query? Be strict - only approve truly relevant memories.

Query: {query}
Memory Title: {title}
Memory Summary: {summary}
Memory Topics: {topics}

Return JSON (no other text):
{{"is_relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""


class RelevanceGuard:
    """
    Service for filtering low-confidence memories before MCP context injection.

    Uses Backboard.io to check if a memory is actually relevant to the query,
    blocking memories that are below the confidence threshold.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://app.backboard.io/api",
        model: str = "gpt-4.1",
        threshold: float = 0.5,
        log_blocked: bool = True,
    ):
        """
        Initialize the Relevance Guard.

        Args:
            api_key: Backboard API key (required for guard functionality)
            api_url: Backboard API base URL
            model: LLM model to use for relevance checking
            threshold: Minimum confidence threshold for relevance (0.0-1.0)
            log_blocked: Whether to log blocked memories
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.threshold = max(0.0, min(1.0, threshold))
        self.log_blocked = log_blocked
        self._assistant_id: Optional[str] = None
        self._thread_id: Optional[str] = None

        if not self.api_key:
            logger.warning("Backboard API key not configured - guard will be disabled")

    @property
    def is_available(self) -> bool:
        """Check if the guard is available."""
        return bool(self.api_key)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def _send_to_backboard(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to Backboard.io and get the response.

        Optimized for speed with shorter retry intervals.

        Args:
            prompt: The guard prompt to send

        Returns:
            Parsed JSON response from the LLM
        """
        if not self.api_key:
            return {"is_relevant": True, "confidence": 1.0, "reason": "Guard disabled"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Create thread/assistant if needed (reuse for efficiency)
            if not self._thread_id:
                if not self._assistant_id:
                    assistant_payload = {
                        "name": "CORTEX Guard",
                        "llm_provider": "openai",
                        "llm_model_name": self.model,
                        "instructions": "You are a relevance guard. Only respond with valid JSON. Be strict.",
                    }
                    response = await client.post(
                        f"{self.api_url}/assistants",
                        headers=headers,
                        json=assistant_payload,
                    )
                    response.raise_for_status()
                    self._assistant_id = response.json().get("id")

                thread_response = await client.post(
                    f"{self.api_url}/threads",
                    headers=headers,
                    json={"assistant_id": self._assistant_id},
                )
                thread_response.raise_for_status()
                self._thread_id = thread_response.json().get("id")

            # Send message
            message_payload = {
                "content": prompt,
                "role": "user",
            }

            response = await client.post(
                f"{self.api_url}/threads/{self._thread_id}/messages",
                headers=headers,
                json=message_payload,
            )
            response.raise_for_status()

            # Get response
            messages_response = await client.get(
                f"{self.api_url}/threads/{self._thread_id}/messages",
                headers=headers,
            )
            messages_response.raise_for_status()

            messages = messages_response.json().get("data", [])
            if messages:
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        try:
                            # Parse JSON from response
                            if "```json" in content:
                                json_start = content.find("```json") + 7
                                json_end = content.find("```", json_start)
                                content = content[json_start:json_end].strip()
                            elif "```" in content:
                                json_start = content.find("```") + 3
                                json_end = content.find("```", json_start)
                                content = content[json_start:json_end].strip()

                            return json.loads(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse guard response: {content[:100]}")
                            # Default to allowing if we can't parse
                            return {"is_relevant": True, "confidence": 0.5, "reason": "Parse error"}

            # Default to allowing if no response
            return {"is_relevant": True, "confidence": 0.5, "reason": "No response"}

    async def check_relevance(
        self,
        query: str,
        memory: Dict[str, Any],
    ) -> GuardResult:
        """
        Check if a single memory is relevant to the query.

        Args:
            query: The user's search query
            memory: Memory dict with title, summary, topics fields

        Returns:
            GuardResult with is_relevant, confidence, and reason
        """
        if not self.is_available:
            return GuardResult(
                is_relevant=True,
                confidence=1.0,
                reason="Guard disabled - API key not configured",
            )

        title = memory.get("title", "Untitled")
        summary = memory.get("summary", "No summary available")
        topics = memory.get("topics", [])
        topics_str = ", ".join(topics) if topics else "None"

        prompt = GUARD_PROMPT.format(
            query=query,
            title=title,
            summary=summary,
            topics=topics_str,
        )

        try:
            response = await self._send_to_backboard(prompt)

            is_relevant = response.get("is_relevant", True)
            confidence = float(response.get("confidence", 0.5))
            reason = response.get("reason", "No reason provided")

            # Apply threshold
            if confidence < self.threshold:
                is_relevant = False
                reason = f"Below threshold ({confidence:.2f} < {self.threshold:.2f}): {reason}"

            return GuardResult(
                is_relevant=is_relevant,
                confidence=confidence,
                reason=reason,
            )
        except Exception as e:
            logger.error(f"Guard check failed: {e}")
            # Default to allowing on error to prevent blocking legitimate content
            return GuardResult(
                is_relevant=True,
                confidence=0.5,
                reason=f"Guard error: {str(e)}",
            )

    async def filter_relevant_memories(
        self,
        query: str,
        memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of memories, keeping only those that are relevant.

        Args:
            query: The user's search query
            memories: List of memory dicts to filter

        Returns:
            Filtered list containing only relevant memories
        """
        if not self.is_available or not memories:
            return memories

        filtered = []
        blocked_count = 0

        for memory in memories:
            result = await self.check_relevance(query, memory)

            if result.is_relevant:
                filtered.append(memory)
            else:
                blocked_count += 1
                if self.log_blocked:
                    logger.info(
                        f"[GUARD] Blocked memory: '{memory.get('title', 'Unknown')}' "
                        f"(confidence: {result.confidence:.2f}, reason: {result.reason})"
                    )

        if blocked_count > 0:
            logger.info(f"[GUARD] Filtered {blocked_count} low-confidence memories from {len(memories)} total")

        return filtered


# Singleton instance
_guard_instance: Optional[RelevanceGuard] = None


def get_relevance_guard() -> RelevanceGuard:
    """
    Get the singleton RelevanceGuard instance.

    Initializes with configuration from environment variables.
    """
    global _guard_instance

    if _guard_instance is None:
        import os
        _guard_instance = RelevanceGuard(
            api_key=os.getenv("BACKBOARD_API_KEY"),
            api_url=os.getenv("BACKBOARD_API_URL", "https://app.backboard.io/api"),
            model=os.getenv("BACKBOARD_EVAL_MODEL", "gpt-4.1"),
            threshold=float(os.getenv("BACKBOARD_GUARD_THRESHOLD", "0.5")),
            log_blocked=os.getenv("BACKBOARD_GUARD_LOG_BLOCKED", "true").lower() == "true",
        )

    return _guard_instance
