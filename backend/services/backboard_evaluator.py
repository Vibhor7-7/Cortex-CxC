"""
Backboard.io Evaluator Service for CORTEX.

Provides retrieval quality evaluation by sending (query, retrieved_chats, summaries)
to Backboard.io's LLM API to score:
- Relevance: How well each result matches the query
- Redundancy: Overlap between retrieved results
- Coverage: How completely results address the query
"""

import json
import time
import logging
from typing import List, Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.schemas import (
    SearchResultItem,
    RetrievalEvaluation,
    RelevanceScore,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Prompts
# ============================================================================

RELEVANCE_PROMPT = """You are evaluating the relevance of retrieved memories to a user query.
For each memory, score its relevance from 0.0 to 1.0 where:
- 1.0 = Highly relevant, directly addresses the query
- 0.7-0.9 = Relevant, contains useful information for the query
- 0.4-0.6 = Partially relevant, tangentially related
- 0.1-0.3 = Slightly relevant, shares some keywords but not useful
- 0.0 = Not relevant at all

Query: {query}

Retrieved Memories:
{memories}

Evaluate each memory and return a JSON object with this exact structure:
{{
  "scores": [
    {{"id": "conversation_id_1", "score": 0.85, "reason": "brief explanation"}},
    {{"id": "conversation_id_2", "score": 0.60, "reason": "brief explanation"}}
  ]
}}

Be strict in your scoring. Only give high scores to truly relevant results."""

REDUNDANCY_PROMPT = """You are evaluating the redundancy between retrieved memories.
Redundancy measures how much overlap exists between the results.

Query: {query}

Retrieved Memories:
{memories}

Score the overall redundancy from 0.0 to 1.0 where:
- 0.0 = No redundancy, all results are unique and complementary
- 0.3-0.5 = Some overlap, a few results cover similar ground
- 0.6-0.8 = High redundancy, many results cover the same information
- 1.0 = Complete redundancy, all results say the same thing

Return a JSON object with this exact structure:
{{
  "redundancy_score": 0.35,
  "reason": "brief explanation of the overlap found"
}}"""

COVERAGE_PROMPT = """You are evaluating how completely the retrieved memories address a query.
Coverage measures whether the results together provide a comprehensive answer.

Query: {query}

Retrieved Memories:
{memories}

Score the overall coverage from 0.0 to 1.0 where:
- 1.0 = Complete coverage, results fully address all aspects of the query
- 0.7-0.9 = Good coverage, most aspects are addressed
- 0.4-0.6 = Partial coverage, some aspects are missing
- 0.1-0.3 = Poor coverage, most aspects are not addressed
- 0.0 = No coverage, results don't address the query at all

Return a JSON object with this exact structure:
{{
  "coverage_score": 0.75,
  "reason": "brief explanation of what aspects are covered or missing"
}}"""


class BackboardEvaluator:
    """
    Service for evaluating retrieval quality using Backboard.io's LLM API.

    Provides scoring for relevance, redundancy, and coverage of search results.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://app.backboard.io/api",
        model: str = "gpt-4.1",
    ):
        """
        Initialize the Backboard evaluator.

        Args:
            api_key: Backboard API key (required for evaluation)
            api_url: Backboard API base URL
            model: LLM model to use for evaluation
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.model = model
        self._assistant_id: Optional[str] = None
        self._thread_id: Optional[str] = None

        if not self.api_key:
            logger.warning("Backboard API key not configured - evaluation will be skipped")

    @property
    def is_available(self) -> bool:
        """Check if Backboard is available for evaluation."""
        return bool(self.api_key)

    def _format_memories_for_prompt(self, results: List[SearchResultItem]) -> str:
        """Format search results for inclusion in evaluation prompts."""
        formatted = []
        for i, r in enumerate(results, 1):
            memory_text = f"""
Memory {i}:
- ID: {r.conversation_id}
- Title: {r.title}
- Summary: {r.summary or 'No summary available'}
- Topics: {', '.join(r.topics) if r.topics else 'None'}
- Preview: {r.message_preview or 'No preview available'}
"""
            formatted.append(memory_text.strip())
        return "\n\n".join(formatted)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def _send_to_backboard(self, prompt: str) -> Dict[str, Any]:
        """
        Send a prompt to Backboard.io and get the response.

        Args:
            prompt: The evaluation prompt to send

        Returns:
            Parsed JSON response from the LLM
        """
        if not self.api_key:
            return {}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create a thread if we don't have one
        async with httpx.AsyncClient(timeout=30.0) as client:
            if not self._thread_id:
                # Create assistant if needed
                if not self._assistant_id:
                    assistant_payload = {
                        "name": "CORTEX Evaluator",
                        "llm_provider": "openai",
                        "llm_model_name": self.model,
                        "instructions": "You are a retrieval quality evaluator. Always respond with valid JSON.",
                    }
                    response = await client.post(
                        f"{self.api_url}/assistants",
                        headers=headers,
                        json=assistant_payload,
                    )
                    response.raise_for_status()
                    self._assistant_id = response.json().get("id")

                # Create thread
                thread_response = await client.post(
                    f"{self.api_url}/threads",
                    headers=headers,
                    json={"assistant_id": self._assistant_id},
                )
                thread_response.raise_for_status()
                self._thread_id = thread_response.json().get("id")

            # Send message and get response
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

            # Get the assistant's response
            messages_response = await client.get(
                f"{self.api_url}/threads/{self._thread_id}/messages",
                headers=headers,
            )
            messages_response.raise_for_status()

            messages = messages_response.json().get("data", [])
            if messages:
                # Get the latest assistant message
                for msg in messages:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        try:
                            # Try to parse JSON from the response
                            # Handle cases where JSON is wrapped in markdown code blocks
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
                            logger.warning(f"Failed to parse JSON from Backboard response: {content[:200]}")
                            return {}

            return {}

    async def evaluate_relevance(
        self,
        query: str,
        results: List[SearchResultItem],
    ) -> List[RelevanceScore]:
        """
        Evaluate the relevance of each search result to the query.

        Args:
            query: The search query
            results: List of search results to evaluate

        Returns:
            List of RelevanceScore objects with scores and reasons
        """
        if not results or not self.is_available:
            return []

        memories_text = self._format_memories_for_prompt(results)
        prompt = RELEVANCE_PROMPT.format(query=query, memories=memories_text)

        try:
            response = await self._send_to_backboard(prompt)
            scores = []
            for score_data in response.get("scores", []):
                scores.append(RelevanceScore(
                    conversation_id=score_data.get("id", ""),
                    score=float(score_data.get("score", 0.0)),
                    reason=score_data.get("reason", "No reason provided"),
                ))
            return scores
        except Exception as e:
            logger.error(f"Failed to evaluate relevance: {e}")
            return []

    async def evaluate_redundancy(
        self,
        query: str,
        results: List[SearchResultItem],
    ) -> float:
        """
        Evaluate the redundancy between search results.

        Args:
            query: The search query
            results: List of search results to evaluate

        Returns:
            Redundancy score from 0.0 to 1.0
        """
        if not results or not self.is_available:
            return 0.0

        memories_text = self._format_memories_for_prompt(results)
        prompt = REDUNDANCY_PROMPT.format(query=query, memories=memories_text)

        try:
            response = await self._send_to_backboard(prompt)
            return float(response.get("redundancy_score", 0.0))
        except Exception as e:
            logger.error(f"Failed to evaluate redundancy: {e}")
            return 0.0

    async def evaluate_coverage(
        self,
        query: str,
        results: List[SearchResultItem],
    ) -> float:
        """
        Evaluate how completely the results address the query.

        Args:
            query: The search query
            results: List of search results to evaluate

        Returns:
            Coverage score from 0.0 to 1.0
        """
        if not results or not self.is_available:
            return 0.0

        memories_text = self._format_memories_for_prompt(results)
        prompt = COVERAGE_PROMPT.format(query=query, memories=memories_text)

        try:
            response = await self._send_to_backboard(prompt)
            return float(response.get("coverage_score", 0.0))
        except Exception as e:
            logger.error(f"Failed to evaluate coverage: {e}")
            return 0.0

    async def evaluate_retrieval(
        self,
        query: str,
        results: List[SearchResultItem],
    ) -> RetrievalEvaluation:
        """
        Perform full retrieval evaluation including relevance, redundancy, and coverage.

        Args:
            query: The search query
            results: List of search results to evaluate

        Returns:
            RetrievalEvaluation with all scores and metrics
        """
        start_time = time.time()

        if not results or not self.is_available:
            return RetrievalEvaluation(
                relevance_scores=[],
                redundancy_score=0.0,
                coverage_score=0.0,
                evaluation_time_ms=0.0,
            )

        try:
            # Run all evaluations (could parallelize for speed)
            relevance_scores = await self.evaluate_relevance(query, results)
            redundancy_score = await self.evaluate_redundancy(query, results)
            coverage_score = await self.evaluate_coverage(query, results)

            evaluation_time = (time.time() - start_time) * 1000

            return RetrievalEvaluation(
                relevance_scores=relevance_scores,
                redundancy_score=redundancy_score,
                coverage_score=coverage_score,
                evaluation_time_ms=evaluation_time,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate retrieval: {e}")
            evaluation_time = (time.time() - start_time) * 1000
            return RetrievalEvaluation(
                relevance_scores=[],
                redundancy_score=0.0,
                coverage_score=0.0,
                evaluation_time_ms=evaluation_time,
            )


# Singleton instance
_evaluator_instance: Optional[BackboardEvaluator] = None


def get_backboard_evaluator() -> BackboardEvaluator:
    """
    Get the singleton BackboardEvaluator instance.

    Initializes with configuration from environment variables.
    """
    global _evaluator_instance

    if _evaluator_instance is None:
        import os
        _evaluator_instance = BackboardEvaluator(
            api_key=os.getenv("BACKBOARD_API_KEY"),
            api_url=os.getenv("BACKBOARD_API_URL", "https://app.backboard.io/api"),
            model=os.getenv("BACKBOARD_EVAL_MODEL", "gpt-4.1"),
        )

    return _evaluator_instance


def rerank_by_scores(
    results: List[SearchResultItem],
    relevance_scores: List[RelevanceScore],
) -> List[SearchResultItem]:
    """
    Rerank search results based on Backboard relevance scores.

    Args:
        results: Original search results
        relevance_scores: Relevance scores from Backboard evaluation

    Returns:
        Results reordered by relevance score, with scores attached
    """
    # Create a mapping of conversation_id to relevance score
    score_map = {s.conversation_id: s for s in relevance_scores}

    # Attach relevance scores to results
    for result in results:
        if result.conversation_id in score_map:
            score_data = score_map[result.conversation_id]
            result.relevance_score = score_data.score
            result.relevance_reason = score_data.reason

    # Sort by relevance score (descending), falling back to original score
    return sorted(
        results,
        key=lambda r: (r.relevance_score or 0.0, r.score),
        reverse=True,
    )
