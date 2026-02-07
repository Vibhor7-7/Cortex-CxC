"""
Integration tests for Backboard.io integration in CORTEX.

Tests:
- BackboardEvaluator service
- RelevanceGuard service
- Graceful degradation when Backboard unavailable
- Search API with evaluation
- MCP server with guard filtering
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict

# Configure anyio for async tests
pytestmark = pytest.mark.anyio

from backend.schemas import (
    SearchResultItem,
    RetrievalEvaluation,
    RelevanceScore,
    GuardResult,
    SearchRequest,
    SearchResponse,
)
from backend.services.backboard_evaluator import (
    BackboardEvaluator,
    get_backboard_evaluator,
    rerank_by_scores,
)
from backend.services.backboard_guard import (
    RelevanceGuard,
    get_relevance_guard,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_search_results() -> List[SearchResultItem]:
    """Create sample search results for testing."""
    return [
        SearchResultItem(
            conversation_id="conv-001",
            title="Machine Learning Basics",
            summary="Introduction to neural networks and deep learning concepts.",
            topics=["machine learning", "neural networks", "AI"],
            cluster_id=1,
            cluster_name="Tech",
            score=0.85,
            message_preview="Let's discuss how neural networks work...",
        ),
        SearchResultItem(
            conversation_id="conv-002",
            title="Python Web Development",
            summary="Building web apps with Flask and Django.",
            topics=["python", "web development", "flask"],
            cluster_id=1,
            cluster_name="Tech",
            score=0.72,
            message_preview="Flask is a micro web framework...",
        ),
        SearchResultItem(
            conversation_id="conv-003",
            title="Cooking Italian Pasta",
            summary="Traditional Italian pasta recipes and techniques.",
            topics=["cooking", "italian", "pasta"],
            cluster_id=2,
            cluster_name="Lifestyle",
            score=0.45,
            message_preview="To make authentic carbonara...",
        ),
    ]


@pytest.fixture
def sample_memory_dicts() -> List[Dict]:
    """Create sample memory dicts for guard testing."""
    return [
        {
            "conversation_id": "conv-001",
            "title": "Machine Learning Basics",
            "summary": "Introduction to neural networks and deep learning concepts.",
            "topics": ["machine learning", "neural networks", "AI"],
        },
        {
            "conversation_id": "conv-002",
            "title": "Python Web Development",
            "summary": "Building web apps with Flask and Django.",
            "topics": ["python", "web development", "flask"],
        },
        {
            "conversation_id": "conv-003",
            "title": "Cooking Italian Pasta",
            "summary": "Traditional Italian pasta recipes and techniques.",
            "topics": ["cooking", "italian", "pasta"],
        },
    ]


@pytest.fixture
def mock_backboard_relevance_response():
    """Mock Backboard relevance evaluation response."""
    return {
        "scores": [
            {"id": "conv-001", "score": 0.92, "reason": "Directly about ML and neural networks"},
            {"id": "conv-002", "score": 0.65, "reason": "Python related but not ML focused"},
            {"id": "conv-003", "score": 0.15, "reason": "Unrelated topic about cooking"},
        ]
    }


@pytest.fixture
def mock_backboard_redundancy_response():
    """Mock Backboard redundancy evaluation response."""
    return {"redundancy_score": 0.25, "reason": "Some overlap in tech topics"}


@pytest.fixture
def mock_backboard_coverage_response():
    """Mock Backboard coverage evaluation response."""
    return {"coverage_score": 0.78, "reason": "Good coverage of ML concepts"}


# ============================================================================
# BackboardEvaluator Tests
# ============================================================================

class TestBackboardEvaluator:
    """Tests for BackboardEvaluator service."""

    def test_evaluator_initialization_without_api_key(self):
        """Test evaluator initializes without API key (disabled state)."""
        evaluator = BackboardEvaluator(api_key=None)
        assert evaluator.is_available is False

    def test_evaluator_initialization_with_api_key(self):
        """Test evaluator initializes with API key (enabled state)."""
        evaluator = BackboardEvaluator(api_key="test-key-123")
        assert evaluator.is_available is True
        assert evaluator.api_key == "test-key-123"

    def test_format_memories_for_prompt(self, sample_search_results):
        """Test memory formatting for prompts."""
        evaluator = BackboardEvaluator(api_key="test")
        formatted = evaluator._format_memories_for_prompt(sample_search_results)

        assert "Machine Learning Basics" in formatted
        assert "conv-001" in formatted
        assert "neural networks" in formatted
        assert "Memory 1:" in formatted
        assert "Memory 2:" in formatted

    async def test_evaluate_retrieval_without_api_key(self, sample_search_results):
        """Test that evaluation returns empty result when API key not set."""
        evaluator = BackboardEvaluator(api_key=None)
        result = await evaluator.evaluate_retrieval("machine learning", sample_search_results)

        assert isinstance(result, RetrievalEvaluation)
        assert result.relevance_scores == []
        assert result.redundancy_score == 0.0
        assert result.coverage_score == 0.0

    async def test_evaluate_retrieval_with_empty_results(self):
        """Test evaluation with empty results."""
        evaluator = BackboardEvaluator(api_key="test")
        result = await evaluator.evaluate_retrieval("test query", [])

        assert isinstance(result, RetrievalEvaluation)
        assert result.relevance_scores == []

    async def test_evaluate_relevance_with_mock(
        self, sample_search_results, mock_backboard_relevance_response
    ):
        """Test relevance evaluation with mocked Backboard response."""
        evaluator = BackboardEvaluator(api_key="test")

        with patch.object(
            evaluator, "_send_to_backboard", return_value=mock_backboard_relevance_response
        ):
            scores = await evaluator.evaluate_relevance("machine learning", sample_search_results)

            assert len(scores) == 3
            assert scores[0].conversation_id == "conv-001"
            assert scores[0].score == 0.92
            assert "ML" in scores[0].reason

    async def test_evaluate_redundancy_with_mock(
        self, sample_search_results, mock_backboard_redundancy_response
    ):
        """Test redundancy evaluation with mocked Backboard response."""
        evaluator = BackboardEvaluator(api_key="test")

        with patch.object(
            evaluator, "_send_to_backboard", return_value=mock_backboard_redundancy_response
        ):
            score = await evaluator.evaluate_redundancy("machine learning", sample_search_results)

            assert score == 0.25

    async def test_evaluate_coverage_with_mock(
        self, sample_search_results, mock_backboard_coverage_response
    ):
        """Test coverage evaluation with mocked Backboard response."""
        evaluator = BackboardEvaluator(api_key="test")

        with patch.object(
            evaluator, "_send_to_backboard", return_value=mock_backboard_coverage_response
        ):
            score = await evaluator.evaluate_coverage("machine learning", sample_search_results)

            assert score == 0.78


class TestReranking:
    """Tests for result reranking based on Backboard scores."""

    def test_rerank_by_scores(self, sample_search_results):
        """Test that results are reranked based on relevance scores."""
        relevance_scores = [
            RelevanceScore(conversation_id="conv-001", score=0.5, reason="Medium relevance"),
            RelevanceScore(conversation_id="conv-002", score=0.9, reason="High relevance"),
            RelevanceScore(conversation_id="conv-003", score=0.3, reason="Low relevance"),
        ]

        reranked = rerank_by_scores(sample_search_results, relevance_scores)

        # conv-002 should be first (highest relevance score)
        assert reranked[0].conversation_id == "conv-002"
        assert reranked[0].relevance_score == 0.9

        # conv-001 should be second
        assert reranked[1].conversation_id == "conv-001"
        assert reranked[1].relevance_score == 0.5

        # conv-003 should be last
        assert reranked[2].conversation_id == "conv-003"
        assert reranked[2].relevance_score == 0.3

    def test_rerank_preserves_original_order_without_scores(self, sample_search_results):
        """Test that original order is preserved when no scores provided."""
        reranked = rerank_by_scores(sample_search_results, [])

        assert reranked[0].conversation_id == "conv-001"
        assert reranked[1].conversation_id == "conv-002"
        assert reranked[2].conversation_id == "conv-003"


# ============================================================================
# RelevanceGuard Tests
# ============================================================================

class TestRelevanceGuard:
    """Tests for RelevanceGuard service."""

    def test_guard_initialization_without_api_key(self):
        """Test guard initializes without API key (disabled state)."""
        guard = RelevanceGuard(api_key=None)
        assert guard.is_available is False

    def test_guard_initialization_with_api_key(self):
        """Test guard initializes with API key (enabled state)."""
        guard = RelevanceGuard(api_key="test-key-123", threshold=0.6)
        assert guard.is_available is True
        assert guard.threshold == 0.6

    def test_threshold_bounds(self):
        """Test threshold is bounded between 0 and 1."""
        guard_low = RelevanceGuard(api_key="test", threshold=-0.5)
        assert guard_low.threshold == 0.0

        guard_high = RelevanceGuard(api_key="test", threshold=1.5)
        assert guard_high.threshold == 1.0

    async def test_check_relevance_without_api_key(self, sample_memory_dicts):
        """Test that guard allows all when API key not set."""
        guard = RelevanceGuard(api_key=None)
        result = await guard.check_relevance("test query", sample_memory_dicts[0])

        assert isinstance(result, GuardResult)
        assert result.is_relevant is True
        assert "disabled" in result.reason.lower()

    async def test_check_relevance_with_mock_relevant(self, sample_memory_dicts):
        """Test relevance check with mocked relevant response."""
        guard = RelevanceGuard(api_key="test", threshold=0.5)

        mock_response = {"is_relevant": True, "confidence": 0.85, "reason": "Highly relevant"}

        with patch.object(guard, "_send_to_backboard", return_value=mock_response):
            result = await guard.check_relevance("machine learning", sample_memory_dicts[0])

            assert result.is_relevant is True
            assert result.confidence == 0.85

    async def test_check_relevance_with_mock_not_relevant(self, sample_memory_dicts):
        """Test relevance check with mocked not relevant response."""
        guard = RelevanceGuard(api_key="test", threshold=0.5)

        mock_response = {"is_relevant": False, "confidence": 0.2, "reason": "Unrelated topic"}

        with patch.object(guard, "_send_to_backboard", return_value=mock_response):
            result = await guard.check_relevance("machine learning", sample_memory_dicts[2])

            assert result.is_relevant is False
            assert result.confidence == 0.2

    async def test_check_relevance_below_threshold(self, sample_memory_dicts):
        """Test that low confidence results are marked as not relevant."""
        guard = RelevanceGuard(api_key="test", threshold=0.7)

        mock_response = {"is_relevant": True, "confidence": 0.5, "reason": "Marginally relevant"}

        with patch.object(guard, "_send_to_backboard", return_value=mock_response):
            result = await guard.check_relevance("machine learning", sample_memory_dicts[0])

            # Should be marked not relevant due to low confidence
            assert result.is_relevant is False
            assert "threshold" in result.reason.lower()

    async def test_filter_relevant_memories(self, sample_memory_dicts):
        """Test filtering a list of memories."""
        guard = RelevanceGuard(api_key="test", threshold=0.5, log_blocked=False)

        async def mock_check(query, memory):
            # Only ML related is relevant
            if "Machine Learning" in memory.get("title", ""):
                return GuardResult(is_relevant=True, confidence=0.9, reason="Relevant")
            return GuardResult(is_relevant=False, confidence=0.3, reason="Not relevant")

        with patch.object(guard, "check_relevance", side_effect=mock_check):
            filtered = await guard.filter_relevant_memories("machine learning", sample_memory_dicts)

            # Only ML memory should remain
            assert len(filtered) == 1
            assert filtered[0]["title"] == "Machine Learning Basics"

    async def test_filter_preserves_all_without_api_key(self, sample_memory_dicts):
        """Test that all memories are preserved when guard is disabled."""
        guard = RelevanceGuard(api_key=None)
        filtered = await guard.filter_relevant_memories("test", sample_memory_dicts)

        assert len(filtered) == len(sample_memory_dicts)


# ============================================================================
# Graceful Degradation Tests
# ============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation when Backboard is unavailable."""

    async def test_evaluator_handles_api_error(self, sample_search_results):
        """Test evaluator handles API errors gracefully."""
        evaluator = BackboardEvaluator(api_key="test")

        with patch.object(
            evaluator, "_send_to_backboard", side_effect=Exception("API Error")
        ):
            result = await evaluator.evaluate_retrieval("test", sample_search_results)

            # Should return empty evaluation, not raise
            assert isinstance(result, RetrievalEvaluation)
            assert result.relevance_scores == []

    async def test_guard_handles_api_error(self, sample_memory_dicts):
        """Test guard allows content when API errors occur."""
        guard = RelevanceGuard(api_key="test")

        with patch.object(
            guard, "_send_to_backboard", side_effect=Exception("API Error")
        ):
            result = await guard.check_relevance("test", sample_memory_dicts[0])

            # Should allow on error (fail open)
            assert result.is_relevant is True
            assert "error" in result.reason.lower()

    async def test_guard_filter_handles_api_error(self, sample_memory_dicts):
        """Test guard filter handles errors gracefully by returning allowed results."""
        guard = RelevanceGuard(api_key="test")

        # When check_relevance returns an error result (is_relevant=True, error in reason)
        # the filter should allow the memory through
        async def mock_check_returns_error_result(query, memory):
            return GuardResult(
                is_relevant=True,
                confidence=0.5,
                reason="Guard error: API Error"
            )

        with patch.object(guard, "check_relevance", side_effect=mock_check_returns_error_result):
            filtered = await guard.filter_relevant_memories("test", sample_memory_dicts)

            # When guard returns error results with is_relevant=True, all should pass
            assert len(filtered) == len(sample_memory_dicts)


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemas:
    """Tests for Backboard-related schemas."""

    def test_relevance_score_validation(self):
        """Test RelevanceScore schema validation."""
        score = RelevanceScore(
            conversation_id="test-id",
            score=0.85,
            reason="Very relevant"
        )
        assert score.score == 0.85
        assert score.conversation_id == "test-id"

    def test_relevance_score_bounds(self):
        """Test RelevanceScore rejects out-of-bounds values."""
        with pytest.raises(ValueError):
            RelevanceScore(conversation_id="test", score=1.5, reason="Invalid")

        with pytest.raises(ValueError):
            RelevanceScore(conversation_id="test", score=-0.1, reason="Invalid")

    def test_retrieval_evaluation_schema(self):
        """Test RetrievalEvaluation schema."""
        evaluation = RetrievalEvaluation(
            relevance_scores=[
                RelevanceScore(conversation_id="test", score=0.8, reason="Good")
            ],
            redundancy_score=0.2,
            coverage_score=0.9,
            evaluation_time_ms=150.5
        )
        assert len(evaluation.relevance_scores) == 1
        assert evaluation.redundancy_score == 0.2
        assert evaluation.coverage_score == 0.9

    def test_guard_result_schema(self):
        """Test GuardResult schema."""
        result = GuardResult(
            is_relevant=True,
            confidence=0.85,
            reason="Relevant content"
        )
        assert result.is_relevant is True
        assert result.confidence == 0.85

    def test_search_request_with_evaluate(self):
        """Test SearchRequest with evaluate parameter."""
        request = SearchRequest(
            query="machine learning",
            evaluate=True
        )
        assert request.evaluate is True

        request_default = SearchRequest(query="test")
        assert request_default.evaluate is False


# ============================================================================
# Integration with Search API Tests
# ============================================================================

class TestSearchAPIIntegration:
    """Tests for Backboard integration with search API."""

    def test_search_result_item_with_relevance(self):
        """Test SearchResultItem includes relevance fields."""
        item = SearchResultItem(
            conversation_id="test",
            title="Test",
            summary="Test summary",
            topics=["test"],
            score=0.8,
            relevance_score=0.9,
            relevance_reason="Highly relevant"
        )
        assert item.relevance_score == 0.9
        assert item.relevance_reason == "Highly relevant"

    def test_search_response_with_evaluation(self):
        """Test SearchResponse includes evaluation field."""
        evaluation = RetrievalEvaluation(
            relevance_scores=[
                RelevanceScore(conversation_id="test", score=0.85, reason="Good")
            ],
            redundancy_score=0.1,
            coverage_score=0.9,
            evaluation_time_ms=100.0
        )

        response = SearchResponse(
            query="test",
            results=[],
            total_results=0,
            search_time_ms=50.0,
            evaluation=evaluation
        )

        assert response.evaluation is not None
        assert response.evaluation.coverage_score == 0.9


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_evaluator_with_single_result(self):
        """Test evaluation with only one result."""
        evaluator = BackboardEvaluator(api_key="test")
        single_result = [
            SearchResultItem(
                conversation_id="only-one",
                title="Single Result",
                summary="Only result",
                topics=["test"],
                score=0.9,
            )
        ]

        mock_response = {
            "scores": [{"id": "only-one", "score": 0.8, "reason": "Relevant"}]
        }

        with patch.object(evaluator, "_send_to_backboard", return_value=mock_response):
            scores = await evaluator.evaluate_relevance("test", single_result)
            assert len(scores) == 1

    async def test_guard_with_empty_memory_fields(self):
        """Test guard handles memories with empty fields."""
        guard = RelevanceGuard(api_key="test", threshold=0.3)

        empty_memory = {
            "conversation_id": "empty",
            "title": "",
            "summary": "",
            "topics": [],
        }

        mock_response = {"is_relevant": True, "confidence": 0.5, "reason": "Cannot determine"}

        with patch.object(guard, "_send_to_backboard", return_value=mock_response):
            result = await guard.check_relevance("test", empty_memory)
            assert isinstance(result, GuardResult)

    def test_rerank_with_partial_scores(self, sample_search_results):
        """Test reranking when only some results have scores."""
        partial_scores = [
            RelevanceScore(conversation_id="conv-001", score=0.7, reason="Good"),
            # conv-002 and conv-003 missing
        ]

        reranked = rerank_by_scores(sample_search_results, partial_scores)

        # conv-001 has score, others don't
        assert reranked[0].relevance_score == 0.7
        assert reranked[1].relevance_score is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
