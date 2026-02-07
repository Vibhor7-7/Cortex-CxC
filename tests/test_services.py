"""
Unit tests for services module.

Tests all service components:
- normalizer
- summarizer (with mocked Ollama calls)
- embedder (with mocked OpenAI calls)
- dimensionality_reducer
- clusterer
"""

import unittest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Import services
from backend.services.normalizer import (
    normalize_conversation,
    _clean_message,
    _generate_title,
    _extract_timestamp,
    validate_normalized_conversation
)

from backend.services.summarizer import (
    summarize_conversation,
    _format_conversation,
    generate_cache_key,
    clear_cache as clear_summary_cache
)

from backend.services.embedder import (
    generate_embedding,
    generate_embeddings_batch,
    prepare_text_for_embedding,
    clear_cache as clear_embedding_cache
)

from backend.services.dimensionality_reducer import (
    fit_umap_model,
    reduce_embeddings,
    normalize_coordinates,
    refit_and_reduce_all
)

from backend.services.clusterer import (
    cluster_conversations,
    predict_clusters,
    generate_cluster_names_from_topics,
    get_cluster_statistics
)


class TestNormalizer(unittest.TestCase):
    """Test conversation normalization."""
    
    def test_normalize_conversation_with_title(self):
        """Test normalizing a conversation with existing title."""
        parsed_data = {
            "title": "Test Conversation",
            "timestamp": "2026-02-04 10:30:00"
        }
        messages = [
            {"role": "user", "content": "Hello", "sequence_number": 0},
            {"role": "assistant", "content": "Hi there!", "sequence_number": 1}
        ]
        
        result = normalize_conversation(parsed_data, messages)
        
        self.assertEqual(result["title"], "Test Conversation")
        self.assertEqual(result["message_count"], 2)
        self.assertEqual(len(result["messages"]), 2)
        self.assertIsInstance(result["created_at"], datetime)
    
    def test_normalize_conversation_generates_title(self):
        """Test title generation from first user message."""
        parsed_data = {}
        messages = [
            {"role": "user", "content": "How do I learn Python?", "sequence_number": 0},
            {"role": "assistant", "content": "Start with basics", "sequence_number": 1}
        ]
        
        result = normalize_conversation(parsed_data, messages)
        
        self.assertEqual(result["title"], "How do I learn Python?")
    
    def test_normalize_conversation_empty_messages(self):
        """Test error handling for empty messages."""
        with self.assertRaises(ValueError):
            normalize_conversation({}, [])
    
    def test_clean_message(self):
        """Test message cleaning and validation."""
        # Valid message
        msg = {"role": "user", "content": "  Test  content  ", "sequence_number": 0}
        cleaned = _clean_message(msg)
        self.assertEqual(cleaned["content"], "Test content")
        
        # Invalid role
        msg = {"role": "invalid", "content": "Test"}
        self.assertIsNone(_clean_message(msg))
        
        # Empty content
        msg = {"role": "user", "content": "   "}
        self.assertIsNone(_clean_message(msg))
    
    def test_generate_title_truncation(self):
        """Test title truncation for long messages."""
        messages = [
            {
                "role": "user",
                "content": "A" * 100,
                "sequence_number": 0
            }
        ]
        
        title = _generate_title(None, messages)
        self.assertEqual(len(title), 50)
        self.assertTrue(title.endswith("..."))
    
    def test_extract_timestamp(self):
        """Test timestamp parsing."""
        # Valid timestamp
        ts = _extract_timestamp("2026-02-04 10:30:00")
        self.assertIsInstance(ts, datetime)
        self.assertEqual(ts.year, 2026)
        
        # Invalid timestamp returns current time
        ts = _extract_timestamp("invalid")
        self.assertIsInstance(ts, datetime)
    
    def test_validate_normalized_conversation(self):
        """Test validation of normalized conversation."""
        valid = {
            "title": "Test",
            "messages": [{"role": "user", "content": "Hi"}],
            "message_count": 1,
            "created_at": datetime.now()
        }
        self.assertTrue(validate_normalized_conversation(valid))
        
        # Missing field
        invalid = valid.copy()
        del invalid["title"]
        self.assertFalse(validate_normalized_conversation(invalid))


class TestSummarizer(unittest.TestCase):
    """Test LLM-based summarization with mocked Ollama/Qwen 2.5 calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.messages = [
            {"role": "user", "content": "How do I learn Python?"},
            {"role": "assistant", "content": "Start with the basics like variables and functions."},
            {"role": "user", "content": "What about data structures?"},
            {"role": "assistant", "content": "Learn lists, dictionaries, and sets."}
        ]
    
    @patch('backend.services.summarizer.httpx.AsyncClient')
    def test_summarize_conversation(self, mock_async_client):
        """Test conversation summarization via Ollama."""
        # Mock Ollama HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "content": json.dumps({
                    "summary": "Discussion about learning Python and data structures.",
                    "topics": ["Python", "Learning", "Data Structures"]
                })
            }
        }
        
        # Set up the async context manager mock
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Run async test
        async def run_test():
            return await summarize_conversation(self.messages, use_cache=False)
        
        summary, topics = asyncio.run(run_test())
        
        self.assertIsInstance(summary, str)
        self.assertIsInstance(topics, list)
        self.assertGreater(len(summary), 10)
        self.assertGreater(len(topics), 0)
    
    def test_format_conversation(self):
        """Test conversation formatting."""
        formatted = _format_conversation(self.messages)
        
        self.assertIn("USER:", formatted)
        self.assertIn("ASSISTANT:", formatted)
        self.assertIn("How do I learn Python?", formatted)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = generate_cache_key(self.messages)
        key2 = generate_cache_key(self.messages)
        
        # Same messages should produce same key
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 64)  # SHA256 hash length


class TestEmbedder(unittest.TestCase):
    """Test embedding generation with mocked Ollama/nomic-embed-text calls."""
    
    @patch('backend.services.embedder.httpx.AsyncClient')
    def test_generate_embedding(self, mock_async_client):
        """Test single embedding generation via Ollama."""
        # Mock Ollama HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Run async test
        async def run_test():
            return await generate_embedding("Test text", use_cache=False)
        
        embedding = asyncio.run(run_test())
        
        self.assertEqual(len(embedding), 768)
        self.assertIsInstance(embedding[0], float)
    
    @patch('backend.services.embedder.httpx.AsyncClient')
    def test_generate_embeddings_batch(self, mock_async_client):
        """Test batch embedding generation via Ollama."""
        # Mock Ollama HTTP response (called once per text)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_async_client.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Run async test
        async def run_test():
            return await generate_embeddings_batch(["Text 1", "Text 2"], use_cache=False)
        
        embeddings = asyncio.run(run_test())
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 768)
        self.assertEqual(len(embeddings[1]), 768)
    
    def test_prepare_text_for_embedding(self):
        """Test text preparation for embedding."""
        text = prepare_text_for_embedding(
            title="Test Conversation",
            summary="This is a test summary.",
            topics=["Python", "Testing"],
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        
        self.assertIn("Title: Test Conversation", text)
        self.assertIn("Topics: Python, Testing", text)
        self.assertIn("Summary: This is a test summary.", text)
        self.assertIn("Hello", text)


class TestDimensionalityReducer(unittest.TestCase):
    """Test UMAP dimensionality reduction."""
    
    def test_fit_umap_model(self):
        """Test UMAP model fitting."""
        # Create sample embeddings
        embeddings = [[float(i) for _ in range(384)] for i in range(10)]
        
        model = fit_umap_model(embeddings, save_model=False)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.n_components, 3)
    
    def test_reduce_embeddings(self):
        """Test embedding reduction."""
        # Create and fit model
        embeddings = [[float(i) for _ in range(384)] for i in range(10)]
        model = fit_umap_model(embeddings, save_model=False)
        
        # Reduce embeddings
        results = reduce_embeddings(embeddings, model=model)
        
        self.assertEqual(len(results), 10)
        self.assertEqual(len(results[0]["vector_3d"]), 3)
        self.assertIn("start_x", results[0])
        self.assertIn("magnitude", results[0])
    
    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        coords_list = [
            {"vector_3d": [1.0, 2.0, 3.0]},
            {"vector_3d": [4.0, 5.0, 6.0]}
        ]
        
        normalized = normalize_coordinates(coords_list, scale=10.0)
        
        self.assertEqual(len(normalized), 2)
        # Check that coordinates are scaled
        self.assertNotEqual(normalized[0]["vector_3d"], [1.0, 2.0, 3.0])


class TestClusterer(unittest.TestCase):
    """Test K-means clustering."""
    
    def test_cluster_conversations(self):
        """Test conversation clustering."""
        # Create sample 3D coordinates
        coords_3d = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [5.0, 6.0, 7.0],
            [5.1, 6.1, 7.1],
            [10.0, 11.0, 12.0]
        ]
        
        results = cluster_conversations(coords_3d, n_clusters=3, save_model=False)
        
        self.assertEqual(len(results), 5)
        self.assertIn("cluster_id", results[0])
        self.assertIn("cluster_name", results[0])
        self.assertIn("color", results[0])
        
        # Check color format
        self.assertTrue(results[0]["color"].startswith("#"))
    
    def test_generate_cluster_names_from_topics(self):
        """Test cluster name generation from topics."""
        cluster_assignments = [
            {"cluster_id": 0},
            {"cluster_id": 0},
            {"cluster_id": 1},
            {"cluster_id": 1}
        ]
        
        all_topics = [
            ["Python", "Coding"],
            ["Python", "Programming"],
            ["Career", "Jobs"],
            ["Career", "Interview"]
        ]
        
        names = generate_cluster_names_from_topics(cluster_assignments, all_topics)
        
        self.assertIn(0, names)
        self.assertIn(1, names)
        self.assertIn("Python", names[0])
        self.assertIn("Career", names[1])
    
    def test_get_cluster_statistics(self):
        """Test cluster statistics calculation."""
        assignments = [
            {"cluster_id": 0, "cluster_name": "Python", "color": "#9333ea"},
            {"cluster_id": 0, "cluster_name": "Python", "color": "#9333ea"},
            {"cluster_id": 1, "cluster_name": "Career", "color": "#3b82f6"},
        ]
        
        stats = get_cluster_statistics(assignments)
        
        self.assertEqual(stats[0]["count"], 2)
        self.assertEqual(stats[1]["count"], 1)
        self.assertAlmostEqual(stats[0]["percentage"], 66.7, places=1)
        self.assertAlmostEqual(stats[1]["percentage"], 33.3, places=1)


if __name__ == "__main__":
    unittest.main()
