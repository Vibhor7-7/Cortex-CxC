"""
Comprehensive tests for Task 3.1 and 3.2: Local Vector Store & Search

Tests:
Task 3.1: Vector Store Setup & Indexing
- 3.1.1: Store initialisation and persistence
- 3.1.2: Conversation upsert and retrieval
- 3.1.3: Delete and statistics

Task 3.2: Semantic Search
- 3.2.1: Cosine-similarity search
- 3.2.2: Score threshold filtering
- 3.2.3: Result mapping back to conversation metadata

Search API endpoint integration tests.
"""

import os
import sys
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, drop_db, get_db_context
from backend.models import Conversation, Embedding
from backend.services.vector_store import VectorStoreService


class TestTask3_1_VectorStoreSetup(unittest.TestCase):
    """Test Task 3.1: Local Vector Store Setup & Indexing."""

    def setUp(self):
        """Create a temp store for each test."""
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)  # start with no file
        self.service = VectorStoreService(store_path=self.tmpfile.name)

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_initialization_creates_empty_store(self):
        """Test that a new store starts empty."""
        self.assertEqual(self.service.count(), 0)

    def test_upsert_and_retrieve(self):
        """Test upserting a conversation and verifying it exists."""
        self.service.upsert_conversation(
            "conv_001",
            "Title: Test\nSummary: A summary\nTopics: Python",
            [0.1] * 768,
            {"title": "Test", "topic_count": 1},
        )
        self.assertEqual(self.service.count(), 1)

        # Verify on-disk persistence
        with open(self.tmpfile.name) as f:
            data = json.load(f)
        self.assertIn("conv_001", data)
        self.assertEqual(data["conv_001"]["metadata"]["title"], "Test")

    def test_upsert_updates_existing(self):
        """Test that upserting with the same ID overwrites the doc."""
        self.service.upsert_conversation("conv_002", "Version 1", [0.1] * 768, {"v": "1"})
        self.service.upsert_conversation("conv_002", "Version 2", [0.2] * 768, {"v": "2"})

        self.assertEqual(self.service.count(), 1)  # still one doc
        results = self.service.search([0.2] * 768, max_results=1)
        self.assertEqual(results[0]["content"], "Version 2")

    def test_delete_conversation(self):
        """Test deleting a conversation from the store."""
        self.service.upsert_conversation("conv_003", "doc", [0.1] * 768, {})
        self.assertEqual(self.service.count(), 1)
        self.assertTrue(self.service.delete_conversation("conv_003"))
        self.assertEqual(self.service.count(), 0)
        # deleting again returns False
        self.assertFalse(self.service.delete_conversation("conv_003"))

    def test_persistence_across_reloads(self):
        """Test that data survives a new VectorStoreService instance."""
        self.service.upsert_conversation("conv_004", "persisted", [0.5] * 768, {})
        # Create a NEW instance pointing at same file
        reloaded = VectorStoreService(store_path=self.tmpfile.name)
        self.assertEqual(reloaded.count(), 1)
        results = reloaded.search([0.5] * 768, max_results=1)
        self.assertEqual(results[0]["conversation_id"], "conv_004")

    def test_get_stats(self):
        """Test collection statistics."""
        self.service.upsert_conversation("c1", "doc1", [0.1] * 768, {})
        self.service.upsert_conversation("c2", "doc2", [0.2] * 768, {})
        stats = self.service.get_stats()
        self.assertEqual(stats["document_count"], 2)


class TestTask3_2_VectorSearch(unittest.TestCase):
    """Test Task 3.2: Semantic Search via cosine similarity."""

    @classmethod
    def setUpClass(cls):
        """Seed a temp store with three distinct-direction conversations."""
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        cls.tmpfile.close()
        os.unlink(cls.tmpfile.name)
        cls.service = VectorStoreService(store_path=cls.tmpfile.name)

        cls.service.upsert_conversation(
            "conv_python",
            "Discussion about Python programming and data structures",
            [1.0] + [0.0] * 767,
            {"title": "Python Help"},
        )
        cls.service.upsert_conversation(
            "conv_career",
            "Career advice and interview preparation tips",
            [0.0] + [1.0] + [0.0] * 766,
            {"title": "Career Tips"},
        )
        cls.service.upsert_conversation(
            "conv_cooking",
            "Recipes and cooking techniques for beginners",
            [0.0] * 2 + [1.0] + [0.0] * 765,
            {"title": "Cooking 101"},
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tmpfile.name):
            os.unlink(cls.tmpfile.name)

    def test_search_returns_results_sorted_by_similarity(self):
        """Query closest to conv_python should return it first."""
        query = [0.9] + [0.1] * 767
        results = self.service.search(query, max_results=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["conversation_id"], "conv_python")

    def test_search_max_results(self):
        """max_results limits the output length."""
        results = self.service.search([0.5] * 768, max_results=1)
        self.assertEqual(len(results), 1)

    def test_search_score_threshold(self):
        """Only results above the threshold are returned."""
        query = [1.0] + [0.0] * 767
        results = self.service.search(query, max_results=10, score_threshold=0.95)
        conv_ids = [r["conversation_id"] for r in results]
        self.assertIn("conv_python", conv_ids)
        self.assertNotIn("conv_career", conv_ids)

    def test_search_result_structure(self):
        """Each result must have the expected keys."""
        results = self.service.search([0.5] * 768, max_results=1)
        r = results[0]
        self.assertIn("conversation_id", r)
        self.assertIn("score", r)
        self.assertIn("content", r)
        self.assertIn("metadata", r)
        self.assertIsInstance(r["score"], float)

    def test_search_empty_store(self):
        """Search on empty store returns empty list."""
        empty_svc = VectorStoreService(store_path="/tmp/_cortex_empty_test.json")
        results = empty_svc.search([0.5] * 768)
        self.assertEqual(results, [])
        if os.path.exists("/tmp/_cortex_empty_test.json"):
            os.unlink("/tmp/_cortex_empty_test.json")


class TestTask3_SearchEndpoint(unittest.TestCase):
    """Test the /api/search/ endpoint wired to the local vector store."""

    @classmethod
    def setUpClass(cls):
        drop_db()
        init_db()
        cls.client = TestClient(app)

    @patch("backend.api.search.search_store")
    @patch("backend.api.search.generate_embedding")
    def test_search_endpoint_empty(self, mock_embed, mock_search):
        """200 with zero results when the store is empty."""
        async def _embed(text, use_cache=False):
            return [0.1] * 768
        mock_embed.side_effect = _embed

        async def _search(**kw):
            return []
        mock_search.side_effect = _search

        resp = self.client.post("/api/search/", json={"query": "Python", "limit": 10})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["query"], "Python")
        self.assertEqual(data["total_results"], 0)

    @patch("backend.api.search.get_vector_store_service")
    def test_search_stats_endpoint(self, mock_get_svc):
        mock_svc = Mock()
        mock_svc.get_stats.return_value = {
            "collection_name": ".vector_store",
            "document_count": 5,
            "store_path": "/tmp/test.json",
        }
        mock_get_svc.return_value = mock_svc

        resp = self.client.get("/api/search/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["document_count"], 5)


class TestTask3_IntegrationWithIngestion(unittest.TestCase):
    """Test integration of vector store with the ingestion pipeline."""

    @classmethod
    def setUpClass(cls):
        db_path = "backend/cortex.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        init_db()
        cls.client = TestClient(app)

    @patch("backend.api.ingest.upsert_conversation_to_store")
    @patch("backend.api.ingest.generate_embedding")
    @patch("backend.api.ingest.summarize_conversation")
    def test_ingestion_upserts_to_store(self, mock_summarize, mock_embed, mock_upsert):
        """Ingestion pipeline should call upsert_conversation_to_store."""
        async def _summarize(messages):
            return "Test summary", ["Test topic"]
        mock_summarize.side_effect = _summarize

        async def _embed(text, conversation_id=None):
            return [0.1] * 768
        mock_embed.side_effect = _embed

        mock_upsert.return_value = None

        html = """
        <html>
        <head><title>ChatGPT - Test</title></head>
        <body>
            <div data-message-author-role="user"><p>Test message</p></div>
            <div data-message-author-role="assistant"><p>Test response</p></div>
        </body>
        </html>
        """
        files = {"file": ("test.html", BytesIO(html.encode()), "text/html")}
        resp = self.client.post("/api/ingest/", files=files)

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        mock_upsert.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
