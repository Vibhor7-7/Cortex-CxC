"""
Integration tests for CORTEX API.

Tests the complete API functionality including:
- FastAPI app initialization
- Health check endpoint
- Database operations
- API endpoints (once implemented)
"""

import unittest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import Base, engine, get_db_context, init_db
from backend.models import Conversation, Message, Embedding


class TestAPIIntegration(unittest.TestCase):
    """Test API integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        # Initialize database (create tables)
        init_db()
        cls.client = TestClient(app)

    def test_app_initialization(self):
        """Test that the FastAPI app initializes correctly."""
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "CORTEX API")
        self.assertEqual(app.version, "1.0.0")

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertIn("description", data)
        self.assertEqual(data["name"], "CORTEX API")

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check required fields
        self.assertIn("status", data)
        self.assertIn("version", data)
        self.assertIn("database_connected", data)
        self.assertIn("ollama_connected", data)
        self.assertIn("chroma_ready", data)

        # Database should be connected
        self.assertTrue(data["database_connected"])

        # Status should be healthy or degraded
        self.assertIn(data["status"], ["healthy", "degraded"])

    def test_openapi_docs(self):
        """Test that OpenAPI documentation is available."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)

    def test_database_connection(self):
        """Test database connection and basic operations."""
        # Test connection
        with engine.connect() as conn:
            self.assertIsNotNone(conn)

        # Test context manager
        with get_db_context() as db:
            self.assertIsNotNone(db)

            # Test query (should return empty list if no data)
            conversations = db.query(Conversation).all()
            self.assertIsInstance(conversations, list)


class TestChatEndpoints(unittest.TestCase):
    """Test chat-related endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        cls.client = TestClient(app)

    def test_get_all_conversations_empty(self):
        """Test getting all conversations when database is empty."""
        response = self.client.get("/api/chats/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Should return a list (may be empty)
        self.assertIsInstance(data, list)

    def test_get_visualization_data_empty(self):
        """Test getting visualization data when database is empty."""
        response = self.client.get("/api/chats/visualization")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check structure
        self.assertIn("nodes", data)
        self.assertIn("total_nodes", data)
        self.assertIn("clusters", data)

        # Should be empty lists
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["clusters"], list)

    def test_get_nonexistent_conversation(self):
        """Test getting a conversation that doesn't exist."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = self.client.get(f"/api/chats/{fake_id}")

        # Should return 404
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)


class TestIngestEndpoint(unittest.TestCase):
    """Test ingestion endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        cls.client = TestClient(app)

    def test_ingest_endpoint_exists(self):
        """Test that ingest endpoint is registered."""
        # Test with no file (should fail but endpoint should exist)
        response = self.client.post("/api/ingest/")

        # Should not be 404 (endpoint exists)
        # Will be 422 (validation error) because no file provided
        self.assertNotEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
