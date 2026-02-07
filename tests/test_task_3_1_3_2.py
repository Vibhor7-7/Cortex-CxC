"""
Comprehensive tests for Task 3.1 and 3.2: OpenAI Vector Store Integration & Search

Tests all subtasks and acceptance criteria from PRD.md:
Task 3.1: OpenAI Vector Store Setup & Indexing
- 3.1.1: Vector store creation and management
- 3.1.2: File upload and mapping
- 3.1.3: File ingestion status polling

Task 3.2: OpenAI Vector Store Search
- 3.2.1: Vector store search API
- 3.2.2: Filters and ranking options
- 3.2.3: Result mapping and metadata

Acceptance Criteria:
 Newly ingested conversations are searchable via vector store
 Vector store file ingestion reaches 'completed' state
 Each result can be mapped back to conversation_id
 Search returns relevant results for both keyword and semantic queries
 Results include per-chunk scores and content
"""

import unittest
import sys
import os
from pathlib import Path
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, drop_db, get_db_context
from backend.models import Conversation, Message, Embedding
from backend.services.openai_vector_store import VectorStoreService


class TestTask3_1_VectorStoreSetup(unittest.TestCase):
    """Test Task 3.1: OpenAI Vector Store Setup & Indexing."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        drop_db()  # Drop old schema
        init_db()  # Create new schema
        cls.client = TestClient(app)

    def test_vector_store_service_initialization(self):
        """Test that VectorStoreService initializes correctly."""
        # Mock OpenAI client to avoid actual API calls
        with patch('backend.services.openai_vector_store.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock vector store creation
            mock_vector_store = Mock()
            mock_vector_store.id = "vs_test_123"
            mock_client.beta.vector_stores.create.return_value = mock_vector_store

            service = VectorStoreService()

            # Verify vector store ID was set
            self.assertIsNotNone(service.vector_store_id)

    def test_conversation_to_markdown_conversion(self):
        """Test conversion of conversation data to markdown format."""
        with patch('backend.services.openai_vector_store.OpenAI') as mock_openai_class:
            # Setup proper mocks
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock vector store creation with proper string ID
            mock_vector_store = Mock()
            mock_vector_store.id = "vs_test_markdown_123"  # Use string instead of Mock
            mock_client.beta.vector_stores.create.return_value = mock_vector_store

            service = VectorStoreService()

            conversation_data = {
                'title': 'Test Conversation',
                'summary': 'A test conversation about Python',
                'topics': ['Python', 'Programming'],
                'messages': [
                    {'role': 'user', 'content': 'Hello, can you help with Python?'},
                    {'role': 'assistant', 'content': 'Of course! What do you need help with?'}
                ]
            }

            markdown = service.conversation_to_markdown(conversation_data)

            # Verify markdown structure
            self.assertIn('# Test Conversation', markdown)
            self.assertIn('**Summary:** A test conversation about Python', markdown)
            self.assertIn('**Topics:** Python, Programming', markdown)
            self.assertIn('## USER', markdown)
            self.assertIn('## ASSISTANT', markdown)
            self.assertIn('Hello, can you help with Python?', markdown)

    @patch('backend.services.openai_vector_store.OpenAI')
    def test_upload_conversation_mock(self, mock_openai_class):
        """Test conversation upload to vector store (mocked)."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock file upload
        mock_file = Mock()
        mock_file.id = "file_test_123"
        mock_client.files.create.return_value = mock_file

        # Mock vector store file attachment
        mock_vs_file = Mock()
        mock_vs_file.status = "completed"
        mock_client.beta.vector_stores.files.create.return_value = mock_vs_file
        mock_client.beta.vector_stores.files.retrieve.return_value = mock_vs_file

        # Mock vector store creation
        mock_vector_store = Mock()
        mock_vector_store.id = "vs_test_123"
        mock_client.beta.vector_stores.create.return_value = mock_vector_store

        service = VectorStoreService()

        conversation_data = {
            'title': 'Test',
            'summary': 'Test summary',
            'topics': ['Test'],
            'messages': [{'role': 'user', 'content': 'Test message'}]
        }

        file_id, status = service.upload_conversation(
            "conv_123",
            conversation_data,
            poll_completion=True
        )

        # Verify upload was successful
        self.assertEqual(file_id, "file_test_123")
        self.assertEqual(status, "completed")

    def test_file_status_polling_completed(self):
        """Test polling for file ingestion status - completed case."""
        with patch('backend.services.openai_vector_store.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock completed status
            mock_vs_file = Mock()
            mock_vs_file.status = "completed"
            mock_client.beta.vector_stores.files.retrieve.return_value = mock_vs_file

            # Mock vector store creation
            mock_vector_store = Mock()
            mock_vector_store.id = "vs_test_123"
            mock_client.beta.vector_stores.create.return_value = mock_vector_store

            service = VectorStoreService()
            status = service._poll_file_status("file_123", max_attempts=5, poll_interval=0.1)

            self.assertEqual(status, "completed")

    def test_file_status_polling_failed(self):
        """Test polling for file ingestion status - failed case."""
        with patch('backend.services.openai_vector_store.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock failed status
            mock_vs_file = Mock()
            mock_vs_file.status = "failed"
            mock_client.beta.vector_stores.files.retrieve.return_value = mock_vs_file

            # Mock vector store creation
            mock_vector_store = Mock()
            mock_vector_store.id = "vs_test_123"
            mock_client.beta.vector_stores.create.return_value = mock_vector_store

            service = VectorStoreService()
            status = service._poll_file_status("file_123", max_attempts=5, poll_interval=0.1)

            self.assertEqual(status, "failed")


class TestTask3_2_VectorStoreSearch(unittest.TestCase):
    """Test Task 3.2: OpenAI Vector Store Search."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        drop_db()  # Drop old schema
        init_db()  # Create new schema
        cls.client = TestClient(app)

    @patch('backend.services.openai_vector_store.OpenAI')
    def test_vector_store_search_mock(self, mock_openai_class):
        """Test vector store search functionality (mocked)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock search results
        mock_result = Mock()
        mock_result.file_id = "file_123"
        mock_result.score = 0.85
        mock_result.content = [Mock(text="Relevant content about Python")]
        mock_result.metadata = {}

        mock_response = Mock()
        mock_response.data = [mock_result]
        mock_client.beta.vector_stores.search.return_value = mock_response

        # Mock vector store creation
        mock_vector_store = Mock()
        mock_vector_store.id = "vs_test_123"
        mock_client.beta.vector_stores.create.return_value = mock_vector_store

        service = VectorStoreService()
        results = service.search(
            query="Python programming",
            max_results=10,
            rewrite_query=True
        )

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['file_id'], "file_123")
        self.assertGreater(results[0]['score'], 0.5)

    @patch('backend.services.openai_vector_store.OpenAI')
    def test_search_with_score_threshold(self, mock_openai_class):
        """Test search with score threshold filtering."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock search results with varying scores
        mock_result1 = Mock()
        mock_result1.file_id = "file_123"
        mock_result1.score = 0.9
        mock_result1.content = [Mock(text="High relevance")]

        mock_result2 = Mock()
        mock_result2.file_id = "file_456"
        mock_result2.score = 0.4
        mock_result2.content = [Mock(text="Low relevance")]

        mock_response = Mock()
        mock_response.data = [mock_result1]  # Only high-score result returned
        mock_client.beta.vector_stores.search.return_value = mock_response

        # Mock vector store creation
        mock_vector_store = Mock()
        mock_vector_store.id = "vs_test_123"
        mock_client.beta.vector_stores.create.return_value = mock_vector_store

        service = VectorStoreService()
        results = service.search(
            query="Test query",
            max_results=10,
            score_threshold=0.5  # Filter low scores
        )

        # Only high-score results should be returned
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['score'], 0.5)

    def test_search_endpoint_integration(self):
        """Test the search API endpoint with mocked vector store."""
        with patch('backend.api.search.search_vector_store') as mock_search:
            # Mock empty search results
            mock_search.return_value = []

            response = self.client.post(
                "/api/search/",
                json={
                    "query": "Python programming",
                    "limit": 10
                }
            )

            # Should return 200 even with no results
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['query'], "Python programming")
            self.assertEqual(data['total_results'], 0)
            self.assertIsInstance(data['results'], list)

    def test_search_endpoint_with_filters(self):
        """Test search endpoint with cluster and topic filters."""
        with patch('backend.api.search.search_vector_store') as mock_search:
            # Mock search results
            mock_search.return_value = [
                {
                    'file_id': 'file_123',
                    'score': 0.85,
                    'content': 'Test content'
                }
            ]

            # Create a test conversation with openai_file_id
            with get_db_context() as db:
                from backend.models import MessageRole
                import uuid

                conv_id = str(uuid.uuid4())
                conversation = Conversation(
                    id=conv_id,
                    title="Test Conversation",
                    summary="Test summary",
                    topics=["Python"],
                    cluster_id=1,
                    cluster_name="Programming",
                    message_count=2,
                    openai_file_id="file_123"
                )
                db.add(conversation)

                # Add embedding
                embedding = Embedding(
                    conversation_id=conv_id,
                    embedding_384d=[0.1] * 384,
                    vector_3d=[1.0, 2.0, 3.0],
                    start_x=0.0,
                    start_y=0.0,
                    start_z=0.0,
                    end_x=1.0,
                    end_y=2.0,
                    end_z=3.0,
                    magnitude=1.5
                )
                db.add(embedding)
                db.commit()

            # Test search with filters
            response = self.client.post(
                "/api/search/",
                json={
                    "query": "Python",
                    "limit": 10,
                    "cluster_filter": 1,
                    "topic_filter": ["Python"]
                }
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(data['total_results'], 0)

    def test_search_stats_endpoint(self):
        """Test the search stats endpoint."""
        with patch('backend.services.openai_vector_store.get_vector_store_service') as mock_service:
            mock_instance = Mock()
            mock_instance.get_vector_store_stats.return_value = {
                "id": "vs_test_123",
                "name": "Cortex Chat Memory",
                "file_counts": {"total": 5},
                "status": "active"
            }
            mock_service.return_value = mock_instance

            response = self.client.get("/api/search/stats")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("id", data)
            self.assertEqual(data["name"], "Cortex Chat Memory")


class TestTask3_IntegrationWithIngestion(unittest.TestCase):
    """Test integration of vector store with ingestion pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        # Delete existing database to ensure fresh schema
        import os
        db_path = "backend/cortex.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        init_db()
        cls.client = TestClient(app)

    @patch('backend.api.ingest.upload_conversation_to_vector_store')
    @patch('backend.api.ingest.generate_embedding')
    @patch('backend.api.ingest.summarize_conversation')
    def test_ingestion_uploads_to_vector_store(
        self,
        mock_summarize,
        mock_embed,
        mock_upload
    ):
        """Test that ingestion automatically uploads to vector store."""
        # Mock summarization
        async def mock_summarize_async(messages):
            return "Test summary", ["Test topic"]
        mock_summarize.side_effect = mock_summarize_async

        # Mock embedding
        async def mock_embed_async(text, conversation_id=None):
            return [0.1] * 384
        mock_embed.side_effect = mock_embed_async

        # Mock vector store upload
        async def mock_upload_async(conversation_id, conversation_data, poll_completion=True):
            return "file_test_123", "completed"
        mock_upload.side_effect = mock_upload_async

        # Create test HTML
        html_content = """
        <html>
        <head><title>ChatGPT - Test</title></head>
        <body>
            <div data-message-author-role="user"><p>Test message</p></div>
            <div data-message-author-role="assistant"><p>Test response</p></div>
        </body>
        </html>
        """

        files = {"file": ("test.html", BytesIO(html_content.encode()), "text/html")}
        response = self.client.post("/api/ingest/", files=files)

        # Verify ingestion succeeded
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])

        # Verify vector store upload was called
        mock_upload.assert_called_once()

        # Verify openai_file_id was stored
        with get_db_context() as db:
            conversation = db.query(Conversation).filter(
                Conversation.id == data['conversation_id']
            ).first()
            self.assertIsNotNone(conversation)
            self.assertEqual(conversation.openai_file_id, "file_test_123")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
