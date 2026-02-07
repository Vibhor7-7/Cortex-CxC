"""
Comprehensive tests for Task 2.3: Ingest API Endpoint

Tests all subtasks and acceptance criteria from PRD.md:
- 2.3.1: Single file ingestion
- 2.3.2: Batch ingestion
- 2.3.3: Re-clustering functionality
- 2.3.4: Progress tracking (optional - not implemented)
- 2.3.5: Error handling

Acceptance Criteria:
✅ Successfully ingest sample ChatGPT HTML file
✅ Conversation appears in database with all fields populated
✅ 3D coordinates are generated and stored
✅ Error cases return appropriate HTTP status codes
"""

import unittest
import sys
from pathlib import Path
from io import BytesIO

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, get_db_context
from backend.models import Conversation, Message, Embedding


class TestTask2_3_1_SingleIngestion(unittest.TestCase):
    """Test Task 2.3.1: Single file ingestion."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        init_db()
        cls.client = TestClient(app)

    def test_ingest_valid_html_file(self):
        """Test ingesting a valid ChatGPT HTML file."""
        # Create a minimal valid ChatGPT HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChatGPT - Test Conversation</title>
        </head>
        <body>
            <div class="conversation">
                <div data-message-author-role="user">
                    <p>Hello, how are you?</p>
                </div>
                <div data-message-author-role="assistant">
                    <p>I'm doing well, thank you!</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Create file upload
        files = {
            "file": ("test.html", BytesIO(html_content.encode()), "text/html")
        }

        # Make request
        response = self.client.post("/api/ingest/", files=files)

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["conversation_id"])
        self.assertEqual(data["title"], "Test Conversation")
        self.assertEqual(data["message_count"], 2)
        self.assertIsNone(data["error"])

        # Verify conversation in database
        with get_db_context() as db:
            conversation = db.query(Conversation).filter(
                Conversation.id == data["conversation_id"]
            ).first()

            self.assertIsNotNone(conversation)
            self.assertEqual(conversation.title, "Test Conversation")
            self.assertEqual(conversation.message_count, 2)

            # Verify messages exist
            messages = db.query(Message).filter(
                Message.conversation_id == data["conversation_id"]
            ).all()
            self.assertEqual(len(messages), 2)

            # Verify embedding exists
            embedding = db.query(Embedding).filter(
                Embedding.conversation_id == data["conversation_id"]
            ).first()
            self.assertIsNotNone(embedding)
            self.assertIsNotNone(embedding.embedding_384d)
            self.assertEqual(len(embedding.embedding_384d), 384)

    def test_ingest_real_chatgpt_export(self):
        """Test ingesting the real ChatGPT export file (570KB)."""
        # Load the real test file
        test_file = Path(__file__).parent / "chat.html"
        if not test_file.exists():
            self.skipTest("chat.html test file not found")

        with open(test_file, "rb") as f:
            files = {"file": ("chat.html", f, "text/html")}
            response = self.client.post("/api/ingest/", files=files)

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify success
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["conversation_id"])
        self.assertGreater(data["message_count"], 0)

        # Verify in database
        with get_db_context() as db:
            conversation = db.query(Conversation).filter(
                Conversation.id == data["conversation_id"]
            ).first()

            self.assertIsNotNone(conversation)
            self.assertIsNotNone(conversation.summary)
            self.assertGreater(len(conversation.topics), 0)


class TestTask2_3_2_BatchIngestion(unittest.TestCase):
    """Test Task 2.3.2: Batch ingestion."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        init_db()
        cls.client = TestClient(app)

    def test_batch_ingest_multiple_files(self):
        """Test ingesting multiple files in batch."""
        # Create multiple test HTML files
        html_template = """
        <html>
        <head><title>ChatGPT - Conversation {}</title></head>
        <body>
            <div data-message-author-role="user"><p>Test message {}</p></div>
            <div data-message-author-role="assistant"><p>Response {}</p></div>
        </body>
        </html>
        """

        files = [
            ("files", ("test1.html", BytesIO(html_template.format(1, 1, 1).encode()), "text/html")),
            ("files", ("test2.html", BytesIO(html_template.format(2, 2, 2).encode()), "text/html")),
            ("files", ("test3.html", BytesIO(html_template.format(3, 3, 3).encode()), "text/html")),
        ]

        # Make request with auto_reprocess disabled for speed
        response = self.client.post(
            "/api/ingest/batch",
            files=files,
            data={"auto_reprocess": "false"}
        )

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify batch response
        self.assertEqual(data["total_processed"], 3)
        self.assertEqual(data["successful"], 3)
        self.assertEqual(data["failed"], 0)
        self.assertEqual(len(data["conversations"]), 3)

        # Verify all conversations are in database
        with get_db_context() as db:
            for conv_data in data["conversations"]:
                if conv_data["success"]:
                    conversation = db.query(Conversation).filter(
                        Conversation.id == conv_data["conversation_id"]
                    ).first()
                    self.assertIsNotNone(conversation)


class TestTask2_3_3_Reprocessing(unittest.TestCase):
    """Test Task 2.3.3: Re-clustering functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        init_db()
        cls.client = TestClient(app)

        # Ingest multiple conversations first
        html_template = """
        <html>
        <head><title>ChatGPT - Topic {}</title></head>
        <body>
            <div data-message-author-role="user"><p>{}</p></div>
            <div data-message-author-role="assistant"><p>Response about {}</p></div>
        </body>
        </html>
        """

        topics = [
            ("Python", "Tell me about Python programming"),
            ("JavaScript", "Tell me about JavaScript"),
            ("Cooking", "How do I cook pasta?"),
            ("Exercise", "What are good exercises?"),
            ("Travel", "Where should I travel?"),
        ]

        for topic_name, user_msg in topics:
            html = html_template.format(topic_name, user_msg, topic_name)
            files = {"file": (f"{topic_name}.html", BytesIO(html.encode()), "text/html")}
            cls.client.post("/api/ingest/", files=files, data={"auto_reprocess": "false"})

    def test_reprocess_generates_3d_coordinates(self):
        """Test that reprocessing generates 3D coordinates."""
        # Trigger reprocessing
        response = self.client.post("/api/ingest/reprocess")

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertTrue(data["success"])
        self.assertGreater(data["conversations_processed"], 0)
        self.assertGreater(data["conversations_updated"], 0)
        self.assertGreater(data["n_clusters"], 0)
        self.assertIn("cluster_statistics", data)

        # Verify embeddings have non-zero 3D coordinates
        with get_db_context() as db:
            embeddings = db.query(Embedding).all()

            # At least some should have non-zero coordinates
            non_zero_count = sum(
                1 for emb in embeddings
                if emb.end_x != 0 or emb.end_y != 0 or emb.end_z != 0
            )
            self.assertGreater(non_zero_count, 0)

    def test_reprocess_assigns_clusters(self):
        """Test that reprocessing assigns cluster IDs and names."""
        # Trigger reprocessing
        response = self.client.post("/api/ingest/reprocess")
        self.assertEqual(response.status_code, 200)

        # Verify conversations have cluster assignments
        with get_db_context() as db:
            conversations = db.query(Conversation).all()

            for conv in conversations:
                self.assertIsNotNone(conv.cluster_id)
                self.assertIsNotNone(conv.cluster_name)
                self.assertNotEqual(conv.cluster_name, "Unclustered")

    def test_reprocess_with_too_few_conversations(self):
        """Test that reprocessing fails gracefully with < 2 conversations."""
        # Clear database
        with get_db_context() as db:
            db.query(Embedding).delete()
            db.query(Message).delete()
            db.query(Conversation).delete()
            db.commit()

        # Try to reprocess with no conversations
        response = self.client.post("/api/ingest/reprocess")

        # Should return 422
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("at least 2 conversations", data["detail"].lower())


class TestTask2_3_5_ErrorHandling(unittest.TestCase):
    """Test Task 2.3.5: Error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        init_db()
        cls.client = TestClient(app)

    def test_invalid_file_format_returns_400(self):
        """Test that non-HTML files return 400 Bad Request."""
        # Create a non-HTML file
        files = {"file": ("test.txt", BytesIO(b"Not HTML"), "text/plain")}

        response = self.client.post("/api/ingest/", files=files)

        # Should return 400
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("HTML", data["detail"])

    def test_empty_conversation_returns_422(self):
        """Test that empty conversations return 422 Unprocessable Entity."""
        # Create HTML with no messages
        html_content = """
        <html>
        <head><title>ChatGPT - Empty</title></head>
        <body><div></div></body>
        </html>
        """

        files = {"file": ("empty.html", BytesIO(html_content.encode()), "text/html")}
        response = self.client.post("/api/ingest/", files=files)

        # Should return 422
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("empty", data["detail"].lower())

    def test_unrecognized_format_returns_422(self):
        """Test that unrecognized HTML format returns 422."""
        # Create HTML that doesn't match ChatGPT or Claude format
        html_content = """
        <html>
        <head><title>Random HTML</title></head>
        <body><p>This is not a chat export</p></body>
        </html>
        """

        files = {"file": ("random.html", BytesIO(html_content.encode()), "text/html")}
        response = self.client.post("/api/ingest/", files=files)

        # Should return 422
        self.assertEqual(response.status_code, 422)

    def test_malformed_html_handled_gracefully(self):
        """Test that malformed HTML is handled gracefully."""
        # Create malformed HTML
        html_content = "<html><div>Incomplete"

        files = {"file": ("malformed.html", BytesIO(html_content.encode()), "text/html")}
        response = self.client.post("/api/ingest/", files=files)

        # Should return error status (not crash)
        self.assertIn(response.status_code, [400, 422, 500])


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
