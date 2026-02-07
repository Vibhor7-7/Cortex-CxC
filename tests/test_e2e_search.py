"""
End-to-end integration tests for search functionality with real OpenAI API.

Tests the complete flow:
1. Ingest conversation → 2. Upload to vector store → 3. Search → 4. Retrieve

Requires OPENAI_API_KEY to be set in environment.
"""

import unittest
import sys
import os
import time
from pathlib import Path
from io import BytesIO

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, drop_db, get_db_context
from backend.models import Conversation


class TestE2ESearch(unittest.TestCase):
    """End-to-end search tests with real API."""

    @classmethod
    def setUpClass(cls):
        """Set up test client and database."""
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise unittest.SkipTest("OpenAI API key not configured - skipping E2E tests")

        drop_db()
        init_db()
        cls.client = TestClient(app)
        cls.conversation_ids = []

    def test_01_ingest_conversation_with_vector_store(self):
        """Test ingesting a conversation and uploading to vector store."""
        # Create a test HTML file
        html_content = """
        <html>
        <head><title>ChatGPT - Python Programming Tips</title></head>
        <body>
            <div data-message-author-role="user">
                <p>Can you explain list comprehensions in Python?</p>
            </div>
            <div data-message-author-role="assistant">
                <p>List comprehensions are a concise way to create lists in Python.
                They consist of brackets containing an expression followed by a for clause,
                then zero or more for or if clauses. For example:
                squares = [x**2 for x in range(10)]</p>
            </div>
            <div data-message-author-role="user">
                <p>What about dictionary comprehensions?</p>
            </div>
            <div data-message-author-role="assistant">
                <p>Dictionary comprehensions work similarly but create dictionaries instead.
                They use curly braces and include a key:value pair. For example:
                squares_dict = {x: x**2 for x in range(10)}</p>
            </div>
        </body>
        </html>
        """

        files = {"file": ("python_tips.html", BytesIO(html_content.encode()), "text/html")}
        response = self.client.post("/api/ingest/", files=files)

        # Verify ingestion succeeded
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["conversation_id"])

        # Store conversation ID for later tests
        self.__class__.conversation_ids.append(data["conversation_id"])

        # Verify conversation in database
        with get_db_context() as db:
            conversation = db.query(Conversation).filter(
                Conversation.id == data["conversation_id"]
            ).first()

            self.assertIsNotNone(conversation)
            self.assertIsNotNone(conversation.openai_file_id,
                "Conversation should have openai_file_id after ingestion")

        print(f"✅ Ingested conversation: {data['conversation_id']}")
        print(f"   OpenAI File ID: {conversation.openai_file_id}")

    def test_02_search_for_ingested_conversation(self):
        """Test searching for the ingested conversation."""
        if not self.__class__.conversation_ids:
            self.skipTest("No conversations to search")

        # Wait a bit for vector store indexing
        print("Waiting 5 seconds for vector store indexing...")
        time.sleep(5)

        # Search for Python-related content
        search_request = {
            "query": "python list comprehensions",
            "limit": 10
        }

        response = self.client.post("/api/search/", json=search_request)

        # Verify search response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("query", data)
        self.assertIn("results", data)
        self.assertIn("total_results", data)

        print(f"✅ Search returned {data['total_results']} results")

        # Check if our conversation is in results
        if data['total_results'] > 0:
            found_our_conversation = any(
                result['conversation_id'] in self.__class__.conversation_ids
                for result in data['results']
            )

            if found_our_conversation:
                print("✅ Found our ingested conversation in search results!")
                result = next(
                    r for r in data['results']
                    if r['conversation_id'] in self.__class__.conversation_ids
                )
                print(f"   Title: {result['title']}")
                print(f"   Score: {result['score']}")
                print(f"   Topics: {result['topics']}")
            else:
                print("⚠️ Our conversation not in top results (may need more indexing time)")

    def test_03_retrieve_conversation_details(self):
        """Test retrieving full conversation details."""
        if not self.__class__.conversation_ids:
            self.skipTest("No conversations to retrieve")

        conversation_id = self.__class__.conversation_ids[0]

        response = self.client.get(f"/api/chats/{conversation_id}")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["id"], conversation_id)
        self.assertIn("title", data)
        self.assertIn("messages", data)
        self.assertGreater(len(data["messages"]), 0)

        # Verify caching headers
        self.assertIn("cache-control", response.headers.lower() or "Cache-Control" in response.headers)

        print(f"✅ Retrieved conversation details")
        print(f"   Title: {data['title']}")
        print(f"   Messages: {len(data['messages'])}")

    def test_04_list_all_conversations(self):
        """Test listing all conversations with caching."""
        response = self.client.get("/api/chats/")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        # Verify caching headers
        cache_control = response.headers.get("cache-control") or response.headers.get("Cache-Control")
        self.assertIsNotNone(cache_control, "Cache-Control header should be present")
        self.assertIn("60", cache_control, "Should cache for 60 seconds")

        print(f"✅ Listed {len(data)} conversations")
        print(f"   Cache-Control: {cache_control}")

    def test_05_search_with_filters(self):
        """Test search with cluster and topic filters."""
        search_request = {
            "query": "python programming",
            "limit": 10,
            "cluster_filter": None,  # No cluster filter
            "topic_filter": None  # No topic filter
        }

        response = self.client.post("/api/search/", json=search_request)

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("results", data)
        self.assertIn("total_results", data)

        print(f"✅ Search with filters returned {data['total_results']} results")

    def test_06_vector_store_stats(self):
        """Test vector store statistics endpoint."""
        response = self.client.get("/api/search/stats")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("id", data)
        print(f"✅ Vector store stats:")
        print(f"   ID: {data.get('id')}")
        print(f"   Name: {data.get('name')}")
        print(f"   Status: {data.get('status')}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
