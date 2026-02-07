"""
OpenAI Vector Store service for hybrid search.

Handles:
- Creating and managing OpenAI Vector Stores
- Uploading conversation documents to vector store
- Searching conversations using hybrid retrieval
- File ingestion status polling
"""

import os
import time
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()


class VectorStoreService:
    """Service for managing OpenAI Vector Store operations."""

    def __init__(self):
        """Initialize OpenAI client and load configuration."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store_id = None
        self.config_file = Path("backend/.vector_store_config.json")

        # Load or create vector store
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """
        Load existing vector store ID from env/config file, or create new one.

        Priority order:
        1. OPENAI_VECTOR_STORE_ID environment variable
        2. Local config file (.vector_store_config.json)
        3. Create new vector store
        """
        # Check environment variable first
        env_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
        if env_store_id and env_store_id.strip():
            self.vector_store_id = env_store_id.strip()
            print(f" Using vector store from env: {self.vector_store_id}")
            return

        # Check local config file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.vector_store_id = config.get("vector_store_id")
                    if self.vector_store_id:
                        print(f" Using vector store from config: {self.vector_store_id}")
                        return
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")

        # Create new vector store
        print("Creating new OpenAI Vector Store...")
        self._create_new_vector_store()

    def _create_new_vector_store(self):
        """Create a new OpenAI Vector Store and persist the ID."""
        try:
            vector_store = self.client.beta.vector_stores.create(
                name="Cortex Chat Memory",
                expires_after=None  # No expiration
            )
            self.vector_store_id = vector_store.id

            # Save to config file
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump({
                    "vector_store_id": self.vector_store_id,
                    "created_at": time.time()
                }, f, indent=2)

            print(f" Created new vector store: {self.vector_store_id}")
            print(f"   Saved to: {self.config_file}")

        except Exception as e:
            raise Exception(f"Failed to create vector store: {str(e)}")

    def conversation_to_markdown(self, conversation_data: Dict) -> str:
        """
        Convert a conversation to markdown format for vector store upload.

        Args:
            conversation_data: Dictionary containing:
                - title: Conversation title
                - summary: Conversation summary
                - topics: List of topics
                - messages: List of message dicts with 'role' and 'content'

        Returns:
            Markdown-formatted conversation text
        """
        md_lines = []

        # Title
        md_lines.append(f"# {conversation_data['title']}\n")

        # Summary
        if conversation_data.get('summary'):
            md_lines.append(f"**Summary:** {conversation_data['summary']}\n")

        # Topics
        if conversation_data.get('topics'):
            topics_str = ", ".join(conversation_data['topics'])
            md_lines.append(f"**Topics:** {topics_str}\n")

        md_lines.append("\n---\n")

        # Messages
        for msg in conversation_data.get('messages', []):
            role = msg['role'].upper()
            content = msg['content']
            md_lines.append(f"\n## {role}\n\n{content}\n")

        return "\n".join(md_lines)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upload_conversation(
        self,
        conversation_id: str,
        conversation_data: Dict,
        poll_completion: bool = True
    ) -> Tuple[str, str]:
        """
        Upload a conversation to the vector store.

        Args:
            conversation_id: Unique conversation identifier
            conversation_data: Conversation data dict (title, summary, topics, messages)
            poll_completion: If True, poll until file ingestion completes

        Returns:
            Tuple of (file_id, status)

        Raises:
            Exception: If upload fails
        """
        try:
            # Convert conversation to markdown
            markdown_content = self.conversation_to_markdown(conversation_data)

            # Create temporary file
            temp_file = Path(f"/tmp/cortex_conv_{conversation_id}.md")
            temp_file.write_text(markdown_content, encoding='utf-8')

            # Upload file to OpenAI
            with open(temp_file, 'rb') as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )

            # Delete temporary file
            temp_file.unlink()

            # Attach file to vector store with metadata
            vector_store_file = self.client.beta.vector_stores.files.create(
                vector_store_id=self.vector_store_id,
                file_id=file_obj.id
            )

            # Poll for completion if requested
            if poll_completion:
                status = self._poll_file_status(file_obj.id)
                return file_obj.id, status

            return file_obj.id, "pending"

        except Exception as e:
            raise Exception(f"Failed to upload conversation: {str(e)}")

    def _poll_file_status(
        self,
        file_id: str,
        max_attempts: int = 30,
        poll_interval: float = 2.0
    ) -> str:
        """
        Poll file ingestion status until completed or failed.

        Args:
            file_id: OpenAI file ID
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between polls

        Returns:
            Final status: 'completed', 'failed', or 'timeout'
        """
        for attempt in range(max_attempts):
            try:
                vector_store_file = self.client.beta.vector_stores.files.retrieve(
                    vector_store_id=self.vector_store_id,
                    file_id=file_id
                )

                status = vector_store_file.status

                if status == "completed":
                    print(f" File {file_id} ingestion completed")
                    return "completed"
                elif status == "failed":
                    print(f" File {file_id} ingestion failed")
                    return "failed"
                elif status in ["in_progress", "pending"]:
                    print(f"⏳ File {file_id} status: {status} (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(poll_interval)
                else:
                    print(f"Unknown status: {status}")
                    time.sleep(poll_interval)

            except Exception as e:
                print(f"Error polling file status: {e}")
                time.sleep(poll_interval)

        print(f" File {file_id} ingestion timed out after {max_attempts} attempts")
        return "timeout"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(
        self,
        query: str,
        max_results: int = 10,
        rewrite_query: bool = True,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search the vector store using hybrid retrieval.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            rewrite_query: Whether to use OpenAI query rewriting for better semantic matching
            score_threshold: Minimum score threshold for results (0.0 to 1.0)
            filters: Optional filters dict (not directly supported by vector store search API)

        Returns:
            List of search result dictionaries containing:
                - file_id: OpenAI file ID
                - content: Matching text chunk
                - score: Relevance score
                - metadata: File metadata
        """
        try:
            # Prepare search request
            search_params = {
                "query": query,
                "max_num_results": max_results
            }

            # Add optional parameters
            if rewrite_query is not None:
                search_params["rewrite_query"] = rewrite_query

            if score_threshold is not None:
                search_params["ranking_options"] = {
                    "score_threshold": score_threshold
                }

            # Execute search using the vector store search API
            response = self.client.beta.vector_stores.search(
                vector_store_id=self.vector_store_id,
                **search_params
            )

            # Parse results
            results = []
            for result in response.data:
                results.append({
                    "file_id": result.file_id,
                    "content": result.content[0].text if result.content else "",
                    "score": result.score,
                    "metadata": result.metadata if hasattr(result, 'metadata') else {}
                })

            return results

        except Exception as e:
            # If vector store search fails, return empty results
            print(f"Vector store search error: {e}")
            return []

    def delete_conversation_file(self, file_id: str) -> bool:
        """
        Delete a conversation file from the vector store.

        Args:
            file_id: OpenAI file ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from vector store
            self.client.beta.vector_stores.files.delete(
                vector_store_id=self.vector_store_id,
                file_id=file_id
            )

            # Delete the file itself
            self.client.files.delete(file_id=file_id)

            print(f" Deleted file {file_id} from vector store")
            return True

        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            return False

    def get_vector_store_stats(self) -> Dict:
        """
        Get statistics about the current vector store.

        Returns:
            Dictionary with vector store statistics
        """
        try:
            vector_store = self.client.beta.vector_stores.retrieve(
                vector_store_id=self.vector_store_id
            )

            return {
                "id": vector_store.id,
                "name": vector_store.name,
                "file_counts": vector_store.file_counts.dict() if hasattr(vector_store, 'file_counts') else {},
                "status": vector_store.status if hasattr(vector_store, 'status') else "active",
                "created_at": vector_store.created_at
            }

        except Exception as e:
            return {
                "error": str(e),
                "id": self.vector_store_id
            }


# Singleton instance
_vector_store_service = None


def get_vector_store_service() -> VectorStoreService:
    """Get or create the singleton VectorStoreService instance."""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service


# Convenience functions
async def upload_conversation_to_vector_store(
    conversation_id: str,
    conversation_data: Dict,
    poll_completion: bool = True
) -> Tuple[str, str]:
    """
    Upload a conversation to the vector store (async wrapper).

    Args:
        conversation_id: Conversation ID
        conversation_data: Conversation data dict
        poll_completion: Whether to poll for completion

    Returns:
        Tuple of (file_id, status)
    """
    service = get_vector_store_service()
    return service.upload_conversation(conversation_id, conversation_data, poll_completion)


async def search_vector_store(
    query: str,
    max_results: int = 10,
    rewrite_query: bool = True,
    score_threshold: Optional[float] = None
) -> List[Dict]:
    """
    Search the vector store (async wrapper).

    Args:
        query: Search query
        max_results: Maximum results
        rewrite_query: Use query rewriting
        score_threshold: Minimum score threshold

    Returns:
        List of search results
    """
    service = get_vector_store_service()
    return service.search(query, max_results, rewrite_query, score_threshold)
