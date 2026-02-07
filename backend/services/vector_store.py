"""
Local Vector Store Service for CORTEX.

A lightweight, pure-numpy vector store that persists to a JSON file on disk.
Replaces the previous OpenAI Vector Store with zero external service dependencies.

Uses cosine similarity for semantic search over nomic-embed-text embeddings (768D).

Storage format:
    {
        "conversation_id": {
            "document": "...",
            "embedding": [float, ...],
            "metadata": {...}
        },
        ...
    }
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".vector_store.json"),
)


# ---------------------------------------------------------------------------
# Service Class
# ---------------------------------------------------------------------------
class VectorStoreService:
    """
    In-process vector store backed by a JSON file and numpy cosine similarity.

    Thread-safe via a simple reentrant lock.
    """

    def __init__(self, store_path: Optional[str] = None):
        self.store_path = store_path or VECTOR_STORE_PATH
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load store from disk (if it exists)."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Persist store to disk."""
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self._data, f)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def upsert_conversation(
        self,
        conversation_id: str,
        document: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a conversation in the store."""
        with self._lock:
            self._data[conversation_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata or {},
            }
            self._save()

    def delete_conversation(self, conversation_id: str) -> bool:
        """Remove a conversation. Returns True if it existed."""
        with self._lock:
            removed = self._data.pop(conversation_id, None) is not None
            if removed:
                self._save()
            return removed

    def search(
        self,
        query_embedding: List[float],
        max_results: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the store by cosine similarity.

        Args:
            query_embedding: 768-D query vector.
            max_results: Maximum number of results to return.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            List of dicts with keys:
                conversation_id, score, content, metadata
            sorted by descending score.
        """
        with self._lock:
            if not self._data:
                return []

            ids = list(self._data.keys())
            docs = [self._data[cid]["document"] for cid in ids]
            metas = [self._data[cid]["metadata"] for cid in ids]
            matrix = np.array(
                [self._data[cid]["embedding"] for cid in ids], dtype=np.float32
            )

        query = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        # Avoid division by zero
        denom = norms * query_norm
        denom[denom == 0] = 1e-10
        scores = matrix @ query / denom

        # Sort descending
        sorted_idx = np.argsort(-scores)

        results: List[Dict[str, Any]] = []
        for idx in sorted_idx:
            sim = float(scores[idx])
            if sim < score_threshold:
                continue
            results.append(
                {
                    "conversation_id": ids[idx],
                    "score": sim,
                    "content": docs[idx],
                    "metadata": metas[idx],
                }
            )
            if len(results) >= max_results:
                break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the store."""
        with self._lock:
            return {
                "collection_name": Path(self.store_path).stem,
                "document_count": len(self._data),
                "store_path": self.store_path,
            }

    def count(self) -> int:
        """Return number of stored documents."""
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[VectorStoreService] = None
_instance_lock = threading.Lock()


def get_vector_store_service(store_path: Optional[str] = None) -> VectorStoreService:
    """Get (or create) the singleton VectorStoreService."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = VectorStoreService(store_path)
        return _instance


# ---------------------------------------------------------------------------
# Async convenience wrappers (used in FastAPI endpoints)
# ---------------------------------------------------------------------------
async def upsert_conversation_to_store(
    conversation_id: str,
    conversation_data: Dict[str, Any],
    embedding: List[float],
) -> None:
    """
    Async wrapper around VectorStoreService.upsert_conversation().

    Builds a searchable document from conversation_data and stores it
    alongside the embedding vector.
    """
    service = get_vector_store_service()

    # Build a flat text document for search
    title = conversation_data.get("title", "Untitled")
    summary = conversation_data.get("summary", "")
    topics = ", ".join(conversation_data.get("topics", []))
    messages_text = ""
    for msg in conversation_data.get("messages", [])[:20]:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:500]
        messages_text += f"\n{role}: {content}"

    document = f"Title: {title}\nSummary: {summary}\nTopics: {topics}\n{messages_text}"

    metadata = {
        "title": title,
        "topic_count": len(conversation_data.get("topics", [])),
        "message_count": len(conversation_data.get("messages", [])),
    }

    service.upsert_conversation(conversation_id, document, embedding, metadata)


async def search_store(
    query_embedding: List[float],
    max_results: int = 10,
    score_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Async wrapper around VectorStoreService.search()."""
    service = get_vector_store_service()
    return service.search(query_embedding, max_results, score_threshold)
