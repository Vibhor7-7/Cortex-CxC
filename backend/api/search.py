"""
Search API endpoint.

Provides semantic search using a local vector store + nomic-embed-text embeddings via Ollama.
"""

import time
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from sqlalchemy import or_
from backend.database import get_db_context
from backend.models import Conversation, Embedding
from backend.schemas import SearchRequest, SearchResponse, SearchResultItem
from backend.services.vector_store import search_store, get_vector_store_service
from backend.services.embedder import generate_embedding


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_conversations(request: SearchRequest):
    """
    Search conversations using semantic similarity via local vector store.

    Embeds the query with nomic-embed-text, searches the vector store for nearest
    neighbours, then fetches full conversation metadata from SQLite.

    Args:
        request: SearchRequest with query, limit, and filters

    Returns:
        SearchResponse with matching conversations and metadata

    Process:
        1. Generate query embedding via Ollama (nomic-embed-text)
        2. Search local vector store for nearest conversations
        3. Fetch full conversation metadata from SQLite
        4. Apply additional filters (cluster, topics)
        5. Return ranked results with 3D coordinates
    """
    start_time = time.time()

    try:
        # Step 1: Generate query embedding
        query_embedding = await generate_embedding(request.query, use_cache=False)

        # Step 2: Search vector store
        chroma_results = await search_store(
            query_embedding=query_embedding,
            max_results=request.limit * 3,
            score_threshold=0.3,
        )

        if not chroma_results:
            search_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time_ms=search_time,
            )

        # Step 3: Build score + preview maps keyed by conversation_id
        conv_scores = {}
        conv_previews = {}
        for r in chroma_results:
            cid = r["conversation_id"]
            conv_scores[cid] = r["score"]
            content = r.get("content", "")
            conv_previews[cid] = content[:200] + "..." if content else None

        # Step 4: Fetch full metadata from SQLite
        with get_db_context() as db:
            conversations = (
                db.query(Conversation)
                .join(Embedding)
                .filter(Conversation.id.in_(list(conv_scores.keys())))
                .all()
            )

            results = []
            for conv in conversations:
                score = conv_scores.get(conv.id, 0.0)

                # Apply filters
                if request.cluster_filter is not None:
                    if conv.cluster_id != request.cluster_filter:
                        continue
                if request.topic_filter:
                    conv_topics = set(conv.topics or [])
                    if not any(t in conv_topics for t in request.topic_filter):
                        continue

                emb = conv.embedding
                results.append(
                    SearchResultItem(
                        conversation_id=conv.id,
                        title=conv.title,
                        summary=conv.summary or "",
                        topics=conv.topics or [],
                        message_count=conv.message_count,
                        created_at=conv.created_at,
                        start_x=emb.start_x if emb else 0.0,
                        start_y=emb.start_y if emb else 0.0,
                        start_z=emb.start_z if emb else 0.0,
                        end_x=emb.end_x if emb else 0.0,
                        end_y=emb.end_y if emb else 0.0,
                        end_z=emb.end_z if emb else 0.0,
                        magnitude=emb.magnitude if emb else 1.0,
                        cluster_id=conv.cluster_id,
                        cluster_name=conv.cluster_name,
                        score=score,
                        message_preview=conv_previews.get(conv.id),
                    )
                )

            results.sort(key=lambda x: x.score, reverse=True)
            results = results[: request.limit]

            search_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time,
            )

    except Exception as e:
        search_time = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/stats")
async def get_search_stats():
    """
    Get local vector store statistics.

    Returns:
        Dictionary with collection stats including document count
    """
    try:
        service = get_vector_store_service()
        stats = service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )
