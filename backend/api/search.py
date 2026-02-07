"""
Search API endpoint.

Provides hybrid semantic + keyword search using OpenAI Vector Store.
"""

import time
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from sqlalchemy import or_
from backend.database import get_db_context
from backend.models import Conversation, Embedding
from backend.schemas import SearchRequest, SearchResponse, SearchResultItem
from backend.services.openai_vector_store import search_vector_store


router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_conversations(request: SearchRequest):
    """
    Search conversations using hybrid retrieval.

    Uses OpenAI Vector Store for hybrid semantic + keyword search, then
    fetches full conversation metadata from the database.

    Args:
        request: SearchRequest with query, limit, and filters

    Returns:
        SearchResponse with matching conversations and metadata

    Process:
        1. Search OpenAI Vector Store with query
        2. Extract file_ids from search results
        3. Map file_ids to conversation_ids via database
        4. Fetch full conversation metadata
        5. Apply additional filters (cluster, topics)
        6. Aggregate scores and deduplicate
        7. Return ranked results with 3D coordinates
    """
    start_time = time.time()

    try:
        # Step 1: Search vector store
        # Note: The vector store search uses its own hybrid ranking
        vector_results = await search_vector_store(
            query=request.query,
            max_results=request.limit * 3,  # Get more results for filtering
            rewrite_query=True,
            score_threshold=0.3  # Filter low-quality results
        )

        if not vector_results:
            # No results from vector store
            search_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time_ms=search_time
            )

        # Step 2: Extract file IDs and aggregate scores
        file_scores = {}  # file_id -> aggregated score
        file_chunks = {}   # file_id -> list of content chunks

        for result in vector_results:
            file_id = result.get('file_id')
            score = result.get('score', 0.0)
            content = result.get('content', '')

            if file_id:
                if file_id not in file_scores:
                    file_scores[file_id] = 0.0
                    file_chunks[file_id] = []

                # Aggregate scores (take max score across chunks)
                file_scores[file_id] = max(file_scores[file_id], score)
                if content:
                    file_chunks[file_id].append(content)

        # Step 3: Map file_ids to conversation_ids and fetch metadata
        with get_db_context() as db:
            # Get conversations with matching file IDs
            conversations = db.query(Conversation).join(Embedding).filter(
                Conversation.openai_file_id.in_(list(file_scores.keys()))
            ).all()

            # Build results list
            results = []
            for conv in conversations:
                # Get score for this conversation
                score = file_scores.get(conv.openai_file_id, 0.0)

                # Apply filters
                if request.cluster_filter is not None:
                    if conv.cluster_id != request.cluster_filter:
                        continue

                if request.topic_filter:
                    # Check if any requested topic is in conversation topics
                    conv_topics = set(conv.topics or [])
                    if not any(topic in conv_topics for topic in request.topic_filter):
                        continue

                # Get embedding for 3D coordinates
                embedding = conv.embedding

                # Get message preview (first chunk)
                chunks = file_chunks.get(conv.openai_file_id, [])
                preview = chunks[0][:200] + "..." if chunks else None

                # Create result item
                result_item = SearchResultItem(
                    conversation_id=conv.id,
                    title=conv.title,
                    summary=conv.summary or "",
                    topics=conv.topics or [],
                    message_count=conv.message_count,
                    created_at=conv.created_at,
                    start_x=embedding.start_x if embedding else 0.0,
                    start_y=embedding.start_y if embedding else 0.0,
                    start_z=embedding.start_z if embedding else 0.0,
                    end_x=embedding.end_x if embedding else 0.0,
                    end_y=embedding.end_y if embedding else 0.0,
                    end_z=embedding.end_z if embedding else 0.0,
                    magnitude=embedding.magnitude if embedding else 1.0,
                    cluster_id=conv.cluster_id,
                    cluster_name=conv.cluster_name,
                    score=score,
                    message_preview=preview
                )

                results.append(result_item)

            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)

            # Limit results
            results = results[:request.limit]

            search_time = (time.time() - start_time) * 1000

            return SearchResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                search_time_ms=search_time
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
    Get vector store statistics.

    Returns:
        Dictionary with vector store stats including file count
    """
    try:
        from backend.services.openai_vector_store import get_vector_store_service
        service = get_vector_store_service()
        stats = service.get_vector_store_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )
