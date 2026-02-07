"""
Chats API endpoint.

Provides endpoints for retrieving conversation data.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy.orm import Session

from backend.database import get_db_context
from backend.models import Conversation, Message, Embedding
from backend.schemas import (
    ConversationResponse,
    ConversationDetailResponse,
    MessageResponse,
    VisualizationResponse,
    VisualizationNode
)


router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.get("/", response_model=List[ConversationResponse])
async def get_all_conversations(
    response: Response,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    Get all conversations with pagination.

    Returns conversation metadata without full message history.
    Use GET /api/chats/{id} to get full conversation details.

    Cache-Control: max-age=60 (60 seconds)
    """
    # Add caching header - cache for 60 seconds
    response.headers["Cache-Control"] = "public, max-age=60"

    with get_db_context() as db:
        conversations = (
            db.query(Conversation)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

        return [
            ConversationResponse.model_validate(conv)
            for conv in conversations
        ]


@router.get("/visualization", response_model=VisualizationResponse)
async def get_visualization_data():
    """
    Get all conversations with 3D coordinates for visualization.
    
    Returns data formatted for the 3D frontend visualization.
    """
    with get_db_context() as db:
        conversations = db.query(Conversation).all()
        
        nodes = []
        clusters = {}
        
        for conv in conversations:
            # Get embedding data
            embedding = db.query(Embedding).filter(
                Embedding.conversation_id == conv.id
            ).first()
            
            if embedding:
                # Create visualization node
                node = VisualizationNode(
                    id=conv.id,
                    title=conv.title,
                    summary=conv.summary,
                    topics=conv.topics or [],
                    cluster_id=conv.cluster_id,
                    cluster_name=conv.cluster_name,
                    message_count=conv.message_count,
                    position=[embedding.end_x, embedding.end_y, embedding.end_z],
                    start_position=[embedding.start_x, embedding.start_y, embedding.start_z],
                    magnitude=embedding.magnitude,
                    created_at=conv.created_at
                )
                nodes.append(node)
                
                # Track cluster info
                if conv.cluster_id is not None:
                    if conv.cluster_id not in clusters:
                        clusters[conv.cluster_id] = {
                            "cluster_id": conv.cluster_id,
                            "cluster_name": conv.cluster_name,
                            "count": 0
                        }
                    clusters[conv.cluster_id]["count"] += 1
        
        return VisualizationResponse(
            nodes=nodes,
            total_nodes=len(nodes),
            clusters=list(clusters.values())
        )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_details(conversation_id: str, response: Response):
    """
    Get full conversation details including all messages.

    Args:
        conversation_id: UUID of the conversation

    Returns:
        Full conversation with all messages in sequence

    Cache-Control: max-age=300 (5 minutes)
    """
    # Add caching header - cache for 5 minutes (300 seconds)
    response.headers["Cache-Control"] = "public, max-age=300"

    with get_db_context() as db:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found"
            )
        
        # Get messages
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.sequence_number)
            .all()
        )
        
        # Convert to response models
        message_responses = [
            MessageResponse.model_validate(msg)
            for msg in messages
        ]
        
        # Create detailed response
        response = ConversationDetailResponse.model_validate(conversation)
        response.messages = message_responses
        
        return response


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: UUID of the conversation to delete
        
    Returns:
        Success message
    """
    with get_db_context() as db:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found"
            )
        
        db.delete(conversation)
        db.commit()
        
        return {
            "success": True,
            "message": f"Conversation {conversation_id} deleted successfully"
        }
