"""
Pydantic schemas for request/response validation in Cortex API.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============================================================================
# Message Schemas
# ============================================================================

class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    role: MessageRole
    content: str = Field(..., min_length=1)
    sequence_number: int = Field(..., ge=0)


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    sequence_number: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Conversation Schemas
# ============================================================================

class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    title: str = Field(..., max_length=200, min_length=1)
    summary: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    message_count: int = Field(default=0, ge=0)


class ConversationResponse(BaseModel):
    """Schema for conversation response (without messages)."""
    id: str
    title: str
    summary: Optional[str] = None
    topics: List[str]
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    message_count: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(ConversationResponse):
    """Schema for detailed conversation response (includes messages)."""
    messages: List[MessageResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Embedding Schemas
# ============================================================================

class EmbeddingCreate(BaseModel):
    """Schema for creating an embedding."""
    conversation_id: str
    embedding_384d: List[float] = Field(..., min_length=768, max_length=768)
    vector_3d: List[float] = Field(..., min_length=3, max_length=3)
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    end_x: float
    end_y: float
    end_z: float
    magnitude: float = 1.0


class EmbeddingResponse(BaseModel):
    """Schema for embedding response."""
    conversation_id: str
    embedding_384d: List[float]
    vector_3d: List[float]
    start_x: float
    start_y: float
    start_z: float
    end_x: float
    end_y: float
    end_z: float
    magnitude: float
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Search Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """Schema for hybrid search request."""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    cluster_filter: Optional[int] = None
    topic_filter: Optional[List[str]] = None
    evaluate: bool = Field(default=False, description="Enable Backboard.io retrieval quality evaluation")


class SearchResultItem(BaseModel):
    """Schema for a single search result."""
    conversation_id: str
    title: str
    summary: Optional[str] = None
    topics: List[str]
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    score: float
    message_preview: Optional[str] = None  # Preview of relevant message content
    relevance_score: Optional[float] = None  # Backboard evaluation score (0.0-1.0)
    relevance_reason: Optional[str] = None  # Reason for relevance score


# ============================================================================
# Backboard Evaluation Schemas
# ============================================================================

class RelevanceScore(BaseModel):
    """Schema for individual relevance score from Backboard evaluation."""
    conversation_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score from 0.0 to 1.0")
    reason: str = Field(..., description="Brief explanation for the score")


class RetrievalEvaluation(BaseModel):
    """Schema for full retrieval evaluation from Backboard."""
    relevance_scores: List[RelevanceScore] = Field(default_factory=list)
    redundancy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How much overlap exists between results")
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How completely results address the query")
    evaluation_time_ms: float = Field(default=0.0, description="Time taken for evaluation in milliseconds")


class GuardResult(BaseModel):
    """Schema for MCP guard relevance check result."""
    is_relevant: bool = Field(..., description="Whether the memory is relevant to the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the relevance decision")
    reason: str = Field(..., description="Brief explanation for the decision")


class SearchResponse(BaseModel):
    """Schema for search response."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float
    evaluation: Optional[RetrievalEvaluation] = None  # Backboard evaluation (when evaluate=true)


# ============================================================================
# Ingestion Schemas
# ============================================================================

class IngestRequest(BaseModel):
    """Schema for chat ingestion request."""
    source: str = Field(..., description="Source of the chat (e.g., 'chatgpt', 'claude')")
    file_content: Optional[str] = None  # For direct HTML content
    file_path: Optional[str] = None  # For file upload


class IngestResponse(BaseModel):
    """Schema for ingestion response."""
    success: bool
    conversation_id: Optional[str] = None
    title: Optional[str] = None
    message_count: int = 0
    error: Optional[str] = None
    processing_time_ms: float


class IngestBatchResponse(BaseModel):
    """Schema for batch ingestion response."""
    total_processed: int
    successful: int
    failed: int
    conversations: List[IngestResponse]
    total_time_ms: float


# ============================================================================
# Prompt Generation Schemas
# ============================================================================

class GeneratePromptRequest(BaseModel):
    """Schema for system-prompt generation request."""
    conversation_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="IDs of selected conversations to build a system prompt from"
    )


class GeneratePromptResponse(BaseModel):
    """Schema for system-prompt generation response."""
    prompt: str = Field(..., description="LLM-generated system prompt")
    conversations_used: int = Field(..., description="Number of conversations included")
    processing_time_ms: float = Field(..., description="LLM processing time in ms")


# ============================================================================
# Visualization Data Schemas
# ============================================================================

class VisualizationNode(BaseModel):
    """Schema for a single conversation node in 3D visualization."""
    id: str
    title: str
    summary: Optional[str] = None
    topics: List[str]
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    message_count: int
    position: List[float] = Field(..., min_length=3, max_length=3)  # [x, y, z]
    start_position: List[float] = Field(..., min_length=3, max_length=3)  # [start_x, start_y, start_z]
    magnitude: float
    created_at: datetime


class VisualizationResponse(BaseModel):
    """Schema for visualization data response."""
    nodes: List[VisualizationNode]
    total_nodes: int
    clusters: List[dict]  # Cluster metadata


# ============================================================================
# Health Check Schema
# ============================================================================

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    version: str = "1.0.0"
    database_connected: bool
    ollama_connected: bool
    chroma_ready: bool
    