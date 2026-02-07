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
    embedding_384d: List[float] = Field(..., min_length=384, max_length=384)
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


class SearchResponse(BaseModel):
    """Schema for search response."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float


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
    openai_configured: bool
    vector_store_configured: bool
    