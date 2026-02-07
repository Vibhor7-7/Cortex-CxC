"""
SQLAlchemy database models for Cortex.
"""
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database import Base
import uuid
import enum
from datetime import datetime


def generate_uuid():
    """Generate a UUID string for use as primary key."""
    return str(uuid.uuid4())


class MessageRole(str, enum.Enum):
    """Enum for message roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """
    Conversation model representing a chat conversation.
    
    Stores metadata about conversations including title, summary,
    topics, and clustering information.
    """
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(200), nullable=False, index=True)
    summary = Column(Text, nullable=True)
    topics = Column(JSON, default=list)  # Array of topic strings
    cluster_id = Column(Integer, nullable=True, index=True)
    cluster_name = Column(String(100), nullable=True)
    message_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    embedding = relationship("Embedding", back_populates="conversation", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title[:30]}...)>"


class Message(Base):
    """
    Message model representing individual messages within a conversation.
    
    Messages are ordered by sequence_number and have a role (user/assistant/system).
    """
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, seq={self.sequence_number})>"


class Embedding(Base):
    """
    Embedding model storing vector representations and 3D coordinates.
    
    Stores:
    - Original 384-dimensional embeddings
    - UMAP-reduced 3D coordinates for visualization
    - Animation start/end points for frontend transitions
    - Magnitude for visualization scaling
    """
    __tablename__ = "embeddings"

    conversation_id = Column(String(36), ForeignKey("conversations.id"), primary_key=True)
    
    # Original high-dimensional embedding (768d for nomic-embed-text)
    embedding_384d = Column(JSON, nullable=False)  # Array of 768 floats (column name kept for migration compat)
    
    # UMAP-reduced 3D coordinates
    vector_3d = Column(JSON, nullable=False)  # Array of [x, y, z]
    
    # Visualization coordinates for animation
    start_x = Column(Float, nullable=False, default=0.0)
    start_y = Column(Float, nullable=False, default=0.0)
    start_z = Column(Float, nullable=False, default=0.0)
    end_x = Column(Float, nullable=False)
    end_y = Column(Float, nullable=False)
    end_z = Column(Float, nullable=False)
    
    # Vector magnitude for scaling
    magnitude = Column(Float, nullable=False, default=1.0)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="embedding")

    def __repr__(self):
        return f"<Embedding(conversation_id={self.conversation_id}, magnitude={self.magnitude})>"


# Index definitions for better query performance
from sqlalchemy import Index

# Create indexes for common query patterns
Index('idx_conversation_cluster', Conversation.cluster_id)
Index('idx_conversation_created', Conversation.created_at)
Index('idx_message_conversation', Message.conversation_id, Message.sequence_number)