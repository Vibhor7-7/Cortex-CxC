"""
Services package for chat processing and analysis.

This package contains services for:
- Normalizing conversation data
- Generating summaries with LLM
- Generating embeddings
- Reducing dimensionality (384D -> 3D)
- Clustering conversations
"""

from .normalizer import normalize_conversation
from .summarizer import summarize_conversation
from .embedder import generate_embedding
from .dimensionality_reducer import reduce_embeddings, fit_umap_model
from .clusterer import cluster_conversations

__all__ = [
    "normalize_conversation",
    "summarize_conversation",
    "generate_embedding",
    "reduce_embeddings",
    "fit_umap_model",
    "cluster_conversations",
]
