"""
K-means clustering service for 3D conversation coordinates.

This module handles:
- Clustering conversations in 3D space using K-means
- Assigning cluster IDs and names
- Assigning colors based on frontend color scheme
- Re-clustering when new conversations are added
"""

import os
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


# Default number of clusters
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "5"))

# Frontend color scheme for clusters (from memory-visualizer.tsx)
CLUSTER_COLORS = [
    "#9333ea",  # Purple
    "#3b82f6",  # Blue
    "#10b981",  # Green
    "#f59e0b",  # Orange
    "#ef4444",  # Red
    "#8b5cf6",  # Violet
    "#06b6d4",  # Cyan
    "#84cc16",  # Lime
    "#f97316",  # Deep Orange
    "#ec4899",  # Pink
]

# Default cluster names
DEFAULT_CLUSTER_NAMES = [
    "Cluster 0",
    "Cluster 1", 
    "Cluster 2",
    "Cluster 3",
    "Cluster 4",
    "Cluster 5",
    "Cluster 6",
    "Cluster 7",
    "Cluster 8",
    "Cluster 9",
]

# Model storage directory
MODEL_DIR = Path(__file__).parent.parent.parent / ".models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"


def cluster_conversations(
    coords_3d: List[List[float]],
    n_clusters: int | None = None,
    cluster_names: List[str] | None = None,
    save_model: bool = True
) -> List[Dict[str, Any]]:
    """
    Cluster conversations in 3D space using K-means.
    
    Args:
        coords_3d: List of [x, y, z] coordinate arrays
        n_clusters: Number of clusters (uses N_CLUSTERS env var if None)
        cluster_names: Optional custom cluster names
        save_model: Whether to save the fitted model
    
    Returns:
        List of dictionaries containing:
        - cluster_id: int (0 to n_clusters-1)
        - cluster_name: str
        - color: str (hex color code)
    
    Raises:
        ValueError: If coords_3d is empty or invalid
    """
    if not coords_3d:
        raise ValueError("Cannot cluster: coordinates list is empty")
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = N_CLUSTERS
    
    # Convert to numpy array
    X = np.array(coords_3d)
    
    # Validate dimensions
    if X.ndim != 2:
        raise ValueError("Coordinates must be 2D array")
    
    if X.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got {X.shape[1]}D")
    
    # Adjust n_clusters if we have fewer samples
    n_clusters = min(n_clusters, X.shape[0])
    
    if n_clusters < 1:
        raise ValueError("Need at least 1 cluster")
    
    # Fit K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(X)
    
    # Save model if requested
    if save_model:
        _save_model(kmeans)
    
    # Build cluster assignments
    results = []
    for label in labels:
        cluster_id = int(label)
        
        # Get cluster name
        if cluster_names and cluster_id < len(cluster_names):
            cluster_name = cluster_names[cluster_id]
        else:
            cluster_name = DEFAULT_CLUSTER_NAMES[cluster_id % len(DEFAULT_CLUSTER_NAMES)]
        
        # Get color
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        
        results.append({
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "color": color
        })
    
    return results


def predict_clusters(
    coords_3d: List[List[float]],
    model: KMeans | None = None
) -> List[Dict[str, Any]]:
    """
    Predict cluster assignments for new coordinates using existing model.
    
    Useful when adding a few new conversations without re-clustering all.
    
    Args:
        coords_3d: List of [x, y, z] coordinate arrays
        model: Optional pre-fitted KMeans model (loads from disk if None)
    
    Returns:
        List of cluster assignment dictionaries
    
    Raises:
        ValueError: If coordinates are invalid or model not found
    """
    if not coords_3d:
        raise ValueError("Cannot predict: coordinates list is empty")
    
    # Load or use provided model
    if model is None:
        model = load_cluster_model()
        if model is None:
            raise ValueError("No clustering model found. Call cluster_conversations first.")
    
    # Convert to numpy array
    X = np.array(coords_3d)
    
    # Validate dimensions
    if X.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got {X.shape[1]}D")
    
    # Predict clusters
    labels = model.predict(X)
    
    # Build results
    results = []
    for label in labels:
        cluster_id = int(label)
        cluster_name = DEFAULT_CLUSTER_NAMES[cluster_id % len(DEFAULT_CLUSTER_NAMES)]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        
        results.append({
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "color": color
        })
    
    return results


def generate_cluster_names_from_topics(
    cluster_assignments: List[Dict[str, Any]],
    all_topics: List[List[str]]
) -> Dict[int, str]:
    """
    Generate meaningful cluster names based on common topics.
    
    For each cluster, finds the most common topics and creates a name.
    
    Args:
        cluster_assignments: List of cluster assignment dicts with cluster_id
        all_topics: List of topic lists (one per conversation)
    
    Returns:
        Dictionary mapping cluster_id to cluster_name
    """
    if len(cluster_assignments) != len(all_topics):
        raise ValueError("cluster_assignments and all_topics must have same length")
    
    # Group topics by cluster
    cluster_topics: Dict[int, List[str]] = {}
    
    for assignment, topics in zip(cluster_assignments, all_topics):
        cluster_id = assignment["cluster_id"]
        
        if cluster_id not in cluster_topics:
            cluster_topics[cluster_id] = []
        
        cluster_topics[cluster_id].extend(topics)
    
    # Generate names for each cluster
    cluster_names = {}
    
    for cluster_id, topics in cluster_topics.items():
        if not topics:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Count topic frequency
        topic_counts: Dict[str, int] = {}
        for topic in topics:
            topic_lower = topic.lower()
            topic_counts[topic_lower] = topic_counts.get(topic_lower, 0) + 1
        
        # Get top 2 most common topics
        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_topics = [t[0].title() for t in sorted_topics[:2]]
        
        # Create cluster name
        if len(top_topics) == 1:
            cluster_names[cluster_id] = top_topics[0]
        else:
            cluster_names[cluster_id] = f"{top_topics[0]} & {top_topics[1]}"
    
    return cluster_names


def update_cluster_names(
    cluster_assignments: List[Dict[str, Any]],
    cluster_names: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Update cluster names in assignment dictionaries.
    
    Args:
        cluster_assignments: List of cluster assignment dicts
        cluster_names: Dictionary mapping cluster_id to new name
    
    Returns:
        Updated list of cluster assignments
    """
    updated = []
    
    for assignment in cluster_assignments:
        cluster_id = assignment["cluster_id"]
        
        new_assignment = assignment.copy()
        if cluster_id in cluster_names:
            new_assignment["cluster_name"] = cluster_names[cluster_id]
        
        updated.append(new_assignment)
    
    return updated


def load_cluster_model() -> KMeans | None:
    """
    Load a previously fitted K-means model from disk.
    
    Returns:
        Fitted KMeans model or None if not found
    """
    if not CLUSTER_MODEL_PATH.exists():
        return None
    
    try:
        with open(CLUSTER_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, IOError):
        return None


def _save_model(model: KMeans) -> None:
    """
    Save a fitted K-means model to disk.
    
    Args:
        model: Fitted KMeans model
    """
    try:
        with open(CLUSTER_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    except (pickle.PickleError, IOError):
        pass  # Silently fail if save fails


def get_cluster_statistics(
    cluster_assignments: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """
    Get statistics about cluster distribution.
    
    Args:
        cluster_assignments: List of cluster assignment dicts
    
    Returns:
        Dictionary mapping cluster_id to statistics:
        - count: Number of conversations in cluster
        - percentage: Percentage of total
        - name: Cluster name
        - color: Cluster color
    """
    if not cluster_assignments:
        return {}
    
    total = len(cluster_assignments)
    cluster_counts: Dict[int, int] = {}
    cluster_info: Dict[int, Dict[str, Any]] = {}
    
    # Count conversations per cluster
    for assignment in cluster_assignments:
        cluster_id = assignment["cluster_id"]
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        # Store name and color
        if cluster_id not in cluster_info:
            cluster_info[cluster_id] = {
                "name": assignment.get("cluster_name", f"Cluster {cluster_id}"),
                "color": assignment.get("color", CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])
            }
    
    # Build statistics
    stats = {}
    for cluster_id, count in cluster_counts.items():
        stats[cluster_id] = {
            "count": count,
            "percentage": round(count / total * 100, 1),
            "name": cluster_info[cluster_id]["name"],
            "color": cluster_info[cluster_id]["color"]
        }
    
    return stats


def clear_model() -> bool:
    """
    Delete the saved K-means model.
    
    Returns:
        True if model was deleted, False otherwise
    """
    if CLUSTER_MODEL_PATH.exists():
        try:
            CLUSTER_MODEL_PATH.unlink()
            return True
        except IOError:
            return False
    return False
