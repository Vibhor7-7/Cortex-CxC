"""
UMAP dimensionality reduction service for conversation embeddings.

This module handles:
- Reducing 384D embeddings to 3D coordinates for visualization
- Fitting UMAP models on all conversation embeddings
- Saving/loading fitted UMAP models for consistency
- Generating vector arrows for 3D visualization
"""

import os
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
from umap import UMAP


# UMAP configuration from environment or defaults
UMAP_N_NEIGHBORS = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
UMAP_MIN_DIST = float(os.getenv("UMAP_MIN_DIST", "0.1"))
UMAP_RANDOM_STATE = 42  # For reproducibility

# Model storage directory
MODEL_DIR = Path(__file__).parent.parent.parent / ".models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "umap_model.pkl"


def fit_umap_model(
    embeddings: List[List[float]],
    save_model: bool = True
) -> UMAP:
    """
    Fit a UMAP model on a collection of embeddings.
    
    This should be called when:
    1. First time processing conversations
    2. After adding new conversations (to re-fit on all data)
    
    Args:
        embeddings: List of 384D embedding vectors
        save_model: Whether to save the fitted model to disk
    
    Returns:
        Fitted UMAP model
    
    Raises:
        ValueError: If embeddings list is empty or has wrong dimensions
    """
    if not embeddings:
        raise ValueError("Cannot fit UMAP: embeddings list is empty")
    
    # Convert to numpy array
    X = np.array(embeddings)
    
    # Validate dimensions
    if X.ndim != 2:
        raise ValueError("Embeddings must be 2D array")
    
    if X.shape[1] not in (384, 768):
        raise ValueError(f"Expected 384D or 768D embeddings, got {X.shape[1]}D")
    
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 embeddings to fit UMAP")
    
    # Adjust n_neighbors if we have fewer samples
    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, X.shape[0] - 1))
    
    # For very small datasets, use random init instead of spectral
    # (spectral fails when k >= N in the sparse eigensolver)
    init_method = "spectral" if X.shape[0] > n_neighbors + 1 else "random"
    
    # Create and fit UMAP model
    umap_model = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
        init=init_method,
        verbose=False
    )
    
    # Fit the model
    umap_model.fit(X)
    
    # Save model if requested
    if save_model:
        _save_model(umap_model)
    
    return umap_model


def reduce_embeddings(
    embeddings: List[List[float]],
    model: UMAP | None = None,
    generate_vectors: bool = True
) -> List[Dict[str, Any]]:
    """
    Reduce embeddings from 384D to 3D coordinates.
    
    Args:
        embeddings: List of 384D embedding vectors
        model: Optional pre-fitted UMAP model (loads from disk if None)
        generate_vectors: Whether to generate vector arrow coordinates
    
    Returns:
        List of dictionaries containing:
        - vector_3d: [x, y, z] coordinates
        - start_x, start_y, start_z: Vector start point (origin)
        - end_x, end_y, end_z: Vector end point
        - magnitude: Vector magnitude
    
    Raises:
        ValueError: If embeddings list is empty or model not found
    """
    if not embeddings:
        raise ValueError("Cannot reduce embeddings: list is empty")
    
    # Load or use provided model
    if model is None:
        model = load_umap_model()
        if model is None:
            raise ValueError("No UMAP model found. Call fit_umap_model first.")
    
    # Convert to numpy array
    X = np.array(embeddings)
    
    # Validate dimensions
    if X.shape[1] not in (384, 768):
        raise ValueError(f"Expected 384D or 768D embeddings, got {X.shape[1]}D")
    
    # Transform to 3D
    coords_3d = model.transform(X)
    
    # Build result dictionaries
    results = []
    for i, coords in enumerate(coords_3d):
        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
        
        result = {
            "vector_3d": [x, y, z]
        }
        
        if generate_vectors:
            # Generate vector arrow from origin to point
            # This creates a visual arrow in the 3D space
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            
            result.update({
                "start_x": 0.0,
                "start_y": 0.0,
                "start_z": 0.0,
                "end_x": x,
                "end_y": y,
                "end_z": z,
                "magnitude": float(magnitude)
            })
        
        results.append(result)
    
    return results


def load_umap_model() -> UMAP | None:
    """
    Load a previously fitted UMAP model from disk.
    
    Returns:
        Fitted UMAP model or None if not found
    """
    if not MODEL_PATH.exists():
        return None
    
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, IOError):
        return None


def _save_model(model: UMAP) -> None:
    """
    Save a fitted UMAP model to disk.
    
    Args:
        model: Fitted UMAP model
    """
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    except (pickle.PickleError, IOError):
        pass  # Silently fail if save fails


def normalize_coordinates(
    coords_list: List[Dict[str, Any]],
    scale: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Normalize 3D coordinates to a consistent scale for visualization.
    
    Centers the data at origin and scales to the specified range.
    
    Args:
        coords_list: List of coordinate dictionaries from reduce_embeddings
        scale: Target scale for the coordinates (default: 10.0)
    
    Returns:
        List of normalized coordinate dictionaries
    """
    if not coords_list:
        return coords_list
    
    # Extract all 3D coordinates
    coords_array = np.array([c["vector_3d"] for c in coords_list])
    
    # Center at origin
    mean = coords_array.mean(axis=0)
    centered = coords_array - mean
    
    # Scale to target range
    max_dist = np.abs(centered).max()
    if max_dist > 0:
        scaled = centered * (scale / max_dist)
    else:
        scaled = centered
    
    # Update coordinate dictionaries
    normalized = []
    for i, coords_dict in enumerate(coords_list):
        x, y, z = float(scaled[i][0]), float(scaled[i][1]), float(scaled[i][2])
        
        result = {
            "vector_3d": [x, y, z]
        }
        
        # Update vector arrow if present
        if "start_x" in coords_dict:
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            result.update({
                "start_x": 0.0,
                "start_y": 0.0,
                "start_z": 0.0,
                "end_x": x,
                "end_y": y,
                "end_z": z,
                "magnitude": float(magnitude)
            })
        
        normalized.append(result)
    
    return normalized


def refit_and_reduce_all(
    embeddings: List[List[float]],
    normalize: bool = True,
    scale: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Convenience function to fit UMAP and reduce embeddings in one call.
    
    Use this when re-clustering after adding new conversations.
    
    Args:
        embeddings: List of 384D embedding vectors
        normalize: Whether to normalize coordinates
        scale: Scale for normalization
    
    Returns:
        List of coordinate dictionaries
    """
    # Fit UMAP model
    model = fit_umap_model(embeddings, save_model=True)
    
    # Reduce embeddings
    coords_list = reduce_embeddings(
        embeddings,
        model=model,
        generate_vectors=True
    )
    
    # Normalize if requested
    if normalize:
        coords_list = normalize_coordinates(coords_list, scale=scale)
    
    return coords_list


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the current UMAP model.
    
    Returns:
        Dictionary with model information or None if no model exists
    """
    model = load_umap_model()
    
    if model is None:
        return {
            "exists": False,
            "path": str(MODEL_PATH)
        }
    
    return {
        "exists": True,
        "path": str(MODEL_PATH),
        "n_components": model.n_components,
        "n_neighbors": model.n_neighbors,
        "min_dist": model.min_dist,
        "metric": model.metric,
        "random_state": model.random_state,
    }


def clear_model() -> bool:
    """
    Delete the saved UMAP model.
    
    Returns:
        True if model was deleted, False otherwise
    """
    if MODEL_PATH.exists():
        try:
            MODEL_PATH.unlink()
            return True
        except IOError:
            return False
    return False
