"""Utility functions for step signatures - minimal."""

import json
from typing import Optional, Union
import numpy as np

from mycelium.config import CENTROID_CACHE_MAX_SIZE

# Global centroid cache
_centroid_cache: dict[int, np.ndarray] = {}


def get_cached_centroid(sig_id: int, centroid_json: str) -> np.ndarray:
    """Get centroid from cache or parse and cache it."""
    if sig_id in _centroid_cache:
        return _centroid_cache[sig_id]

    centroid = unpack_embedding(centroid_json)
    if centroid is not None:
        if len(_centroid_cache) >= CENTROID_CACHE_MAX_SIZE:
            keys_to_remove = list(_centroid_cache.keys())[:CENTROID_CACHE_MAX_SIZE // 2]
            for k in keys_to_remove:
                del _centroid_cache[k]
        _centroid_cache[sig_id] = centroid

    return centroid


def invalidate_centroid_cache(sig_id: int = None):
    """Invalidate centroid cache entry or entire cache."""
    global _centroid_cache
    if sig_id is not None:
        _centroid_cache.pop(sig_id, None)
    else:
        _centroid_cache.clear()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pack_embedding(embedding: Optional[Union[np.ndarray, list]]) -> Optional[str]:
    """Pack a numpy array into JSON string for SQLite storage."""
    if embedding is None:
        return None
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    return json.dumps(embedding)


def unpack_embedding(data) -> Optional[np.ndarray]:
    """Unpack JSON string or bytes into numpy array."""
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    if isinstance(data, bytes):
        return np.frombuffer(data, dtype=np.float32).copy()
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    return None
