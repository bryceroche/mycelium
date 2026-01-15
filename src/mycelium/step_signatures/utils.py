"""Utility functions for step signatures."""

import hashlib
import json
from functools import lru_cache
from typing import Optional, Union
import numpy as np


# Global centroid cache - keyed by signature_id
# Using a simple dict with manual eviction for flexibility
_centroid_cache: dict[int, np.ndarray] = {}
_CENTROID_CACHE_MAX_SIZE = 10000

# Preloaded centroid matrix for batch similarity (much faster)
_centroid_matrix_cache: dict[str, tuple] = {}  # db_path -> (sig_ids, matrix, timestamp)


def get_cached_centroid(sig_id: int, centroid_json: str) -> np.ndarray:
    """Get centroid from cache or parse and cache it.

    This avoids repeated JSON parsing of the same centroid.
    ~10x faster for repeated lookups.
    """
    if sig_id in _centroid_cache:
        return _centroid_cache[sig_id]

    # Parse and cache
    centroid = unpack_embedding(centroid_json)
    if centroid is not None:
        # Simple eviction: clear half when full
        if len(_centroid_cache) >= _CENTROID_CACHE_MAX_SIZE:
            # Keep most recent half (roughly)
            keys_to_remove = list(_centroid_cache.keys())[:_CENTROID_CACHE_MAX_SIZE // 2]
            for k in keys_to_remove:
                del _centroid_cache[k]
        _centroid_cache[sig_id] = centroid

    return centroid


def invalidate_centroid_cache(sig_id: int = None):
    """Invalidate centroid cache entry or entire cache.

    Call this when a centroid is updated.
    """
    global _centroid_cache
    if sig_id is not None:
        _centroid_cache.pop(sig_id, None)
    else:
        _centroid_cache.clear()


def get_centroid_cache_stats() -> dict:
    """Get cache statistics for monitoring."""
    return {
        "size": len(_centroid_cache),
        "max_size": _CENTROID_CACHE_MAX_SIZE,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray, matrix_normalized: bool = False) -> np.ndarray:
    """Compute cosine similarity between query and all rows in matrix.

    This is ~10-50x faster than looping for large matrices.

    Args:
        query: (768,) query vector
        matrix: (n, 768) matrix of vectors to compare against
        matrix_normalized: If True, skip matrix normalization (already pre-normalized)

    Returns:
        (n,) array of similarities
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    query_normalized = query / query_norm

    if matrix_normalized:
        # Matrix already normalized - just dot product (30% faster!)
        return matrix @ query_normalized

    # Normalize matrix rows (avoid division by zero)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid div by zero
    matrix_normalized_arr = matrix / norms

    # Batch dot product
    return matrix_normalized_arr @ query_normalized


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit length for fast cosine similarity."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def pack_embedding(embedding: Optional[Union[np.ndarray, list]]) -> Optional[str]:
    """Pack a numpy array into JSON string for SQLite storage."""
    if embedding is None:
        return None
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    return json.dumps(embedding)


def unpack_embedding(data: str) -> Optional[np.ndarray]:
    """Unpack JSON string into numpy array."""
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    if isinstance(data, (list, tuple)):
        return np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    return None


# =============================================================================
# CENTROID BUCKET: Coarse-grained uniqueness for embeddings
# =============================================================================
# Quantize embeddings to buckets so near-identical vectors hash to same bucket.
# This provides DB-level uniqueness without exact TEXT matching issues.

# Bucket precision: 1 decimal place means vectors within ~0.05 cosine distance
# will likely share a bucket. This is intentionally coarse - fine-grained
# dedup is handled by application-level similarity checks.
BUCKET_DECIMAL_PLACES = 1


def compute_centroid_bucket(embedding: np.ndarray) -> str:
    """Compute a coarse hash bucket for an embedding.

    Quantizes embedding to BUCKET_DECIMAL_PLACES precision, then hashes.
    Two embeddings with high cosine similarity (>0.95) will likely share a bucket.

    This enables DB-level UNIQUE constraint on centroid_bucket to prevent
    gross duplicates, while allowing minor centroid drift within the bucket.

    Args:
        embedding: 768-dim embedding vector

    Returns:
        Hex string hash (16 chars) suitable for UNIQUE index
    """
    if embedding is None:
        return None

    # Normalize first (so direction matters, not magnitude)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        normalized = embedding / norm
    else:
        normalized = embedding

    # Quantize to N decimal places
    quantized = np.round(normalized, BUCKET_DECIMAL_PLACES)

    # Hash the quantized values (use first 64 dims for speed - still unique enough)
    # Converting to bytes is faster than JSON serialization
    bucket_bytes = quantized[:64].astype(np.float32).tobytes()
    bucket_hash = hashlib.md5(bucket_bytes).hexdigest()[:16]

    return bucket_hash


def embeddings_in_same_bucket(emb1: np.ndarray, emb2: np.ndarray) -> bool:
    """Check if two embeddings would fall in the same bucket."""
    return compute_centroid_bucket(emb1) == compute_centroid_bucket(emb2)
