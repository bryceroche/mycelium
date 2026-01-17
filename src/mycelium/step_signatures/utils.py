"""Utility functions for step signatures."""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union, TypeVar, Generic
import numpy as np


# Global centroid cache - keyed by signature_id
# Using a simple dict with manual eviction for flexibility
_centroid_cache: dict[int, np.ndarray] = {}
_CENTROID_CACHE_MAX_SIZE = 10000


# =============================================================================
# LRU CACHE WITH TTL
# =============================================================================
# Generic LRU cache with time-to-live support for signature lookups.

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and timestamp."""
    value: T
    timestamp: float


class LRUCacheWithTTL(Generic[T]):
    """LRU cache with TTL expiration.

    Features:
    - O(1) get/put via OrderedDict
    - Automatic TTL expiration on access
    - LRU eviction when max_size exceeded
    - Manual invalidation by key or full clear
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0):
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value if exists and not expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        now = time.monotonic()

        # Check TTL
        if now - entry.timestamp > self._ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    def put(self, key: str, value: T) -> None:
        """Put value into cache."""
        now = time.monotonic()

        # Update existing or add new
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = CacheEntry(value=value, timestamp=now)

        # Evict oldest if over max size
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# =============================================================================
# SIGNATURE LOOKUP CACHES
# =============================================================================
# Separate caches for get_signature and get_children to allow independent tuning.

from mycelium.config import (
    SIGNATURE_CACHE_MAX_SIZE,
    SIGNATURE_CACHE_TTL_SECONDS,
    CHILDREN_CACHE_MAX_SIZE,
)

# Cache for get_signature(id) -> StepSignature
_signature_cache: LRUCacheWithTTL = LRUCacheWithTTL(
    max_size=SIGNATURE_CACHE_MAX_SIZE,
    ttl_seconds=SIGNATURE_CACHE_TTL_SECONDS,
)

# Cache for get_children(parent_id) -> list of children
_children_cache: LRUCacheWithTTL = LRUCacheWithTTL(
    max_size=CHILDREN_CACHE_MAX_SIZE,
    ttl_seconds=SIGNATURE_CACHE_TTL_SECONDS,
)


def get_cached_signature(sig_id: int):
    """Get signature from cache (returns None if not cached)."""
    return _signature_cache.get(str(sig_id))


def cache_signature(sig_id: int, signature) -> None:
    """Cache a signature lookup result."""
    _signature_cache.put(str(sig_id), signature)


def get_cached_children(parent_id: int, for_routing: bool = False):
    """Get children from cache (returns None if not cached)."""
    key = f"{parent_id}:{for_routing}"
    return _children_cache.get(key)


def cache_children(parent_id: int, children: list, for_routing: bool = False) -> None:
    """Cache a get_children result."""
    key = f"{parent_id}:{for_routing}"
    _children_cache.put(key, children)


def invalidate_signature_cache(sig_id: int = None) -> None:
    """Invalidate signature cache entry or entire cache.

    Call this when a signature is updated.
    Also invalidates children cache entries that might reference this signature.
    """
    if sig_id is not None:
        _signature_cache.invalidate(str(sig_id))
        # Also clear children cache since it may contain this signature
        # (more aggressive but simpler than tracking reverse dependencies)
        _children_cache.clear()
    else:
        _signature_cache.clear()
        _children_cache.clear()


def invalidate_children_cache(parent_id: int = None) -> None:
    """Invalidate children cache for a specific parent or all.

    Call this when children relationships change.
    """
    if parent_id is not None:
        _children_cache.invalidate(f"{parent_id}:True")
        _children_cache.invalidate(f"{parent_id}:False")
    else:
        _children_cache.clear()


def get_signature_cache_stats() -> dict:
    """Get statistics for signature lookup caches."""
    return {
        "signature_cache": _signature_cache.stats(),
        "children_cache": _children_cache.stats(),
    }

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
