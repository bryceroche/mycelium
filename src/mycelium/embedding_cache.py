"""Embedding Cache: Fast, persistent caching for MathBERT embeddings.

Embeddings are expensive to compute (~50ms each). This module provides:
- Two-tier cache: fast in-memory LRU + persistent SQLite
- Cache warming from signature DB on startup
- Text normalization for consistent cache keys
- Batch operations for efficient bulk lookups
- Statistics tracking for cache performance monitoring

Per CLAUDE.md: "everything must be automated - system must be independent"
The system should avoid redundant computation automatically.

Per CLAUDE.md "New Favorite Pattern": Uses data_layer connection factory
for centralized database connection management.
"""

import hashlib
import logging
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from mycelium.config import (
    DB_PATH,
    EMBEDDING_DIM,
    EMBEDDING_CACHE_ENABLED,
    EMBEDDING_CACHE_MEMORY_SIZE,
    EMBEDDING_CACHE_PERSIST,
    EMBEDDING_CACHE_WARM_ON_START,
    EMBEDDING_CACHE_TTL_DAYS,
)
from mycelium.data_layer.connection import create_connection_manager

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE STATISTICS
# =============================================================================

@dataclass
class CacheStats:
    """Statistics for embedding cache performance."""
    memory_hits: int = 0
    memory_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    computes: int = 0  # Actually computed via model
    total_requests: int = 0

    # Timing
    total_compute_time_ms: float = 0.0
    total_lookup_time_ms: float = 0.0

    # Size tracking
    memory_size: int = 0
    disk_size: int = 0

    @property
    def memory_hit_rate(self) -> float:
        """Memory cache hit rate."""
        total = self.memory_hits + self.memory_misses
        return self.memory_hits / total if total > 0 else 0.0

    @property
    def disk_hit_rate(self) -> float:
        """Disk cache hit rate (of memory misses)."""
        total = self.disk_hits + self.disk_misses
        return self.disk_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Overall cache hit rate (memory + disk)."""
        hits = self.memory_hits + self.disk_hits
        return hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_compute_time_ms(self) -> float:
        """Average time to compute an embedding."""
        return self.total_compute_time_ms / self.computes if self.computes > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Cache Stats: {self.total_requests} requests\n"
            f"  Memory: {self.memory_hits}/{self.memory_hits + self.memory_misses} "
            f"({self.memory_hit_rate:.1%} hit rate)\n"
            f"  Disk: {self.disk_hits}/{self.disk_hits + self.disk_misses} "
            f"({self.disk_hit_rate:.1%} hit rate)\n"
            f"  Overall: {self.overall_hit_rate:.1%} hit rate\n"
            f"  Computes: {self.computes} (avg {self.avg_compute_time_ms:.1f}ms)\n"
            f"  Size: {self.memory_size} memory, {self.disk_size} disk"
        )


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_cache_key(text: str) -> str:
    """Normalize text for consistent cache keys.

    Applies transformations to ensure semantically similar texts
    get the same cache key:
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Remove punctuation variations

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable as cache key
    """
    if not text:
        return ""

    # Lowercase and strip
    normalized = text.lower().strip()

    # Collapse whitespace
    normalized = " ".join(normalized.split())

    return normalized


def text_hash(text: str) -> str:
    """Generate stable hash for text (used as DB key).

    Args:
        text: Text to hash (should be pre-normalized)

    Returns:
        Hex string hash (32 chars)
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# =============================================================================
# EMBEDDING SERIALIZATION
# =============================================================================

def pack_embedding(embedding: np.ndarray) -> bytes:
    """Pack numpy embedding to bytes for storage.

    Args:
        embedding: Float32 numpy array of shape (EMBEDDING_DIM,)

    Returns:
        Packed bytes (EMBEDDING_DIM * 4 bytes)
    """
    return embedding.astype(np.float32).tobytes()


def unpack_embedding(data: bytes) -> np.ndarray:
    """Unpack bytes to numpy embedding.

    Args:
        data: Packed bytes from pack_embedding()

    Returns:
        Float32 numpy array of shape (EMBEDDING_DIM,)
    """
    return np.frombuffer(data, dtype=np.float32).copy()


# =============================================================================
# LRU MEMORY CACHE
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache for embeddings.

    Uses OrderedDict for O(1) access and LRU eviction.
    """

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache, moving to end (most recent).

        Args:
            key: Normalized text key

        Returns:
            Embedding array or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache, evicting oldest if full.

        Args:
            key: Normalized text key
            embedding: Embedding array to store
        """
        with self._lock:
            if key in self._cache:
                # Update and move to end
                self._cache.move_to_end(key)
                self._cache[key] = embedding
            else:
                # Evict oldest if at capacity
                while len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = embedding

    def get_batch(self, keys: list[str]) -> dict[str, Optional[np.ndarray]]:
        """Get multiple embeddings from cache.

        Args:
            keys: List of normalized text keys

        Returns:
            Dict mapping key -> embedding (or None if not found)
        """
        results = {}
        with self._lock:
            for key in keys:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    results[key] = self._cache[key]
                else:
                    results[key] = None
        return results

    def put_batch(self, items: dict[str, np.ndarray]) -> None:
        """Store multiple embeddings in cache.

        Args:
            items: Dict mapping key -> embedding
        """
        with self._lock:
            for key, embedding in items.items():
                if key in self._cache:
                    self._cache.move_to_end(key)
                    self._cache[key] = embedding
                else:
                    while len(self._cache) >= self.maxsize:
                        self._cache.popitem(last=False)
                    self._cache[key] = embedding

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


# =============================================================================
# PERSISTENT DISK CACHE
# =============================================================================

class DiskCache:
    """SQLite-backed persistent cache for embeddings.

    Survives process restarts. Used as L2 cache behind LRU.

    Per CLAUDE.md "New Favorite Pattern": Uses data_layer connection factory
    for thread-safe access to embedding_cache.db.
    """

    def __init__(self, db_path: Optional[Path] = None, auto_prune: bool = True):
        self.db_path = db_path or Path(DB_PATH).parent / "embedding_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn_mgr = create_connection_manager(str(self.db_path))
        self._init_db()

        # Auto-prune old entries on startup to prevent unbounded growth
        if auto_prune:
            self.prune_old(ttl_days=EMBEDDING_CACHE_TTL_DAYS)

    def _init_db(self) -> None:
        """Initialize the cache database."""
        with self._conn_mgr.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_accessed
                ON embedding_cache(last_accessed_at)
            """)

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from disk cache.

        Args:
            text: Normalized text

        Returns:
            Embedding array or None if not found
        """
        key = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._conn_mgr.connection() as conn:
                row = conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                    (key,)
                ).fetchone()

                if row:
                    # Update access time and count
                    conn.execute("""
                        UPDATE embedding_cache
                        SET last_accessed_at = ?, access_count = access_count + 1
                        WHERE text_hash = ?
                    """, (now, key))
                    return unpack_embedding(row["embedding"])

                return None

        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Disk read error: %s", e)
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in disk cache.

        Args:
            text: Normalized text
            embedding: Embedding array to store
        """
        key = text_hash(text)
        now = datetime.now(timezone.utc).isoformat()
        data = pack_embedding(embedding)

        try:
            with self._conn_mgr.connection() as conn:
                conn.execute("""
                    INSERT INTO embedding_cache
                        (text_hash, text, embedding, created_at, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(text_hash) DO UPDATE SET
                        last_accessed_at = excluded.last_accessed_at,
                        access_count = access_count + 1
                """, (key, text, data, now, now))
        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Disk write error: %s", e)

    def get_batch(self, texts: list[str]) -> dict[str, Optional[np.ndarray]]:
        """Get multiple embeddings from disk cache.

        Args:
            texts: List of normalized texts

        Returns:
            Dict mapping text -> embedding (or None if not found)
        """
        if not texts:
            return {}

        results = {text: None for text in texts}
        keys = {text_hash(text): text for text in texts}
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._conn_mgr.connection() as conn:
                placeholders = ",".join("?" * len(keys))
                rows = conn.execute(
                    f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders})",
                    list(keys.keys())
                ).fetchall()

                found_keys = []
                for row in rows:
                    text = keys[row["text_hash"]]
                    results[text] = unpack_embedding(row["embedding"])
                    found_keys.append(row["text_hash"])

                # Update access times
                if found_keys:
                    placeholders = ",".join("?" * len(found_keys))
                    conn.execute(f"""
                        UPDATE embedding_cache
                        SET last_accessed_at = ?, access_count = access_count + 1
                        WHERE text_hash IN ({placeholders})
                    """, [now] + found_keys)

                return results

        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Disk batch read error: %s", e)
            return results

    def put_batch(self, items: dict[str, np.ndarray]) -> None:
        """Store multiple embeddings in disk cache.

        Args:
            items: Dict mapping normalized text -> embedding
        """
        if not items:
            return

        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._conn_mgr.connection() as conn:
                for text, embedding in items.items():
                    key = text_hash(text)
                    data = pack_embedding(embedding)
                    conn.execute("""
                        INSERT INTO embedding_cache
                            (text_hash, text, embedding, created_at, last_accessed_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(text_hash) DO UPDATE SET
                            last_accessed_at = excluded.last_accessed_at,
                            access_count = access_count + 1
                    """, (key, text, data, now, now))
        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Disk batch write error: %s", e)

    def count(self) -> int:
        """Get number of entries in disk cache."""
        try:
            with self._conn_mgr.connection() as conn:
                row = conn.execute("SELECT COUNT(*) as cnt FROM embedding_cache").fetchone()
                return row["cnt"]
        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Failed to get disk count: %s", e)
            return 0

    def prune_old(self, ttl_days: int = 30) -> int:
        """Remove entries not accessed in ttl_days.

        Args:
            ttl_days: Days since last access before pruning

        Returns:
            Number of entries pruned
        """
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()

        try:
            with self._conn_mgr.connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM embedding_cache WHERE last_accessed_at < ?",
                    (cutoff,)
                )
                count = cursor.rowcount
                logger.info("[embedding_cache] Pruned %d old entries (>%d days)", count, ttl_days)
                return count
        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Prune error: %s", e)
            return 0

    def clear(self) -> None:
        """Clear all entries from disk cache."""
        try:
            with self._conn_mgr.connection() as conn:
                conn.execute("DELETE FROM embedding_cache")
        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Clear error: %s", e)


# =============================================================================
# UNIFIED EMBEDDING CACHE
# =============================================================================

class EmbeddingCache:
    """Two-tier embedding cache with statistics.

    Provides a unified interface to:
    - L1: Fast in-memory LRU cache
    - L2: Persistent SQLite disk cache
    - Statistics tracking for monitoring
    - Cache warming from signature DB

    Usage:
        cache = EmbeddingCache.get_instance()
        embedding = cache.get_or_compute("some text", embedder.embed)
    """

    _instance: Optional["EmbeddingCache"] = None

    def __init__(
        self,
        memory_size: int = None,
        persist: bool = None,
        db_path: Optional[Path] = None,
    ):
        self.memory_size = memory_size or EMBEDDING_CACHE_MEMORY_SIZE
        self.persist = persist if persist is not None else EMBEDDING_CACHE_PERSIST

        self._memory = LRUCache(maxsize=self.memory_size)
        self._disk = DiskCache(db_path) if self.persist else None
        self._stats = CacheStats()
        self._lock = Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingCache":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache (memory then disk).

        Args:
            text: Text to look up (will be normalized)

        Returns:
            Embedding array or None if not cached
        """
        if not EMBEDDING_CACHE_ENABLED:
            return None

        key = normalize_cache_key(text)
        if not key:
            return None

        with self._lock:
            self._stats.total_requests += 1
            start = time.time()

            # Try memory first
            embedding = self._memory.get(key)
            if embedding is not None:
                self._stats.memory_hits += 1
                self._stats.total_lookup_time_ms += (time.time() - start) * 1000
                return embedding

            self._stats.memory_misses += 1

            # Try disk
            if self._disk:
                embedding = self._disk.get(key)
                if embedding is not None:
                    self._stats.disk_hits += 1
                    # Promote to memory
                    self._memory.put(key, embedding)
                    self._stats.total_lookup_time_ms += (time.time() - start) * 1000
                    return embedding

                self._stats.disk_misses += 1

            self._stats.total_lookup_time_ms += (time.time() - start) * 1000
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache (both memory and disk).

        Args:
            text: Text key (will be normalized)
            embedding: Embedding array to store
        """
        if not EMBEDDING_CACHE_ENABLED:
            return

        key = normalize_cache_key(text)
        if not key:
            return

        with self._lock:
            self._memory.put(key, embedding)
            if self._disk:
                self._disk.put(key, embedding)

    def get_or_compute(
        self,
        text: str,
        compute_fn,
    ) -> np.ndarray:
        """Get embedding from cache or compute if missing.

        This is the main entry point for cached embedding lookups.

        Args:
            text: Text to embed
            compute_fn: Function to compute embedding if not cached.
                        Should accept text and return np.ndarray.

        Returns:
            Embedding array (from cache or freshly computed)
        """
        # Try cache first
        embedding = self.get(text)
        if embedding is not None:
            return embedding

        # Compute and cache
        with self._lock:
            self._stats.computes += 1
            start = time.time()

        embedding = compute_fn(text)

        with self._lock:
            self._stats.total_compute_time_ms += (time.time() - start) * 1000

        self.put(text, embedding)
        return embedding

    async def get_or_compute_async(
        self,
        text: str,
        compute_fn,
    ) -> np.ndarray:
        """Async version of get_or_compute.

        Per CLAUDE.md "New Favorite Pattern": Single entry point for async
        cached embedding lookups.

        Args:
            text: Text to embed
            compute_fn: Async function to compute embedding if not cached.
                        Should accept text and return list[float] or np.ndarray.

        Returns:
            Embedding array (from cache or freshly computed)
        """
        # Try cache first (sync - cache is in-memory/disk)
        embedding = self.get(text)
        if embedding is not None:
            return embedding

        # Compute async and cache
        with self._lock:
            self._stats.computes += 1
            start = time.time()

        result = await compute_fn(text)

        # Convert list to numpy if needed
        if isinstance(result, list):
            embedding = np.array(result, dtype=np.float32)
        else:
            embedding = result

        with self._lock:
            self._stats.total_compute_time_ms += (time.time() - start) * 1000

        self.put(text, embedding)
        return embedding

    def get_batch(
        self,
        texts: list[str],
        compute_fn=None,
    ) -> dict[str, np.ndarray]:
        """Get multiple embeddings, computing missing ones.

        Args:
            texts: List of texts to embed
            compute_fn: Function to compute embeddings for missing texts.
                        Should accept list[str] and return np.ndarray (2D).
                        If None, missing texts return None.

        Returns:
            Dict mapping text -> embedding
        """
        if not texts:
            return {}

        # Normalize keys
        keys = {normalize_cache_key(t): t for t in texts if t}
        results = {}

        # Check memory cache
        memory_results = self._memory.get_batch(list(keys.keys()))
        missing_keys = []

        with self._lock:
            for key, orig_text in keys.items():
                self._stats.total_requests += 1
                if memory_results.get(key) is not None:
                    self._stats.memory_hits += 1
                    results[orig_text] = memory_results[key]
                else:
                    self._stats.memory_misses += 1
                    missing_keys.append((key, orig_text))

        # Check disk cache for misses
        if self._disk and missing_keys:
            disk_results = self._disk.get_batch([k for k, _ in missing_keys])
            still_missing = []

            with self._lock:
                for key, orig_text in missing_keys:
                    if disk_results.get(key) is not None:
                        self._stats.disk_hits += 1
                        results[orig_text] = disk_results[key]
                        # Promote to memory
                        self._memory.put(key, disk_results[key])
                    else:
                        self._stats.disk_misses += 1
                        still_missing.append((key, orig_text))

            missing_keys = still_missing

        # Compute missing if function provided
        if compute_fn and missing_keys:
            missing_texts = [orig for _, orig in missing_keys]

            with self._lock:
                self._stats.computes += len(missing_texts)
                start = time.time()

            embeddings = compute_fn(missing_texts)

            with self._lock:
                self._stats.total_compute_time_ms += (time.time() - start) * 1000

            # Store computed embeddings
            to_cache = {}
            for i, (key, orig_text) in enumerate(missing_keys):
                emb = embeddings[i] if len(embeddings.shape) > 1 else embeddings
                results[orig_text] = emb
                to_cache[key] = emb

            self._memory.put_batch(to_cache)
            if self._disk:
                self._disk.put_batch(to_cache)

        return results

    def warm_from_signatures(self, db_path: str = DB_PATH) -> int:
        """Pre-load embeddings from signature database.

        Loads existing signature centroids into memory cache
        for faster startup routing.

        Args:
            db_path: Path to signature database

        Returns:
            Number of embeddings loaded
        """
        if not EMBEDDING_CACHE_WARM_ON_START:
            return 0

        logger.info("[embedding_cache] Warming cache from signatures...")

        try:
            # Use main data layer for signature DB access
            # Per CLAUDE.md "New Favorite Pattern": Centralized connections
            from mycelium.data_layer import get_db
            db = get_db()

            rows = db.fetchall("""
                SELECT description, centroid
                FROM step_signatures
                WHERE centroid IS NOT NULL
                LIMIT 10000
            """)

            count = 0
            for row in rows:
                desc = row["description"]
                centroid_json = row["centroid"]

                if desc and centroid_json:
                    try:
                        import json
                        centroid = np.array(json.loads(centroid_json), dtype=np.float32)
                        key = normalize_cache_key(desc)
                        self._memory.put(key, centroid)
                        count += 1
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("[embedding_cache] Failed to parse centroid JSON: %s", e)

            logger.info("[embedding_cache] Warmed %d embeddings into memory", count)
            return count

        except sqlite3.Error as e:
            logger.warning("[embedding_cache] Failed to warm cache: %s", e)
            return 0

    def prune(self, ttl_days: int = None) -> int:
        """Prune old entries from disk cache.

        Args:
            ttl_days: Days since last access (default from config)

        Returns:
            Number of entries pruned
        """
        if not self._disk:
            return 0

        ttl = ttl_days or EMBEDDING_CACHE_TTL_DAYS
        return self._disk.prune_old(ttl)

    def clear(self) -> None:
        """Clear all caches."""
        self._memory.clear()
        if self._disk:
            self._disk.clear()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.memory_size = len(self._memory)
        self._stats.disk_size = self._disk.count() if self._disk else 0
        return self._stats

    def get_stats_dict(self) -> dict:
        """Get statistics as dict (for JSON serialization)."""
        s = self.stats
        return {
            "memory_hits": s.memory_hits,
            "memory_misses": s.memory_misses,
            "disk_hits": s.disk_hits,
            "disk_misses": s.disk_misses,
            "computes": s.computes,
            "total_requests": s.total_requests,
            "memory_hit_rate": s.memory_hit_rate,
            "disk_hit_rate": s.disk_hit_rate,
            "overall_hit_rate": s.overall_hit_rate,
            "avg_compute_time_ms": s.avg_compute_time_ms,
            "memory_size": s.memory_size,
            "disk_size": s.disk_size,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_embedding_cache() -> EmbeddingCache:
    """Get singleton embedding cache instance."""
    return EmbeddingCache.get_instance()


def cached_embed(text: str, embedder=None) -> np.ndarray:
    """Get embedding with caching.

    Convenience function that uses the singleton cache and embedder.

    Args:
        text: Text to embed
        embedder: Optional embedder instance (uses singleton if None)

    Returns:
        Embedding array
    """
    if embedder is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

    cache = get_embedding_cache()
    return cache.get_or_compute(text, embedder.embed)


async def cached_embed_async(text: str, embed_fn) -> np.ndarray:
    """Async version of cached_embed.

    Per CLAUDE.md "New Favorite Pattern": Single entry point for async
    cached embedding lookups.

    Args:
        text: Text to embed
        embed_fn: Async function that takes text and returns embedding.
                  Typically `embedding_client.embed` where client has
                  async embed() method.

    Returns:
        Embedding array
    """
    cache = get_embedding_cache()
    return await cache.get_or_compute_async(text, embed_fn)


def cached_embed_batch(texts: list[str], embedder=None) -> dict[str, np.ndarray]:
    """Get multiple embeddings with caching.

    Args:
        texts: List of texts to embed
        embedder: Optional embedder instance (uses singleton if None)

    Returns:
        Dict mapping text -> embedding
    """
    if embedder is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

    cache = get_embedding_cache()
    return cache.get_batch(texts, embedder.embed_batch)


def get_cache_stats() -> dict:
    """Get cache statistics as dict."""
    return get_embedding_cache().get_stats_dict()


def warm_embedding_cache(db_path: str = DB_PATH) -> int:
    """Warm the embedding cache from signature database."""
    return get_embedding_cache().warm_from_signatures(db_path)
