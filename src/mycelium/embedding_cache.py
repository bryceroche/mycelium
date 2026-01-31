"""Embedding cache - minimal in-memory implementation."""

import hashlib
from typing import Optional, Callable
import numpy as np


def normalize_cache_key(text: str) -> str:
    """Normalize text for cache key."""
    return text.strip().lower()


def text_hash(text: str) -> str:
    """Hash text for cache key."""
    return hashlib.sha256(normalize_cache_key(text).encode()).hexdigest()[:32]


class EmbeddingCache:
    """Simple in-memory embedding cache."""

    _instance: Optional["EmbeddingCache"] = None

    def __init__(self, maxsize: int = 10000):
        self._cache: dict[str, np.ndarray] = {}
        self._maxsize = maxsize

    @classmethod
    def get_instance(cls) -> "EmbeddingCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def get(self, text: str) -> Optional[np.ndarray]:
        key = text_hash(text)
        return self._cache.get(key)

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = text_hash(text)
        if len(self._cache) >= self._maxsize:
            # Simple eviction: remove first half
            keys = list(self._cache.keys())[:self._maxsize // 2]
            for k in keys:
                del self._cache[k]
        self._cache[key] = embedding

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        """Get from cache or compute and cache."""
        cached = self.get(text)
        if cached is not None:
            return cached
        embedding = compute_fn(text)
        self.put(text, embedding)
        return embedding

    def get_batch(
        self,
        texts: list[str],
        compute_fn: Callable[[list[str]], list[np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Get batch from cache or compute missing."""
        results = {}
        missing = []

        for text in texts:
            cached = self.get(text)
            if cached is not None:
                results[text] = cached
            else:
                missing.append(text)

        if missing:
            computed = compute_fn(missing)
            for text, emb in zip(missing, computed):
                self.put(text, emb)
                results[text] = emb

        return results

    def clear(self) -> None:
        self._cache.clear()


def get_embedding_cache() -> EmbeddingCache:
    """Get singleton cache instance."""
    return EmbeddingCache.get_instance()


def cached_embed(text: str, embedder=None) -> np.ndarray:
    """Get embedding with caching."""
    if embedder is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

    cache = get_embedding_cache()
    return cache.get_or_compute(text, embedder.embed)


async def cached_embed_async(text: str, embedder=None) -> np.ndarray:
    """Async version of cached_embed using asyncio.to_thread to avoid blocking.

    This wraps the synchronous embedder in a thread to prevent blocking the event loop.
    """
    import asyncio

    if embedder is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

    cache = get_embedding_cache()
    cached = cache.get(text)
    if cached is not None:
        return cached

    # Run sync embedding in thread pool to avoid blocking event loop
    embedding = await asyncio.to_thread(embedder.embed, text)
    cache.put(text, embedding)
    return embedding


def cached_embed_batch(texts: list[str], embedder=None) -> dict[str, np.ndarray]:
    """Get multiple embeddings with caching."""
    if embedder is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

    cache = get_embedding_cache()
    return cache.get_batch(texts, embedder.embed_batch)
