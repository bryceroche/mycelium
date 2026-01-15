"""Tests for the embedding cache system."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from mycelium.embedding_cache import (
    normalize_cache_key,
    text_hash,
    pack_embedding,
    unpack_embedding,
    LRUCache,
    DiskCache,
    EmbeddingCache,
    CacheStats,
)


class TestNormalization:
    """Tests for text normalization."""

    def test_lowercase(self):
        assert normalize_cache_key("HELLO") == "hello"

    def test_strip_whitespace(self):
        assert normalize_cache_key("  hello  ") == "hello"

    def test_collapse_spaces(self):
        assert normalize_cache_key("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_cache_key("  HELLO   WORLD  ") == "hello world"

    def test_empty_string(self):
        assert normalize_cache_key("") == ""

    def test_none_like(self):
        assert normalize_cache_key("   ") == ""


class TestTextHash:
    """Tests for text hashing."""

    def test_consistent(self):
        """Same text should always produce same hash."""
        h1 = text_hash("hello world")
        h2 = text_hash("hello world")
        assert h1 == h2

    def test_different(self):
        """Different text should produce different hash."""
        h1 = text_hash("hello")
        h2 = text_hash("world")
        assert h1 != h2

    def test_length(self):
        """Hash should be 32 hex chars (MD5)."""
        h = text_hash("test")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


class TestEmbeddingSerialization:
    """Tests for embedding pack/unpack."""

    def test_round_trip(self):
        """Pack then unpack should preserve embedding."""
        original = np.random.randn(768).astype(np.float32)
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)
        np.testing.assert_array_almost_equal(original, unpacked)

    def test_packed_size(self):
        """Packed embedding should be 768 * 4 bytes."""
        emb = np.zeros(768, dtype=np.float32)
        packed = pack_embedding(emb)
        assert len(packed) == 768 * 4


class TestCacheStats:
    """Tests for CacheStats."""

    def test_default_values(self):
        stats = CacheStats()
        assert stats.memory_hits == 0
        assert stats.total_requests == 0

    def test_memory_hit_rate(self):
        stats = CacheStats(memory_hits=80, memory_misses=20)
        assert stats.memory_hit_rate == 0.8

    def test_memory_hit_rate_zero(self):
        stats = CacheStats()
        assert stats.memory_hit_rate == 0.0

    def test_overall_hit_rate(self):
        stats = CacheStats(
            memory_hits=50,
            memory_misses=50,
            disk_hits=30,
            disk_misses=20,
            total_requests=100,
        )
        assert stats.overall_hit_rate == 0.8  # (50 + 30) / 100

    def test_summary(self):
        stats = CacheStats(total_requests=100, memory_hits=80)
        summary = stats.summary()
        assert "100 requests" in summary


class TestLRUCache:
    """Tests for in-memory LRU cache."""

    def test_put_get(self):
        cache = LRUCache(maxsize=10)
        emb = np.random.randn(768).astype(np.float32)
        cache.put("test", emb)
        result = cache.get("test")
        np.testing.assert_array_equal(emb, result)

    def test_get_missing(self):
        cache = LRUCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.put("b", np.ones(768, dtype=np.float32))
        cache.put("c", np.full(768, 2, dtype=np.float32))

        # "a" should be evicted (oldest)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_lru_order(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.put("b", np.ones(768, dtype=np.float32))

        # Access "a" to make it most recent
        cache.get("a")

        # Add "c" - should evict "b" (now oldest)
        cache.put("c", np.full(768, 2, dtype=np.float32))

        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_len(self):
        cache = LRUCache(maxsize=10)
        assert len(cache) == 0
        cache.put("a", np.zeros(768, dtype=np.float32))
        assert len(cache) == 1

    def test_contains(self):
        cache = LRUCache(maxsize=10)
        cache.put("a", np.zeros(768, dtype=np.float32))
        assert "a" in cache
        assert "b" not in cache

    def test_clear(self):
        cache = LRUCache(maxsize=10)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.clear()
        assert len(cache) == 0

    def test_get_batch(self):
        cache = LRUCache(maxsize=10)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.put("b", np.ones(768, dtype=np.float32))

        results = cache.get_batch(["a", "b", "c"])
        assert results["a"] is not None
        assert results["b"] is not None
        assert results["c"] is None

    def test_put_batch(self):
        cache = LRUCache(maxsize=10)
        cache.put_batch({
            "a": np.zeros(768, dtype=np.float32),
            "b": np.ones(768, dtype=np.float32),
        })
        assert cache.get("a") is not None
        assert cache.get("b") is not None


class TestDiskCache:
    """Tests for persistent disk cache."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cache.db"

    def test_put_get(self, temp_db):
        cache = DiskCache(temp_db)
        emb = np.random.randn(768).astype(np.float32)
        cache.put("test", emb)
        result = cache.get("test")
        np.testing.assert_array_almost_equal(emb, result)

    def test_get_missing(self, temp_db):
        cache = DiskCache(temp_db)
        assert cache.get("nonexistent") is None

    def test_persistence(self, temp_db):
        """Data should survive cache recreation."""
        emb = np.random.randn(768).astype(np.float32)

        cache1 = DiskCache(temp_db)
        cache1.put("test", emb)

        # Create new cache instance
        cache2 = DiskCache(temp_db)
        result = cache2.get("test")
        np.testing.assert_array_almost_equal(emb, result)

    def test_count(self, temp_db):
        cache = DiskCache(temp_db)
        assert cache.count() == 0
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.put("b", np.ones(768, dtype=np.float32))
        assert cache.count() == 2

    def test_clear(self, temp_db):
        cache = DiskCache(temp_db)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.clear()
        assert cache.count() == 0

    def test_get_batch(self, temp_db):
        cache = DiskCache(temp_db)
        cache.put("a", np.zeros(768, dtype=np.float32))
        cache.put("b", np.ones(768, dtype=np.float32))

        results = cache.get_batch(["a", "b", "c"])
        assert results["a"] is not None
        assert results["b"] is not None
        assert results["c"] is None

    def test_put_batch(self, temp_db):
        cache = DiskCache(temp_db)
        cache.put_batch({
            "a": np.zeros(768, dtype=np.float32),
            "b": np.ones(768, dtype=np.float32),
        })
        assert cache.get("a") is not None
        assert cache.get("b") is not None


class TestEmbeddingCache:
    """Tests for unified two-tier cache."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cache.db"

    @pytest.fixture
    def cache(self, temp_db):
        """Create cache with temp disk storage."""
        # Reset singleton
        EmbeddingCache.reset_instance()
        return EmbeddingCache(memory_size=100, persist=True, db_path=temp_db)

    def test_get_put(self, cache):
        emb = np.random.randn(768).astype(np.float32)
        cache.put("test text", emb)
        result = cache.get("test text")
        np.testing.assert_array_almost_equal(emb, result)

    def test_normalization(self, cache):
        """Text should be normalized before caching."""
        emb = np.random.randn(768).astype(np.float32)
        cache.put("  HELLO  WORLD  ", emb)
        # Should find with different whitespace/case
        result = cache.get("hello world")
        np.testing.assert_array_almost_equal(emb, result)

    def test_memory_hit(self, cache):
        emb = np.random.randn(768).astype(np.float32)
        cache.put("test", emb)
        cache.get("test")
        assert cache.stats.memory_hits == 1

    def test_disk_hit(self, cache, temp_db):
        emb = np.random.randn(768).astype(np.float32)
        cache.put("test", emb)

        # Clear memory to force disk lookup
        cache._memory.clear()

        result = cache.get("test")
        np.testing.assert_array_almost_equal(emb, result)
        assert cache.stats.disk_hits == 1

    def test_get_or_compute(self, cache):
        computed = []

        def compute_fn(text):
            computed.append(text)
            return np.random.randn(768).astype(np.float32)

        # First call should compute
        emb1 = cache.get_or_compute("test", compute_fn)
        assert len(computed) == 1

        # Second call should use cache
        emb2 = cache.get_or_compute("test", compute_fn)
        assert len(computed) == 1  # Still 1, no new compute
        np.testing.assert_array_equal(emb1, emb2)

    def test_stats(self, cache):
        cache.put("test", np.zeros(768, dtype=np.float32))
        cache.get("test")
        cache.get("nonexistent")

        stats = cache.stats
        assert stats.memory_hits == 1
        assert stats.memory_misses == 1
        assert stats.total_requests == 2

    def test_get_stats_dict(self, cache):
        cache.put("test", np.zeros(768, dtype=np.float32))
        cache.get("test")

        stats_dict = cache.get_stats_dict()
        assert isinstance(stats_dict, dict)
        assert "memory_hits" in stats_dict
        assert "overall_hit_rate" in stats_dict

    def test_clear(self, cache):
        cache.put("test", np.zeros(768, dtype=np.float32))
        cache.clear()
        assert cache.get("test") is None
        assert cache.stats.memory_hits == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_embedding_cache_singleton(self):
        from mycelium.embedding_cache import get_embedding_cache, EmbeddingCache

        EmbeddingCache.reset_instance()
        c1 = get_embedding_cache()
        c2 = get_embedding_cache()
        assert c1 is c2
