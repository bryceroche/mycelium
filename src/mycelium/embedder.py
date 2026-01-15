"""Embedder: Generate embeddings for problem signatures.

Uses sentence-transformers for fast, local embeddings.
The model is loaded lazily on first use.
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model - math-specific for better signature matching
# Options:
#   "all-MiniLM-L6-v2"   - 384 dims, fast (original)
#   "all-mpnet-base-v2"  - 768 dims, better quality
#   "tbs17/MathBERT"     - 768 dims, math-specific (current)
DEFAULT_MODEL = "tbs17/MathBERT"  # Math-trained, better for math steps


class Embedder:
    """Generate embeddings using sentence-transformers."""

    _instance: Optional["Embedder"] = None
    _model = None

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL) -> "Embedder":
        """Get singleton instance of embedder."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"[embedder] Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"[embedder] Model loaded, dim={self._model.get_sentence_embedding_dimension()}")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            2D array of embeddings (num_texts x embedding_dim)
        """
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# =============================================================================
# Convenience functions with caching
# =============================================================================
#
# These functions use the two-tier embedding cache (memory LRU + disk SQLite)
# for efficient caching across process restarts.


def get_embedding(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Get embedding with two-tier caching (memory + disk).

    This is the recommended function for most use cases. Uses the
    EmbeddingCache for efficient caching across restarts.

    Args:
        text: Text to embed
        model_name: Model to use for embedding

    Returns:
        Embedding vector as np.ndarray of shape (embedding_dim,)
    """
    from mycelium.embedding_cache import cached_embed
    embedder = Embedder.get_instance(model_name)
    return cached_embed(text, embedder)


def get_embeddings_batch(texts: list[str], model_name: str = DEFAULT_MODEL) -> dict[str, np.ndarray]:
    """Get multiple embeddings with caching.

    Args:
        texts: List of texts to embed
        model_name: Model to use for embedding

    Returns:
        Dict mapping text -> embedding vector
    """
    from mycelium.embedding_cache import cached_embed_batch
    embedder = Embedder.get_instance(model_name)
    return cached_embed_batch(texts, embedder)


# Legacy function for backward compatibility
@lru_cache(maxsize=1000)
def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> tuple[float, ...]:
    """Embed text with simple LRU caching (legacy).

    Prefer get_embedding() for better caching. This exists for
    backward compatibility.

    Args:
        text: Text to embed
        model_name: Model to use for embedding

    Returns:
        Tuple of floats (embedding vector).
    """
    embedder = Embedder.get_instance(model_name)
    return tuple(embedder.embed(text).tolist())
