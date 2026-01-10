"""Embedder: Generate embeddings for problem signatures.

Uses sentence-transformers for fast, local embeddings.
The model is loaded lazily on first use.
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model - small but effective for semantic similarity
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast


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
# Design note: embed_text() returns tuple for LRU cache hashability (np.ndarray
# is not hashable). Use get_embedding() for the typical use case - it calls
# embed_text() internally and converts back to np.ndarray.


@lru_cache(maxsize=1000)
def embed_text(text: str, model_name: str = DEFAULT_MODEL) -> tuple[float, ...]:
    """Embed text with caching.

    Returns tuple (not ndarray) because LRU cache requires hashable return type.
    For most use cases, prefer get_embedding() which returns np.ndarray.

    Args:
        text: Text to embed
        model_name: Model to use for embedding

    Returns:
        Tuple of floats (embedding vector). Use get_embedding() for np.ndarray.
    """
    embedder = Embedder.get_instance(model_name)
    return tuple(embedder.embed(text).tolist())


def get_embedding(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Get embedding as numpy array (cached via embed_text).

    This is the recommended function for most use cases. It uses embed_text()
    internally for caching and converts the result to np.ndarray.

    Args:
        text: Text to embed
        model_name: Model to use for embedding

    Returns:
        Embedding vector as np.ndarray of shape (embedding_dim,)
    """
    return np.array(embed_text(text, model_name), dtype=np.float32)
