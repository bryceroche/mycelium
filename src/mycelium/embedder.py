"""Embedder: Generate embeddings for problem signatures.

Supports multiple backends:
- sentence-transformers (local): MathBERT, MiniLM, etc.
- Gemini API: gemini-embedding-001

The model is loaded lazily on first use.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np

from mycelium.config import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


def is_gemini_model(model_name: str) -> bool:
    """Check if model name is a Gemini embedding model."""
    return model_name.startswith("gemini-") or model_name.startswith("models/")


class GeminiEmbedder:
    """Generate embeddings using Google Gemini API."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy load the Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self._client = genai
            logger.info(f"[embedder] Gemini client configured for model: {self.model_name}")
        return self._client

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        client = self._get_client()
        result = client.embed_content(
            model=f"models/{self.model_name}" if not self.model_name.startswith("models/") else self.model_name,
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return np.array(result['embedding'], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        model_name = f"models/{self.model_name}" if not self.model_name.startswith("models/") else self.model_name
        embeddings = []
        for text in texts:
            result = client.embed_content(
                model=model_name,
                content=text,
                task_type="SEMANTIC_SIMILARITY",
            )
            embeddings.append(result['embedding'])
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (Gemini embedding-001 is 768)."""
        return EMBEDDING_DIM


class SentenceTransformerEmbedder:
    """Generate embeddings using sentence-transformers (local)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"[embedder] Loading model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"[embedder] Model loaded, dim={self._model.get_sentence_embedding_dimension()}")
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this model."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class Embedder:
    """Unified embedder interface supporting multiple backends."""

    _instance: Optional["Embedder"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        if is_gemini_model(model_name):
            self._backend = GeminiEmbedder(model_name)
        else:
            self._backend = SentenceTransformerEmbedder(model_name)

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL) -> "Embedder":
        """Get singleton instance of embedder."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self._backend.embed_batch(texts)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this model."""
        return self._backend.embedding_dim


# =============================================================================
# Convenience functions with caching
# =============================================================================
#
# These functions use the two-tier embedding cache (memory LRU + disk SQLite)
# for efficient caching across process restarts.


def get_embedding(text: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
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


def get_embeddings_batch(texts: list[str], model_name: str = EMBEDDING_MODEL) -> dict[str, np.ndarray]:
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
def embed_text(text: str, model_name: str = EMBEDDING_MODEL) -> tuple[float, ...]:
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
