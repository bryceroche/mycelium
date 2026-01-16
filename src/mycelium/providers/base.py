"""Abstract base classes for LLM and Embedding providers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional format spec, e.g. {"type": "json_object"}

        Returns:
            Generated text content
        """
        pass

    @abstractmethod
    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate a JSON response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON dict
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass

    async def __aenter__(self) -> "LLMProvider":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this provider."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            2D array of embeddings (num_texts x embedding_dim)
        """
        pass
