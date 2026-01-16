"""Abstract base classes for providers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

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


class DatabaseProvider(ABC):
    """Abstract base class for database providers."""

    @property
    @abstractmethod
    def is_postgresql(self) -> bool:
        """Return True if this is a PostgreSQL backend."""
        pass

    @abstractmethod
    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Get a database connection with automatic commit/rollback."""
        pass

    @abstractmethod
    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Get a database connection in a transaction."""
        pass

    @abstractmethod
    def q(self, sql: str) -> str:
        """Adapt SQL for the database backend.

        Converts SQLite-style ? placeholders to PostgreSQL-style $1, $2, etc.
        """
        pass

    @abstractmethod
    def pack_vector(self, embedding: Union[list, np.ndarray]) -> Any:
        """Pack an embedding for storage.

        SQLite: JSON string
        PostgreSQL with pgvector: native vector type
        """
        pass

    @abstractmethod
    def unpack_vector(self, data: Any) -> Optional[np.ndarray]:
        """Unpack an embedding from storage."""
        pass

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a SQL statement."""
        pass

    @abstractmethod
    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        """Execute and fetch one row."""
        pass

    @abstractmethod
    def fetchall(self, sql: str, params: tuple = ()) -> list:
        """Execute and fetch all rows."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
