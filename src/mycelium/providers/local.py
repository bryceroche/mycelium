"""Local providers: SQLite, OpenAI, and local MathBERT embeddings.

These wrap the existing mycelium implementations to conform to the provider interface.
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import numpy as np

from .base import LLMProvider, EmbeddingProvider, DatabaseProvider

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Provider - OpenAI
# =============================================================================


class OpenAILLMProvider(LLMProvider):
    """OpenAI-based LLM provider (wraps existing LLMClient)."""

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mycelium.client import LLMClient
            self._client = LLMClient(model=self.model)
        return self._client

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        client = self._get_client()
        return await client.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        client = self._get_client()
        return await client.generate_json(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


# =============================================================================
# Embedding Provider - Local MathBERT
# =============================================================================


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers (MathBERT)."""

    def __init__(self, model_name: str = "tbs17/MathBERT"):
        self.model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from mycelium.embedder import Embedder
            self._embedder = Embedder.get_instance(self.model_name)
        return self._embedder

    @property
    def embedding_dim(self) -> int:
        return self._get_embedder().embedding_dim

    def embed(self, text: str) -> np.ndarray:
        return self._get_embedder().embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._get_embedder().embed_batch(texts)


# =============================================================================
# Database Provider - SQLite
# =============================================================================


class SQLiteProvider(DatabaseProvider):
    """SQLite database provider (wraps existing ConnectionManager)."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.getenv("MYCELIUM_DB_PATH", "mycelium.db")
        self._manager = None

    def _get_manager(self):
        if self._manager is None:
            from mycelium.data_layer.connection import ConnectionManager
            self._manager = ConnectionManager()
        return self._manager

    @property
    def is_postgresql(self) -> bool:
        return False

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        with self._get_manager().connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        with self._get_manager().transaction() as conn:
            yield conn

    def q(self, sql: str) -> str:
        """SQLite uses ? placeholders - no conversion needed."""
        return sql

    def pack_vector(self, embedding: Union[list, np.ndarray]) -> str:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return json.dumps(embedding)

    def unpack_vector(self, data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, str):
            return np.array(json.loads(data), dtype=np.float32)
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return None

    def execute(self, sql: str, params: tuple = ()) -> Any:
        return self._get_manager().execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        return self._get_manager().fetchone(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list:
        return self._get_manager().fetchall(sql, params)

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
