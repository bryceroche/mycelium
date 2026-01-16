"""Provider abstraction layer for DB, LLM, and Embeddings.

Supports local (SQLite, OpenAI, MathBERT) and GCP (Cloud SQL, Vertex AI) backends.

Usage:
    from mycelium.providers import get_llm_provider, get_embedding_provider, get_db_provider

    # Providers are configured via MYCELIUM_PROVIDER environment variable
    # "local" (default) or "gcp"

    llm = get_llm_provider()
    response = await llm.generate(messages)

    embedder = get_embedding_provider()
    embedding = embedder.embed("some text")

    db = get_db_provider()
    with db.connection() as conn:
        ...
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider, EmbeddingProvider, DatabaseProvider

# Provider mode: "local" or "gcp"
PROVIDER_MODE = os.getenv("MYCELIUM_PROVIDER", "local")

_llm_provider: "LLMProvider | None" = None
_embedding_provider: "EmbeddingProvider | None" = None
_db_provider: "DatabaseProvider | None" = None


def get_llm_provider() -> "LLMProvider":
    """Get the configured LLM provider (singleton)."""
    global _llm_provider
    if _llm_provider is None:
        if PROVIDER_MODE == "gcp":
            from .gcp import VertexAILLMProvider
            _llm_provider = VertexAILLMProvider()
        else:
            from .local import OpenAILLMProvider
            _llm_provider = OpenAILLMProvider()
    return _llm_provider


def get_embedding_provider() -> "EmbeddingProvider":
    """Get the configured embedding provider (singleton)."""
    global _embedding_provider
    if _embedding_provider is None:
        if PROVIDER_MODE == "gcp":
            from .gcp import VertexAIEmbeddingProvider
            _embedding_provider = VertexAIEmbeddingProvider()
        else:
            from .local import LocalEmbeddingProvider
            _embedding_provider = LocalEmbeddingProvider()
    return _embedding_provider


def get_db_provider() -> "DatabaseProvider":
    """Get the configured database provider (singleton)."""
    global _db_provider
    if _db_provider is None:
        if PROVIDER_MODE == "gcp":
            from .gcp import CloudSQLProvider
            _db_provider = CloudSQLProvider()
        else:
            from .local import SQLiteProvider
            _db_provider = SQLiteProvider()
    return _db_provider


def reset_providers():
    """Reset all provider singletons (for testing)."""
    global _llm_provider, _embedding_provider, _db_provider
    if _db_provider is not None:
        _db_provider.close()
    _llm_provider = None
    _embedding_provider = None
    _db_provider = None


__all__ = [
    "PROVIDER_MODE",
    "get_llm_provider",
    "get_embedding_provider",
    "get_db_provider",
    "reset_providers",
]
