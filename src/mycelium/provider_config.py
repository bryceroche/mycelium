"""Provider-specific configuration.

This module provides unified access to providers (LLM, Embeddings, DB) based on
the MYCELIUM_PROVIDER environment variable.

Usage:
    from mycelium.provider_config import llm, embedder, db

    # These work regardless of local or GCP mode
    response = await llm.generate(messages)
    embedding = embedder.embed("text")
    with db.connection() as conn:
        ...

Environment variables:
    MYCELIUM_PROVIDER: "local" (default) or "gcp"

Local mode uses:
    - SQLite database (MYCELIUM_DB_PATH)
    - OpenAI API (OPENAI_API_KEY)
    - Local MathBERT embeddings

GCP mode uses:
    - Cloud SQL PostgreSQL (CLOUD_SQL_* vars)
    - Vertex AI Gemini (GCP_PROJECT_ID, GCP_REGION)
    - Vertex AI text embeddings
"""

import os

# Provider mode
PROVIDER_MODE = os.getenv("MYCELIUM_PROVIDER", "local")
IS_GCP = PROVIDER_MODE == "gcp"

# Lazy-loaded provider instances
_llm = None
_embedder = None
_db = None


def get_llm():
    """Get the LLM provider (lazy singleton)."""
    global _llm
    if _llm is None:
        from mycelium.providers import get_llm_provider
        _llm = get_llm_provider()
    return _llm


def get_embedder():
    """Get the embedding provider (lazy singleton)."""
    global _embedder
    if _embedder is None:
        from mycelium.providers import get_embedding_provider
        _embedder = get_embedding_provider()
    return _embedder


def get_db():
    """Get the database provider (lazy singleton)."""
    global _db
    if _db is None:
        from mycelium.providers import get_db_provider
        _db = get_db_provider()
    return _db


# Convenience accessors (property-like)
class _LLMProxy:
    """Proxy for lazy LLM access."""
    def __getattr__(self, name):
        return getattr(get_llm(), name)

class _EmbedderProxy:
    """Proxy for lazy embedder access."""
    def __getattr__(self, name):
        return getattr(get_embedder(), name)

class _DBProxy:
    """Proxy for lazy DB access."""
    def __getattr__(self, name):
        return getattr(get_db(), name)


# Export proxies for easy import
llm = _LLMProxy()
embedder = _EmbedderProxy()
db = _DBProxy()


def reset_all():
    """Reset all provider singletons (for testing)."""
    global _llm, _embedder, _db
    from mycelium.providers import reset_providers
    reset_providers()
    _llm = None
    _embedder = None
    _db = None
