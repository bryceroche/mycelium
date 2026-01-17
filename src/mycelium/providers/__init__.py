"""Provider abstractions for LLM and Embeddings.

When MYCELIUM_PROVIDER=gcp, uses Vertex AI APIs for LLM and embeddings.
Otherwise, uses OpenAI or Gemini APIs.

The database is always SQLite (managed separately in data_layer/).
"""

from .base import LLMProvider, EmbeddingProvider

__all__ = ["LLMProvider", "EmbeddingProvider"]
