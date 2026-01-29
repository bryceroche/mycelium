"""LLM Client stub - minimal for cleanup.

Note: Real LLM functionality removed for local decomposition architecture.
This stub exists only for import compatibility.
"""

from typing import Optional

class LLMClient:
    """Stub LLM client - raises NotImplementedError."""

    def __init__(self, **kwargs):
        pass

    async def generate(self, messages, **kwargs) -> str:
        raise NotImplementedError("LLM calls removed - use local decomposition")

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Use embedder.py for embeddings")

def get_client(**kwargs) -> LLMClient:
    """Get stub client."""
    return LLMClient()

async def ask_llama(prompt: str, **kwargs) -> str:
    """Stub - raises NotImplementedError."""
    raise NotImplementedError("LLM calls removed - use local decomposition")
