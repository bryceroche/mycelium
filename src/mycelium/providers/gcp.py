"""GCP providers: Vertex AI (Gemini LLM + Embeddings).

Simple setup: SQLite database + Vertex AI APIs for LLM and embeddings.

Required environment variables:
    GCP_PROJECT_ID: Your GCP project ID
    GCP_REGION: GCP region (e.g., us-central1)

Optional:
    TRAINING_MODE: "true" for beefier training model (default: true)
    VERTEX_AI_MODEL_TRAINING: Training model (default: gemini-2.0-flash)
    VERTEX_AI_MODEL_INFERENCE: Inference model (default: gemini-1.5-flash)
    VERTEX_AI_EMBEDDING_MODEL: Embedding model (default: text-embedding-004)
"""

import asyncio
import json
import logging
import os
from typing import Optional

import numpy as np

from mycelium.config import EMBEDDING_DIM
from .base import LLMProvider, EmbeddingProvider

logger = logging.getLogger(__name__)

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

# Vertex AI Configuration
# Training mode uses a beefier model for higher-quality signature generation
# Once signatures are mature, zero-LLM routing bypasses the model anyway
_TRAINING_MODE = os.getenv("TRAINING_MODE", "true").lower() == "true"
VERTEX_AI_MODEL_TRAINING = os.getenv("VERTEX_AI_MODEL_TRAINING", "gemini-2.5-pro")
VERTEX_AI_MODEL_INFERENCE = os.getenv("VERTEX_AI_MODEL_INFERENCE", "gemini-2.0-flash")
VERTEX_AI_MODEL = VERTEX_AI_MODEL_TRAINING if _TRAINING_MODE else VERTEX_AI_MODEL_INFERENCE
VERTEX_AI_EMBEDDING_MODEL = os.getenv("VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001")  # 3072d


# =============================================================================
# LLM Provider - Vertex AI (Gemini)
# =============================================================================


class VertexAILLMProvider(LLMProvider):
    """Vertex AI Gemini LLM provider.

    Uses google-cloud-aiplatform SDK for Vertex AI inference.
    """

    def __init__(
        self,
        model: str = VERTEX_AI_MODEL,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.model = model
        self.project_id = project_id or GCP_PROJECT_ID
        self.region = region or GCP_REGION
        self._client = None
        self._initialized = False

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable required")

    def _initialize(self):
        """Lazy initialization of Vertex AI."""
        if self._initialized:
            return

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project_id, location=self.region)
        self._client = GenerativeModel(self.model)
        self._initialized = True
        logger.info(f"[vertex-ai] Initialized model={self.model} region={self.region}")

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate completion using Vertex AI Gemini."""
        self._initialize()

        from vertexai.generative_models import GenerationConfig, Content, Part

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(Content(role="user", parts=[Part.from_text(content)]))
            elif role == "assistant":
                contents.append(Content(role="model", parts=[Part.from_text(content)]))

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Add JSON mode if requested
        if response_format and response_format.get("type") == "json_object":
            generation_config.response_mime_type = "application/json"

        # Create model with system instruction if provided
        if system_instruction:
            from vertexai.generative_models import GenerativeModel
            model = GenerativeModel(
                self.model,
                system_instruction=system_instruction,
            )
        else:
            model = self._client

        # Run generation (sync call wrapped for async compatibility)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                contents,
                generation_config=generation_config,
            )
        )

        return response.text

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate JSON response using Vertex AI Gemini."""
        content = await self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(content)

    async def close(self) -> None:
        """No persistent connection to close for Vertex AI."""
        self._client = None
        self._initialized = False


# =============================================================================
# Embedding Provider - Vertex AI Text Embeddings
# =============================================================================


class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Vertex AI text embedding provider.

    Uses gemini-embedding-001 (3072 dims) for state-of-the-art semantic matching.
    """

    def __init__(
        self,
        model: str = VERTEX_AI_EMBEDDING_MODEL,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.model = model
        self.project_id = project_id or GCP_PROJECT_ID
        self.region = region or GCP_REGION
        self._client = None
        self._initialized = False
        self._embedding_dim = EMBEDDING_DIM

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable required")

    def _initialize(self):
        """Lazy initialization of Vertex AI."""
        if self._initialized:
            return

        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.region,
        )
        self._initialized = True
        logger.info(f"[vertex-ai-embed] Initialized model={self.model} dim={self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        self._initialize()
        from google.genai.types import EmbedContentConfig

        response = self._client.models.embed_content(
            model=self.model,
            contents=[text],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._embedding_dim,
            ),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Vertex AI supports batches of up to 250 texts.
        """
        self._initialize()
        from google.genai.types import EmbedContentConfig

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        # Batch in chunks of 100 (conservative for gemini-embedding-001)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self._embedding_dim,
                ),
            )
            all_embeddings.extend([np.array(e.values, dtype=np.float32) for e in response.embeddings])

        return np.stack(all_embeddings)
