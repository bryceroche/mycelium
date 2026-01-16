"""GCP providers: Cloud SQL (PostgreSQL), Vertex AI (Gemini + Embeddings).

All components on one GCP cluster for minimal latency.

Required environment variables:
    GCP_PROJECT_ID: Your GCP project ID
    GCP_REGION: GCP region (e.g., us-central1)
    CLOUD_SQL_CONNECTION_NAME: project:region:instance
    CLOUD_SQL_DB_NAME: Database name (default: mycelium)
    CLOUD_SQL_USER: Database user
    CLOUD_SQL_PASSWORD: Database password

Optional:
    VERTEX_AI_MODEL: Gemini model (default: gemini-1.5-flash)
    VERTEX_AI_EMBEDDING_MODEL: Embedding model (default: text-embedding-004)
"""

import asyncio
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import numpy as np

from .base import LLMProvider, EmbeddingProvider, DatabaseProvider

logger = logging.getLogger(__name__)

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

# Cloud SQL Configuration
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME")
CLOUD_SQL_DB_NAME = os.getenv("CLOUD_SQL_DB_NAME", "mycelium")
CLOUD_SQL_USER = os.getenv("CLOUD_SQL_USER")
CLOUD_SQL_PASSWORD = os.getenv("CLOUD_SQL_PASSWORD")

# Vertex AI Configuration
# Training mode uses a beefier model for higher-quality signature generation
# Once signatures are mature, zero-LLM routing bypasses the model anyway
_TRAINING_MODE = os.getenv("TRAINING_MODE", "true").lower() == "true"
VERTEX_AI_MODEL_TRAINING = os.getenv("VERTEX_AI_MODEL_TRAINING", "gemini-3.0-flash")  # Beefier for training
VERTEX_AI_MODEL_INFERENCE = os.getenv("VERTEX_AI_MODEL_INFERENCE", "gemini-1.5-flash")  # Cheaper for inference
VERTEX_AI_MODEL = VERTEX_AI_MODEL_TRAINING if _TRAINING_MODE else VERTEX_AI_MODEL_INFERENCE
VERTEX_AI_EMBEDDING_MODEL = os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-004")  # 768d (pgvector 2000d limit)


# =============================================================================
# LLM Provider - Vertex AI (Gemini)
# =============================================================================


class VertexAILLMProvider(LLMProvider):
    """Vertex AI Gemini LLM provider.

    Uses google-cloud-aiplatform SDK for low-latency inference
    when running on the same GCP cluster.
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

    Uses text-embedding-004 (768 dims) for semantic matching.
    Matches MathBERT dimensions and stays within pgvector's 2000-dim limit.
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
        self._embedding_dim = 768  # text-embedding-004 (pgvector compatible)

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable required")

    def _initialize(self):
        """Lazy initialization of Vertex AI."""
        if self._initialized:
            return

        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        vertexai.init(project=self.project_id, location=self.region)
        self._client = TextEmbeddingModel.from_pretrained(self.model)
        self._initialized = True
        logger.info(f"[vertex-ai-embed] Initialized model={self.model}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        self._initialize()
        embeddings = self._client.get_embeddings([text])
        return np.array(embeddings[0].values, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Vertex AI supports batches of up to 250 texts.
        """
        self._initialize()

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dim)

        # Batch in chunks of 250 (Vertex AI limit)
        batch_size = 250
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._client.get_embeddings(batch)
            all_embeddings.extend([np.array(e.values, dtype=np.float32) for e in embeddings])

        return np.stack(all_embeddings)


# =============================================================================
# Database Provider - Cloud SQL (PostgreSQL with pgvector)
# =============================================================================


class CloudSQLProvider(DatabaseProvider):
    """Cloud SQL PostgreSQL provider with pgvector for embeddings.

    Supports both direct connection (for Cloud Run/GKE on same VPC)
    and Cloud SQL Auth Proxy connection.
    """

    def __init__(
        self,
        connection_name: Optional[str] = None,
        db_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.connection_name = connection_name or CLOUD_SQL_CONNECTION_NAME
        self.db_name = db_name or CLOUD_SQL_DB_NAME
        self.user = user or CLOUD_SQL_USER
        self.password = password or CLOUD_SQL_PASSWORD
        self._pool = None
        self._initialized = False

        if not all([self.connection_name, self.db_name, self.user, self.password]):
            raise ValueError(
                "Cloud SQL configuration required. Set environment variables: "
                "CLOUD_SQL_CONNECTION_NAME, CLOUD_SQL_DB_NAME, CLOUD_SQL_USER, CLOUD_SQL_PASSWORD"
            )

    def _initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return

        import pg8000
        from google.cloud.sql.connector import Connector

        # Create connector (handles IAM auth and SSL automatically)
        connector = Connector()

        def get_conn():
            return connector.connect(
                self.connection_name,
                "pg8000",
                user=self.user,
                password=self.password,
                db=self.db_name,
            )

        # Create connection pool using sqlalchemy
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool

        self._engine = create_engine(
            "postgresql+pg8000://",
            creator=get_conn,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )

        # Ensure pgvector extension exists
        from sqlalchemy import text
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        self._initialized = True
        logger.info(f"[cloud-sql] Connected to {self.connection_name}/{self.db_name}")

    @property
    def is_postgresql(self) -> bool:
        return True

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Get a database connection with automatic commit/rollback."""
        self._initialize()
        conn = self._engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Get a database connection in a transaction."""
        self._initialize()
        with self._engine.begin() as conn:
            yield conn

    def q(self, sql: str) -> str:
        """Convert SQLite ? placeholders to PostgreSQL $1, $2, etc."""
        # Count placeholders and replace sequentially
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            return f"${count}"

        return re.sub(r"\?", replace, sql)

    def pack_vector(self, embedding: Union[list, np.ndarray]) -> str:
        """Pack embedding for pgvector storage.

        pgvector expects vectors as string: '[1.0, 2.0, 3.0]'
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return "[" + ",".join(str(x) for x in embedding) + "]"

    def unpack_vector(self, data: Any) -> Optional[np.ndarray]:
        """Unpack embedding from pgvector storage."""
        if data is None:
            return None
        if isinstance(data, str):
            # pgvector returns as string '[1.0, 2.0, ...]'
            data = data.strip("[]")
            return np.array([float(x) for x in data.split(",")], dtype=np.float32)
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return None

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a SQL statement."""
        self._initialize()
        with self.connection() as conn:
            from sqlalchemy import text
            return conn.execute(text(self.q(sql)), dict(enumerate(params, 1)))

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[Any]:
        """Execute and fetch one row."""
        self._initialize()
        with self.connection() as conn:
            from sqlalchemy import text
            # Convert ? to $1, $2, etc and params tuple to dict
            pg_sql = self.q(sql)
            # Create param dict: {1: val1, 2: val2, ...}
            param_dict = {str(i): v for i, v in enumerate(params, 1)}
            # Replace $1 with :1, $2 with :2 for SQLAlchemy
            sa_sql = re.sub(r"\$(\d+)", r":\1", pg_sql)
            result = conn.execute(text(sa_sql), param_dict)
            return result.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list:
        """Execute and fetch all rows."""
        self._initialize()
        with self.connection() as conn:
            from sqlalchemy import text
            pg_sql = self.q(sql)
            param_dict = {str(i): v for i, v in enumerate(params, 1)}
            sa_sql = re.sub(r"\$(\d+)", r":\1", pg_sql)
            result = conn.execute(text(sa_sql), param_dict)
            return result.fetchall()

    def close(self) -> None:
        """Close the connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._initialized = False
