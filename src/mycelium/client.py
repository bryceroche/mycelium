"""Groq client for Llama-3.3-70B inference."""

import asyncio
import logging
import os
import random
from typing import Optional

import httpx

from mycelium.config import (
    CLIENT_DEFAULT_TIMEOUT,
    CLIENT_DEFAULT_TEMPERATURE,
    SOLVER_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Connection pool defaults
DEFAULT_MAX_CONNECTIONS = 10  # Max concurrent connections
DEFAULT_MAX_KEEPALIVE = 5  # Max idle connections to keep alive
DEFAULT_CONNECT_TIMEOUT = 10.0  # Connection timeout in seconds

# Default model - our flagship (imported from config)
DEFAULT_MODEL = SOLVER_DEFAULT_MODEL


class GroqClient:
    """Async client for Groq API with Llama-3.3-70B.

    Uses connection pooling for efficient parallel LLM calls.
    The pool is lazily initialized on first request and reused.

    Usage:
        # Simple usage (auto-manages pool)
        client = GroqClient()
        response = await client.generate(messages)

        # Explicit cleanup (recommended for long-running processes)
        async with GroqClient() as client:
            response = await client.generate(messages)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = CLIENT_DEFAULT_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        self.timeout = timeout
        self.base_url = "https://api.groq.com/openai/v1"

        # Connection pool configuration
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._connect_timeout = connect_timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the pooled HTTP client (thread-safe)."""
        if self._client is None:
            async with self._client_lock:
                # Double-check after acquiring lock
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive,
                    )
                    timeout = httpx.Timeout(
                        timeout=self.timeout,
                        connect=self._connect_timeout,
                    )
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    logger.debug(
                        f"[groq] Created connection pool: "
                        f"max_conn={self._max_connections} keepalive={self._max_keepalive}"
                    )
        return self._client

    async def close(self) -> None:
        """Close the connection pool and release resources."""
        async with self._client_lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
                logger.debug("[groq] Connection pool closed")

    async def __aenter__(self) -> "GroqClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - closes the pool."""
        await self.close()

    async def generate(
        self,
        messages: list[dict],
        temperature: float = CLIENT_DEFAULT_TEMPERATURE,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text content

        Raises:
            httpx.HTTPStatusError: After max retries exhausted
        """
        logger.debug(f"[groq] model={self.model} temp={temperature} max_tokens={max_tokens}")

        client = await self._get_client()
        last_exception: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                logger.debug(
                    f"[groq] tokens: prompt={usage.get('prompt_tokens', 0)} "
                    f"completion={usage.get('completion_tokens', 0)}"
                )
                # Validate response structure to prevent KeyError/IndexError
                choices = data.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"Invalid API response: missing or empty 'choices' array")
                message = choices[0].get("message")
                if not message or not isinstance(message, dict):
                    raise ValueError(f"Invalid API response: missing 'message' in first choice")
                content = message.get("content")
                if content is None:
                    raise ValueError(f"Invalid API response: missing 'content' in message")
                return content

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise  # Non-retryable error

                if attempt < MAX_RETRIES:
                    delay = self._calculate_backoff(attempt, e.response)
                    logger.warning(
                        f"[groq] Retry {attempt + 1}/{MAX_RETRIES} after {e.response.status_code}, "
                        f"waiting {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                # RemoteProtocolError: "Server disconnected without sending a response"
                last_exception = e
                if attempt < MAX_RETRIES:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"[groq] Retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}, "
                        f"waiting {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        # Exhausted retries
        logger.error(f"[groq] All {MAX_RETRIES} retries exhausted")
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _calculate_backoff(
        self, attempt: int, response: Optional[httpx.Response] = None
    ) -> float:
        """Calculate backoff delay with jitter.

        Uses exponential backoff: base * 2^attempt + random jitter.
        Respects Retry-After header if present (for 429s).
        """
        # Check for Retry-After header (rate limits)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), MAX_DELAY)
                except ValueError:
                    pass

        # Exponential backoff with jitter
        delay = BASE_DELAY * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.1)
        return min(delay + jitter, MAX_DELAY)


# Convenience function
async def ask_llama(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = CLIENT_DEFAULT_TEMPERATURE,
) -> str:
    """Simple interface to ask Llama a question.

    Args:
        prompt: User prompt
        system: Optional system prompt
        temperature: Sampling temperature

    Returns:
        Model response
    """
    async with GroqClient() as client:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await client.generate(messages, temperature=temperature)
