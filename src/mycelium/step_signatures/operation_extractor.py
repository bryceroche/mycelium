"""Operation Extraction from Problem Text.

Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.

This module extracts the computational operation needed from problem text.
The extracted operation is then embedded and compared against computation
graph embeddings for routing.

Flow:
    Problem text → LLM extracts operation → Embed operation → Compare to graph embeddings

Example:
    "Calculate 25% of 200" → "multiply percentage by base then divide by 100"
    → embed → matches MUL(DIV(param_0, CONST(100)), param_1)
"""

import json
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.client import LLMClient

logger = logging.getLogger(__name__)

# Prompt for operation extraction
OPERATION_EXTRACTION_PROMPT = '''You are an operation extractor for a math problem solver.

Given a math problem or step, extract WHAT COMPUTATION is needed - not the specific numbers or context.

## Rules

1. Focus on the OPERATION TYPE, not the specific values
2. Use generic terms: "multiply", "divide", "add", "subtract", "raise to power", "find GCD", etc.
3. Describe the computation structure: "multiply two numbers then add a third"
4. Be concise - just the operation, not a full explanation
5. If multiple operations, describe the sequence: "divide A by B, then multiply by C"

## Examples

Problem: "Calculate 25% of 200"
Operation: "divide percentage by 100, multiply by base value"

Problem: "Find 2^8"
Operation: "raise base to exponent power"

Problem: "What is the GCD of 24 and 36?"
Operation: "find greatest common divisor of two numbers"

Problem: "If a train travels 60 mph for 3 hours, how far does it go?"
Operation: "multiply rate by time"

Problem: "A rectangle has length 5 and width 3. What is its area?"
Operation: "multiply length by width"

Problem: "What is 15% more than 200?"
Operation: "multiply base by percentage divided by 100, add to base"

Problem: "Simplify (3x + 2) - (x - 4)"
Operation: "subtract polynomials by distributing negative and combining like terms"

Now extract the operation for this problem:

Problem: {problem_text}
Operation:'''


async def extract_operation_needed(
    client,  # LLMClient
    problem_text: str,
    timeout: float = 10.0,
) -> Optional[str]:
    """Extract the computational operation needed from problem text.

    This is the key for routing: we embed this operation description and
    compare it to computation graph embeddings to find matching signatures.

    Args:
        client: LLM client for extraction
        problem_text: The problem or step text
        timeout: Request timeout in seconds

    Returns:
        Operation description (e.g., "multiply base by exponent power") or None

    Example:
        >>> op = await extract_operation_needed(client, "Calculate 2^8")
        >>> op
        'raise base to exponent power'
    """
    if not problem_text or not problem_text.strip():
        return None

    # Clean and truncate input
    problem_text = problem_text.strip()
    if len(problem_text) > 500:
        problem_text = problem_text[:500] + "..."

    prompt = OPERATION_EXTRACTION_PROMPT.format(problem_text=problem_text)

    try:
        messages = [
            {
                "role": "system",
                "content": "You extract computational operations. Respond with only the operation description, nothing else."
            },
            {"role": "user", "content": prompt},
        ]

        response = await client.generate(messages, temperature=0.0)

        if not response:
            logger.warning("[op_extract] Empty response from LLM")
            return None

        # Clean up response
        operation = response.strip()

        # Remove any quotes that might wrap the response
        if operation.startswith('"') and operation.endswith('"'):
            operation = operation[1:-1]
        if operation.startswith("'") and operation.endswith("'"):
            operation = operation[1:-1]

        # Validate - should be a reasonable length
        if len(operation) < 3:
            logger.warning("[op_extract] Operation too short: %s", operation)
            return None
        if len(operation) > 200:
            # Truncate overly verbose responses
            operation = operation[:200]

        logger.debug("[op_extract] Extracted operation: %s", operation)
        return operation

    except Exception as e:
        logger.error("[op_extract] Failed to extract operation: %s", e)
        return None


# Cache for operation embeddings
# Key: operation description, Value: embedding vector
_operation_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX_SIZE = 1000


async def get_operation_embedding(
    embedding_client,  # Embedding client (e.g., gemini)
    operation: str,
) -> Optional[list[float]]:
    """Get embedding for an operation description, with caching.

    Args:
        embedding_client: Client for embedding generation
        operation: Operation description to embed

    Returns:
        Embedding vector or None
    """
    global _operation_embedding_cache

    if not operation:
        return None

    # Normalize for cache lookup
    cache_key = operation.lower().strip()

    # Check cache
    if cache_key in _operation_embedding_cache:
        logger.debug("[op_embed] Cache hit for: %s", cache_key[:50])
        return _operation_embedding_cache[cache_key]

    # Generate embedding
    try:
        embedding = await embedding_client.embed(operation)

        if embedding:
            # Cache with LRU-like behavior
            if len(_operation_embedding_cache) >= _CACHE_MAX_SIZE:
                # Remove oldest entries (first 10%)
                keys_to_remove = list(_operation_embedding_cache.keys())[:_CACHE_MAX_SIZE // 10]
                for k in keys_to_remove:
                    del _operation_embedding_cache[k]

            _operation_embedding_cache[cache_key] = embedding
            logger.debug("[op_embed] Cached embedding for: %s", cache_key[:50])

        return embedding

    except Exception as e:
        logger.error("[op_embed] Failed to embed operation: %s", e)
        return None


def clear_operation_cache() -> None:
    """Clear the operation embedding cache."""
    global _operation_embedding_cache
    _operation_embedding_cache.clear()
    logger.debug("[op_embed] Cache cleared")


async def extract_and_embed_operation(
    llm_client,
    embedding_client,
    problem_text: str,
) -> tuple[Optional[str], Optional[list[float]]]:
    """Extract operation from problem and embed it in one call.

    Convenience function combining extraction and embedding.

    Args:
        llm_client: LLM client for operation extraction
        embedding_client: Client for embedding generation
        problem_text: The problem or step text

    Returns:
        Tuple of (operation_description, embedding) or (None, None)

    Example:
        >>> op, emb = await extract_and_embed_operation(llm, gemini, "Calculate 2^8")
        >>> op
        'raise base to exponent power'
        >>> len(emb)
        3072
    """
    operation = await extract_operation_needed(llm_client, problem_text)
    if not operation:
        return None, None

    embedding = await get_operation_embedding(embedding_client, operation)
    return operation, embedding
