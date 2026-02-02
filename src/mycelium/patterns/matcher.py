"""Embedding-based pattern matching."""
import logging
import numpy as np
from typing import Tuple, Optional, List, Dict

from .registry import PATTERNS, Pattern
from mycelium.embedding_cache import cached_embed

logger = logging.getLogger(__name__)

# Cache for pattern example embeddings
_pattern_embeddings: Dict[str, List[Tuple[str, np.ndarray]]] = {}


def _get_pattern_embeddings() -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Get or compute embeddings for all pattern examples."""
    global _pattern_embeddings

    if _pattern_embeddings:
        return _pattern_embeddings

    logger.info("[matcher] Computing pattern example embeddings...")

    for name, pattern in PATTERNS.items():
        examples_with_embeddings = []
        for example in pattern.examples:
            embedding = cached_embed(example)
            if embedding is not None:
                examples_with_embeddings.append((example, embedding))
        _pattern_embeddings[name] = examples_with_embeddings
        logger.debug(f"[matcher] Loaded {len(examples_with_embeddings)} examples for '{name}'")

    total = sum(len(v) for v in _pattern_embeddings.values())
    logger.info(f"[matcher] Loaded {total} total examples across {len(PATTERNS)} patterns")

    return _pattern_embeddings


def match_pattern(problem: str) -> Tuple[Optional[Pattern], float, Optional[str]]:
    """
    Find the best matching pattern for a problem using embedding similarity.

    Args:
        problem: The problem text

    Returns:
        Tuple of (Pattern, similarity_score, example_id) or (None, 0.0, None) if no match
        example_id is a hash of the matched example text for Welford tracking
    """
    # Embed the problem
    problem_embedding = cached_embed(problem)
    if problem_embedding is None:
        logger.warning("[matcher] Failed to embed problem")
        return None, 0.0, None

    # Normalize
    problem_embedding = problem_embedding / (np.linalg.norm(problem_embedding) + 1e-9)

    # Get pattern embeddings
    pattern_embeddings = _get_pattern_embeddings()

    # Find best match across all patterns
    best_pattern_name = None
    best_similarity = -1.0
    best_example = None

    for pattern_name, examples in pattern_embeddings.items():
        for example_text, example_embedding in examples:
            # Normalize example embedding
            example_norm = example_embedding / (np.linalg.norm(example_embedding) + 1e-9)

            # Compute similarity
            similarity = float(np.dot(problem_embedding, example_norm))

            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern_name = pattern_name
                best_example = example_text

    if best_pattern_name is None:
        logger.warning("[matcher] No pattern examples found")
        return PATTERNS.get("sequential"), 0.0, None  # Default fallback

    pattern = PATTERNS[best_pattern_name]
    # Create example_id from hash of example text (stable identifier)
    example_id = f"{best_pattern_name}:{hash(best_example)}"

    logger.info(f"[matcher] Matched '{best_pattern_name}' (sim={best_similarity:.3f})")
    logger.debug(f"[matcher] Best example: {best_example[:60]}...")

    return pattern, best_similarity, example_id


def clear_cache():
    """Clear the pattern embeddings cache."""
    global _pattern_embeddings
    _pattern_embeddings = {}
