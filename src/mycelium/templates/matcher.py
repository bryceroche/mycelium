"""Template matching via embedding similarity."""
import logging
from typing import Tuple, Optional

from .models import Template
from .db import find_nearest_example, get_template, get_all_templates
from mycelium.embedding_cache import cached_embed

logger = logging.getLogger(__name__)


def match_template(problem: str) -> Tuple[Optional[Template], float]:
    """
    Find the best matching template for a problem.

    Uses embedding similarity to find the nearest example in the database,
    then returns that example's template.

    Returns:
        Tuple of (Template, similarity_score) or (None, 0.0) if no match
    """
    # Embed the problem
    embedding = cached_embed(problem)
    if embedding is None:
        logger.warning("[matcher] Failed to embed problem")
        return None, 0.0

    # Find nearest example
    results = find_nearest_example(embedding, top_k=1)

    if not results:
        # No examples yet - fall back to first template or None
        templates = get_all_templates()
        if templates:
            logger.info("[matcher] No examples found, using first template")
            return templates[0], 0.0
        return None, 0.0

    example, similarity = results[0]
    template = get_template(example.template_id)

    logger.info(f"[matcher] Matched template '{template.name}' with similarity {similarity:.3f}")

    return template, similarity
