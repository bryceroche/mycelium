"""DSL Templates and Inference for Auto-Assignment.

DESIGN PHILOSOPHY: No hardcoded mappings. The system learns which DSLs work
for which step_types from execution history. This scales automatically as
new patterns are discovered.

When creating new signatures:
1. Query existing signatures with successful DSL executions
2. Find semantically similar step_types using embeddings
3. Clone the DSL from the best match
4. Fall back to decompose for truly novel patterns (LLM handles it)
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def infer_dsl_for_signature(
    step_type: str,
    description: str,
    db=None,
) -> tuple[Optional[str], str]:
    """Infer DSL script and type for a new signature.

    Uses semantic similarity to find successful DSLs from existing signatures.
    No hardcoded templates - learns from what has worked before.

    Args:
        step_type: The signature's step type (e.g., "compute_sum")
        description: The step description text
        db: Optional signature database for similarity lookup

    Returns:
        Tuple of (dsl_script_json, dsl_type)
    """
    # Try to find similar successful signatures
    if db is not None:
        similar_dsl = _find_similar_successful_dsl(step_type, description, db)
        if similar_dsl:
            return similar_dsl

    # Default fallback: decompose (let LLM handle novel patterns)
    # This is not a failure - it's how the system learns new patterns
    fallback = {
        "type": "decompose",
        "script": "reason_step",
        "params": ["context"],
        "purpose": f"Execute: {description[:50]}",
    }
    return json.dumps(fallback), "decompose"


def _find_similar_successful_dsl(
    step_type: str,
    description: str,
    db,
    min_success_rate: float = 0.6,
    min_uses: int = 3,
) -> Optional[tuple[str, str]]:
    """Find a successful DSL from semantically similar signatures.

    Queries the database for signatures with:
    1. Similar step_type or description (embedding similarity)
    2. Good success rate (DSL actually works)
    3. Enough uses to be reliable

    Returns:
        (dsl_script_json, dsl_type) or None if no good match
    """
    try:
        from mycelium.embedder import Embedder
        import numpy as np

        embedder = Embedder.get_instance()

        # Get embedding for this description
        query_embedding = embedder.embed(f"{step_type}: {description}")

        # Query signatures with successful DSLs
        candidates = db.get_signatures_with_successful_dsls(
            min_success_rate=min_success_rate,
            min_uses=min_uses,
            limit=50,
        )

        if not candidates:
            return None

        # Find best semantic match
        best_match = None
        best_similarity = 0.0

        for sig in candidates:
            # Get or compute signature embedding
            sig_text = f"{sig.step_type}: {sig.description}"
            sig_embedding = embedder.embed(sig_text)

            # Compute cosine similarity
            similarity = float(np.dot(query_embedding, sig_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(sig_embedding)
            ))

            if similarity > best_similarity and sig.dsl_script:
                best_similarity = similarity
                best_match = sig

        # Threshold for accepting a match (0.7 = reasonably similar)
        if best_match and best_similarity >= 0.7:
            logger.info(
                "[dsl_infer] Found similar DSL: '%s' -> '%s' (sim=%.3f)",
                step_type, best_match.step_type, best_similarity
            )
            # Parse and return the DSL
            try:
                dsl_data = json.loads(best_match.dsl_script)
                return best_match.dsl_script, dsl_data.get("type", "math")
            except json.JSONDecodeError:
                pass

        return None

    except Exception as e:
        logger.debug("[dsl_infer] Similarity lookup failed: %s", e)
        return None
