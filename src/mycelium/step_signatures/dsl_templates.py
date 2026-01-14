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
    extracted_values: dict = None,
) -> tuple[Optional[str], str]:
    """Infer DSL script and type for a new signature.

    Priority order:
    1. Generate from extracted_values (planner already knows the structure)
    2. Find similar successful DSLs from database
    3. Fall back to decompose for truly novel patterns

    Args:
        step_type: The signature's step type (e.g., "compute_sum")
        description: The step description text
        db: Optional signature database for similarity lookup
        extracted_values: Dict of semantic param names -> values from planner

    Returns:
        Tuple of (dsl_script_json, dsl_type)
    """
    # Priority 1: Generate DSL from extracted_values structure
    # The planner already extracted semantic param names - use them!
    if extracted_values:
        dsl = _infer_dsl_from_values(step_type, description, extracted_values)
        if dsl:
            return dsl

    # Priority 2: Find similar successful signatures
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


def _infer_dsl_from_values(
    step_type: str,
    description: str,
    extracted_values: dict,
) -> Optional[tuple[str, str]]:
    """Generate DSL from planner's extracted values.

    The planner extracts values with semantic names. These names encode
    the mathematical structure - we just need to connect them with operators.

    Examples:
        {"base": 2, "exponent": 10} -> "base ** exponent"
        {"dividend": 100, "divisor": 5} -> "dividend / divisor"
        {"total_km": "{step_1}", "circumference_km": 40000} -> "total_km / circumference_km"

    Args:
        step_type: The signature's step type
        description: The step description
        extracted_values: Dict of semantic param names -> values

    Returns:
        (dsl_json, dsl_type) or None if can't infer
    """
    if not extracted_values:
        return None

    # Filter out step references for param list (keep only numeric values as params)
    params = []
    for key, val in extracted_values.items():
        if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
            # This is a step reference like "{step_1}" - still a valid param
            params.append(key)
        elif isinstance(val, (int, float)):
            params.append(key)
        elif isinstance(val, str):
            try:
                float(val)
                params.append(key)
            except ValueError:
                pass

    if not params:
        return None

    # Infer operation from step_type and description
    step_lower = step_type.lower()
    desc_lower = description.lower()

    # Two-param operations
    if len(params) >= 2:
        p1, p2 = params[0], params[1]

        # PRIORITY 1: Step type is the strongest signal (from planner)
        # Check step_type first before falling back to description keywords
        if "sum" in step_lower or "add" in step_lower:
            script = f"{p1} + {p2}"
            return _make_dsl_json(script, params, f"add {p1} and {p2}"), "math"

        if "subtract" in step_lower or "difference" in step_lower:
            script = f"{p1} - {p2}"
            return _make_dsl_json(script, params, f"subtract {p2} from {p1}"), "math"

        if "product" in step_lower or "multiply" in step_lower:
            script = f"{p1} * {p2}"
            return _make_dsl_json(script, params, f"multiply {p1} and {p2}"), "math"

        if "divide" in step_lower or "quotient" in step_lower or "ratio" in step_lower:
            script = f"{p1} / {p2}"
            return _make_dsl_json(script, params, f"divide {p1} by {p2}"), "math"

        # PRIORITY 2: Description keywords (less reliable, check more carefully)
        # Division patterns (high confidence keywords)
        if any(kw in desc_lower for kw in ["divide", "divided by", "quotient", "per "]):
            script = f"{p1} / {p2}"
            return _make_dsl_json(script, params, f"divide {p1} by {p2}"), "math"

        # Subtraction patterns (check before addition - "left", "remaining" are strong signals)
        if any(kw in desc_lower for kw in ["subtract", "minus", "remaining", " left", "difference"]):
            script = f"{p1} - {p2}"
            return _make_dsl_json(script, params, f"subtract {p2} from {p1}"), "math"

        # Multiplication patterns (avoid "total" - too ambiguous)
        if any(kw in desc_lower for kw in ["multiply", "product", "times", " x "]):
            script = f"{p1} * {p2}"
            return _make_dsl_json(script, params, f"multiply {p1} and {p2}"), "math"

        # Addition patterns
        if any(kw in desc_lower for kw in ["add", "sum", "plus", "combine", "together", "and"]):
            script = f"{p1} + {p2}"
            return _make_dsl_json(script, params, f"add {p1} and {p2}"), "math"

        # "total" with "used" or "spent" context → addition (consuming resources)
        if "total" in desc_lower and any(kw in desc_lower for kw in ["used", "spent", "consumed"]):
            script = f"{p1} + {p2}"
            return _make_dsl_json(script, params, f"add {p1} and {p2}"), "math"

        # "total" with "earn" or "make" context → multiplication (computing amounts)
        if "total" in desc_lower and any(kw in desc_lower for kw in ["earn", "make", "cost", "price"]):
            script = f"{p1} * {p2}"
            return _make_dsl_json(script, params, f"multiply {p1} and {p2}"), "math"

        # Power/exponent patterns
        if any(kw in step_lower or kw in desc_lower for kw in [
            "power", "exponent", "raise", "squared", "cubed"
        ]):
            # Try to identify which param is base vs exponent
            base_param = None
            exp_param = None
            for p in params:
                p_lower = p.lower()
                if any(kw in p_lower for kw in ["base", "number"]):
                    base_param = p
                elif any(kw in p_lower for kw in ["exp", "power", "n"]):
                    exp_param = p
            if base_param and exp_param:
                script = f"{base_param} ** {exp_param}"
            else:
                script = f"{p1} ** {p2}"
            return _make_dsl_json(script, params, f"raise to power"), "math"

        # Modulo patterns
        if any(kw in step_lower or kw in desc_lower for kw in [
            "modulo", "remainder", "mod "
        ]):
            script = f"{p1} % {p2}"
            return _make_dsl_json(script, params, f"compute {p1} mod {p2}"), "math"

        # GCD patterns
        if any(kw in step_lower or kw in desc_lower for kw in ["gcd", "greatest common"]):
            script = f"gcd({p1}, {p2})"
            return _make_dsl_json(script, params, f"compute GCD"), "math"

        # LCM patterns
        if any(kw in step_lower or kw in desc_lower for kw in ["lcm", "least common"]):
            script = f"lcm({p1}, {p2})"
            return _make_dsl_json(script, params, f"compute LCM"), "math"

        # Combination/permutation patterns
        if any(kw in step_lower or kw in desc_lower for kw in ["combination", "choose", "c("]):
            script = f"comb({p1}, {p2})"
            return _make_dsl_json(script, params, f"compute combinations"), "math"

        if any(kw in step_lower or kw in desc_lower for kw in ["permutation", "perm", "p("]):
            script = f"perm({p1}, {p2})"
            return _make_dsl_json(script, params, f"compute permutations"), "math"

    # Single-param operations
    if len(params) >= 1:
        p1 = params[0]

        # Square root
        if any(kw in step_lower or kw in desc_lower for kw in ["sqrt", "square root"]):
            script = f"sqrt({p1})"
            return _make_dsl_json(script, params[:1], f"compute square root"), "math"

        # Factorial
        if "factorial" in step_lower or "factorial" in desc_lower:
            script = f"factorial({p1})"
            return _make_dsl_json(script, params[:1], f"compute factorial"), "math"

    # Couldn't infer operation - let other methods try
    logger.debug(
        "[dsl_infer] Could not infer DSL from values: step_type=%s, params=%s",
        step_type, params
    )
    return None


def _make_dsl_json(script: str, params: list, purpose: str) -> str:
    """Create DSL JSON with standard format."""
    dsl = {
        "type": "math",
        "script": script,
        "params": params,
        "purpose": purpose,
    }
    return json.dumps(dsl)


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
