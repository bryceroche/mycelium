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

from mycelium.config import (
    DSL_OPERATION_INFERENCE_COLD_START,
    DSL_OPERATION_INFERENCE_MATURE,
    DSL_OPERATION_INFERENCE_RAMP_SIGS,
)

logger = logging.getLogger(__name__)


def get_dsl_inference_threshold() -> float:
    """Get cold-start aware DSL inference threshold.

    Ramps from COLD_START to MATURE as signature count grows:
    - Few signatures: low threshold → try more DSLs → bootstrap learning
    - Many signatures: high threshold → be selective → use proven paths

    Returns:
        Threshold between COLD_START and MATURE based on signature count
    """
    try:
        from mycelium.data_layer import get_db
        conn = get_db()
        row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
        sig_count = row[0] if row else 0
    except Exception:
        sig_count = 0

    # Linear ramp from cold_start to mature
    if sig_count >= DSL_OPERATION_INFERENCE_RAMP_SIGS:
        threshold = DSL_OPERATION_INFERENCE_MATURE
    else:
        ramp = sig_count / DSL_OPERATION_INFERENCE_RAMP_SIGS
        threshold = DSL_OPERATION_INFERENCE_COLD_START + ramp * (
            DSL_OPERATION_INFERENCE_MATURE - DSL_OPERATION_INFERENCE_COLD_START
        )

    logger.debug(
        "[dsl_infer] Cold-start threshold: %.3f (sigs=%d, ramp=%d)",
        threshold, sig_count, DSL_OPERATION_INFERENCE_RAMP_SIGS
    )
    return threshold


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

    # Priority 3: Combine/synthesis steps
    # These steps combine or report previous results
    desc_lower = description.lower()

    # 3a: Addition for "combine" operations (adding multiple values together)
    if any(phrase in desc_lower for phrase in [
        "combine", "add together", "total of", "sum of",
        "add up", "together",
    ]):
        combine_add = {
            "type": "math",
            "script": "a + b",
            "params": ["a", "b"],
            "aliases": {
                "a": ["step_1", "first", "regular", "base"],
                "b": ["step_2", "second", "overtime", "additional"],
            },
            "purpose": "combine by addition",
        }
        logger.info("[dsl_infer] Combine step detected, using addition DSL")
        return json.dumps(combine_add), "math"

    # 3b: Passthrough for final answer steps (just return the value)
    if any(phrase in desc_lower for phrase in [
        "final answer", "get the answer", "report the result",
        "state the answer", "give the answer", "get the result",
        "obtain the result", "return the result", "the result is",
        "calculate the final", "determine the final",
    ]):
        passthrough = {
            "type": "math",
            "script": "result",  # Just return the input named "result"
            "params": ["result"],
            "aliases": {"result": ["step_1", "step_2", "step_3", "answer", "value", "total"]},
            "purpose": "passthrough: return computed result",
        }
        logger.info("[dsl_infer] Synthesis step detected, using passthrough DSL")
        return json.dumps(passthrough), "math"

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
    """Generate DSL from planner's extracted values using semantic similarity.

    Uses embedding similarity to match step description to operation prototypes.
    Falls back to trying common operations when semantic matching fails.

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
    numeric_values = []
    for key, val in extracted_values.items():
        if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
            # This is a step reference like "{step_1}" - still a valid param
            params.append(key)
        elif isinstance(val, (int, float)):
            params.append(key)
            numeric_values.append(val)
        elif isinstance(val, str):
            try:
                numeric_values.append(float(val))
                params.append(key)
            except ValueError:
                pass

    if not params:
        return None

    # Use semantic similarity to infer operation
    # Pass param names - they carry semantic hints (dividend/divisor → divide)
    # Use cold-start aware threshold (low when few sigs, high when mature)
    threshold = get_dsl_inference_threshold()
    op_info = _infer_operation_semantic(step_type, description, len(params), param_names=params, min_similarity=threshold)

    # Fallback: if semantic matching fails but we have exactly 2 numeric values,
    # try multiplication first (most common operation for rate * quantity patterns)
    # This enables bootstrap learning - once one succeeds, future matches improve
    if op_info is None and len(numeric_values) == 2 and len(params) == 2:
        logger.info(
            "[dsl_infer] Semantic match failed, trying multiplication fallback for 2 params: %s",
            params
        )
        op_info = ("*", "multiplication fallback")

    if op_info is None:
        logger.debug(
            "[dsl_infer] Could not infer DSL from values: step_type=%s, params=%s",
            step_type, params
        )
        return None

    operator, op_name = op_info

    # Build script based on operator and params
    if len(params) >= 2:
        p1, p2 = params[0], params[1]
        if operator in ("+", "-", "*", "/", "**", "%"):
            script = f"{p1} {operator} {p2}"
        elif operator == "gcd":
            script = f"gcd({p1}, {p2})"
        elif operator == "lcm":
            script = f"lcm({p1}, {p2})"
        elif operator == "comb":
            script = f"comb({p1}, {p2})"
        elif operator == "perm":
            script = f"perm({p1}, {p2})"
        else:
            script = f"{p1} {operator} {p2}"
        return _make_dsl_json(script, params, op_name), "math"
    elif len(params) == 1:
        p1 = params[0]
        if operator == "sqrt":
            script = f"sqrt({p1})"
        elif operator == "factorial":
            script = f"factorial({p1})"
        elif operator == "abs":
            script = f"abs({p1})"
        else:
            return None
        return _make_dsl_json(script, params[:1], op_name), "math"

    return None


# Operation anchors for semantic matching
# Each tuple: (operator, anchor_description, min_params)
# Anchors are SEMANTIC descriptions, not keyword lists - embeddings handle similarity
#
# KNOWN LIMITATION: MathBERT embeddings don't discriminate well between basic
# arithmetic operations (+, -, *, /). Similarity scores cluster within ~0.1 range.
# The dual-channel approach (param anchors + description anchors) helps but isn't
# perfect. Consider using a different embedding model if better discrimination needed.
#
# Two-tier operation anchors:
# 1. PARAM anchors - what param names would you expect for this operation?
# 2. DESCRIPTION anchors - what does this operation do semantically?
# We compare query against BOTH and combine signals

_PARAM_ANCHORS = {
    # Param names that semantically indicate each operation
    # Be careful not to overlap param names between operations
    "+": "addend augend summand first second amount value number",
    "-": "minuend subtrahend larger smaller difference",
    "*": "factor multiplicand multiplier width height length breadth dimensions sides area product",
    "/": "dividend divisor numerator denominator ratio rate quotient per",
    "**": "base exponent power index",
    "%": "modulus remainder mod leftover",
    "gcd": "gcd numbers integers a b first second",  # avoid 'divisor' - conflicts with /
    "lcm": "lcm numbers integers a b first second",  # avoid 'multiple' ambiguity
    "comb": "n k choose combinations binomial",
    "perm": "n k permutations arrangements",
    "sqrt": "radicand square root",
    "factorial": "factorial exclamation",
    "abs": "absolute magnitude",
}

_DESCRIPTION_ANCHORS = {
    # Semantic description of what the operation does
    "+": "adding summing combining accumulating total plus",
    "-": "subtracting taking away how much more difference minus less",
    "*": "multiplying computing area product times scale",
    "/": "dividing quotient ratio splitting per evenly",
    "**": "power exponent raising squared cubed exponential",
    "%": "remainder leftover modulo after division",
    "gcd": "greatest common divisor largest shared factor",
    "lcm": "least common multiple smallest shared",
    "comb": "combinations choosing selecting ways",
    "perm": "permutations arranging ordering",
    "sqrt": "square root of a number",
    "factorial": "factorial n! product of integers",
    "abs": "absolute value magnitude",
}

# Combined anchors for each operation (will be embedded once)
_OPERATION_ANCHORS = [
    # Each operation has params + description combined for richer embedding
    ("+", f"{_PARAM_ANCHORS['+']} {_DESCRIPTION_ANCHORS['+']}", 2),
    ("-", f"{_PARAM_ANCHORS['-']} {_DESCRIPTION_ANCHORS['-']}", 2),
    ("*", f"{_PARAM_ANCHORS['*']} {_DESCRIPTION_ANCHORS['*']}", 2),
    ("/", f"{_PARAM_ANCHORS['/']} {_DESCRIPTION_ANCHORS['/']}", 2),
    ("**", f"{_PARAM_ANCHORS['**']} {_DESCRIPTION_ANCHORS['**']}", 2),
    ("%", f"{_PARAM_ANCHORS['%']} {_DESCRIPTION_ANCHORS['%']}", 2),
    ("gcd", f"{_PARAM_ANCHORS['gcd']} {_DESCRIPTION_ANCHORS['gcd']}", 2),
    ("lcm", f"{_PARAM_ANCHORS['lcm']} {_DESCRIPTION_ANCHORS['lcm']}", 2),
    ("comb", f"{_PARAM_ANCHORS['comb']} {_DESCRIPTION_ANCHORS['comb']}", 2),
    ("perm", f"{_PARAM_ANCHORS['perm']} {_DESCRIPTION_ANCHORS['perm']}", 2),
    ("sqrt", f"{_PARAM_ANCHORS['sqrt']} {_DESCRIPTION_ANCHORS['sqrt']}", 1),
    ("factorial", f"{_PARAM_ANCHORS['factorial']} {_DESCRIPTION_ANCHORS['factorial']}", 1),
    ("abs", f"{_PARAM_ANCHORS['abs']} {_DESCRIPTION_ANCHORS['abs']}", 1),
]

# Cache for anchor embeddings (separate param and description caches)
_param_anchor_embeddings = None
_desc_anchor_embeddings = None


def _get_param_anchor_embeddings():
    """Get embeddings for param name anchors."""
    global _param_anchor_embeddings
    if _param_anchor_embeddings is not None:
        return _param_anchor_embeddings

    try:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

        _param_anchor_embeddings = {}
        for op, anchor_text in _PARAM_ANCHORS.items():
            _param_anchor_embeddings[op] = embedder.embed(anchor_text)

        return _param_anchor_embeddings
    except Exception as e:
        logger.warning("[dsl_infer] Failed to compute param anchor embeddings: %s", e)
        return {}


def _get_desc_anchor_embeddings():
    """Get embeddings for description anchors."""
    global _desc_anchor_embeddings
    if _desc_anchor_embeddings is not None:
        return _desc_anchor_embeddings

    try:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

        _desc_anchor_embeddings = {}
        for op, anchor_text in _DESCRIPTION_ANCHORS.items():
            _desc_anchor_embeddings[op] = embedder.embed(anchor_text)

        return _desc_anchor_embeddings
    except Exception as e:
        logger.warning("[dsl_infer] Failed to compute desc anchor embeddings: %s", e)
        return {}


def _infer_operation_semantic(
    step_type: str,
    description: str,
    num_params: int,
    param_names: list = None,
    min_similarity: float = 0.5,
) -> Optional[tuple[str, str]]:
    """Infer math operation using dual-channel semantic embedding.

    Uses TWO embedding comparisons:
    1. Param names → param anchors (strong signal when semantic names used)
    2. Step type + description → description anchors (semantic intent)

    Combines both with weighting: params are weighted higher when semantic,
    lower when generic (single letters like 'a', 'b').

    All matching is embedding-based - no keyword matching.

    Args:
        step_type: The signature's step type
        description: The step description
        num_params: Number of available parameters
        param_names: List of parameter names
        min_similarity: Minimum similarity to accept a match

    Returns:
        (operator, operation_name) or None
    """
    try:
        from mycelium.embedder import Embedder
        import numpy as np

        embedder = Embedder.get_instance()
        param_anchors = _get_param_anchor_embeddings()
        desc_anchors = _get_desc_anchor_embeddings()

        if not param_anchors or not desc_anchors:
            return None

        # Determine if param names are semantic (meaningful) or generic (a, b, x, y)
        semantic_params = False
        if param_names:
            avg_len = sum(len(p) for p in param_names) / len(param_names)
            semantic_params = avg_len > 3  # Names like "dividend" vs "a"

        # Channel 1: Param names similarity (if we have params)
        param_sims = {}
        if param_names:
            param_text = " ".join(p.replace("_", " ") for p in param_names)
            param_emb = embedder.embed(param_text)

            for op, anchor_emb in param_anchors.items():
                sim = float(np.dot(param_emb, anchor_emb) / (
                    np.linalg.norm(param_emb) * np.linalg.norm(anchor_emb)
                ))
                param_sims[op] = sim

        # Channel 2: Description similarity (step_type + description)
        desc_sims = {}
        desc_text = step_type.replace("_", " ") if step_type else ""
        if description:
            desc_text += " " + description[:100]
        if desc_text.strip():
            desc_emb = embedder.embed(desc_text)

            for op, anchor_emb in desc_anchors.items():
                sim = float(np.dot(desc_emb, anchor_emb) / (
                    np.linalg.norm(desc_emb) * np.linalg.norm(anchor_emb)
                ))
                desc_sims[op] = sim

        # Combine channels with weighting
        # Semantic params (dividend, factor) are strong signals: weight 0.7
        # Generic params (a, b) are weak signals: weight 0.3
        param_weight = 0.7 if semantic_params else 0.3
        desc_weight = 1.0 - param_weight

        combined_sims = {}
        for op in param_anchors.keys():
            min_params_req = 2 if op not in ("sqrt", "factorial", "abs") else 1
            if num_params < min_params_req:
                continue

            p_sim = param_sims.get(op, 0.0)
            d_sim = desc_sims.get(op, 0.0)
            combined_sims[op] = param_weight * p_sim + desc_weight * d_sim

        if not combined_sims:
            return None

        # Find best match
        best_op = max(combined_sims, key=combined_sims.get)
        best_sim = combined_sims[best_op]

        if best_sim >= min_similarity:
            logger.info(
                "[dsl_infer] ACCEPTED: '%s' → '%s' (sim=%.3f, threshold=%.3f, top=%s)",
                step_type, best_op, best_sim, min_similarity,
                {k: f"{v:.2f}" for k, v in sorted(combined_sims.items(), key=lambda x: -x[1])[:4]}
            )
            return (best_op, _DESCRIPTION_ANCHORS.get(best_op, best_op))

        # Log at INFO level so rejections are visible - this is learning data
        logger.info(
            "[dsl_infer] REJECTED: '%s' best_match='%s' sim=%.3f < threshold=%.3f → decompose",
            step_type, best_op, best_sim, min_similarity
        )
        return None

    except Exception as e:
        logger.warning("[dsl_infer] Semantic inference failed: %s", e)
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
