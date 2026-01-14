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
    """Generate DSL from planner's extracted values using semantic similarity.

    Uses embedding similarity to match step description to operation prototypes.
    No keyword matching - purely semantic inference.

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

    # Use semantic similarity to infer operation
    # Pass param names - they carry semantic hints (dividend/divisor → divide)
    op_info = _infer_operation_semantic(step_type, description, len(params), param_names=params)

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


# Operation prototypes for semantic matching
# Each tuple: (operator, prototype_text, min_params)
# Prototypes include common param names to help semantic matching when params are in query
_OPERATION_PROTOTYPES = [
    # Two-param arithmetic - include typical param names for each operation
    ("+", "compute sum addition add plus total addend augend summand combine", 2),
    ("-", "compute difference subtraction subtract minus minuend subtrahend take away reduce", 2),
    ("*", "compute product multiplication multiply times multiplicand multiplier", 2),
    ("/", "compute quotient division divide ratio dividend divisor", 2),
    ("**", "compute power exponentiation exponent squared cubed base raise", 2),
    ("%", "compute remainder modulo mod modulus", 2),
    # Special two-param functions
    ("gcd", "compute gcd greatest common divisor", 2),
    ("lcm", "compute lcm least common multiple", 2),
    ("comb", "compute combinations choose binomial coefficient", 2),
    ("perm", "compute permutations arrangements ordering", 2),
    # Single-param functions - be specific to avoid overmatch
    ("sqrt", "compute square root sqrt radical", 1),
    ("factorial", "compute factorial exclamation permutation single", 1),
    ("abs", "compute absolute value magnitude abs", 1),
]

# Cache for prototype embeddings
_prototype_embeddings = None


def _get_prototype_embeddings():
    """Lazily compute and cache prototype embeddings."""
    global _prototype_embeddings
    if _prototype_embeddings is not None:
        return _prototype_embeddings

    try:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()

        _prototype_embeddings = []
        for operator, prototype_text, min_params in _OPERATION_PROTOTYPES:
            emb = embedder.embed(prototype_text)
            _prototype_embeddings.append((operator, emb, min_params, prototype_text))

        return _prototype_embeddings
    except Exception as e:
        logger.warning("[dsl_infer] Failed to compute prototype embeddings: %s", e)
        return []


# Param name patterns that strongly indicate specific operations
# These are checked before embedding similarity for reliable matching
_PARAM_OPERATION_HINTS = {
    # Division indicators
    "dividend": "/",
    "divisor": "/",
    "numerator": "/",
    "denominator": "/",
    # Subtraction indicators
    "minuend": "-",
    "subtrahend": "-",
    # Addition indicators
    "addend": "+",
    "augend": "+",
    "summand": "+",
    # Multiplication indicators
    "multiplicand": "*",
    "multiplier": "*",
    "factor": "*",  # "factor_a" → multiplication
}

# Step type patterns that indicate specific operations
# Checked after param hints, before embedding
_STEP_TYPE_HINTS = {
    # Addition
    "sum": "+",
    "add": "+",
    "total": "+",
    "combine": "+",
    # Subtraction
    "difference": "-",
    "subtract": "-",
    "minus": "-",
    # Multiplication
    "product": "*",
    "multiply": "*",
    "times": "*",
    "area": "*",  # compute_area → multiplication
    # Division
    "quotient": "/",
    "divide": "/",
    "ratio": "/",
    # Power
    "power": "**",
    "exponent": "**",
    "square": "**",
    # Modulo
    "remainder": "%",
    "modulo": "%",
    "mod": "%",
}


def _check_param_hints(param_names: list) -> Optional[tuple[str, str]]:
    """Check if param names indicate a specific operation.

    Returns (operator, operation_name) if strong signal found.
    """
    if not param_names:
        return None

    # Check each param name for operation hints
    op_votes = {}
    for param in param_names:
        param_lower = param.lower()
        for hint, op in _PARAM_OPERATION_HINTS.items():
            if hint in param_lower:
                op_votes[op] = op_votes.get(op, 0) + 1

    # If we have a clear winner with at least one strong hint
    if op_votes:
        best_op = max(op_votes, key=op_votes.get)
        if op_votes[best_op] >= 1:
            op_names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division", "**": "power", "%": "modulo"}
            return (best_op, op_names.get(best_op, best_op))

    return None


def _check_step_type_hints(step_type: str) -> Optional[tuple[str, str]]:
    """Check if step_type contains operation hints.

    Returns (operator, operation_name) if clear signal found.
    """
    if not step_type:
        return None

    step_lower = step_type.lower().replace("_", " ")

    # Check for operation hints in step_type
    for hint, op in _STEP_TYPE_HINTS.items():
        if hint in step_lower:
            op_names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division", "**": "power", "%": "modulo"}
            return (op, op_names.get(op, op))

    return None


def _infer_operation_semantic(
    step_type: str,
    description: str,
    num_params: int,
    param_names: list = None,
    min_similarity: float = 0.35,
) -> Optional[tuple[str, str]]:
    """Infer math operation using hints + semantic similarity.

    Priority:
    1. Param name hints (dividend/divisor → division)
    2. Step type hints (compute_product → multiplication)
    3. Embedding similarity to operation prototypes

    Args:
        step_type: The signature's step type
        description: The step description
        num_params: Number of available parameters
        param_names: List of parameter names (carry semantic hints)
        min_similarity: Minimum similarity to accept a match

    Returns:
        (operator, operation_name) or None
    """
    # Priority 1: Check param name hints (most reliable signal)
    hint_result = _check_param_hints(param_names)
    if hint_result:
        logger.debug(
            "[dsl_infer] Param hint match: params=%s → '%s'",
            param_names, hint_result[0]
        )
        return hint_result

    # Priority 2: Check step_type hints
    step_hint = _check_step_type_hints(step_type)
    if step_hint:
        logger.debug(
            "[dsl_infer] Step type hint match: '%s' → '%s'",
            step_type, step_hint[0]
        )
        return step_hint

    # Priority 3: Embedding similarity (fallback for unknown patterns)
    try:
        from mycelium.embedder import Embedder
        import numpy as np

        embedder = Embedder.get_instance()
        prototypes = _get_prototype_embeddings()

        if not prototypes:
            return None

        # Build query from step_type + param names
        step_type_readable = step_type.replace("_", " ")
        if param_names:
            param_text = " ".join(p.replace("_", " ") for p in param_names)
            query_text = f"{step_type_readable} {param_text}"
        else:
            query_text = step_type_readable
        query_emb = embedder.embed(query_text)

        # Find best matching prototype
        best_match = None
        best_sim = 0.0

        for operator, proto_emb, min_params_req, proto_text in prototypes:
            # Skip if not enough params
            if num_params < min_params_req:
                continue

            # Compute cosine similarity
            sim = float(np.dot(query_emb, proto_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(proto_emb)
            ))

            if sim > best_sim:
                best_sim = sim
                best_match = (operator, proto_text.split(",")[0].strip())

        if best_match and best_sim >= min_similarity:
            logger.debug(
                "[dsl_infer] Semantic match: '%s' → '%s' (sim=%.3f)",
                step_type, best_match[0], best_sim
            )
            return best_match

        logger.debug(
            "[dsl_infer] No semantic match above threshold: best_sim=%.3f < %.3f",
            best_sim, min_similarity
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
