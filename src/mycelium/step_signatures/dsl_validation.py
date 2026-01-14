"""DSL Validation: Semantic validation for DSL execution.

This module contains:
- Negative param mapping cache
- Algebraic manipulation blocking
- Value type classification
- Step-type alignment validation
- Param type and result validation
"""

__all__ = [
    # Negative param cache
    "record_negative_param_mapping",
    "check_negative_param_mapping",
    # Algebraic manipulation
    "is_algebraic_manipulation",
    # Value classification
    "classify_value_heuristic",
    "classify_value_embedding",
    "is_numeric_input",
    # Step-type alignment
    "validate_step_type_alignment",
    # DSL validation
    "validate_dsl_task_match",
    "validate_param_mapping_semantic",
    "validate_param_types",
    "validate_result_bounds",
    "is_valid_dsl_result",
]

import logging
import re
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from mycelium.step_signatures.dsl_types import DSLSpec, DSLLayer, ValueType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# NEGATIVE PARAM MAPPING CACHE
# =============================================================================
# In-memory cache of param mappings that produced wrong results.
# Keyed by (dsl_type, param_name) -> list of context description embeddings that failed.
# This allows rejecting mappings similar to known failures without GPU training.

_negative_param_cache: dict[tuple[str, str], list[Any]] = {}
_NEGATIVE_PARAM_SIMILARITY_THRESHOLD = 0.85  # Reject if very similar to known failure


def record_negative_param_mapping(
    dsl_type: str,
    param: str,
    context_embedding: Any,
) -> None:
    """Record a failed param mapping for future rejection."""
    key = (dsl_type, param)
    if key not in _negative_param_cache:
        _negative_param_cache[key] = []
    # Limit cache size per key to prevent memory bloat
    if len(_negative_param_cache[key]) < 50:
        _negative_param_cache[key].append(context_embedding)
        logger.debug("[neg_param] Recorded negative: type=%s param=%s", dsl_type, param)


def check_negative_param_mapping(
    dsl_type: str,
    param: str,
    context_embedding: Any,
) -> tuple[bool, float]:
    """Check if a param mapping is similar to known failures.

    Returns:
        (is_negative, max_similarity) - True if should reject, with similarity score
    """
    key = (dsl_type, param)
    if key not in _negative_param_cache:
        return False, 0.0

    max_sim = 0.0
    for neg_emb in _negative_param_cache[key]:
        sim = float(np.dot(context_embedding, neg_emb) / (
            np.linalg.norm(context_embedding) * np.linalg.norm(neg_emb)
        ))
        max_sim = max(max_sim, sim)
        if sim >= _NEGATIVE_PARAM_SIMILARITY_THRESHOLD:
            return True, sim

    return False, max_sim


# =============================================================================
# ALGEBRAIC MANIPULATION BLOCKING
# =============================================================================
# For arithmetic DSLs (a+b, a*b, etc.), we need to block algebraic manipulation
# tasks like "multiply both sides of equation" which use the same vocabulary
# but are conceptually different operations.
#
# This anchor embedding catches algebraic manipulation contexts.
_ALGEBRAIC_ANCHOR_TEXT = "transform equation by applying operation to both sides or each term"
_ALGEBRAIC_BLOCKING_THRESHOLD = 0.34  # Aggressive blocking - false positives ok (LLM fallback), false negatives bad
_algebraic_anchor_embedding = None  # Computed lazily


def _get_algebraic_anchor_embedding():
    """Get or compute the algebraic manipulation anchor embedding."""
    global _algebraic_anchor_embedding
    if _algebraic_anchor_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        _algebraic_anchor_embedding = embedder.embed(_ALGEBRAIC_ANCHOR_TEXT)
    return _algebraic_anchor_embedding


def is_algebraic_manipulation(task: str, task_embedding=None) -> tuple[bool, float]:
    """Check if a task describes algebraic manipulation (not suitable for arithmetic DSL).

    This is a public function for use in solver formula execution path.

    In TRAINING_MODE, this always returns False to collect failure data.

    Args:
        task: The task description
        task_embedding: Pre-computed embedding (optional)

    Returns:
        (is_algebraic, similarity) - True if task looks like algebraic manipulation
    """
    from mycelium.config import TRAINING_MODE

    if not task:
        return False, 0.0

    algebraic_anchor = _get_algebraic_anchor_embedding()

    if task_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        task_embedding = embedder.embed(task)

    algebraic_sim = float(np.dot(task_embedding, algebraic_anchor) / (
        np.linalg.norm(task_embedding) * np.linalg.norm(algebraic_anchor)
    ))

    # In training mode, don't block - let it fail and collect data
    if TRAINING_MODE:
        logger.debug("[algebraic] TRAINING_MODE: would block sim=%.2f but allowing", algebraic_sim)
        return False, algebraic_sim

    return algebraic_sim >= _ALGEBRAIC_BLOCKING_THRESHOLD, algebraic_sim


# =============================================================================
# VALUE TYPE CLASSIFICATION (Number vs Expression)
# =============================================================================
# Classifies parameter values as numbers vs algebraic expressions.
# Uses fast heuristics first, falls back to embeddings only for uncertain cases.

# Anchor embeddings for value classification (lazy-computed)
_NUMBER_ANCHOR_TEXT = "a numeric value like 42 or 3.14 or -7"
_EXPRESSION_ANCHOR_TEXT = "an algebraic expression with variables like x+2 or 3n-1 or a*b"
_number_anchor_embedding = None
_expression_anchor_embedding = None

# Value embedding cache (only for uncertain cases that hit embedding path)
_value_embedding_cache: dict[str, Any] = {}
_VALUE_CACHE_MAX = 200

# Threshold for embedding-based classification (difference in similarities)
_VALUE_TYPE_THRESHOLD = 0.15


def _get_number_anchor():
    """Get or compute the number anchor embedding."""
    global _number_anchor_embedding
    if _number_anchor_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        _number_anchor_embedding = embedder.embed(_NUMBER_ANCHOR_TEXT)
    return _number_anchor_embedding


def _get_expression_anchor():
    """Get or compute the expression anchor embedding."""
    global _expression_anchor_embedding
    if _expression_anchor_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        _expression_anchor_embedding = embedder.embed(_EXPRESSION_ANCHOR_TEXT)
    return _expression_anchor_embedding


def _get_value_embedding(value: str):
    """Get embedding for a value string, with caching."""
    if value in _value_embedding_cache:
        return _value_embedding_cache[value]

    from mycelium.embedder import Embedder
    embedder = Embedder.get_instance()
    emb = embedder.embed(value)

    # Evict oldest if full (simple FIFO)
    if len(_value_embedding_cache) >= _VALUE_CACHE_MAX:
        _value_embedding_cache.pop(next(iter(_value_embedding_cache)))

    _value_embedding_cache[value] = emb
    return emb


def classify_value_heuristic(value: Any) -> ValueType:
    """Fast heuristic classification. Returns UNCERTAIN when can't decide."""
    # Already numeric - definitely a number
    if isinstance(value, (int, float)):
        return ValueType.NUMBER

    if not isinstance(value, str):
        return ValueType.UNCERTAIN

    s = str(value).strip()
    if not s:
        return ValueType.UNCERTAIN

    # Direct numeric parse - definitely a number
    try:
        float(s)
        return ValueType.NUMBER
    except ValueError:
        pass

    # Fraction "3/4" - definitely a number
    if '/' in s and s.count('/') == 1:
        parts = s.split('/')
        try:
            float(parts[0].strip())
            float(parts[1].strip())
            return ValueType.NUMBER
        except ValueError:
            pass

    s_lower = s.lower()

    # Text with common natural language patterns - likely expression/text
    # Use simple heuristics; embedding fallback handles edge cases
    if any(s_lower.startswith(w) for w in ['the ', 'a ', 'an ', 'if ', 'when ']):
        return ValueType.EXPRESSION

    # Contains '=' (equation) - definitely expression
    if '=' in s:
        return ValueType.EXPRESSION

    # Contains variable letters with operators - likely expression
    # Include all letters (a-z) since in math context letters are usually variables
    if re.search(r'[a-zA-Z]', s):
        if re.search(r'[+\-*/^()]', s):  # has operators too
            return ValueType.EXPRESSION
        # Just letters without operators - uncertain (could be units like "5m")
        return ValueType.UNCERTAIN

    # High alpha ratio - likely expression/text
    alpha_count = sum(1 for c in s if c.isalpha())
    if len(s) > 3 and alpha_count / len(s) > 0.7:
        return ValueType.EXPRESSION

    # Low alpha ratio - likely number with units
    if len(s) > 0 and alpha_count / len(s) < 0.2:
        return ValueType.NUMBER

    # Can't decide
    return ValueType.UNCERTAIN


def classify_value_embedding(value: str) -> tuple[ValueType, float]:
    """Embedding-based classification. Only called when heuristics uncertain."""
    value_emb = _get_value_embedding(value)
    number_anchor = _get_number_anchor()
    expr_anchor = _get_expression_anchor()

    num_sim = float(np.dot(value_emb, number_anchor) / (
        np.linalg.norm(value_emb) * np.linalg.norm(number_anchor)
    ))
    expr_sim = float(np.dot(value_emb, expr_anchor) / (
        np.linalg.norm(value_emb) * np.linalg.norm(expr_anchor)
    ))

    diff = num_sim - expr_sim

    if diff > _VALUE_TYPE_THRESHOLD:
        return ValueType.NUMBER, diff
    elif diff < -_VALUE_TYPE_THRESHOLD:
        return ValueType.EXPRESSION, -diff
    else:
        # Too close to call - default to number (conservative for DSL execution)
        return ValueType.NUMBER, abs(diff)


def is_numeric_input(value: Any) -> bool:
    """Check if a value can be used as numeric input.

    Uses fast heuristics first, falls back to embedding-based classification
    only for uncertain cases (~5% of inputs).
    """
    # Fast path: heuristics handle 95%+ of cases
    classification = classify_value_heuristic(value)

    if classification == ValueType.NUMBER:
        return True
    if classification == ValueType.EXPRESSION:
        return False

    # Slow path: embedding fallback for uncertain cases
    if isinstance(value, str) and len(value.strip()) > 0:
        emb_class, confidence = classify_value_embedding(value.strip())
        logger.debug(
            "[value_classify] Embedding fallback: '%s' -> %s (conf=%.3f)",
            str(value)[:30], emb_class.value, confidence
        )
        return emb_class == ValueType.NUMBER

    return False


# =============================================================================
# STEP-TYPE ALIGNMENT VALIDATION
# =============================================================================
# Validates that the signature's step_type semantically matches the actual task.
# This catches cases where signature similarity is high but operation type differs.
#
# Step type descriptions are inferred dynamically from step_type names
# (e.g., "compute_gcd" -> "compute gcd operation") rather than hardcoded.
# This scales automatically as new step_types are added to the signature database.

# Cache for step_type embeddings
_step_type_embedding_cache: dict[str, Any] = {}

# Threshold for step-type alignment (lowered from 0.30 to let system breathe)
_STEP_TYPE_ALIGNMENT_THRESHOLD = 0.20


def _get_step_type_description(step_type: str) -> str:
    """Infer description from step_type name dynamically.

    Converts snake_case to human-readable sentence.
    e.g., "compute_gcd" -> "compute gcd operation"
    """
    description = step_type.replace("_", " ").lower()
    return f"{description} operation"


def _get_step_type_embedding(step_type: str):
    """Get or compute embedding for a step_type description."""
    if step_type in _step_type_embedding_cache:
        return _step_type_embedding_cache[step_type]

    description = _get_step_type_description(step_type)

    from mycelium.embedder import Embedder
    embedder = Embedder.get_instance()
    embedding = embedder.embed(description)
    _step_type_embedding_cache[step_type] = embedding
    return embedding


def validate_step_type_alignment(
    step_type: str,
    task: str,
    task_embedding=None,
) -> tuple[bool, float, str]:
    """Validate that step_type semantically aligns with the task.

    Args:
        step_type: The signature's step_type (e.g., "compute_product")
        task: The actual task description
        task_embedding: Pre-computed task embedding (optional)

    Returns:
        (valid, similarity, reason) - True if aligned, with similarity score
    """
    if not step_type or not task:
        return True, 1.0, "ok (no step_type or task)"

    # Get step_type embedding
    step_type_emb = _get_step_type_embedding(step_type)

    # Get task embedding
    if task_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        task_embedding = embedder.embed(task)

    # Compute similarity
    similarity = float(np.dot(step_type_emb, task_embedding) / (
        np.linalg.norm(step_type_emb) * np.linalg.norm(task_embedding)
    ))

    description = _get_step_type_description(step_type)

    if similarity < _STEP_TYPE_ALIGNMENT_THRESHOLD:
        logger.info(
            "[step_type_align] Mismatch: type='%s' ('%s') vs task='%s' sim=%.3f < %.3f",
            step_type, description[:30], task[:40], similarity, _STEP_TYPE_ALIGNMENT_THRESHOLD
        )
        return False, similarity, f"step_type '{step_type}' misaligned with task (sim={similarity:.3f})"

    return True, similarity, "ok"


# =============================================================================
# DSL TASK AND PARAM VALIDATION
# =============================================================================


def validate_dsl_task_match(dsl_spec: DSLSpec, step_task: str, task_embedding=None) -> tuple[bool, str]:
    """Validate that DSL script is appropriate for the step task using semantic gate.

    Uses embedding similarity between DSL purpose and task to catch mismatches.
    This replaces brittle keyword guards with learned semantic understanding.
    Uses per-DSL-type thresholds for more precise filtering.

    Args:
        dsl_spec: The DSL specification
        step_task: The task description
        task_embedding: Pre-computed task embedding (optional, will compute if needed)

    Returns:
        (valid, reason) tuple
    """
    if not step_task:
        return True, "ok"

    from mycelium.config import DSL_SEMANTIC_GATE_THRESHOLD, DSL_THRESHOLDS_BY_TYPE, TRAINING_MODE

    # Get per-type threshold (stricter for error-prone DSL types)
    dsl_type = dsl_spec.get_dsl_type()
    type_thresholds = DSL_THRESHOLDS_BY_TYPE.get(dsl_type, DSL_THRESHOLDS_BY_TYPE["default"])
    threshold = type_thresholds.get("gate", DSL_SEMANTIC_GATE_THRESHOLD)

    # Get DSL purpose embedding
    purpose_embedding = dsl_spec.get_purpose_embedding()
    if purpose_embedding is None:
        # Can't validate semantically, allow through
        return True, "ok (no purpose embedding)"

    # Get task embedding (compute if not provided)
    if task_embedding is None:
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        task_embedding = embedder.embed(step_task)

    # Compute cosine similarity
    similarity = np.dot(purpose_embedding, task_embedding) / (
        np.linalg.norm(purpose_embedding) * np.linalg.norm(task_embedding)
    )

    # Log for debugging
    purpose_text = dsl_spec.purpose or dsl_spec._infer_purpose_from_script()
    logger.debug(
        "[dsl_semantic_gate] type=%s purpose='%s' task='%s' sim=%.3f threshold=%.3f",
        dsl_type, purpose_text[:40], step_task[:40], similarity, threshold
    )

    # For arithmetic DSLs, use algebraic blocking instead of purpose matching
    # The purpose descriptions are too generic ("multiply two numbers") to distinguish
    # valid arithmetic from algebraic manipulation - rely on blocking anchor instead
    if dsl_type == "arithmetic":
        algebraic_anchor = _get_algebraic_anchor_embedding()
        algebraic_sim = float(np.dot(task_embedding, algebraic_anchor) / (
            np.linalg.norm(task_embedding) * np.linalg.norm(algebraic_anchor)
        ))

        if algebraic_sim >= _ALGEBRAIC_BLOCKING_THRESHOLD:
            if TRAINING_MODE:
                # Training mode: don't block, let it fail and collect data
                logger.debug(
                    "[dsl_algebraic_block] TRAINING_MODE: would block task='%s' sim=%.3f but allowing",
                    step_task[:40], algebraic_sim
                )
            else:
                logger.debug(
                    "[dsl_algebraic_block] task='%s' sim=%.3f >= %.3f - blocking arithmetic DSL",
                    step_task[:40], algebraic_sim, _ALGEBRAIC_BLOCKING_THRESHOLD
                )
                return False, f"algebraic manipulation detected: sim={algebraic_sim:.3f} >= {_ALGEBRAIC_BLOCKING_THRESHOLD}"
        # For arithmetic, don't check purpose similarity - algebraic blocking is sufficient
        return True, "ok (arithmetic, not algebraic manipulation)"

    # For non-arithmetic DSLs, use purpose similarity check
    if similarity < threshold:
        return False, f"semantic mismatch: DSL purpose vs task sim={similarity:.3f} < {threshold}"

    return True, "ok"


def validate_param_mapping_semantic(
    dsl_spec: DSLSpec,
    param_mapping: dict[str, str],
    step_descriptions: Optional[dict[str, str]] = None,
    threshold: Optional[float] = None,
) -> tuple[bool, str, dict[str, float]]:
    """Validate parameter mappings using embedding similarity.

    Compares each parameter's expected semantic role against the context
    from which the mapped value came (via step descriptions).
    Uses per-DSL-type thresholds for more precise filtering.

    Args:
        dsl_spec: The DSL specification with param role info
        param_mapping: Map of param_name -> context_key (e.g., {'base': 'step_1'})
        step_descriptions: Map of context_key -> task description
        threshold: Override threshold (if None, uses per-type config)

    Returns:
        (valid, reason, similarities) tuple with per-param similarity scores
    """
    if not step_descriptions or not param_mapping:
        return True, "ok (no descriptions)", {}

    from mycelium.config import DSL_PARAM_SEMANTIC_THRESHOLD, DSL_THRESHOLDS_BY_TYPE
    from mycelium.embedder import Embedder

    # Get per-type threshold if not overridden
    if threshold is None:
        dsl_type = dsl_spec.get_dsl_type()
        type_thresholds = DSL_THRESHOLDS_BY_TYPE.get(dsl_type, DSL_THRESHOLDS_BY_TYPE["default"])
        threshold = type_thresholds.get("param", DSL_PARAM_SEMANTIC_THRESHOLD)

    embedder = Embedder.get_instance()

    similarities: dict[str, float] = {}
    min_sim = 1.0
    worst_param = None
    worst_context = None

    dsl_type = dsl_spec.get_dsl_type()

    for param, context_key in param_mapping.items():
        # Get the step description for this context key
        step_desc = step_descriptions.get(context_key)
        if not step_desc:
            continue

        # Get param role embedding
        role_embedding = dsl_spec.get_param_role_embedding(param)
        if role_embedding is None:
            continue

        # Get step description embedding
        desc_embedding = embedder.embed(step_desc)

        # Check against known negative mappings first (fast rejection)
        is_negative, neg_sim = check_negative_param_mapping(dsl_type, param, desc_embedding)
        if is_negative:
            logger.info(
                "[neg_param] Rejected known-bad mapping: type=%s param=%s context='%s' neg_sim=%.3f",
                dsl_type, param, step_desc[:40], neg_sim
            )
            return False, f"param '{param}' matches known failure (sim={neg_sim:.3f})", {}

        # Compute cosine similarity
        similarity = float(np.dot(role_embedding, desc_embedding) / (
            np.linalg.norm(role_embedding) * np.linalg.norm(desc_embedding)
        ))
        similarities[param] = similarity

        if similarity < min_sim:
            min_sim = similarity
            worst_param = param
            worst_context = step_desc[:50]

    # Check if any mapping is below threshold
    if min_sim < threshold and worst_param:
        role_text = dsl_spec.param_roles.get(worst_param) or dsl_spec._infer_param_role(worst_param)
        logger.info(
            "[dsl_param_semantic] Param '%s' role mismatch: role='%s' context='%s' sim=%.3f < %.3f",
            worst_param, role_text[:40], worst_context, min_sim, threshold
        )
        return False, f"param '{worst_param}' semantic mismatch: sim={min_sim:.3f} < {threshold}", similarities

    return True, "ok", similarities


def validate_param_types(
    dsl_spec: DSLSpec,
    inputs: dict[str, Any],
    step_task: str = "",
    extract_numeric_fn=None,
) -> tuple[bool, str]:
    """Validate that inputs match expected parameter types.

    Also detects semantic mapping errors:
    - Duplicate values: multiple params mapped to same value (bad semantic matching)
    - Binomial errors: n == r or n < r in combination/permutation DSLs
    - Task/DSL mismatch: DSL operation doesn't match task intent

    Args:
        dsl_spec: The DSL specification
        inputs: Mapped input values
        step_task: Task description for semantic validation
        extract_numeric_fn: Function to extract numeric values (injected to avoid circular import)

    Returns:
        (valid, reason) - True if valid, False with reason if not
    """
    # First check task/DSL semantic match
    valid, reason = validate_dsl_task_match(dsl_spec, step_task)
    if not valid:
        return valid, reason

    # Import extract function if not provided
    if extract_numeric_fn is None:
        from mycelium.step_signatures.math_layer import extract_numeric_value
        extract_numeric_fn = extract_numeric_value

    # Type validation
    if dsl_spec.param_types:
        for param, expected_type in dsl_spec.param_types.items():
            if param not in inputs:
                continue

            value = inputs[param]

            if expected_type == "numeric":
                if not is_numeric_input(value):
                    return False, f"param '{param}' expected numeric, got: {str(value)[:30]}"

            elif expected_type == "symbolic":
                # Symbolic can be anything (expressions, equations)
                pass

            # "any" type accepts anything

    # Semantic mapping validation: detect duplicate values
    if len(inputs) >= 2:
        # Extract numeric values for comparison
        numeric_values: dict[str, float] = {}
        for param, value in inputs.items():
            extracted = extract_numeric_fn(value)
            if extracted is not None:
                numeric_values[param] = extracted

        # Check for duplicates (all params having same value = bad mapping)
        if len(numeric_values) >= 2:
            unique_values = set(numeric_values.values())
            if len(unique_values) == 1:
                # All params have same value - likely bad semantic mapping
                dup_val = list(unique_values)[0]
                return False, f"duplicate values: all params = {dup_val} (bad semantic mapping)"

    # Power/exponent validation - catch huge exponents that produce garbage
    script_lower = dsl_spec.script.lower()
    is_power = '**' in script_lower or 'pow' in script_lower or 'exponent' in script_lower

    if is_power and len(inputs) >= 2:
        # Find the likely exponent value
        exp_val = None
        for param in ['exponent', 'exp', 'power', 'n']:
            if param in inputs:
                exp_val = extract_numeric_fn(inputs[param])
                break

        # Fallback: assume second numeric value is exponent
        if exp_val is None:
            numeric_vals = [extract_numeric_fn(v) for v in inputs.values()]
            numeric_vals = [v for v in numeric_vals if v is not None]
            if len(numeric_vals) >= 2:
                exp_val = numeric_vals[1]  # Second value often the exponent

        if exp_val is not None and abs(exp_val) > 50:
            return False, f"exponent too large ({exp_val}) - likely bad param mapping"

    # Binomial/combination specific validation
    is_binomial = any(kw in script_lower for kw in [
        'factorial', 'comb', 'perm', 'binomial', 'choose', 'c(', 'p('
    ])

    if is_binomial and len(inputs) >= 2:
        # Look for n, r params (or first two numeric values)
        n_val = None
        r_val = None

        # Try to identify n and r by name
        for param in ['n', 'N']:
            if param in inputs:
                n_val = extract_numeric_fn(inputs[param])
                break
        for param in ['r', 'k', 'R', 'K']:
            if param in inputs:
                r_val = extract_numeric_fn(inputs[param])
                break

        # Fallback to positional (first two params for binomial operations)
        if n_val is None or r_val is None:
            numeric_inputs = [(p, extract_numeric_fn(v)) for p, v in inputs.items()]
            numeric_inputs = [(p, v) for p, v in numeric_inputs if v is not None]
            if len(numeric_inputs) >= 2:
                n_val = numeric_inputs[0][1]
                r_val = numeric_inputs[1][1]

        if n_val is not None and r_val is not None:
            # Validate binomial constraints
            if n_val == r_val:
                return False, f"binomial n == r ({n_val}) produces trivial result"
            if n_val < r_val:
                return False, f"binomial n < r ({n_val} < {r_val}) is invalid"
            # Sanity check: factorial of very large numbers is suspicious
            if n_val > 1000 or r_val > 1000:
                return False, f"binomial params too large (n={n_val}, r={r_val})"

    return True, "ok"


def validate_result_bounds(
    result: Any,
    dsl_spec: DSLSpec,
    inputs: dict[str, Any],
    extract_numeric_fn=None,
) -> tuple[bool, str]:
    """Validate result against operation-specific bounds.

    Catches results that violate mathematical invariants.
    """
    if not isinstance(result, (int, float)):
        return True, "ok"

    # Import extract function if not provided
    if extract_numeric_fn is None:
        from mycelium.step_signatures.math_layer import extract_numeric_value
        extract_numeric_fn = extract_numeric_value

    script_lower = dsl_spec.script.lower()
    numeric_inputs = [extract_numeric_fn(v) for v in inputs.values()]
    numeric_inputs = [v for v in numeric_inputs if v is not None and v > 0]

    if not numeric_inputs:
        return True, "ok"

    # LCM must be >= max(inputs)
    if 'lcm' in script_lower or 'least common' in script_lower:
        max_input = max(numeric_inputs)
        if result < max_input * 0.99:  # Allow small floating point tolerance
            return False, f"LCM {result} < max input {max_input}"

    # GCD must be <= min(inputs)
    if 'gcd' in script_lower or 'greatest common' in script_lower:
        min_input = min(numeric_inputs)
        if result > min_input * 1.01:  # Allow small floating point tolerance
            return False, f"GCD {result} > min input {min_input}"

    # sqrt(x) must be <= x for x >= 1
    if 'sqrt' in script_lower and len(numeric_inputs) == 1:
        x = numeric_inputs[0]
        if x >= 1 and result > x * 1.01:
            return False, f"sqrt({x}) = {result} violates sqrt(x) <= x"

    # Factorial/combinatorics should return integers
    if any(kw in script_lower for kw in ['factorial', 'comb', 'perm', 'choose']):
        if isinstance(result, float) and result != int(result):
            return False, f"combinatorics returned non-integer {result}"

    # Product of same-sign numbers should preserve sign
    if 'a * b' in script_lower or 'product' in script_lower:
        if len(numeric_inputs) >= 2:
            if all(v > 0 for v in numeric_inputs) and result < 0:
                return False, "product of positives cannot be negative"
            if all(v < 0 for v in numeric_inputs) and result < 0:
                return False, "product of negatives cannot be negative"

    return True, "ok"


def is_valid_dsl_result(
    result: Any,
    dsl_spec: DSLSpec = None,
    inputs: dict[str, Any] = None,
) -> bool:
    """Check if a DSL result is valid (not obvious garbage).

    Rejects results that are clearly invalid:
    - Boolean False (sympy comparison/factor failures return False)
    - Astronomically large numbers (> 1e15) - likely param mapping errors
    - Operation-specific bound violations
    - Note: True can be valid for some predicates, so we allow it
    """
    # Boolean False is always invalid (sympy factor/simplify failures)
    # True could be valid for some predicate checks, so allow it
    if result is False:
        logger.debug("[dsl_validate] Rejecting False result (likely sympy failure)")
        return False

    # Check operation-specific bounds if we have the context
    if dsl_spec and inputs:
        valid, reason = validate_result_bounds(result, dsl_spec, inputs)
        if not valid:
            logger.info("[dsl_validate] Result bounds violation: %s", reason)
            return False

    # Reject astronomically large results - usually from bad param mapping
    # Competition math can have large intermediate results (billions, trillions)
    # Relaxed from 1e10 to 1e15 to let the system breathe and learn
    try:
        if isinstance(result, (int, float)):
            if abs(result) > 1e15:
                logger.info("[dsl_validate] Rejecting huge result: %s", result)
                return False
    except (TypeError, OverflowError):
        pass

    return True
