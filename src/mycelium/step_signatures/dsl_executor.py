"""DSL Executor: Layered execution engine for signature DSL scripts.

Three execution layers:
1. Math: Basic arithmetic, trig, log (extends existing _safe_eval_formula)
2. SymPy: Symbolic algebra (solve, simplify, diff, integrate)
3. Custom: Registered domain-specific operators

DSL scripts are JSON-encoded specifications:
{
    "type": "math|sympy|custom",
    "script": "expression or operator call",
    "params": ["required", "input", "names"],
    "aliases": {"param_name": ["alias1", "alias2"]},
    "fallback": "decompose|formula|llm"
}

Aliases enable fuzzy matching between param names and context keys.
The LLM generates aliases once at signature creation time, then matching
is fast dict lookups at runtime.

This module is a facade that re-exports from focused submodules:
- dsl_types: DSLLayer, DSLSpec, ValueType
- dsl_validation: Semantic validation functions
- math_layer: Math DSL execution
- sympy_layer: SymPy DSL execution
- custom_layer: Custom operator execution
"""

import logging
import re
import signal
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Re-exports from submodules (backward compatibility)
# =============================================================================

# Types
from mycelium.step_signatures.dsl_types import (
    DSLLayer,
    DSLSpec,
    ValueType,
)

# Validation
from mycelium.step_signatures.dsl_validation import (
    # Negative param cache
    record_negative_param_mapping,
    check_negative_param_mapping,
    # Algebraic manipulation
    is_algebraic_manipulation,
    # Value classification
    classify_value_heuristic as _classify_value_heuristic,
    classify_value_embedding as _classify_value_embedding,
    is_numeric_input as _is_numeric_input,
    # Step-type alignment
    validate_step_type_alignment,
    # DSL validation
    validate_dsl_task_match as _validate_dsl_task_match,
    validate_param_mapping_semantic as _validate_param_mapping_semantic,
    validate_param_types as _validate_param_types,
    validate_result_bounds as _validate_result_bounds,
    is_valid_dsl_result as _is_valid_dsl_result,
)

# Math layer
from mycelium.step_signatures.math_layer import (
    BINOPS as _BINOPS,
    UNARYOPS as _UNARYOPS,
    FUNCTIONS as _FUNCTIONS,
    CONSTANTS as _CONSTANTS,
    extract_numeric_value as _extract_numeric_value,
    prepare_math_inputs as _prepare_math_inputs,
    try_execute_dsl_math,
)

# SymPy layer
from mycelium.step_signatures.sympy_layer import (
    SYMPY_ALLOWED as _SYMPY_ALLOWED,
    SYMPY_ALLOWED_METHODS as _SYMPY_ALLOWED_METHODS,
    parse_to_sympy as _parse_to_sympy,
    try_execute_dsl_sympy,
)

# Custom layer
from mycelium.step_signatures.custom_layer import (
    register_operator,
    get_operator,
    list_operators,
    try_execute_dsl_custom,
)

# =============================================================================
# Main Entry Point
# =============================================================================

@contextmanager
def _timeout(seconds: float):
    """Timeout context manager for DSL execution."""
    def handler(signum, frame):
        raise TimeoutError(f"DSL execution timed out after {seconds}s")

    # Only use signal-based timeout on Unix
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # Windows or signal not available, skip timeout
        yield


def try_execute_dsl(
    dsl_spec: DSLSpec,
    inputs: dict[str, Any],
    timeout_sec: float = 1.0,
    step_task: str = "",
) -> tuple[Optional[Any], bool]:
    """Execute a DSL script with given inputs.

    Args:
        dsl_spec: The DSL specification
        inputs: Input values (from context/previous steps) - keys may not match param names exactly
        timeout_sec: Maximum execution time
        step_task: The task description for semantic validation

    Returns:
        (result, success) tuple. If success=False, caller should fallback.
    """
    # Map inputs to param names using alias matching
    if dsl_spec.params:
        mapped_inputs = dsl_spec.map_inputs(inputs)
        # Check if we found all required params
        missing = set(dsl_spec.params) - set(mapped_inputs.keys())
        if missing:
            logger.debug("DSL missing params after alias matching: %s (had keys: %s)", missing, list(inputs.keys()))
            return None, False
    else:
        # No explicit params - pass all inputs directly
        # This allows scripts like "result = step_1 * 15" to work
        mapped_inputs = inputs.copy()

    # DECOMPOSE/ROUTER/NONE layers: immediately fall back to LLM (no DSL execution)
    # ROUTER delegates to children at a higher level, not here
    if dsl_spec.layer in (DSLLayer.DECOMPOSE, DSLLayer.ROUTER, DSLLayer.NONE):
        return None, False

    # Filter out empty/None values before validation
    # This handles cascading failures where previous steps returned empty strings
    filtered_inputs = {}
    for k, v in mapped_inputs.items():
        if v is None or v == "" or (isinstance(v, str) and not v.strip()):
            continue
        extracted = _extract_numeric_value(v)
        if extracted is not None:
            filtered_inputs[k] = extracted
        else:
            filtered_inputs[k] = v

    if not filtered_inputs:
        logger.debug("DSL has no valid inputs after filtering empty values")
        return None, False

    # Validate input types and task/DSL semantic match before execution
    valid, reason = _validate_param_types(dsl_spec, filtered_inputs, step_task, _extract_numeric_value)
    if not valid:
        logger.info("[dsl_debug] TYPE_MISMATCH %s | script='%s' | %s",
                    dsl_spec.layer.value, dsl_spec.script[:40], reason)
        return None, False

    # Use filtered inputs for execution
    mapped_inputs = filtered_inputs

    try:
        with _timeout(timeout_sec):
            if dsl_spec.layer == DSLLayer.MATH:
                result = try_execute_dsl_math(dsl_spec.script, mapped_inputs)
            elif dsl_spec.layer == DSLLayer.SYMPY:
                result = try_execute_dsl_sympy(dsl_spec.script, mapped_inputs)
            elif dsl_spec.layer == DSLLayer.CUSTOM:
                result = try_execute_dsl_custom(dsl_spec.script, mapped_inputs)
            else:
                return None, False

            if result is not None:
                # Validate result before accepting (with operation-specific bounds)
                if not _is_valid_dsl_result(result, dsl_spec, mapped_inputs):
                    logger.info(
                        "[dsl_debug] INVALID %s | script='%s' | result=%s (failed validation)",
                        dsl_spec.layer.value, dsl_spec.script[:60], str(result)[:50]
                    )
                    return None, False
                logger.info(
                    "[dsl_debug] EXEC %s | script='%s' | inputs=%s | result=%s",
                    dsl_spec.layer.value, dsl_spec.script[:60],
                    {k: str(v)[:30] for k, v in mapped_inputs.items()}, result
                )
                return result, True
            logger.debug(
                "[dsl_debug] FAIL %s | script='%s' | inputs=%s | result=None",
                dsl_spec.layer.value, dsl_spec.script[:60],
                {k: str(v)[:30] for k, v in mapped_inputs.items()}
            )
            return None, False

    except TimeoutError:
        logger.warning("[dsl_debug] TIMEOUT | script='%s'", dsl_spec.script[:60])
        return None, False
    except Exception as e:
        logger.debug("[dsl_debug] ERROR %s | script='%s' | error=%s",
                     dsl_spec.layer.value, dsl_spec.script[:60], e)
        return None, False


def execute_dsl_from_json(
    dsl_json: str,
    inputs: dict[str, Any],
) -> tuple[Optional[Any], bool]:
    """Convenience function to execute DSL from JSON string."""
    spec = DSLSpec.from_json(dsl_json)
    if not spec:
        return None, False
    return try_execute_dsl(spec, inputs)


def execute_dsl_with_confidence(
    dsl_json: str,
    inputs: dict[str, Any],
    min_confidence: float = 0.7,
) -> tuple[Optional[Any], bool, float]:
    """Execute DSL only if confidence exceeds threshold.

    Args:
        dsl_json: JSON-encoded DSL specification
        inputs: Input values from context
        min_confidence: Minimum confidence to proceed (default 0.7)

    Returns:
        (result, success, confidence) tuple.
        If confidence < min_confidence, returns (None, False, confidence).
    """
    spec = DSLSpec.from_json(dsl_json)
    if not spec:
        return None, False, 0.0

    confidence = spec.compute_confidence(inputs)
    if confidence < min_confidence:
        logger.debug(
            "DSL skipped (low confidence %.2f < %.2f): %s",
            confidence, min_confidence, spec.script[:50]
        )
        return None, False, confidence

    result, success = try_execute_dsl(spec, inputs)
    return result, success, confidence


# =============================================================================
# Semantic Parameter Mapping (replaces LLM guessing)
# =============================================================================

def semantic_rewrite_script(
    dsl_spec: "DSLSpec",
    context: dict[str, Any],
    step_descriptions: Optional[dict[str, str]] = None,
) -> tuple[Optional[str], float]:
    """Rewrite DSL script using semantic matching instead of LLM guessing.

    Maps DSL parameters to context variables by comparing semantic meaning:
    - DSL param: "area_ABC"
    - Step description: "Calculate the area of triangle ABC"
    - Match by string similarity → area_ABC = step_1

    For generic single-letter params (n, a, b, x), falls back to positional
    matching with step_N keys sorted by step number.

    Args:
        dsl_spec: The DSL specification with script to rewrite
        context: Runtime context with available values
        step_descriptions: Dict mapping step_id -> task description

    Returns:
        (rewritten_script, confidence) or (None, 0.0) if no good mapping
    """
    if not dsl_spec.params or not context:
        return None, 0.0

    def _is_generic_param(param: str) -> bool:
        """Check if param is generic (should use positional fallback).

        Generic params are short names that don't carry specific semantics.
        Instead of hardcoding a list, use length as heuristic.
        """
        p = param.lower()
        # Single letters or very short names (<=3 chars) are generic
        return len(p) <= 3

    # Build param -> context_key mapping
    param_mapping: dict[str, str] = {}
    used_keys: set[str] = set()  # Track keys already used to prevent duplicates
    total_score = 0.0
    unmatched_params = []

    # Get sorted step keys for positional fallback
    step_keys = sorted([k for k in context.keys() if k.startswith('step_')],
                       key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)

    for param in dsl_spec.params:
        best_match = None
        best_score = 0.0
        param_lower = param.lower().replace("_", " ")

        # Try semantic matching first (if step_descriptions available)
        if step_descriptions:
            for ctx_key, ctx_value in context.items():
                # Skip already-used keys to prevent duplicate mappings
                if ctx_key in used_keys:
                    continue
                if ctx_key not in step_descriptions:
                    continue

                desc = step_descriptions[ctx_key].lower()
                score = 0.0

                # Score 1: Exact param name in description
                if param_lower in desc:
                    score = 0.95

                # Score 2: Param tokens overlap with description tokens
                param_tokens = set(param_lower.split())
                desc_tokens = set(desc.replace("_", " ").split())
                overlap = param_tokens & desc_tokens
                if overlap and score < 0.9:
                    score = max(score, 0.5 + 0.4 * len(overlap) / max(len(param_tokens), 1))

                # Score 3: Param suffix matches (e.g., "area_ABC" matches "...ABC...")
                if "_" in param:
                    suffix = param.split("_")[-1].lower()
                    if suffix in desc and len(suffix) > 1:
                        score = max(score, 0.8)

                if score > best_score:
                    best_score = score
                    best_match = ctx_key

        if best_match and best_score >= 0.5:
            param_mapping[param] = best_match
            used_keys.add(best_match)  # Mark key as used
            total_score += best_score
        elif _is_generic_param(param):
            # Generic param - save for positional fallback
            unmatched_params.append(param)
        else:
            # Can't map this param semantically and it's not generic
            logger.debug("[dsl_semantic] No semantic match for param '%s'", param)
            return None, 0.0

    # Positional fallback for generic params
    # Map params to available context keys in order (no hardcoded preferences)
    if unmatched_params:
        # Gather all available keys, preferring step_N keys (computed values)
        available_keys = [k for k in step_keys if k not in used_keys]
        # Add task/problem num keys as fallback
        num_keys = sorted([k for k in context.keys()
                          if (k.startswith('task_num_') or k.startswith('problem_num_'))
                          and k not in used_keys])
        available_keys.extend(num_keys)

        for param in unmatched_params:
            if available_keys:
                key = available_keys.pop(0)
                param_mapping[param] = key
                used_keys.add(key)
                total_score += 0.6
                logger.debug("[dsl_semantic] Positional match: %s -> %s", param, key)
            else:
                logger.debug("[dsl_semantic] No positional match for param '%s'", param)
                return None, 0.0

    if len(param_mapping) != len(dsl_spec.params):
        return None, 0.0

    # Validate param mappings semantically using embeddings
    # Each param's expected role must match the context it's mapped to
    # Uses per-DSL-type thresholds (stricter for power, geometry, combinatorics)
    valid, reason, similarities = _validate_param_mapping_semantic(
        dsl_spec, param_mapping, step_descriptions
    )
    if not valid:
        logger.info(
            "[dsl_semantic] Param mapping rejected: %s (mappings=%s, sims=%s)",
            reason, param_mapping, {k: f"{v:.2f}" for k, v in similarities.items()}
        )
        return None, 0.0

    # Rewrite script by replacing params with context keys
    rewritten = dsl_spec.script
    for param, ctx_key in param_mapping.items():
        # Replace param name with context key
        rewritten = re.sub(rf'\b{re.escape(param)}\b', ctx_key, rewritten)

    avg_confidence = total_score / len(dsl_spec.params) if dsl_spec.params else 0.0

    # Log success with param similarities AND actual resolved values
    sim_str = ", ".join(f"{k}={v:.2f}" for k, v in similarities.items()) if similarities else "no sims"

    # Build detailed mapping showing param -> context_key -> actual_value
    detailed_mapping = {
        param: f"{ctx_key}={context.get(ctx_key, '?')}"
        for param, ctx_key in param_mapping.items()
    }

    logger.info(
        "[dsl_param_map] %s | mappings: %s | sims: {%s}",
        dsl_spec.script[:40],
        detailed_mapping,
        sim_str,
    )
    logger.info(
        "[dsl_semantic] '%s' -> '%s' (conf=%.2f)",
        dsl_spec.script[:50], rewritten[:50], avg_confidence
    )

    return rewritten, avg_confidence
