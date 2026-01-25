"""DSL Executor: Math execution engine for signature DSL scripts.

DSL scripts are JSON-encoded specifications:
{
    "type": "math",
    "script": "expression",
    "params": ["required", "input", "names"],
    "aliases": {"param_name": ["alias1", "alias2"]},
    "fallback": "decompose"
}

Aliases enable fuzzy matching between param names and context keys.
The LLM generates aliases once at signature creation time, then matching
is fast dict lookups at runtime.

Architecture (Strategy Pattern):
================================
try_execute_dsl()               <- MAIN ENTRY POINT (this module)
├── DSLLayer.MATH   -> try_execute_dsl_math()   [math_layer.py]
│   └── AST-based safe eval for forward calculations
│   └── No eval() - parses Python AST for security
├── DSLLayer.SYMPY  -> try_execute_dsl_sympy()  [sympy_layer.py]
│   └── SymPy symbolic solver for backwards solving
│   └── Solves equations when unknowns present
└── DSLLayer.DECOMPOSE/ROUTER -> return (None, False)
    └── No execution - these layers delegate to LLM/children

Note: DSL *regeneration* (rewriting failing scripts) is a separate concern
handled by maybe_run_dsl_regeneration() in solver.py.

This module re-exports from:
- dsl_types: DSLLayer, DSLSpec, ValueType
- math_layer: Math DSL execution
"""

import logging
import re
import signal
from contextlib import contextmanager
from typing import Any, Optional

from mycelium.config import DSL_TIMEOUT_SEC

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


# Math layer
from mycelium.step_signatures.math_layer import (
    try_execute_dsl_math,
    extract_numeric_value as _extract_numeric_value,
)

# SymPy layer (algebra / backwards solving)
from mycelium.step_signatures.sympy_layer import (
    try_execute_dsl_sympy,
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
    timeout_sec: float = DSL_TIMEOUT_SEC,
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

    # DECOMPOSE/ROUTER layers: immediately fall back to LLM (no DSL execution)
    # ROUTER delegates to children at a higher level, not here
    if dsl_spec.layer in (DSLLayer.DECOMPOSE, DSLLayer.ROUTER):
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

    # Use filtered inputs for execution (no validation gates - let it fail and learn)
    mapped_inputs = filtered_inputs

    try:
        with _timeout(timeout_sec):
            if dsl_spec.layer == DSLLayer.MATH:
                result = try_execute_dsl_math(dsl_spec.script, mapped_inputs)
            elif dsl_spec.layer == DSLLayer.SYMPY:
                # Algebra / backwards solving via SymPy
                unknown_var = dsl_spec.params[0] if dsl_spec.params else "x"
                result = try_execute_dsl_sympy(dsl_spec.script, mapped_inputs, unknown_var)
            else:
                # Only MATH and SYMPY layers supported for execution
                return None, False

            if result is not None:
                # Basic sanity check: reject False (sympy failure) and astronomically large numbers
                if result is False:
                    logger.warning("[dsl_debug] Rejecting False result (sympy failure)")
                    return None, False
                try:
                    if isinstance(result, (int, float)) and abs(result) > 1e15:
                        logger.warning("[dsl_debug] Rejecting huge result: %s", result)
                        return None, False
                except (TypeError, OverflowError) as e:
                    logger.debug("[dsl_debug] Result size check skipped (non-comparable type): %s", e)
                logger.info(
                    "[dsl_debug] EXEC %s | script='%s' | inputs=%s | result=%s",
                    dsl_spec.layer.value, dsl_spec.script[:60],
                    {k: str(v)[:30] for k, v in mapped_inputs.items()}, result
                )
                return result, True
            logger.warning(
                "[dsl_debug] FAIL %s | script='%s' | inputs=%s | result=None",
                dsl_spec.layer.value, dsl_spec.script[:60],
                {k: str(v)[:30] for k, v in mapped_inputs.items()}
            )
            return None, False

    except TimeoutError:
        logger.warning("[dsl_debug] TIMEOUT | script='%s'", dsl_spec.script[:60])
        return None, False
    except Exception as e:
        logger.warning("[dsl_debug] ERROR %s | script='%s' | error=%s",
                       dsl_spec.layer.value, dsl_spec.script[:60], e)
        return None, False


