"""SymPy Layer: Symbolic algebra DSL execution for backwards solving.

This module contains:
- Equation construction from known/unknown values
- SymPy solve() wrapper with safe execution
- Result extraction and validation

Used when a step has undefined variables that require working backwards
from known results to find unknown inputs.
"""

__all__ = [
    "try_execute_dsl_sympy",
    "build_equation_from_values",
    "solve_for_unknown",
]

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import SymPy - it's already a dependency
try:
    from sympy import Symbol, Eq, solve, simplify, Float, Integer, Rational
    from sympy.core.numbers import NumberSymbol
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("[sympy_layer] SymPy not available - algebra solving disabled")


def try_execute_dsl_sympy(
    script: str,
    inputs: dict[str, Any],
    unknown_var: str = "x",
) -> Optional[float]:
    """Execute a SymPy equation solving script.

    For backwards solving, one input will be None (the unknown).
    The function will:
    1. Identify the unknown (None value)
    2. Create SymPy Symbol for unknown
    3. Parse and execute the solve script
    4. Return numeric result

    Args:
        script: The SymPy script (e.g., "solve(Eq(x + a, b), x)")
        inputs: Dict with known values (numbers) and unknown (None)
        unknown_var: Name of the unknown variable (default "x")

    Returns:
        Solved value as float, or None on failure
    """
    if not SYMPY_AVAILABLE:
        logger.warning("[sympy] SymPy not available")
        return None

    try:
        # Identify unknown variable (value is None)
        unknowns = [k for k, v in inputs.items() if v is None]
        known = {k: v for k, v in inputs.items() if v is not None}

        if not unknowns:
            logger.debug("[sympy] No unknown variables found in inputs")
            return None

        if len(unknowns) > 1:
            logger.warning("[sympy] Multiple unknowns not supported: %s", unknowns)
            return None

        unknown = unknowns[0]

        # Create symbol for unknown
        x = Symbol(unknown)

        # Build safe namespace with known values and sympy functions
        namespace = {
            unknown: x,
            "Eq": Eq,
            "solve": solve,
            "Symbol": Symbol,
        }

        # Add known values to namespace
        for name, value in known.items():
            try:
                namespace[name] = float(value)
            except (ValueError, TypeError):
                namespace[name] = value

        # Execute script
        logger.debug("[sympy] Executing: %s with namespace keys: %s", script, list(namespace.keys()))
        result = eval(script, {"__builtins__": {}}, namespace)

        # Extract numeric value from SymPy result
        return _extract_result(result)

    except Exception as e:
        logger.warning("[sympy] Execution failed: %s", e)
        return None


def solve_for_unknown(
    known_values: dict[str, float],
    unknown_name: str,
    operation: str,
    result_value: float,
) -> Optional[float]:
    """Solve for an unknown variable given known values and operation.

    This is a simplified interface for common algebra patterns:
    - unknown + known = result → unknown = result - known
    - unknown - known = result → unknown = result + known
    - unknown * known = result → unknown = result / known
    - unknown / known = result → unknown = result * known

    Args:
        known_values: Dict of known variable names to values
        unknown_name: Name of the variable to solve for
        operation: The operation ("add", "subtract", "multiply", "divide")
        result_value: The known result of the operation

    Returns:
        Solved value for the unknown, or None on failure
    """
    if not SYMPY_AVAILABLE:
        return None

    try:
        x = Symbol(unknown_name)
        known_list = list(known_values.values())

        if not known_list:
            logger.warning("[sympy] No known values provided")
            return None

        known = known_list[0]  # Use first known value

        # Build equation based on operation
        if operation in ("add", "+"):
            # x + known = result → x = result - known
            eq = Eq(x + known, result_value)
        elif operation in ("subtract", "-"):
            # x - known = result → x = result + known
            eq = Eq(x - known, result_value)
        elif operation in ("multiply", "*"):
            # x * known = result → x = result / known
            eq = Eq(x * known, result_value)
        elif operation in ("divide", "/"):
            # x / known = result → x = result * known
            eq = Eq(x / known, result_value)
        else:
            logger.warning("[sympy] Unknown operation: %s", operation)
            return None

        # Solve
        solutions = solve(eq, x)
        return _extract_result(solutions)

    except Exception as e:
        logger.warning("[sympy] solve_for_unknown failed: %s", e)
        return None


def build_equation_from_values(
    values: dict[str, Any],
    operation: str,
) -> tuple[str, str]:
    """Build SymPy equation script from values and operation.

    Given values like:
        {"initial_vacuums": None, "sold": 3, "remaining": 5}
    And operation: "subtract"

    Returns:
        (script, unknown_name) tuple
        script: "solve(Eq(initial_vacuums - sold, remaining), initial_vacuums)"

    This constructs the equation for backwards solving.
    """
    unknowns = [k for k, v in values.items() if v is None]
    knowns = {k: v for k, v in values.items() if v is not None}

    if not unknowns:
        return "", ""

    if len(unknowns) > 1:
        logger.warning("[sympy] Multiple unknowns in values: %s", unknowns)
        return "", ""

    unknown = unknowns[0]
    known_names = list(knowns.keys())

    if len(known_names) < 2:
        logger.warning("[sympy] Need at least 2 known values, got %d", len(known_names))
        return "", ""

    # The pattern is: unknown OP known_1 = known_2
    # Where known_2 is typically the "result" value

    # Heuristics to identify which known is the "result":
    # Look for names like "result", "remaining", "left", "total", "final"
    result_hints = ["result", "remaining", "left", "total", "final", "answer", "end"]
    result_var = None
    operand_var = None

    for name in known_names:
        name_lower = name.lower()
        if any(hint in name_lower for hint in result_hints):
            result_var = name
        else:
            operand_var = name

    # If no hint found, use positional: first known is operand, second is result
    if result_var is None:
        operand_var = known_names[0]
        result_var = known_names[1] if len(known_names) > 1 else known_names[0]
    elif operand_var is None:
        # Result found but not operand - use the other known
        for name in known_names:
            if name != result_var:
                operand_var = name
                break

    if operand_var is None or result_var is None:
        logger.warning("[sympy] Could not identify operand and result from: %s", known_names)
        return "", ""

    # Build equation expression based on operation
    if operation in ("add", "+"):
        expr = f"{unknown} + {operand_var}"
    elif operation in ("subtract", "-"):
        expr = f"{unknown} - {operand_var}"
    elif operation in ("multiply", "*"):
        expr = f"{unknown} * {operand_var}"
    elif operation in ("divide", "/"):
        expr = f"{unknown} / {operand_var}"
    else:
        # Try to infer from common patterns
        logger.warning("[sympy] Unknown operation '%s', trying generic solve", operation)
        # Generic: assume additive relationship
        expr = f"{unknown} + {operand_var}"

    script = f"solve(Eq({expr}, {result_var}), {unknown})"
    logger.debug("[sympy] Built equation: %s", script)
    return script, unknown


def _extract_result(result: Any) -> Optional[float]:
    """Extract a numeric float from SymPy result.

    SymPy solve() can return:
    - A list of solutions: [5]
    - A single value: 5
    - A SymPy number: Integer(5), Float(5.0)
    - An empty list: [] (no solution)
    - A complex number (reject)
    """
    if result is None:
        return None

    # Handle list of solutions
    if isinstance(result, list):
        if not result:
            logger.debug("[sympy] No solutions found")
            return None
        result = result[0]  # Take first solution

    # Try to convert to float
    try:
        # Check for complex numbers (reject)
        if hasattr(result, 'is_real') and result.is_real is False:
            logger.debug("[sympy] Rejecting complex result: %s", result)
            return None

        value = float(result)

        # Sanity check: reject astronomically large numbers
        if abs(value) > 1e15:
            logger.warning("[sympy] Rejecting huge result: %s", value)
            return None

        logger.debug("[sympy] Extracted result: %s", value)
        return value

    except (TypeError, ValueError) as e:
        logger.warning("[sympy] Could not convert to float: %s (%s)", result, e)
        return None
