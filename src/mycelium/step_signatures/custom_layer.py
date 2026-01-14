"""Custom Layer: Domain-specific operator registration and execution.

This module contains:
- Custom operator registry
- Built-in operator registration
- Custom DSL execution
"""

import ast
import logging
from typing import Any, Callable, Optional

from mycelium.step_signatures.math_layer import FUNCTIONS as MATH_FUNCTIONS, add_function

logger = logging.getLogger(__name__)

# =============================================================================
# Custom operator registry
# =============================================================================

# Registry for custom operators
_CUSTOM_OPERATORS: dict[str, Callable] = {}


def register_operator(name: str, func: Callable) -> None:
    """Register a custom DSL operator."""
    _CUSTOM_OPERATORS[name] = func


def get_operator(name: str) -> Optional[Callable]:
    """Get a registered custom operator."""
    return _CUSTOM_OPERATORS.get(name)


def list_operators() -> list[str]:
    """List all registered custom operators."""
    return list(_CUSTOM_OPERATORS.keys())


# =============================================================================
# Built-in operator registration
# =============================================================================

def _register_builtin_operators():
    """Register built-in custom operators from math_ops module."""
    # Import pure math functions from math_ops module
    from mycelium.step_signatures.math_ops import (
        extract_coefficient,
        apply_quadratic_formula,
        complete_square,
        solve_linear,
        evaluate_polynomial,
        euclidean_gcd,
        modinv,
        divisors,
        divisor_count,
        divisor_count_from_factors,
        factorization_exponents,
        prime_factors,
        is_prime,
        mod_pow,
        int_to_base,
        from_base,
        base_multiply,
        base_add,
        binomial,
        permutations,
        combinations,
        day_of_week,
        triangular_number,
        fibonacci,
    )

    # Register built-in custom operators (imported from math_ops)
    register_operator("extract_coefficient", extract_coefficient)
    register_operator("apply_quadratic_formula", apply_quadratic_formula)
    register_operator("complete_square", complete_square)
    register_operator("solve_linear", solve_linear)
    register_operator("evaluate_polynomial", evaluate_polynomial)
    # Number theory operators
    register_operator("euclidean_gcd", euclidean_gcd)
    register_operator("modinv", modinv)
    register_operator("divisors", divisors)
    register_operator("divisor_count", divisor_count)
    register_operator("count_divisors", divisor_count)  # Alias
    register_operator("divisor_count_from_factors", divisor_count_from_factors)
    register_operator("count_divisors_from_factors", divisor_count_from_factors)  # Alias
    register_operator("factorization_exponents", factorization_exponents)
    register_operator("prime_factors", prime_factors)
    register_operator("is_prime", is_prime)
    register_operator("mod_pow", mod_pow)
    # Base conversion operators
    register_operator("int_to_base", int_to_base)
    register_operator("to_base", int_to_base)  # Alias
    register_operator("from_base", from_base)
    register_operator("base_multiply", base_multiply)
    register_operator("base_add", base_add)
    # Combinatorics operators
    register_operator("binomial", binomial)
    register_operator("permutations", permutations)
    register_operator("combinations", combinations)
    register_operator("C", combinations)  # Alias
    register_operator("P", permutations)  # Alias
    # Misc operators
    register_operator("day_of_week", day_of_week)
    register_operator("triangular_number", triangular_number)
    register_operator("fibonacci", fibonacci)
    # Python built-ins (for compatibility with auto-generated DSLs)
    register_operator("len", len)
    register_operator("sum", sum)
    register_operator("list", list)
    register_operator("set", lambda *args: set(args) if len(args) != 1 or not hasattr(args[0], '__iter__') else set(args[0]))
    register_operator("sorted", sorted)
    register_operator("int", int)
    register_operator("float", float)
    register_operator("abs", abs)
    register_operator("min", min)
    register_operator("max", max)

    # Add base conversion functions to math layer for compatibility
    add_function("int_to_base", int_to_base)
    add_function("to_base", int_to_base)
    add_function("base_multiply", base_multiply)
    add_function("base_add", base_add)


# Register built-in operators on module load
_register_builtin_operators()


# =============================================================================
# Custom DSL execution
# =============================================================================

def try_execute_dsl_custom(script: str, inputs: dict[str, Any]) -> Optional[Any]:
    """Execute custom operator DSL script.

    Format: operator_name(arg1, arg2, ...)
    """
    try:
        tree = ast.parse(script, mode='eval')

        # Must be a single function call
        if not isinstance(tree.body, ast.Call):
            return None
        if not isinstance(tree.body.func, ast.Name):
            return None

        func_name = tree.body.func.id
        if func_name not in _CUSTOM_OPERATORS:
            logger.warning("Unknown custom operator: %s", func_name)
            return None

        # Evaluate arguments
        def eval_arg(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.Str):
                return node.s
            if isinstance(node, ast.Name):
                if node.id in inputs:
                    return inputs[node.id]
                raise KeyError(f"Unknown variable: {node.id}")
            if isinstance(node, ast.List):
                return [eval_arg(elt) for elt in node.elts]
            raise ValueError(f"Unsupported arg type: {type(node).__name__}")

        args = [eval_arg(arg) for arg in tree.body.args]
        kwargs = {kw.arg: eval_arg(kw.value) for kw in tree.body.keywords}

        return _CUSTOM_OPERATORS[func_name](*args, **kwargs)

    except Exception as e:
        logger.debug("Custom operator failed: %s", e)
        return None
