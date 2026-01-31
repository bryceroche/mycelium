"""
Function Registry - Curated Python math function pointers organized by tier.

This module provides a centralized registry of mathematical functions that can be
used throughout the mycelium system for DSL execution and computation graphs.
"""

import operator
import math
import statistics
from typing import Any, Callable, List, Optional

# Try to import sympy for symbolic operations (tier 7)
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

FUNCTION_REGISTRY = {
    # =========================================================================
    # TIER 1: Arithmetic
    # =========================================================================
    "add": {
        "func": operator.add,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Add two numbers",
    },
    "sub": {
        "func": operator.sub,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Subtract second number from first",
    },
    "mul": {
        "func": operator.mul,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Multiply two numbers",
    },
    "truediv": {
        "func": operator.truediv,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Divide first number by second (true division)",
    },
    "floordiv": {
        "func": operator.floordiv,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Divide first number by second (floor division)",
    },
    "mod": {
        "func": operator.mod,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Modulo (remainder) of first number divided by second",
    },
    "pow": {
        "func": operator.pow,
        "arity": 2,
        "tier": 1,
        "module": "operator",
        "description": "Raise first number to the power of second",
    },
    "neg": {
        "func": operator.neg,
        "arity": 1,
        "tier": 1,
        "module": "operator",
        "description": "Negate a number",
    },
    "abs": {
        "func": abs,
        "arity": 1,
        "tier": 1,
        "module": "builtins",
        "description": "Absolute value of a number",
    },
    "sqrt": {
        "func": math.sqrt,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Square root of a number",
    },
    "cbrt": {
        "func": math.cbrt,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Cube root of a number",
    },
    "floor": {
        "func": math.floor,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Floor of a number (largest integer <= x)",
    },
    "ceil": {
        "func": math.ceil,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Ceiling of a number (smallest integer >= x)",
    },
    "trunc": {
        "func": math.trunc,
        "arity": 1,
        "tier": 1,
        "module": "math",
        "description": "Truncate a number (integer part only)",
    },

    # =========================================================================
    # TIER 2: Comparison
    # =========================================================================
    "eq": {
        "func": operator.eq,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test equality of two values",
    },
    "ne": {
        "func": operator.ne,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test inequality of two values",
    },
    "lt": {
        "func": operator.lt,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is less than second",
    },
    "le": {
        "func": operator.le,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is less than or equal to second",
    },
    "gt": {
        "func": operator.gt,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is greater than second",
    },
    "ge": {
        "func": operator.ge,
        "arity": 2,
        "tier": 2,
        "module": "operator",
        "description": "Test if first value is greater than or equal to second",
    },
    "max": {
        "func": max,
        "arity": -1,  # variadic
        "tier": 2,
        "module": "builtins",
        "description": "Return the maximum of the given values",
    },
    "min": {
        "func": min,
        "arity": -1,  # variadic
        "tier": 2,
        "module": "builtins",
        "description": "Return the minimum of the given values",
    },

    # =========================================================================
    # TIER 3: Trigonometry
    # =========================================================================
    "sin": {
        "func": math.sin,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Sine of angle (in radians)",
    },
    "cos": {
        "func": math.cos,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Cosine of angle (in radians)",
    },
    "tan": {
        "func": math.tan,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Tangent of angle (in radians)",
    },
    "asin": {
        "func": math.asin,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc sine (inverse sine), returns radians",
    },
    "acos": {
        "func": math.acos,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc cosine (inverse cosine), returns radians",
    },
    "atan": {
        "func": math.atan,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Arc tangent (inverse tangent), returns radians",
    },
    "atan2": {
        "func": math.atan2,
        "arity": 2,
        "tier": 3,
        "module": "math",
        "description": "Arc tangent of y/x, returns radians in correct quadrant",
    },
    "degrees": {
        "func": math.degrees,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Convert radians to degrees",
    },
    "radians": {
        "func": math.radians,
        "arity": 1,
        "tier": 3,
        "module": "math",
        "description": "Convert degrees to radians",
    },
    "hypot": {
        "func": math.hypot,
        "arity": -1,  # variadic in Python 3.8+
        "tier": 3,
        "module": "math",
        "description": "Euclidean distance (hypotenuse) from origin",
    },

    # =========================================================================
    # TIER 4: Logarithms/Exponentials
    # =========================================================================
    "log": {
        "func": math.log,
        "arity": -1,  # 1 or 2 args (value, optional base)
        "tier": 4,
        "module": "math",
        "description": "Natural logarithm (or log with given base)",
    },
    "log10": {
        "func": math.log10,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "Base-10 logarithm",
    },
    "log2": {
        "func": math.log2,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "Base-2 logarithm",
    },
    "exp": {
        "func": math.exp,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "e raised to the power x",
    },
    "exp2": {
        "func": math.exp2,
        "arity": 1,
        "tier": 4,
        "module": "math",
        "description": "2 raised to the power x",
    },

    # =========================================================================
    # TIER 5: Number Theory
    # =========================================================================
    "gcd": {
        "func": math.gcd,
        "arity": -1,  # variadic in Python 3.9+
        "tier": 5,
        "module": "math",
        "description": "Greatest common divisor",
    },
    "lcm": {
        "func": math.lcm,
        "arity": -1,  # variadic in Python 3.9+
        "tier": 5,
        "module": "math",
        "description": "Least common multiple",
    },
    "factorial": {
        "func": math.factorial,
        "arity": 1,
        "tier": 5,
        "module": "math",
        "description": "Factorial of a non-negative integer",
    },
    "comb": {
        "func": math.comb,
        "arity": 2,
        "tier": 5,
        "module": "math",
        "description": "Number of combinations (n choose k)",
    },
    "perm": {
        "func": math.perm,
        "arity": 2,
        "tier": 5,
        "module": "math",
        "description": "Number of permutations (n permute k)",
    },
    "isqrt": {
        "func": math.isqrt,
        "arity": 1,
        "tier": 5,
        "module": "math",
        "description": "Integer square root (floor of sqrt)",
    },

    # =========================================================================
    # TIER 6: Statistics
    # =========================================================================
    "mean": {
        "func": statistics.mean,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Arithmetic mean of data",
    },
    "median": {
        "func": statistics.median,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Median (middle value) of data",
    },
    "mode": {
        "func": statistics.mode,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Mode (most common value) of data",
    },
    "stdev": {
        "func": statistics.stdev,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Sample standard deviation of data",
    },
    "variance": {
        "func": statistics.variance,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "statistics",
        "description": "Sample variance of data",
    },
    "sum": {
        "func": sum,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "builtins",
        "description": "Sum of all values in an iterable",
    },
    "len": {
        "func": len,
        "arity": 1,  # takes iterable
        "tier": 6,
        "module": "builtins",
        "description": "Number of items in an iterable",
    },
}

# =========================================================================
# TIER 7: Symbolic (optional, requires sympy)
# =========================================================================
if SYMPY_AVAILABLE:
    FUNCTION_REGISTRY.update({
        "solve": {
            "func": sympy.solve,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Solve algebraic equations",
        },
        "simplify": {
            "func": sympy.simplify,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Simplify a symbolic expression",
        },
        "expand": {
            "func": sympy.expand,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Expand a symbolic expression",
        },
        "factor": {
            "func": sympy.factor,
            "arity": 1,
            "tier": 7,
            "module": "sympy",
            "description": "Factor a symbolic expression",
        },
        "diff": {
            "func": sympy.diff,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Differentiate a symbolic expression",
        },
        "integrate": {
            "func": sympy.integrate,
            "arity": -1,  # variadic
            "tier": 7,
            "module": "sympy",
            "description": "Integrate a symbolic expression",
        },
        "limit": {
            "func": sympy.limit,
            "arity": 3,
            "tier": 7,
            "module": "sympy",
            "description": "Compute the limit of a symbolic expression",
        },
    })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_function(name: str) -> Callable:
    """
    Get function by name.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.

    Returns:
        The callable function.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    if name not in FUNCTION_REGISTRY:
        raise KeyError(f"Function '{name}' not found in registry. "
                       f"Available functions: {list(FUNCTION_REGISTRY.keys())}")
    return FUNCTION_REGISTRY[name]["func"]


def list_functions(tier: Optional[int] = None) -> List[str]:
    """
    List function names, optionally filtered by tier.

    Args:
        tier: If provided, only return functions from this tier.
              If None, return all function names.

    Returns:
        List of function names.
    """
    if tier is None:
        return list(FUNCTION_REGISTRY.keys())
    return [name for name, info in FUNCTION_REGISTRY.items() if info["tier"] == tier]


def call_function(name: str, *args) -> Any:
    """
    Call a function by name with given arguments.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.
        *args: Arguments to pass to the function.

    Returns:
        The result of calling the function with the given arguments.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    func = get_function(name)
    return func(*args)


def get_function_info(name: str) -> dict:
    """
    Get full info dict for a function.

    Args:
        name: The function name as registered in FUNCTION_REGISTRY.

    Returns:
        Dictionary containing func, arity, tier, module, and description.

    Raises:
        KeyError: If the function name is not found in the registry.
    """
    if name not in FUNCTION_REGISTRY:
        raise KeyError(f"Function '{name}' not found in registry. "
                       f"Available functions: {list(FUNCTION_REGISTRY.keys())}")
    return FUNCTION_REGISTRY[name].copy()


def get_tiers() -> List[int]:
    """
    Get list of all available tiers.

    Returns:
        Sorted list of unique tier numbers.
    """
    return sorted(set(info["tier"] for info in FUNCTION_REGISTRY.values()))


def get_tier_description(tier: int) -> str:
    """
    Get a human-readable description of a tier.

    Args:
        tier: The tier number.

    Returns:
        Description string for the tier.
    """
    descriptions = {
        1: "Arithmetic",
        2: "Comparison",
        3: "Trigonometry",
        4: "Logarithms/Exponentials",
        5: "Number Theory",
        6: "Statistics",
        7: "Symbolic (requires sympy)",
    }
    return descriptions.get(tier, f"Tier {tier}")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Running function_registry tests...\n")

    # -------------------------------------------------------------------------
    # Tier 1: Arithmetic tests
    # -------------------------------------------------------------------------
    print("Tier 1: Arithmetic")
    assert call_function("add", 2, 3) == 5, "add failed"
    assert call_function("sub", 10, 4) == 6, "sub failed"
    assert call_function("mul", 3, 7) == 21, "mul failed"
    assert call_function("truediv", 15, 4) == 3.75, "truediv failed"
    assert call_function("floordiv", 15, 4) == 3, "floordiv failed"
    assert call_function("mod", 17, 5) == 2, "mod failed"
    assert call_function("pow", 2, 10) == 1024, "pow failed"
    assert call_function("neg", 5) == -5, "neg failed"
    assert call_function("abs", -42) == 42, "abs failed"
    assert call_function("sqrt", 16) == 4.0, "sqrt failed"
    assert call_function("cbrt", 27) == 3.0, "cbrt failed"
    assert call_function("floor", 3.7) == 3, "floor failed"
    assert call_function("ceil", 3.2) == 4, "ceil failed"
    assert call_function("trunc", -3.7) == -3, "trunc failed"
    print("  All Tier 1 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 2: Comparison tests
    # -------------------------------------------------------------------------
    print("Tier 2: Comparison")
    assert call_function("eq", 5, 5) is True, "eq failed"
    assert call_function("ne", 5, 3) is True, "ne failed"
    assert call_function("lt", 3, 5) is True, "lt failed"
    assert call_function("le", 5, 5) is True, "le failed"
    assert call_function("gt", 7, 3) is True, "gt failed"
    assert call_function("ge", 7, 7) is True, "ge failed"
    assert call_function("max", 1, 5, 3) == 5, "max failed"
    assert call_function("min", 1, 5, 3) == 1, "min failed"
    print("  All Tier 2 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 3: Trigonometry tests
    # -------------------------------------------------------------------------
    print("Tier 3: Trigonometry")
    import math as m
    assert abs(call_function("sin", m.pi / 2) - 1.0) < 1e-10, "sin failed"
    assert abs(call_function("cos", 0) - 1.0) < 1e-10, "cos failed"
    assert abs(call_function("tan", 0) - 0.0) < 1e-10, "tan failed"
    assert abs(call_function("asin", 1) - m.pi / 2) < 1e-10, "asin failed"
    assert abs(call_function("acos", 1) - 0.0) < 1e-10, "acos failed"
    assert abs(call_function("atan", 0) - 0.0) < 1e-10, "atan failed"
    assert abs(call_function("atan2", 1, 1) - m.pi / 4) < 1e-10, "atan2 failed"
    assert abs(call_function("degrees", m.pi) - 180.0) < 1e-10, "degrees failed"
    assert abs(call_function("radians", 180) - m.pi) < 1e-10, "radians failed"
    assert call_function("hypot", 3, 4) == 5.0, "hypot failed"
    print("  All Tier 3 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 4: Logarithms/Exponentials tests
    # -------------------------------------------------------------------------
    print("Tier 4: Logarithms/Exponentials")
    assert abs(call_function("log", m.e) - 1.0) < 1e-10, "log (natural) failed"
    assert abs(call_function("log", 100, 10) - 2.0) < 1e-10, "log (base 10) failed"
    assert abs(call_function("log10", 1000) - 3.0) < 1e-10, "log10 failed"
    assert abs(call_function("log2", 8) - 3.0) < 1e-10, "log2 failed"
    assert abs(call_function("exp", 1) - m.e) < 1e-10, "exp failed"
    assert call_function("exp2", 3) == 8.0, "exp2 failed"
    print("  All Tier 4 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 5: Number Theory tests
    # -------------------------------------------------------------------------
    print("Tier 5: Number Theory")
    assert call_function("gcd", 48, 18) == 6, "gcd failed"
    assert call_function("lcm", 4, 6) == 12, "lcm failed"
    assert call_function("factorial", 5) == 120, "factorial failed"
    assert call_function("comb", 5, 2) == 10, "comb failed"
    assert call_function("perm", 5, 2) == 20, "perm failed"
    assert call_function("isqrt", 17) == 4, "isqrt failed"
    print("  All Tier 5 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 6: Statistics tests
    # -------------------------------------------------------------------------
    print("Tier 6: Statistics")
    assert call_function("mean", [1, 2, 3, 4, 5]) == 3.0, "mean failed"
    assert call_function("median", [1, 2, 3, 4, 5]) == 3, "median failed"
    assert call_function("mode", [1, 2, 2, 3, 3, 3]) == 3, "mode failed"
    assert abs(call_function("stdev", [2, 4, 4, 4, 5, 5, 7, 9]) - 2.138089935299395) < 1e-10, "stdev failed"
    assert abs(call_function("variance", [2, 4, 4, 4, 5, 5, 7, 9]) - 4.571428571428571) < 1e-10, "variance failed"
    assert call_function("sum", [1, 2, 3, 4, 5]) == 15, "sum failed"
    assert call_function("len", [1, 2, 3, 4, 5]) == 5, "len failed"
    print("  All Tier 6 tests passed!")

    # -------------------------------------------------------------------------
    # Tier 7: Symbolic tests (if sympy available)
    # -------------------------------------------------------------------------
    if SYMPY_AVAILABLE:
        print("Tier 7: Symbolic (sympy)")
        x = sympy.Symbol('x')
        # Test simplify - verify it simplifies sin^2 + cos^2 to 1
        expr = sympy.sin(x)**2 + sympy.cos(x)**2
        simplified = call_function("simplify", expr)
        assert simplified == 1, "simplify failed"
        # Test expand
        expanded = call_function("expand", (x + 1)**2)
        assert expanded == x**2 + 2*x + 1, "expand failed"
        # Test factor
        factored = call_function("factor", x**2 - 1)
        assert factored == (x - 1)*(x + 1), "factor failed"
        # Test diff
        diff_result = call_function("diff", x**3, x)
        assert diff_result == 3*x**2, "diff failed"
        # Test integrate
        int_result = call_function("integrate", 2*x, x)
        assert int_result == x**2, "integrate failed"
        # Test solve
        solutions = call_function("solve", x**2 - 4, x)
        assert set(solutions) == {-2, 2}, "solve failed"
        print("  All Tier 7 tests passed!")
    else:
        print("Tier 7: Symbolic (sympy) - SKIPPED (sympy not installed)")

    # -------------------------------------------------------------------------
    # Helper function tests
    # -------------------------------------------------------------------------
    print("\nHelper function tests:")

    # Test get_function
    add_func = get_function("add")
    assert add_func(1, 2) == 3, "get_function failed"
    print("  get_function: passed")

    # Test get_function KeyError
    try:
        get_function("nonexistent")
        assert False, "get_function should raise KeyError"
    except KeyError:
        pass
    print("  get_function KeyError: passed")

    # Test list_functions
    all_funcs = list_functions()
    assert len(all_funcs) > 0, "list_functions returned empty"
    assert "add" in all_funcs, "add not in list_functions"
    print(f"  list_functions (all): {len(all_funcs)} functions")

    tier1_funcs = list_functions(tier=1)
    assert "add" in tier1_funcs, "add not in tier 1"
    assert "sin" not in tier1_funcs, "sin should not be in tier 1"
    print(f"  list_functions (tier 1): {len(tier1_funcs)} functions")

    # Test get_function_info
    add_info = get_function_info("add")
    assert add_info["arity"] == 2, "add arity wrong"
    assert add_info["tier"] == 1, "add tier wrong"
    assert add_info["module"] == "operator", "add module wrong"
    print("  get_function_info: passed")

    # Test get_tiers
    tiers = get_tiers()
    assert 1 in tiers and 6 in tiers, "get_tiers missing expected tiers"
    print(f"  get_tiers: {tiers}")

    # Test get_tier_description
    assert get_tier_description(1) == "Arithmetic", "tier 1 description wrong"
    assert get_tier_description(3) == "Trigonometry", "tier 3 description wrong"
    print("  get_tier_description: passed")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
