"""Math Layer: Basic arithmetic DSL execution.

This module contains:
- Safe AST-based math evaluator
- Numeric value extraction from text
- Allowed operators, functions, and constants
"""

import ast
import logging
import math
import operator
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Allowed operators and functions
# =============================================================================

# Allowed binary operators
BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed functions
FUNCTIONS: dict[str, Callable] = {
    # Basic
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
    "sum": lambda *args: sum(args),  # Vararg sum for DSL like sum(a, b, c)
    "len": lambda *args: len(args),  # Count arguments
    # Roots and exponents
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1/3),  # Cube root
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Rounding
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    # Combinatorics (convert floats to ints for integer-only functions)
    "factorial": lambda n: math.factorial(int(n)),
    "gcd": lambda a, b: math.gcd(int(a), int(b)),
    "lcm": lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)) if a and b else 0,
    "C": lambda n, r: math.comb(int(n), int(r)),  # Combinations
    "P": lambda n, r: math.perm(int(n), int(r)),  # Permutations
    "comb": lambda n, r: math.comb(int(n), int(r)),
    "perm": lambda n, r: math.perm(int(n), int(r)),
    "choose": lambda n, r: math.comb(int(n), int(r)),  # Alias
    # Hyperbolic (occasionally needed)
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Modular arithmetic
    "mod": lambda a, b: int(a) % int(b),
    "divmod": lambda a, b: (int(a) // int(b), int(a) % int(b)),
    # Integer operations - int() handles 1 or 2 args for base conversion
    # Smart base detection: if first arg looks like a base (2-36), swap arguments
    "int": lambda *args: (
        int(str(int(args[1])), int(args[0]))  # Swap: n was passed as base, a as number
        if len(args) == 2 and 2 <= int(args[0]) <= 36 and int(args[1]) > 36
        else int(str(int(args[0])), int(args[1])) if len(args) == 2 else int(args[0])
    ),
    "float": lambda x: float(x),
    "str": lambda x: str(int(x)) if isinstance(x, float) and x == int(x) else str(x),
    "isqrt": lambda n: int(math.sqrt(int(n))),  # Integer square root
    # Base conversion - simple from_base only (int_to_base added via register_operator later)
    # Handle float inputs like 2012.0 by converting to int string first
    "from_base": lambda s, base: int(str(int(float(s))) if '.' in str(s) else str(s).strip(), int(base)),
}

# Allowed constants
CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


# =============================================================================
# Numeric value extraction
# =============================================================================

def extract_numeric_value(value: Any) -> Optional[float]:
    """Extract numeric value from various input types.

    Handles:
    - int/float: return directly
    - "42" or "3.14": parse directly
    - "The value is 16": extract 16
    - "Answer = 25": extract 25
    - "3/4": compute 0.75
    """
    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    # Clean common formatting
    cleaned = raw.replace(',', '').replace('$', '').replace('%', '')

    # Try direct parse first (fast path)
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Handle fractions like "3/4"
    if '/' in cleaned and cleaned.count('/') == 1:
        parts = cleaned.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass

    # Skip if it looks like an expression with operators
    if any(c in cleaned for c in ['+', '*', '^', '=']):
        if not re.match(r'^-?\d+\.?\d*$', cleaned):
            # But continue to try extraction below
            pass

    # Try to extract number from text patterns
    answer_patterns = [
        r'(?:answer|result|value|equals?|is|=)\s*[=:]?\s*(-?\d+\.?\d*)',  # "answer is 25"
        r'=\s*(-?\d+\.?\d*)\s*$',  # "x = 25" at end
        r':\s*(-?\d+\.?\d*)\s*$',  # "result: 42" at end
        r'\b(-?\d+\.?\d*)\s*$',  # last number in string
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def prepare_math_inputs(inputs: dict[str, Any]) -> dict[str, float]:
    """Convert input values to floats for math DSL execution.

    Extracts numeric values from text context values.
    """
    result = {}
    for key, value in inputs.items():
        extracted = extract_numeric_value(value)
        if extracted is not None:
            result[key] = extracted
    return result


# =============================================================================
# Safe math evaluation
# =============================================================================

def _safe_eval_math(script: str, inputs: dict[str, float]) -> Optional[float]:
    """AST-based safe math evaluator. No eval() used."""

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {type(node.value)}")

        elif isinstance(node, ast.Num):  # Python 3.7 compat
            return float(node.n)

        elif isinstance(node, ast.Name):
            name = node.id
            if name in inputs:
                return float(inputs[name])
            if name in CONSTANTS:
                return CONSTANTS[name]
            raise KeyError(f"Unknown variable: {name}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return BINOPS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARYOPS:
                raise ValueError(f"Unsupported unary: {op_type.__name__}")
            return UNARYOPS[op_type](_eval_node(node.operand))

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [_eval_node(arg) for arg in node.args]
            return FUNCTIONS[func_name](*args)

        else:
            raise ValueError(f"Unsupported node: {type(node).__name__}")

    try:
        tree = ast.parse(script, mode='eval')
        return _eval_node(tree)
    except Exception as e:
        logger.debug("Math eval failed: %s", e)
        return None


def try_execute_dsl_math(script: str, inputs: dict[str, Any]) -> Optional[float]:
    """Execute basic math DSL script.

    Automatically extracts numeric values from text inputs like "The value is 16".

    Special cases:
    - sum_all(): Returns sum of all numeric inputs
    - Single input with multi-param script: Returns the single value (identity)
    """
    # Prepare inputs: extract numeric values from text
    prepared = prepare_math_inputs(inputs)
    if not prepared and inputs:
        # No numeric values could be extracted
        logger.debug("Math DSL: no numeric values extracted from inputs: %s", list(inputs.keys()))
        return None

    # Special case: sum_all() - sum all available numeric inputs
    if script.strip() == "sum_all()":
        if not prepared:
            return None
        return sum(prepared.values())

    # Note: We intentionally DON'T do identity fallback for multi-var scripts
    # If a sum/product script has only 1 input, it's a semantic mismatch
    # The correct fix is better signature matching, not wrong identity results

    return _safe_eval_math(script, prepared)


def add_function(name: str, func: Callable) -> None:
    """Add a function to the math layer."""
    FUNCTIONS[name] = func
