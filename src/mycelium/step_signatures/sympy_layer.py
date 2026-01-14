"""SymPy Layer: Symbolic algebra DSL execution.

This module contains:
- Safe SymPy evaluator with whitelisted functions
- String to SymPy expression parser
"""

import ast
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Whitelisted SymPy functions and methods
# =============================================================================

# Whitelisted SymPy functions
SYMPY_ALLOWED = {
    # Core
    "Symbol", "symbols", "Eq", "sympify", "N", "Add", "Mul",  # N for numerical evaluation
    # Algebra
    "solve", "simplify", "expand", "factor", "collect", "cancel", "apart",
    # Calculus
    "diff", "integrate", "limit", "series",
    # Functions
    "sqrt", "Abs", "sin", "cos", "tan", "log", "exp",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    # Trigonometry
    "trigsimp", "expand_trig",
    # Combinatorics
    "binomial", "factorial", "ff", "rf",  # falling/rising factorial
    # Number theory
    "gcd", "lcm", "factorint", "divisors", "isprime", "nextprime", "primefactors",
    # Polynomials
    "Poly", "degree", "roots",
}

# Whitelisted method names for SymPy objects
SYMPY_ALLOWED_METHODS = {
    "subs", "evalf", "simplify", "expand", "factor",
    "diff", "integrate", "limit", "series",
    "coeff", "as_coeff_mul", "as_coefficients_dict",
}


# =============================================================================
# String to SymPy parsing
# =============================================================================

def parse_to_sympy(value: str, sympy) -> Any:
    """Parse a string into a sympy expression.

    Handles:
    - "x^2 - 4 = 0" → Eq(x**2 - 4, 0)
    - "x^2 + 2x - 3" → x**2 + 2*x - 3
    - "42" → 42 (number)
    - "The equation x^2 - 4 = 0" → Eq(x**2 - 4, 0) (extract from text)
    - "\frac{x}{2} = 3" → Eq(x/2, 3) (LaTeX)
    - Plain text → original string (unchanged)
    """
    if not value or not value.strip():
        return value

    cleaned = value.strip()

    # Try to parse as number first
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Pre-process: Extract equation from text like "The equation is x^2 = 4"
    # Look for patterns with = sign surrounded by math-like content
    eq_patterns = [
        r'(?:equation|expression|formula)[:\s]+([^,\.]+=[^,\.]+)',
        r'([a-zA-Z0-9\^\*\+\-\s\(\)]+\s*=\s*[a-zA-Z0-9\^\*\+\-\s\(\)]+)',
    ]
    for pattern in eq_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            break

    # Convert LaTeX fractions: \frac{a}{b} → (a)/(b)
    cleaned = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', cleaned)
    # Remove other LaTeX commands
    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)
    cleaned = cleaned.replace('{', '(').replace('}', ')')

    # Normalize notation: ^ to **, implicit multiplication
    normalized = cleaned.replace('^', '**')
    # Add implicit multiplication: 2x → 2*x, x( → x*(
    normalized = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', normalized)
    normalized = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', normalized)
    normalized = re.sub(r'(\d)\(', r'\1*(', normalized)
    normalized = re.sub(r'\)(\d)', r')*\1', normalized)
    normalized = re.sub(r'\)\(', r')*(', normalized)
    normalized = re.sub(r'([a-zA-Z])\((?![a-zA-Z])', r'\1*(', normalized)  # Avoid sin(, cos(

    # Check for equation (contains =)
    if '=' in normalized and '==' not in normalized:
        parts = normalized.split('=')
        if len(parts) == 2:
            try:
                lhs = sympy.sympify(parts[0].strip())
                rhs = sympy.sympify(parts[1].strip())
                return sympy.Eq(lhs, rhs)
            except Exception:
                pass

    # Try to parse as expression
    try:
        expr = sympy.sympify(normalized)
        # Only wrap in Eq(expr, 0) if it's a multi-term expression (not a single symbol)
        # Single symbols like 'x' should stay as Symbol, not become Eq(x, 0)
        if (expr.free_symbols and
            not isinstance(expr, (int, float, sympy.Number, sympy.Symbol)) and
            len(str(expr)) > 2):  # Multi-term expression
            return sympy.Eq(expr, 0)
        return expr
    except Exception:
        pass

    # Return original if parsing fails
    return value


# =============================================================================
# Safe SymPy evaluation
# =============================================================================

def _safe_eval_sympy(script: str, inputs: dict[str, Any]) -> Optional[Any]:
    """AST-based safe SymPy evaluator."""
    try:
        import sympy
    except ImportError:
        logger.warning("SymPy not installed, skipping sympy layer")
        return None

    try:
        tree = ast.parse(script, mode='eval')

        # Validate all function calls and attribute access
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in SYMPY_ALLOWED:
                        logger.warning("Disallowed sympy function: %s", node.func.id)
                        return None
                elif isinstance(node.func, ast.Attribute):
                    # Only allow whitelisted method names
                    if node.func.attr not in SYMPY_ALLOWED_METHODS:
                        logger.warning("Disallowed sympy method: %s", node.func.attr)
                        return None
            # Block dangerous attribute access (e.g., __class__, __bases__)
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    logger.warning("Disallowed private attribute: %s", node.attr)
                    return None

        # Build namespace with allowed SymPy functions
        namespace = {
            name: getattr(sympy, name)
            for name in SYMPY_ALLOWED
            if hasattr(sympy, name)
        }

        # Parse string inputs into sympy expressions
        parsed_inputs = {}
        for key, val in inputs.items():
            if isinstance(val, str):
                parsed_inputs[key] = parse_to_sympy(val, sympy)
            else:
                parsed_inputs[key] = val
        namespace.update(parsed_inputs)

        # Add common symbols (all single letters plus common multi-letter)
        for letter in "abcdefghijklmnopqrstuvwxyz":
            namespace[letter] = sympy.Symbol(letter)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            namespace[letter] = sympy.Symbol(letter)
        # Common multi-letter symbols
        for name in ["pi", "theta", "alpha", "beta", "gamma", "delta", "epsilon"]:
            if hasattr(sympy, name):
                namespace[name] = getattr(sympy, name)
            else:
                namespace[name] = sympy.Symbol(name)

        # Execute in restricted namespace
        result = eval(compile(tree, '<dsl>', 'eval'), {"__builtins__": {}}, namespace)
        return result

    except Exception as e:
        logger.debug("SymPy eval failed: %s", e)
        return None


def try_execute_dsl_sympy(script: str, inputs: dict[str, Any]) -> Optional[Any]:
    """Execute SymPy symbolic algebra DSL script."""
    return _safe_eval_sympy(script, inputs)
