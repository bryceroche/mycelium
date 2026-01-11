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
    "fallback": "guidance|formula|llm"
}

Aliases enable fuzzy matching between param names and context keys.
The LLM generates aliases once at signature creation time, then matching
is fast dict lookups at runtime.
"""

import ast
import json
import logging
import math
import operator
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DSLLayer(Enum):
    """Execution layer for DSL scripts."""
    MATH = "math"
    SYMPY = "sympy"
    CUSTOM = "custom"
    GUIDANCE = "guidance"  # Fallback-only, no execution
    NONE = "none"  # No DSL execution, use LLM fallback


@dataclass
class DSLSpec:
    """Specification for a DSL script."""
    layer: DSLLayer
    script: str
    params: list[str]
    aliases: dict[str, list[str]] = field(default_factory=dict)
    output_type: str = "numeric"
    fallback: str = "guidance"

    @classmethod
    def from_json(cls, json_str: str) -> Optional["DSLSpec"]:
        """Parse JSON string into DSLSpec."""
        if not json_str:
            return None
        try:
            d = json.loads(json_str)
            layer = DSLLayer(d.get("type", "math"))
            # Handle params as list or dict (extract keys if dict)
            raw_params = d.get("params", [])
            if isinstance(raw_params, dict):
                params = list(raw_params.keys()) if raw_params else []
            elif isinstance(raw_params, list):
                params = raw_params
            else:
                params = []
            return cls(
                layer=layer,
                script=d.get("script", ""),
                params=params,
                aliases=d.get("aliases", {}),
                output_type=d.get("output_type", "numeric"),
                fallback=d.get("fallback", "guidance"),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse DSL spec: %s", e)
            return None

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "type": self.layer.value,
            "script": self.script,
            "params": self.params,
            "output_type": self.output_type,
            "fallback": self.fallback,
        }
        if self.aliases:
            data["aliases"] = self.aliases
        return json.dumps(data)

    def match_param(self, param: str, context_keys: list[str]) -> Optional[str]:
        """Find matching context key for a param using aliases.

        Matching priority:
        1. Exact match (param == key)
        2. Param is substring of key (e.g., 'value' in 'base_value')
        3. Key is substring of param
        4. Alias exact match
        5. Alias substring match

        Returns the matching context key, or None if no match.
        """
        param_lower = param.lower()
        aliases = [a.lower() for a in self.aliases.get(param, [])]

        for key in context_keys:
            key_lower = key.lower()

            # Exact match
            if param_lower == key_lower:
                return key

            # Param in key (e.g., 'percentage' in 'step_1_percentage')
            if param_lower in key_lower:
                return key

            # Key in param (e.g., 'pct' in 'percentage')
            if key_lower in param_lower:
                return key

            # Alias matches
            for alias in aliases:
                if alias == key_lower or alias in key_lower or key_lower in alias:
                    return key

        return None

    def map_inputs(self, context: dict[str, Any]) -> dict[str, Any]:
        """Map context values to param names using alias matching.

        Matching strategy:
        1. First try name-based matching (exact, substring, alias)
        2. If no matches found, fall back to positional matching

        Returns dict with param names as keys and context values as values.
        """
        result = {}
        context_keys = list(context.keys())

        # First pass: name-based matching
        for param in self.params:
            matched_key = self.match_param(param, context_keys)
            if matched_key:
                result[param] = context[matched_key]
                context_keys.remove(matched_key)  # Don't reuse

        # Fallback: positional matching if name matching found nothing
        # This handles cases like DSL params ["a", "b"] with context {"step_1": 10, "step_2": 20}
        if not result and self.params and context:
            # Sort context keys to ensure consistent ordering
            sorted_keys = sorted(context.keys())
            for i, param in enumerate(self.params):
                if i < len(sorted_keys):
                    result[param] = context[sorted_keys[i]]

        return result

    def compute_confidence(self, context: dict[str, Any]) -> float:
        """Compute confidence score for executing DSL with given context.

        Score based on:
        - Percentage of required params found
        - Quality of matches (exact vs fuzzy vs positional)
        - Type compatibility (numeric for math DSL)

        Returns:
            Float between 0.0 and 1.0. Higher = more confident.
        """
        if not self.params:
            return 1.0  # No params required

        context_keys = list(context.keys())
        matched_count = 0
        fuzzy_matches = 0
        type_mismatches = 0
        positional_fallback = False

        # Try name-based matching first
        for param in self.params:
            matched_key = self.match_param(param, context_keys)
            if matched_key:
                matched_count += 1
                context_keys.remove(matched_key)

                # Check if fuzzy match (not exact)
                if param.lower() != matched_key.lower():
                    fuzzy_matches += 1

                # Check type compatibility for math DSL
                if self.layer == DSLLayer.MATH:
                    value = context[matched_key]
                    if not isinstance(value, (int, float)):
                        try:
                            float(str(value))
                        except (ValueError, TypeError):
                            type_mismatches += 1

        # If no name matches, check positional fallback viability
        if matched_count == 0 and len(context) >= len(self.params):
            positional_fallback = True
            matched_count = len(self.params)
            # All positional matches count as fuzzy
            fuzzy_matches = len(self.params)

            # Check type compatibility for positional values
            if self.layer == DSLLayer.MATH:
                sorted_keys = sorted(context.keys())
                for i in range(len(self.params)):
                    if i < len(sorted_keys):
                        value = context[sorted_keys[i]]
                        if not isinstance(value, (int, float)):
                            try:
                                float(str(value))
                            except (ValueError, TypeError):
                                type_mismatches += 1

        # Base score: percentage of params found
        score = matched_count / len(self.params) if self.params else 0.0

        # Penalty for fuzzy matches (20% per fuzzy match)
        score *= (0.8 ** fuzzy_matches)

        # Penalty for type mismatches in math DSL (50% per mismatch)
        if self.layer == DSLLayer.MATH:
            score *= (0.5 ** type_mismatches)

        return score


# =============================================================================
# Layer 0: Math (extends existing safe eval)
# =============================================================================

# Allowed binary operators
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed functions
_FUNCTIONS: dict[str, Callable] = {
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
    # Integer operations
    "int": lambda x: int(x),
    "float": lambda x: float(x),
    "isqrt": lambda n: int(math.sqrt(int(n))),  # Integer square root
}

# Allowed constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _extract_numeric_value(value: Any) -> Optional[float]:
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


def _prepare_math_inputs(inputs: dict[str, Any]) -> dict[str, float]:
    """Convert input values to floats for math DSL execution.

    Extracts numeric values from text context values.
    """
    result = {}
    for key, value in inputs.items():
        extracted = _extract_numeric_value(value)
        if extracted is not None:
            result[key] = extracted
    return result


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
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            raise KeyError(f"Unknown variable: {name}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _BINOPS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARYOPS:
                raise ValueError(f"Unsupported unary: {op_type.__name__}")
            return _UNARYOPS[op_type](_eval_node(node.operand))

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in _FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [_eval_node(arg) for arg in node.args]
            return _FUNCTIONS[func_name](*args)

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
    """
    # Prepare inputs: extract numeric values from text
    prepared = _prepare_math_inputs(inputs)
    if not prepared and inputs:
        # No numeric values could be extracted
        logger.debug("Math DSL: no numeric values extracted from inputs: %s", list(inputs.keys()))
        return None
    return _safe_eval_math(script, prepared)


# =============================================================================
# Layer 1: SymPy Symbolic
# =============================================================================

# Whitelisted SymPy functions
_SYMPY_ALLOWED = {
    # Core
    "Symbol", "symbols", "Eq", "sympify",
    # Algebra
    "solve", "simplify", "expand", "factor", "collect", "cancel",
    # Calculus
    "diff", "integrate", "limit", "series",
    # Functions
    "sqrt", "Abs", "sin", "cos", "tan", "log", "exp",
    "asin", "acos", "atan", "sinh", "cosh", "tanh",
    # Number theory
    "gcd", "lcm", "factorial",
    # Polynomials
    "Poly", "degree", "roots",
}

# Whitelisted method names for SymPy objects
_SYMPY_ALLOWED_METHODS = {
    "subs", "evalf", "simplify", "expand", "factor",
    "diff", "integrate", "limit", "series",
    "coeff", "as_coeff_mul", "as_coefficients_dict",
}


def _parse_to_sympy(value: str, sympy) -> Any:
    """Parse a string into a sympy expression.

    Handles:
    - "x^2 - 4 = 0" → Eq(x**2 - 4, 0)
    - "x^2 + 2x - 3" → x**2 + 2*x - 3
    - "42" → 42 (number)
    - Plain text → original string (unchanged)
    """
    import re

    if not value or not value.strip():
        return value

    cleaned = value.strip()

    # Try to parse as number first
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Normalize notation: ^ to **, implicit multiplication
    normalized = cleaned.replace('^', '**')
    # Add implicit multiplication: 2x → 2*x, x( → x*(
    normalized = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', normalized)
    normalized = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', normalized)
    normalized = re.sub(r'(\d)\(', r'\1*(', normalized)
    normalized = re.sub(r'\)(\d)', r')*\1', normalized)
    normalized = re.sub(r'\)\(', r')*(', normalized)
    normalized = re.sub(r'([a-zA-Z])\(', r'\1*(', normalized)

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
        return sympy.sympify(normalized)
    except Exception:
        pass

    # Return original if parsing fails
    return value


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
                    if node.func.id not in _SYMPY_ALLOWED:
                        logger.warning("Disallowed sympy function: %s", node.func.id)
                        return None
                elif isinstance(node.func, ast.Attribute):
                    # Only allow whitelisted method names
                    if node.func.attr not in _SYMPY_ALLOWED_METHODS:
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
            for name in _SYMPY_ALLOWED
            if hasattr(sympy, name)
        }

        # Parse string inputs into sympy expressions
        parsed_inputs = {}
        for key, val in inputs.items():
            if isinstance(val, str):
                parsed_inputs[key] = _parse_to_sympy(val, sympy)
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


# =============================================================================
# Layer 2: Custom Operators
# =============================================================================

# Registry for custom operators
_CUSTOM_OPERATORS: dict[str, Callable] = {}


def register_operator(name: str, func: Callable) -> None:
    """Register a custom DSL operator."""
    _CUSTOM_OPERATORS[name] = func


def _extract_coefficient(expr: str, var: str = "x") -> Optional[float]:
    """Extract coefficient of variable from expression."""
    try:
        import sympy
        x = sympy.Symbol(var)
        parsed = sympy.sympify(expr)
        coeff = parsed.coeff(x)
        return float(coeff) if coeff.is_number else None
    except Exception:
        return None


def _apply_quadratic_formula(a: float, b: float, c: float) -> Optional[tuple[float, float]]:
    """Apply quadratic formula to solve ax^2 + bx + c = 0."""
    if a == 0:
        return None  # Not quadratic
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # No real roots
    sqrt_d = math.sqrt(discriminant)
    return ((-b + sqrt_d) / (2*a), (-b - sqrt_d) / (2*a))


def _complete_square(a: float, b: float, c: float) -> Optional[str]:
    """Complete the square for ax^2 + bx + c.

    Returns string in form: a(x + h)^2 + k
    """
    if a == 0:
        return None  # Not quadratic
    h = -b / (2*a)
    k = c - b**2 / (4*a)
    return f"{a}*(x + {h})^2 + {k}"


def _solve_linear(a: float, b: float) -> Optional[float]:
    """Solve ax + b = 0 for x."""
    if a == 0:
        return None
    return -b / a


def _evaluate_polynomial(coeffs: list[float], x: float) -> float:
    """Evaluate polynomial with coefficients [a_n, ..., a_1, a_0] at x."""
    result = 0.0
    for coeff in coeffs:
        result = result * x + coeff
    return result


def _euclidean_gcd(a: int, b: int) -> int:
    """Euclidean algorithm for GCD."""
    a, b = int(abs(a)), int(abs(b))
    while b:
        a, b = b, a % b
    return a


def _modinv(a: int, m: int) -> Optional[int]:
    """Modular multiplicative inverse of a mod m using extended Euclidean algorithm."""
    a, m = int(a), int(m)
    if m == 1:
        return 0
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


def _divisors(n: int) -> list[int]:
    """Find all divisors of n."""
    n = int(abs(n))
    if n == 0:
        return []
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _prime_factors(n: int) -> list[int]:
    """Find prime factorization of n."""
    n = int(abs(n))
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _is_prime(n: int) -> bool:
    """Check if n is prime."""
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _mod_pow(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod mod."""
    base, exp, mod = int(base), int(exp), int(mod)
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


def _binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    n, k = int(n), int(k)
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _permutations(n: int, r: int) -> int:
    """Permutations P(n, r) = n! / (n-r)!"""
    n, r = int(n), int(r)
    if r < 0 or r > n:
        return 0
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    return result


def _combinations(n: int, r: int) -> int:
    """Combinations C(n, r) = n! / (r! * (n-r)!)"""
    return _binomial(n, r)


def _day_of_week(year: int, month: int, day: int) -> int:
    """Zeller's formula: 0=Saturday, 1=Sunday, ..., 6=Friday."""
    year, month, day = int(year), int(month), int(day)
    if month < 3:
        month += 12
        year -= 1
    k = year % 100
    j = year // 100
    h = (day + (13 * (month + 1)) // 5 + k + k // 4 + j // 4 - 2 * j) % 7
    return h


def _triangular_number(n: int) -> int:
    """nth triangular number: 1 + 2 + ... + n = n(n+1)/2."""
    n = int(n)
    return n * (n + 1) // 2


def _fibonacci(n: int) -> int:
    """nth Fibonacci number (0-indexed: F(0)=0, F(1)=1)."""
    n = int(n)
    if n < 0:
        return 0
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# Register built-in custom operators
register_operator("extract_coefficient", _extract_coefficient)
register_operator("apply_quadratic_formula", _apply_quadratic_formula)
register_operator("complete_square", _complete_square)
register_operator("solve_linear", _solve_linear)
register_operator("evaluate_polynomial", _evaluate_polynomial)
# Number theory operators
register_operator("euclidean_gcd", _euclidean_gcd)
register_operator("modinv", _modinv)
register_operator("divisors", _divisors)
register_operator("prime_factors", _prime_factors)
register_operator("is_prime", _is_prime)
register_operator("mod_pow", _mod_pow)
# Combinatorics operators
register_operator("binomial", _binomial)
register_operator("permutations", _permutations)
register_operator("combinations", _combinations)
register_operator("C", _combinations)  # Alias
register_operator("P", _permutations)  # Alias
# Misc operators
register_operator("day_of_week", _day_of_week)
register_operator("triangular_number", _triangular_number)
register_operator("fibonacci", _fibonacci)


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
) -> tuple[Optional[Any], bool]:
    """Execute a DSL script with given inputs.

    Args:
        dsl_spec: The DSL specification
        inputs: Input values (from context/previous steps) - keys may not match param names exactly
        timeout_sec: Maximum execution time

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

    # GUIDANCE layer: immediately fall back to LLM guidance (no DSL execution)
    if dsl_spec.layer == DSLLayer.GUIDANCE:
        return None, False

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
                return result, True
            return None, False

    except TimeoutError:
        logger.warning("DSL execution timed out")
        return None, False
    except Exception as e:
        logger.debug("DSL execution failed: %s", e)
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

    Args:
        dsl_spec: The DSL specification with script to rewrite
        context: Runtime context with available values
        step_descriptions: Dict mapping step_id -> task description

    Returns:
        (rewritten_script, confidence) or (None, 0.0) if no good mapping
    """
    if not dsl_spec.params or not context or not step_descriptions:
        return None, 0.0

    # Build param -> context_key mapping
    param_mapping: dict[str, str] = {}
    total_score = 0.0

    for param in dsl_spec.params:
        best_match = None
        best_score = 0.0
        param_lower = param.lower().replace("_", " ")

        for ctx_key, ctx_value in context.items():
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
            total_score += best_score
        else:
            # Can't map this param semantically
            logger.debug("[dsl_semantic] No semantic match for param '%s'", param)
            return None, 0.0

    if len(param_mapping) != len(dsl_spec.params):
        return None, 0.0

    # Rewrite script by replacing params with context keys
    rewritten = dsl_spec.script
    for param, ctx_key in param_mapping.items():
        # Replace param name with context key
        rewritten = re.sub(rf'\b{re.escape(param)}\b', ctx_key, rewritten)

    avg_confidence = total_score / len(dsl_spec.params) if dsl_spec.params else 0.0

    logger.info(
        "[dsl_semantic] Semantic rewrite: '%s' -> '%s' (confidence=%.2f, mappings=%s)",
        dsl_spec.script[:50], rewritten[:50], avg_confidence, param_mapping
    )

    return rewritten, avg_confidence


# LLM-based script rewriter for DSL execution (DEPRECATED - use semantic_rewrite_script)
LLM_SCRIPT_REWRITE_PROMPT = """Rewrite this DSL script to use ONLY the available context variable names.

Original script: {script}
DSL parameters and their meanings: {params}

Available context variables:
{context}

Task: Match each DSL parameter to the most semantically appropriate context variable, then rewrite the script.
The rewritten script must be a valid Python expression using ONLY the context variable names.

Example:
Original: "base * height / 2"
Params: ["base", "height"]
Context:
  step_1 (Calculate the width of the rectangle): 10
  step_2 (Calculate the height of the rectangle): 5
Rewritten: step_1 * step_2 / 2

Example:
Original: "price * quantity"
Params: ["price", "quantity"]
Context:
  step_1 (Find the unit price): 25
  step_2 (Count the number of items): 4
Rewritten: step_1 * step_2

Return ONLY the rewritten script, nothing else:"""


async def llm_rewrite_script(
    dsl_spec: DSLSpec,
    context: dict[str, Any],
    client,  # GroqClient
    step_descriptions: Optional[dict[str, str]] = None,
    current_step_task: Optional[str] = None,
) -> Optional[str]:
    """Use LLM to rewrite DSL script using actual context variable names.

    Args:
        dsl_spec: The DSL specification with script to rewrite
        context: Runtime context with available values
        client: GroqClient instance for LLM calls
        step_descriptions: Optional dict mapping step_id -> task description
        current_step_task: Optional current step task for additional context

    Returns:
        Rewritten script string, or original script if rewriting not possible
    """
    if not context:
        # Return original script - let positional fallback try
        logger.debug("[dsl] No context for rewrite, returning original script")
        return dsl_spec.script

    # Format context with descriptions if available
    context_lines = []
    for k, v in context.items():
        val_str = str(v) if not isinstance(v, str) or len(str(v)) < 100 else str(v)[:100] + "..."
        if step_descriptions and k in step_descriptions:
            desc = step_descriptions[k][:80]  # Truncate long descriptions
            context_lines.append(f"  {k} ({desc}): {val_str}")
        else:
            context_lines.append(f"  {k}: {val_str}")
    context_str = "\n".join(context_lines)

    # Format params with any aliases
    params_info = dsl_spec.params if dsl_spec.params else ["(no explicit params)"]

    # Include current step task for additional context
    task_context = f"\nCurrent step task: {current_step_task[:200]}\n" if current_step_task else ""

    prompt = LLM_SCRIPT_REWRITE_PROMPT.format(
        script=dsl_spec.script[:300],
        params=params_info,
        context=context_str
    ) + task_context

    try:
        messages = [{"role": "user", "content": prompt}]
        response = await client.generate(messages, max_tokens=200, temperature=0.0)
        # Clean up response
        rewritten = response.strip()
        # Handle markdown code blocks
        if rewritten.startswith("```"):
            lines = rewritten.split("\n")
            rewritten = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            rewritten = rewritten.strip()
        # Remove language hints
        if rewritten.startswith("python"):
            rewritten = rewritten[6:].strip()
        logger.info("[dsl] LLM rewrote script: '%s' -> '%s'", dsl_spec.script[:50], rewritten[:50])
        return rewritten
    except Exception as e:
        logger.warning("[dsl] LLM script rewrite failed: %s, using original", e)
        return dsl_spec.script  # Return original - let positional fallback try


async def execute_dsl_with_llm_matching(
    dsl_json: str,
    inputs: dict[str, Any],
    client,  # GroqClient for LLM script rewriting
    min_confidence: float = 0.7,
    llm_threshold: float = 0.3,
    step_descriptions: Optional[dict[str, str]] = None,
    step_task: Optional[str] = None,
) -> tuple[Optional[Any], bool, float]:
    """Execute DSL with LLM-based script rewriting fallback.

    If heuristic confidence is below llm_threshold, use LLM to rewrite the script
    using actual context variable names.

    Args:
        dsl_json: JSON-encoded DSL specification
        inputs: Input values from context
        client: GroqClient for LLM calls when needed
        min_confidence: Minimum confidence to execute (default 0.7)
        llm_threshold: Below this confidence, try LLM script rewriting (default 0.3)
        step_descriptions: Optional dict mapping step_id -> task description for better LLM matching
        step_task: Optional current step task for additional context

    Returns:
        (result, success, confidence) tuple
    """
    spec = DSLSpec.from_json(dsl_json)
    if not spec:
        return None, False, 0.0

    confidence = spec.compute_confidence(inputs)

    # PRIORITY 1: If confidence is low, try SEMANTIC matching (deterministic, reliable)
    if confidence < llm_threshold and step_descriptions:
        rewritten_script, semantic_confidence = semantic_rewrite_script(spec, inputs, step_descriptions)
        if rewritten_script and semantic_confidence >= 0.5:
            # Create new DSLSpec with semantically rewritten script
            rewritten_spec = DSLSpec(
                layer=spec.layer,
                script=rewritten_script,
                params=[],  # Script now uses context var names directly
                aliases={},
                output_type=spec.output_type,
                fallback=spec.fallback,
            )
            result, success = try_execute_dsl(rewritten_spec, inputs)
            if success:
                logger.info("[dsl] Semantic rewrite succeeded: result=%s confidence=%.2f", result, semantic_confidence)
                return result, True, semantic_confidence
            else:
                logger.debug("[dsl] Semantic rewrite produced invalid script")

    # PRIORITY 2: Standard execution if confidence is sufficient
    if confidence >= min_confidence:
        result, success = try_execute_dsl(spec, inputs)
        return result, success, confidence

    # PRIORITY 3: LLM rewriting as last resort (DISABLED - produces garbage)
    # The LLM rewriting was causing the 45% accuracy regression by producing
    # incorrect parameter mappings like "area_ENG / area_ABC" -> "step_1 / step_1"
    # Uncomment below to re-enable if semantic matching isn't sufficient:
    #
    # if confidence < llm_threshold and client:
    #     logger.info("[dsl] Low confidence (%.2f), trying LLM script rewriting", confidence)
    #     try:
    #         rewritten_script = await llm_rewrite_script(spec, inputs, client, step_descriptions, step_task)
    #         ... (rest of LLM logic)

    return None, False, confidence
