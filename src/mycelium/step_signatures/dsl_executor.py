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
    # Roots and exponents
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    # Rounding
    "floor": math.floor,
    "ceil": math.ceil,
    # Combinatorics
    "factorial": math.factorial,
    "gcd": math.gcd,
    # Hyperbolic (occasionally needed)
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
}

# Allowed constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


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


def try_execute_dsl_math(script: str, inputs: dict[str, float]) -> Optional[float]:
    """Execute basic math DSL script."""
    return _safe_eval_math(script, inputs)


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
        namespace.update(inputs)

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


# Register built-in custom operators
register_operator("extract_coefficient", _extract_coefficient)
register_operator("apply_quadratic_formula", _apply_quadratic_formula)
register_operator("complete_square", _complete_square)
register_operator("solve_linear", _solve_linear)
register_operator("evaluate_polynomial", _evaluate_polynomial)


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


# LLM-based script rewriter for DSL execution
LLM_SCRIPT_REWRITE_PROMPT = """Rewrite this DSL script to use ONLY the available context variable names.

Original script: {script}

Available context variables and their values:
{context}

Task: Rewrite the script replacing semantic variable names with the appropriate context variable names.
The rewritten script must be a valid Python expression that can be evaluated.

Example:
Original: "base * height / 2"
Context: {{"step_1": 10, "step_2": 5}}
Rewritten: "step_1 * step_2 / 2"

Example:
Original: "solve(equation, x)"
Context: {{"step_1": "x**2 - 4", "step_2": "x"}}
Rewritten: "solve(step_1, x)"

Return ONLY the rewritten script, nothing else:"""


async def llm_rewrite_script(
    dsl_spec: DSLSpec,
    context: dict[str, Any],
    client,  # GroqClient
) -> Optional[str]:
    """Use LLM to rewrite DSL script using actual context variable names.

    Args:
        dsl_spec: The DSL specification with script to rewrite
        context: Runtime context with available values
        client: GroqClient instance for LLM calls

    Returns:
        Rewritten script string, or None if rewriting failed
    """
    if not context:
        return None

    # Format context for prompt (truncate large values)
    context_str = json.dumps({
        k: (v if not isinstance(v, str) or len(str(v)) < 100 else str(v)[:100] + "...")
        for k, v in context.items()
    }, indent=2)

    prompt = LLM_SCRIPT_REWRITE_PROMPT.format(
        script=dsl_spec.script[:300],
        context=context_str
    )

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
        logger.warning("[dsl] LLM script rewrite failed: %s", e)
        return None


async def execute_dsl_with_llm_matching(
    dsl_json: str,
    inputs: dict[str, Any],
    client,  # GroqClient for LLM script rewriting
    min_confidence: float = 0.7,
    llm_threshold: float = 0.3,
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

    Returns:
        (result, success, confidence) tuple
    """
    spec = DSLSpec.from_json(dsl_json)
    if not spec:
        return None, False, 0.0

    confidence = spec.compute_confidence(inputs)

    # If confidence is low, try LLM script rewriting
    if confidence < llm_threshold and client:
        logger.info("[dsl] Low confidence (%.2f), trying LLM script rewriting", confidence)
        try:
            rewritten_script = await llm_rewrite_script(spec, inputs, client)
            if rewritten_script:
                # Create new DSLSpec with rewritten script and no params
                # (since the script now uses actual context variable names)
                rewritten_spec = DSLSpec(
                    layer=spec.layer,
                    script=rewritten_script,
                    params=[],  # No params - script uses context var names directly
                    aliases={},
                    output_type=spec.output_type,
                    fallback=spec.fallback,
                )
                # Execute with all inputs (script uses context var names)
                result, success = try_execute_dsl(rewritten_spec, inputs)
                logger.info("[dsl] Execution after rewrite: success=%s result=%s script=%s",
                           success, result, rewritten_script[:50])
                if success:
                    return result, True, 1.0  # High confidence since LLM rewrote
        except Exception as e:
            logger.debug("[dsl] LLM script rewriting error: %s", e)

    # Fall back to standard execution if confidence is sufficient
    if confidence >= min_confidence:
        result, success = try_execute_dsl(spec, inputs)
        return result, success, confidence

    return None, False, confidence
