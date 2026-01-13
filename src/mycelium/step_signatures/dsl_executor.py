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

# =============================================================================
# NEGATIVE PARAM MAPPING CACHE
# =============================================================================
# In-memory cache of param mappings that produced wrong results.
# Keyed by (dsl_type, param_name) -> list of context description embeddings that failed.
# This allows rejecting mappings similar to known failures without GPU training.

_negative_param_cache: dict[tuple[str, str], list[Any]] = {}
_NEGATIVE_PARAM_SIMILARITY_THRESHOLD = 0.85  # Reject if very similar to known failure

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
    import numpy as np
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


class ValueType(Enum):
    """Classification of a parameter value."""
    NUMBER = "number"
    EXPRESSION = "expression"
    UNCERTAIN = "uncertain"


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


def _classify_value_heuristic(value: Any) -> ValueType:
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

    # Text indicators - definitely an expression/text (check early)
    text_indicators = ['the ', 'is ', 'are ', 'if ', 'when ', 'let ', 'find ', 'solve ']
    if any(ind in s_lower for ind in text_indicators):
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


def _classify_value_embedding(value: str) -> tuple[ValueType, float]:
    """Embedding-based classification. Only called when heuristics uncertain."""
    import numpy as np

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


# =============================================================================
# STEP-TYPE ALIGNMENT VALIDATION
# =============================================================================
# Validates that the signature's step_type semantically matches the actual task.
# This catches cases where signature similarity is high but operation type differs.

# Step type descriptions - maps step_type to semantic description
STEP_TYPE_DESCRIPTIONS = {
    # Arithmetic operations
    "compute_sum": "add numbers together to get a total",
    "compute_product": "multiply numbers together",
    "compute_difference": "subtract one number from another",
    "compute_quotient": "divide one number by another",
    "compute_remainder": "find the remainder after division",
    "compute_power": "raise a number to an exponent or power",
    "compute_sqrt": "calculate the square root of a number",
    "compute_square": "square a number (multiply by itself)",
    "compute_factorial": "calculate factorial (n!)",

    # Number theory
    "compute_gcd": "find the greatest common divisor of numbers",
    "compute_lcm": "find the least common multiple of numbers",
    "check_divisibility": "check if one number divides another evenly",
    "check_prime_number": "determine if a number is prime",

    # Geometry
    "compute_area": "calculate the area of a shape",
    "area_triangle": "calculate the area of a triangle",
    "area_rectangle": "calculate the area of a rectangle",
    "area_circle": "calculate the area of a circle",
    "compute_perimeter": "calculate the perimeter of a shape",
    "compute_distance": "calculate distance between points",
    "compute_length": "calculate the length of a segment",
    "compute_angle": "calculate an angle measurement",
    "compute_radius": "calculate the radius of a circle",

    # Algebra
    "simplify_expression": "simplify an algebraic expression",
    "solve_equation": "solve an equation for a variable",
    "substitute_values": "substitute known values into an expression",
    "compute_derivative": "calculate the derivative of a function",

    # Statistics/combinatorics
    "compute_average": "calculate the mean or average",
    "compute_probability": "calculate a probability",
    "compute_percentage": "calculate a percentage",
    "count_items": "count the number of items or possibilities",

    # Sequences
    "arithmetic_sequence": "work with arithmetic sequences",
    "arith_seq_diff": "find common difference of arithmetic sequence",
    "arith_sum": "calculate sum of arithmetic sequence",
    "geometric_sequence": "work with geometric sequences",
}

# Cache for step_type embeddings
_step_type_embedding_cache: dict[str, Any] = {}

# Threshold for step-type alignment (lowered from 0.30 to let system breathe)
_STEP_TYPE_ALIGNMENT_THRESHOLD = 0.20


def _get_step_type_description(step_type: str) -> str:
    """Get description for a step_type, inferring from name if not in mapping."""
    if step_type in STEP_TYPE_DESCRIPTIONS:
        return STEP_TYPE_DESCRIPTIONS[step_type]

    # Infer from step_type name: convert snake_case to sentence
    # e.g., "compute_logarithm" -> "compute logarithm"
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
    import numpy as np

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
    import numpy as np

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


class DSLLayer(Enum):
    """Execution layer for DSL scripts."""
    MATH = "math"
    SYMPY = "sympy"
    CUSTOM = "custom"
    DECOMPOSE = "decompose"  # Needs decomposition into atomic steps
    ROUTER = "router"  # Routing layer - delegates to children
    NONE = "none"  # No DSL execution, use LLM fallback


@dataclass
class DSLSpec:
    """Specification for a DSL script."""
    layer: DSLLayer
    script: str
    params: list[str]
    aliases: dict[str, list[str]] = field(default_factory=dict)
    param_types: dict[str, str] = field(default_factory=dict)  # param -> "numeric"|"symbolic"|"any"
    param_roles: dict[str, str] = field(default_factory=dict)  # param -> semantic role description
    output_type: str = "numeric"
    fallback: str = "decompose"
    purpose: str = ""  # Human-readable description of what this DSL does
    _purpose_embedding: Optional[Any] = field(default=None, repr=False)  # Cached embedding
    _param_role_embeddings: dict[str, Any] = field(default_factory=dict, repr=False)  # Cached param embeddings

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

            # Infer param types based on DSL layer if not specified
            param_types = d.get("param_types", {})
            if not param_types and layer == DSLLayer.MATH:
                # Math DSLs expect numeric inputs by default
                param_types = {p: "numeric" for p in params}

            # Get param roles (semantic descriptions)
            param_roles = d.get("param_roles", {})

            return cls(
                layer=layer,
                script=d.get("script", ""),
                params=params,
                aliases=d.get("aliases", {}),
                param_types=param_types,
                param_roles=param_roles,
                output_type=d.get("output_type", "numeric"),
                fallback=d.get("fallback", "decompose"),
                purpose=d.get("purpose", ""),
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
        if self.purpose:
            data["purpose"] = self.purpose
        return json.dumps(data)

    def get_purpose_embedding(self):
        """Get or compute the purpose embedding (lazy).

        If no purpose is set, infers one from the script.
        """
        if self._purpose_embedding is not None:
            return self._purpose_embedding

        # Get purpose text (infer from script if not set)
        purpose_text = self.purpose or self._infer_purpose_from_script()
        if not purpose_text:
            return None

        # Compute embedding
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        self._purpose_embedding = embedder.embed(purpose_text)
        return self._purpose_embedding

    def _infer_purpose_from_script(self) -> str:
        """Infer a purpose description from the DSL script."""
        script_lower = self.script.lower()

        # Map common scripts to purposes
        if 'sqrt' in script_lower:
            return "compute square root of a number"
        if '**' in script_lower or 'pow' in script_lower:
            return "compute power or exponentiation"
        if 'a + b' in script_lower:
            return "add two numbers together"
        if 'a - b' in script_lower:
            return "subtract one number from another"
        if 'a * b' in script_lower:
            return "multiply two numbers"
        if 'a / b' in script_lower:
            return "divide one number by another"
        if 'gcd' in script_lower:
            return "find greatest common divisor"
        if 'lcm' in script_lower:
            return "find least common multiple"
        if 'factor' in script_lower:
            return "factor an expression"
        if 'expand' in script_lower:
            return "expand an algebraic expression"
        if 'simplify' in script_lower:
            return "simplify an expression"
        if 'solve' in script_lower:
            return "solve an equation for a variable"

        # Default: use the script itself as a rough purpose
        return f"execute mathematical operation: {self.script[:50]}"

    def get_param_role_embedding(self, param: str):
        """Get or compute embedding for a parameter's semantic role.

        If no role is defined, infers one from the param name and script context.
        """
        if param in self._param_role_embeddings:
            return self._param_role_embeddings[param]

        # Get role text (explicit or inferred)
        role_text = self.param_roles.get(param) or self._infer_param_role(param)
        if not role_text:
            return None

        # Compute embedding
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        embedding = embedder.embed(role_text)
        self._param_role_embeddings[param] = embedding
        return embedding

    def _infer_param_role(self, param: str) -> str:
        """Infer semantic role description for a parameter from its name and script context."""
        param_lower = param.lower()
        script_lower = self.script.lower()

        # Power/exponent DSLs
        if '**' in script_lower or 'pow' in script_lower:
            if param_lower in ('exponent', 'exp', 'power', 'n'):
                return "the exponent or power to raise the base to"
            if param_lower in ('base', 'b', 'a'):
                return "the base number to be raised to a power"

        # Distance/geometry DSLs
        if 'sqrt' in script_lower and ('**2' in script_lower or 'pow' in script_lower):
            if param_lower in ('x1', 'x2', 'y1', 'y2'):
                return "a coordinate value for distance calculation"
            if param_lower in ('a', 'b'):
                return "a length or distance component"

        # Combinatorics
        if 'factorial' in script_lower or 'comb' in script_lower or 'perm' in script_lower:
            if param_lower == 'n':
                return "total number of items to choose from"
            if param_lower in ('r', 'k'):
                return "number of items to select or arrange"

        # Division
        if '/' in script_lower:
            if param_lower in ('numerator', 'dividend', 'a'):
                return "the value being divided"
            if param_lower in ('denominator', 'divisor', 'b'):
                return "the value to divide by"

        # Generic arithmetic
        if param_lower in ('a', 'b'):
            if '+' in script_lower:
                return "a number to add"
            if '-' in script_lower:
                return "a number for subtraction"
            if '*' in script_lower:
                return "a number to multiply"

        # Default based on param name patterns
        role_patterns = {
            'area': "an area measurement",
            'length': "a length or distance measurement",
            'height': "a height measurement",
            'width': "a width measurement",
            'radius': "a radius measurement",
            'angle': "an angle measurement in degrees or radians",
            'count': "a count or quantity of items",
            'rate': "a rate or ratio value",
            'price': "a price or monetary value",
            'time': "a time duration or timestamp",
            'prob': "a probability value between 0 and 1",
        }

        for pattern, role in role_patterns.items():
            if pattern in param_lower:
                return role

        # Fallback: generic numeric input
        return f"a numeric value for the {param} parameter"

    def get_dsl_type(self) -> str:
        """Determine the DSL type category for threshold selection."""
        script_lower = self.script.lower()

        # Power/exponent operations
        if '**' in script_lower or 'pow(' in script_lower or 'exponent' in script_lower:
            return "power"

        # Geometry/distance operations
        if 'sqrt' in script_lower or 'distance' in script_lower or 'hypot' in script_lower:
            return "geometry"

        # Combinatorics
        if any(kw in script_lower for kw in ['factorial', 'comb', 'perm', 'choose', 'binomial']):
            return "combinatorics"

        # Division/modulo
        if '/' in script_lower or '%' in script_lower or 'mod' in script_lower:
            return "division"

        # Simple arithmetic (add, subtract, multiply)
        if any(op in script_lower for op in ['+', '-', '*']) and '**' not in script_lower:
            return "arithmetic"

        return "default"

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
        2. For unfound params, fall back to positional matching from remaining keys
        3. Aggressively match step_N keys to params by position

        Returns dict with param names as keys and context values as values.
        """
        result = {}
        context_keys = list(context.keys())
        used_keys = set()

        # First pass: name-based matching
        for param in self.params:
            matched_key = self.match_param(param, context_keys)
            if matched_key:
                result[param] = context[matched_key]
                used_keys.add(matched_key)
                context_keys.remove(matched_key)  # Don't reuse

        # Second pass: positional matching for unfound params
        # Sort remaining keys (step_1, step_2, etc. will be in order)
        remaining_keys = sorted([k for k in context.keys() if k not in used_keys])
        unfound_params = [p for p in self.params if p not in result]

        for i, param in enumerate(unfound_params):
            if i < len(remaining_keys):
                result[param] = context[remaining_keys[i]]

        # If still no matches and we have generic params (a, b, c, x, y, z, n, m)
        # Try aggressive positional matching
        if not result and self.params and context:
            generic_params = {'a', 'b', 'c', 'x', 'y', 'z', 'n', 'm', 'r', 'k', 'i', 'j'}
            if all(p.lower() in generic_params for p in self.params):
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

    Special cases:
    - sum_all(): Returns sum of all numeric inputs
    - Single input with multi-param script: Returns the single value (identity)
    """
    # Prepare inputs: extract numeric values from text
    prepared = _prepare_math_inputs(inputs)
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


# =============================================================================
# Layer 1: SymPy Symbolic
# =============================================================================

# Whitelisted SymPy functions
_SYMPY_ALLOWED = {
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
    - "The equation x^2 - 4 = 0" → Eq(x**2 - 4, 0) (extract from text)
    - "\frac{x}{2} = 3" → Eq(x/2, 3) (LaTeX)
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

# Registry for custom operators
_CUSTOM_OPERATORS: dict[str, Callable] = {}


def register_operator(name: str, func: Callable) -> None:
    """Register a custom DSL operator."""
    _CUSTOM_OPERATORS[name] = func


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

# Add base conversion functions to _FUNCTIONS for math layer compatibility
_FUNCTIONS["int_to_base"] = int_to_base
_FUNCTIONS["to_base"] = int_to_base
_FUNCTIONS["base_multiply"] = base_multiply
_FUNCTIONS["base_add"] = base_add


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


def _is_numeric_input(value: Any) -> bool:
    """Check if a value can be used as numeric input.

    Uses fast heuristics first, falls back to embedding-based classification
    only for uncertain cases (~5% of inputs).
    """
    # Fast path: heuristics handle 95%+ of cases
    classification = _classify_value_heuristic(value)

    if classification == ValueType.NUMBER:
        return True
    if classification == ValueType.EXPRESSION:
        return False

    # Slow path: embedding fallback for uncertain cases
    if isinstance(value, str) and len(value.strip()) > 0:
        emb_class, confidence = _classify_value_embedding(value.strip())
        logger.debug(
            "[value_classify] Embedding fallback: '%s' -> %s (conf=%.3f)",
            str(value)[:30], emb_class.value, confidence
        )
        return emb_class == ValueType.NUMBER

    return False


def _validate_dsl_task_match(dsl_spec: DSLSpec, step_task: str, task_embedding=None) -> tuple[bool, str]:
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
    import numpy as np
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


def _validate_param_mapping_semantic(
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

    # Get per-type threshold if not overridden
    if threshold is None:
        dsl_type = dsl_spec.get_dsl_type()
        type_thresholds = DSL_THRESHOLDS_BY_TYPE.get(dsl_type, DSL_THRESHOLDS_BY_TYPE["default"])
        threshold = type_thresholds.get("param", DSL_PARAM_SEMANTIC_THRESHOLD)

    import numpy as np
    from mycelium.embedder import Embedder
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


def _validate_param_types(dsl_spec: DSLSpec, inputs: dict[str, Any], step_task: str = "") -> tuple[bool, str]:
    """Validate that inputs match expected parameter types.

    Also detects semantic mapping errors:
    - Duplicate values: multiple params mapped to same value (bad semantic matching)
    - Binomial errors: n == r or n < r in combination/permutation DSLs
    - Task/DSL mismatch: DSL operation doesn't match task intent

    Returns:
        (valid, reason) - True if valid, False with reason if not
    """
    # First check task/DSL semantic match
    valid, reason = _validate_dsl_task_match(dsl_spec, step_task)
    if not valid:
        return valid, reason

    # Type validation
    if dsl_spec.param_types:
        for param, expected_type in dsl_spec.param_types.items():
            if param not in inputs:
                continue

            value = inputs[param]

            if expected_type == "numeric":
                if not _is_numeric_input(value):
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
            extracted = _extract_numeric_value(value)
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
                exp_val = _extract_numeric_value(inputs[param])
                break

        # Fallback: assume second numeric value is exponent
        if exp_val is None:
            numeric_vals = [_extract_numeric_value(v) for v in inputs.values()]
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
                n_val = _extract_numeric_value(inputs[param])
                break
        for param in ['r', 'k', 'R', 'K']:
            if param in inputs:
                r_val = _extract_numeric_value(inputs[param])
                break

        # Fallback to positional (first two params for binomial operations)
        if n_val is None or r_val is None:
            numeric_inputs = [(p, _extract_numeric_value(v)) for p, v in inputs.items()]
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


def _validate_result_bounds(result: Any, dsl_spec: "DSLSpec", inputs: dict[str, Any]) -> tuple[bool, str]:
    """Validate result against operation-specific bounds.

    Catches results that violate mathematical invariants.
    """
    if not isinstance(result, (int, float)):
        return True, "ok"

    script_lower = dsl_spec.script.lower()
    numeric_inputs = [_extract_numeric_value(v) for v in inputs.values()]
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


def _is_valid_dsl_result(result: Any, dsl_spec: "DSLSpec" = None, inputs: dict[str, Any] = None) -> bool:
    """Check if a DSL result is valid (not obvious garbage).

    Rejects results that are clearly invalid:
    - Boolean False (sympy comparison/factor failures return False)
    - Astronomically large numbers (> 1e10) - likely param mapping errors
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
        valid, reason = _validate_result_bounds(result, dsl_spec, inputs)
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
    valid, reason = _validate_param_types(dsl_spec, filtered_inputs, step_task)
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

    # Generic params that should use positional/heuristic matching
    GENERIC_PARAMS = {
        # Single letters
        'a', 'b', 'c', 'n', 'm', 'x', 'y', 'z', 'k', 'i', 'j', 'r',
        # Common math terms
        'expr', 'value', 'result', 'num', 'val',
        # Arithmetic operations
        'dividend', 'divisor', 'base', 'exp', 'exponent',
        'numerator', 'denominator', 'factor',
        # Function params
        'f', 'var', 'equation', 'count',
    }

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
        elif param.lower() in GENERIC_PARAMS:
            # Generic param - save for positional fallback
            unmatched_params.append(param)
        else:
            # Can't map this param semantically and it's not generic
            logger.debug("[dsl_semantic] No semantic match for param '%s'", param)
            return None, 0.0

    # Positional fallback for generic params
    if unmatched_params:
        # Params that typically come from the task/problem (constants), not computed values
        TASK_PREFERRED_PARAMS = {'divisor', 'denominator', 'modulus', 'base'}

        # Map generic params to step_N in order, ensuring no duplicates
        available_step_keys = [k for k in step_keys if k not in used_keys]
        task_num_keys = sorted([k for k in context.keys() if k.startswith('task_num_') and k not in used_keys],
                               key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        problem_num_keys = sorted([k for k in context.keys() if k.startswith('problem_num_') and k not in used_keys],
                                  key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

        for i, param in enumerate(unmatched_params):
            matched = False
            param_lower = param.lower()

            # For divisor-like params, prefer task_num (the value from the problem)
            if param_lower in TASK_PREFERRED_PARAMS:
                # Try last task_num (often the divisor/base mentioned at end of task)
                for key in reversed(task_num_keys):
                    if key not in used_keys:
                        param_mapping[param] = key
                        used_keys.add(key)
                        total_score += 0.7
                        logger.debug("[dsl_semantic] Task-preferred match: %s -> %s", param, key)
                        matched = True
                        break

            # Otherwise, try step keys first (computed values)
            if not matched:
                for key in available_step_keys:
                    if key not in used_keys:
                        param_mapping[param] = key
                        used_keys.add(key)
                        total_score += 0.6
                        logger.debug("[dsl_semantic] Positional match: %s -> %s", param, key)
                        matched = True
                        break

            # Final fallback to num keys
            if not matched:
                num_keys = task_num_keys + problem_num_keys
                for key in num_keys:
                    if key not in used_keys:
                        param_mapping[param] = key
                        used_keys.add(key)
                        total_score += 0.5
                        logger.debug("[dsl_semantic] Num key match: %s -> %s", param, key)
                        matched = True
                        break

            if not matched:
                logger.debug("[dsl_semantic] No positional match for generic param '%s'", param)
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
