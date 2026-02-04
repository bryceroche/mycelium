"""Span templates - normalized patterns with executable DSLs.

Each template has a custom DSL expression that defines exactly how to compute
the result. Expressions use these variables:
- `entity`: current value of the entity in state (default 0)
- `value`: the extracted number from the span
- `ref`: value of a reference entity (for comparisons)

Examples:
- "[NAME] has [N] [ITEM]" → dsl_expr="value" (SET)
- "[NAME] sold [N] [ITEM]" → dsl_expr="entity - value" (SUB)
- "[NAME] has [N] more than [REF]" → dsl_expr="ref + value" (COMPARE_MORE)
- "twice as many as [REF]" → dsl_expr="ref * 2" (RATIO)
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Any
import numpy as np


def execute_dsl_expr(expr: str, state: Dict[str, float], entity: str,
                     value: float, ref_entity: Optional[str] = None) -> float:
    """Execute a custom DSL expression.

    Args:
        expr: DSL expression like "entity - value" or "ref * 2"
        state: Current entity states
        entity: Primary entity name
        value: Extracted numeric value
        ref_entity: Optional reference entity name

    Returns:
        Computed result
    """
    # Build evaluation context
    entity_val = state.get(entity, 0)
    ref_val = state.get(ref_entity, 0) if ref_entity else 0

    context = {
        "entity": entity_val,
        "value": value,
        "ref": ref_val,
        # Math functions
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
    }

    try:
        return float(eval(expr, {"__builtins__": {}}, context))
    except Exception as e:
        # Fallback to value on error
        return value


@dataclass
class SpanTemplate:
    """A normalized span template with its custom DSL.

    Each template has a dsl_expr that defines exactly how to compute results.
    """
    template_id: str              # Unique ID like "sub_sold_item"
    pattern: str                  # Normalized pattern: "[NAME] sold [N] [ITEM]"
    dsl_expr: str = "value"       # Custom DSL expression: "entity - value"
    centroid: Optional[np.ndarray] = None  # 384-dim embedding centroid
    operation: str = "UNKNOWN"    # Primary operation (for fallback)
    dsl_type: str = "simple"      # "simple" or "complex"
    examples: List[str] = field(default_factory=list)  # Original spans
    count: int = 0                # Number of spans in cluster

    def execute(self, state: Dict[str, float], entity: str, value: float,
                ref_entity: Optional[str] = None) -> float:
        """Execute this template's custom DSL expression."""
        if self.dsl_expr:
            return execute_dsl_expr(self.dsl_expr, state, entity, value, ref_entity)
        # Fallback to operation-based DSL
        dsl_fn = get_dsl(self.operation, self.dsl_type)
        return dsl_fn(state, entity, value, ref_entity)


# Simple DSLs - single entity, single value
def dsl_set(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """SET: Initialize entity to value."""
    return value

def dsl_add(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """ADD: Add value to entity's current state."""
    current = state.get(entity, 0)
    return current + value

def dsl_sub(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """SUB: Subtract value from entity's current state."""
    current = state.get(entity, 0)
    return current - value

def dsl_mul(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """MUL: Multiply entity's current state by value."""
    current = state.get(entity, 0)
    return current * value

def dsl_div(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """DIV: Divide entity's current state by value."""
    current = state.get(entity, 0)
    return current / value if value != 0 else current


# Complex DSLs - reference entities, ratios, percentages
def dsl_compare_more(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """COMPARE_MORE: Entity has value more than reference."""
    ref_value = state.get(ref, 0) if ref else 0
    return ref_value + value

def dsl_compare_less(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """COMPARE_LESS: Entity has value less than reference."""
    ref_value = state.get(ref, 0) if ref else 0
    return ref_value - value

def dsl_ratio(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """RATIO: Entity has value times reference (e.g., 'twice as many')."""
    ref_value = state.get(ref, 0) if ref else 0
    return ref_value * value

def dsl_percent_increase(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """PERCENT_INCREASE: Increase entity by value percent."""
    current = state.get(entity, 0)
    return current * (1 + value / 100)

def dsl_percent_decrease(state: Dict[str, float], entity: str, value: float, ref: Optional[str] = None) -> float:
    """PERCENT_DECREASE: Decrease entity by value percent."""
    current = state.get(entity, 0)
    return current * (1 - value / 100)


# DSL registry (for fallback)
SIMPLE_DSLS = {
    "SET": dsl_set,
    "ADD": dsl_add,
    "SUB": dsl_sub,
    "MUL": dsl_mul,
    "DIV": dsl_div,
}

COMPLEX_DSLS = {
    "COMPARE_MORE": dsl_compare_more,
    "COMPARE_LESS": dsl_compare_less,
    "RATIO": dsl_ratio,
    "PERCENT_INCREASE": dsl_percent_increase,
    "PERCENT_DECREASE": dsl_percent_decrease,
}

def get_dsl(operation: str, dsl_type: str = "simple") -> Callable:
    """Get DSL function by operation name."""
    if dsl_type == "complex" and operation in COMPLEX_DSLS:
        return COMPLEX_DSLS[operation]
    return SIMPLE_DSLS.get(operation, dsl_set)


# =============================================================================
# Pattern-to-DSL Expression Inference
# =============================================================================

# Patterns that indicate SET (initialization)
SET_PATTERNS = [
    (r'\bhas\b', 'value'),
    (r'\bhad\b', 'value'),
    (r'\bhave\b', 'value'),
    (r'\bowns\b', 'value'),
    (r'\bstarts?\s+with\b', 'value'),
    (r'\bbegins?\s+with\b', 'value'),
    (r'\binitially\b', 'value'),
    (r'\bcontains?\b', 'value'),
    (r'\bthere\s+(are|is|were|was)\b', 'value'),
]

# Patterns that indicate SUB (subtraction)
SUB_PATTERNS = [
    (r'\bsold\b', 'entity - value'),
    (r'\bgave\b(?!\s+(him|her|them|me|us)\b)', 'entity - value'),  # "gave away" not "gave him"
    (r'\bspent\b', 'entity - value'),
    (r'\blost\b', 'entity - value'),
    (r'\bate\b', 'entity - value'),
    (r'\bused\b', 'entity - value'),
    (r'\bthrew\s+away\b', 'entity - value'),
    (r'\bremoved\b', 'entity - value'),
    (r'\bdropped\b', 'entity - value'),
    (r'\bdonated\b', 'entity - value'),
    (r'\bpaid\b', 'entity - value'),
    (r'\blent\b', 'entity - value'),
    (r'\bless\s+than\b', 'ref - value'),
    (r'\bfewer\s+than\b', 'ref - value'),
]

# Patterns that indicate ADD (addition)
ADD_PATTERNS = [
    (r'\bfound\b', 'entity + value'),
    (r'\breceived\b', 'entity + value'),
    (r'\bearned\b', 'entity + value'),
    (r'\bwon\b', 'entity + value'),
    (r'\bpicked\b', 'entity + value'),
    (r'\bgot\b', 'entity + value'),
    (r'\bgained\b', 'entity + value'),
    (r'\bcollected\b', 'entity + value'),
    (r'\bbought\b', 'entity + value'),
    (r'\badded\b', 'entity + value'),
    (r'\bmore\s+than\b', 'ref + value'),
    (r'\bgave\s+(him|her|them|me|us)\b', 'entity + value'),  # "gave him" = receiving
    (r'\bextra\b', 'entity + value'),
    (r'\badditional\b', 'entity + value'),
]

# Patterns that indicate MUL (multiplication)
MUL_PATTERNS = [
    (r'\btwice\s+as\s+many\b', 'ref * 2'),
    (r'\btwice\s+as\s+much\b', 'ref * 2'),
    (r'\bdouble\b', 'entity * 2'),
    (r'\bdoubled\b', 'entity * 2'),
    (r'\btriple\b', 'entity * 3'),
    (r'\btripled\b', 'entity * 3'),
    (r'\b(\d+)\s+times\s+as\s+many\b', 'ref * value'),
    (r'\btimes\s+as\s+many\b', 'ref * value'),
    (r'\bper\s+(hour|day|week|month|year|minute)\b', 'entity * value'),
    (r'\beach\b.*\b(costs?|pays?)\b', 'entity * value'),
]

# Patterns that indicate DIV (division)
DIV_PATTERNS = [
    (r'\bsplit\b', 'entity / value'),
    (r'\bdivided\b', 'entity / value'),
    (r'\bshared\s+equally\b', 'entity / value'),
    (r'\bdistributed\b', 'entity / value'),
    (r'\bhalf\s+as\s+many\b', 'ref / 2'),
    (r'\bhalf\s+of\b', 'entity / 2'),
    (r'\bquarter\s+of\b', 'entity / 4'),
    (r'\bthird\s+of\b', 'entity / 3'),
]

# Patterns for percentages
PERCENT_PATTERNS = [
    (r'\b(\d+)\s*%\s*(off|discount)\b', 'entity * (1 - value/100)'),
    (r'\b(\d+)\s*%\s*(increase|more|up)\b', 'entity * (1 + value/100)'),
    (r'\b(\d+)\s*%\s*(decrease|less|down)\b', 'entity * (1 - value/100)'),
    (r'\b(\d+)\s*%\s+of\b', 'entity * value/100'),
]


def infer_dsl_expr(pattern: str, operation: str = None) -> str:
    """Infer the DSL expression from a pattern string.

    Args:
        pattern: Normalized pattern like "[NAME] sold [N] [ITEM]"
        operation: Optional operation hint (SET, ADD, SUB, etc.)

    Returns:
        DSL expression like "entity - value"
    """
    pattern_lower = pattern.lower()

    # Check each pattern category in order of specificity
    # (more specific patterns first)

    # Percentages (most specific)
    for regex, expr in PERCENT_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # Multiplication patterns
    for regex, expr in MUL_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # Division patterns
    for regex, expr in DIV_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # Subtraction patterns
    for regex, expr in SUB_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # Addition patterns
    for regex, expr in ADD_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # SET patterns (least specific - check last)
    for regex, expr in SET_PATTERNS:
        if re.search(regex, pattern_lower):
            return expr

    # Fallback based on operation type
    op_to_expr = {
        "SET": "value",
        "ADD": "entity + value",
        "SUB": "entity - value",
        "MUL": "entity * value",
        "DIV": "entity / value" if "value" else "entity",
        "COMPARE_MORE": "ref + value",
        "COMPARE_LESS": "ref - value",
        "RATIO": "ref * value",
        "PERCENT_INCREASE": "entity * (1 + value/100)",
        "PERCENT_DECREASE": "entity * (1 - value/100)",
    }

    if operation and operation.upper() in op_to_expr:
        return op_to_expr[operation.upper()]

    # Ultimate fallback: just return the value (SET operation)
    return "value"


def create_template_with_dsl(
    template_id: str,
    pattern: str,
    operation: str = None,
    centroid: np.ndarray = None,
    examples: List[str] = None,
    count: int = 0,
) -> SpanTemplate:
    """Create a SpanTemplate with auto-inferred DSL expression.

    Args:
        template_id: Unique template identifier
        pattern: Normalized pattern string
        operation: Optional operation hint
        centroid: Optional embedding centroid
        examples: Optional list of example spans
        count: Usage count

    Returns:
        SpanTemplate with custom dsl_expr
    """
    dsl_expr = infer_dsl_expr(pattern, operation)

    # Infer operation from DSL expression if not provided
    if not operation:
        if "entity - " in dsl_expr or "ref - " in dsl_expr:
            operation = "SUB"
        elif "entity + " in dsl_expr or "ref + " in dsl_expr:
            operation = "ADD"
        elif "entity * " in dsl_expr or "ref * " in dsl_expr:
            operation = "MUL"
        elif "entity / " in dsl_expr or "ref / " in dsl_expr:
            operation = "DIV"
        else:
            operation = "SET"

    # Determine DSL type
    dsl_type = "complex" if "ref" in dsl_expr or "%" in dsl_expr else "simple"

    return SpanTemplate(
        template_id=template_id,
        pattern=pattern,
        dsl_expr=dsl_expr,
        centroid=centroid,
        operation=operation,
        dsl_type=dsl_type,
        examples=examples or [],
        count=count,
    )


# Template registry - populated by clustering
_template_registry: Dict[str, SpanTemplate] = {}

def register_template(template: SpanTemplate) -> None:
    """Register a template in the global registry."""
    _template_registry[template.template_id] = template

def get_template(template_id: str) -> Optional[SpanTemplate]:
    """Get template by ID."""
    return _template_registry.get(template_id)

def get_all_templates() -> List[SpanTemplate]:
    """Get all registered templates."""
    return list(_template_registry.values())


if __name__ == "__main__":
    # Test DSL execution
    state = {"Lisa": 12}
    result = dsl_sub(state, "Lisa", 5, None)
    print(f"Lisa: 12 - 5 = {result}")  # 7

    state = {"John": 10}
    result = dsl_compare_more(state, "Mary", 3, "John")
    print(f"Mary has 3 more than John(10) = {result}")  # 13
