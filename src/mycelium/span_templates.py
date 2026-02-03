"""Span templates - normalized patterns with executable DSLs."""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Any
import numpy as np

@dataclass
class SpanTemplate:
    """A normalized span template with its DSL.

    Represents a cluster of similar spans that map to the same operation.
    """
    template_id: str              # Unique ID like "sub_sold_item"
    pattern: str                  # Normalized pattern: "[NAME] sold [N] [ITEM]"
    centroid: Optional[np.ndarray] = None  # 384-dim embedding centroid
    operation: str = "UNKNOWN"    # Primary operation: SET, ADD, SUB, MUL, DIV
    dsl_type: str = "simple"      # "simple" or "complex"
    examples: List[str] = field(default_factory=list)  # Original spans
    count: int = 0                # Number of spans in cluster

    def execute(self, state: Dict[str, float], entity: str, value: float,
                ref_entity: Optional[str] = None) -> float:
        """Execute this template's DSL on the given state."""
        # Get DSL function and execute
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


# DSL registry
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
