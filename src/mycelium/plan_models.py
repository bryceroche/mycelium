"""Plan Models: Data structures for problem decomposition.

These are pure data structures with no LLM dependencies.
Used by both local_decomposer and solver.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Operation to DSL hint mapping (per CLAUDE.md: route by what operations DO)
OPERATION_TO_DSL_HINT = {
    "add": "+",
    "sum": "+",
    "plus": "+",
    "total": "+",
    "combine": "+",
    "subtract": "-",
    "minus": "-",
    "difference": "-",
    "remaining": "-",
    "left": "-",
    "multiply": "*",
    "times": "*",
    "product": "*",
    "divide": "/",
    "quotient": "/",
    "ratio": "/",
    "per": "/",
}


@dataclass
class SignatureHint:
    """Hint about an available signature for routing.

    Describes what the signature does and what parameters it needs.
    """
    step_type: str
    description: str
    param_names: list[str] = field(default_factory=list)
    param_descriptions: dict[str, str] = field(default_factory=dict)
    clarifying_questions: list[str] = field(default_factory=list)
    is_cluster: bool = False
    children: list["SignatureHint"] = field(default_factory=list)


@dataclass
class Step:
    """A single step in the decomposition DAG.

    A step can be either:
    - ATOMIC: Executed directly (sub_plan is None)
    - COMPOSITE: Contains a sub-DAG (sub_plan is not None)
    """
    id: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    result: Optional[str] = None
    success: bool = False
    extracted_values: dict[str, Any] = field(default_factory=dict)
    dsl_hint: Optional[str] = None
    operation: Optional[str] = None
    requires_algebra: bool = False
    sub_plan: Optional["DAGPlan"] = None

    def __post_init__(self):
        """Infer dsl_hint from operation if missing."""
        if self.dsl_hint is None and self.operation:
            op_lower = self.operation.lower()
            if op_lower in OPERATION_TO_DSL_HINT:
                self.dsl_hint = OPERATION_TO_DSL_HINT[op_lower]

        if self.dsl_hint is None and self.task:
            task_lower = self.task.lower()
            if any(kw in task_lower for kw in ["subtract", "difference", "minus", "remaining", "left"]):
                self.dsl_hint = "-"
                self.operation = self.operation or "subtract"
            elif any(kw in task_lower for kw in ["add", "sum", "total", "combine", "plus"]):
                self.dsl_hint = "+"
                self.operation = self.operation or "add"
            elif any(kw in task_lower for kw in ["multiply", "times", "product"]):
                self.dsl_hint = "*"
                self.operation = self.operation or "multiply"
            elif any(kw in task_lower for kw in ["divide", "quotient", "ratio", "per"]):
                self.dsl_hint = "/"
                self.operation = self.operation or "divide"

    @property
    def is_composite(self) -> bool:
        """True if this step contains a sub-plan."""
        return self.sub_plan is not None

    def max_depth(self) -> int:
        """Calculate maximum nesting depth."""
        if self.sub_plan is None:
            return 0
        return 1 + self.sub_plan.max_depth()

    def total_steps(self) -> int:
        """Count total steps including nested sub-plans."""
        count = 1
        if self.sub_plan:
            count += self.sub_plan.total_steps()
        return count

    def flatten(self, prefix: str = "") -> list[tuple[str, "Step"]]:
        """Flatten step and any sub-plan into (path, step) tuples."""
        path = f"{prefix}/{self.id}" if prefix else self.id
        result = [(path, self)]
        if self.sub_plan:
            for sub_step in self.sub_plan.steps:
                result.extend(sub_step.flatten(path))
        return result


@dataclass
class DAGPlan:
    """A decomposition plan as a DAG of steps."""
    steps: list[Step]
    problem: str
    depth: int = 0
    parent_step_id: Optional[str] = None
    phase1_values: dict[str, Any] = field(default_factory=dict)

    def max_depth(self) -> int:
        """Calculate maximum nesting depth across all steps."""
        if not self.steps:
            return 0
        return max(step.max_depth() for step in self.steps)

    def total_steps(self) -> int:
        """Count total steps including all nested sub-plans."""
        return sum(step.total_steps() for step in self.steps)

    def flatten(self) -> list[tuple[str, Step]]:
        """Flatten entire plan into list of (path, step) tuples."""
        result = []
        for step in self.steps:
            result.extend(step.flatten())
        return result

    def resolve_value_references(self, step_results: dict[str, Any] = None) -> dict[str, dict[str, Any]]:
        """Resolve $name and {step_N} references to actual values."""
        step_results = step_results or {}
        resolved = {}
        phase1_ref_pattern = re.compile(r'^\$([a-zA-Z_][a-zA-Z0-9_]*)$')
        step_ref_pattern = re.compile(r'^\{(step_?\d+)\}$', re.IGNORECASE)

        for step in self.steps:
            step_resolved = {}
            for key, value in step.extracted_values.items():
                if isinstance(value, (int, float)):
                    step_resolved[key] = value
                elif isinstance(value, str):
                    phase1_match = phase1_ref_pattern.match(value)
                    if phase1_match:
                        ref_name = phase1_match.group(1)
                        if ref_name in self.phase1_values:
                            step_resolved[key] = self.phase1_values[ref_name]
                        else:
                            step_resolved[key] = value
                        continue

                    step_match = step_ref_pattern.match(value)
                    if step_match:
                        ref_step = step_match.group(1)
                        if ref_step in step_results:
                            step_resolved[key] = step_results[ref_step]
                        else:
                            step_resolved[key] = value
                        continue

                    try:
                        step_resolved[key] = float(value)
                    except (ValueError, TypeError):
                        step_resolved[key] = value
                else:
                    step_resolved[key] = value

            resolved[step.id] = step_resolved

        return resolved
