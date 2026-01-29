"""Planner: Decompose problems into DAG of subtasks.

The planner breaks down complex problems into simpler steps,
creating a directed acyclic graph (DAG) of subtasks.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

from .client import get_client
from mycelium.config import (
    PLANNER_DEFAULT_MODEL,
    PLANNER_DEFAULT_TEMPERATURE,
    TREE_GUIDED_TOP_K_SUGGESTIONS,
    TREE_GUIDED_NOVELTY_K,
    TREE_GUIDED_NOVELTY_MIN_SAMPLES,
    TREE_GUIDED_NOVELTY_DEFAULT_THRESHOLD,
    # Similarity thresholds (per CLAUDE.md "The Flow")
    HINT_ALTERNATIVES_MIN_SIMILARITY,
    TREE_PLANNER_NEGOTIATION_SIMILARITY_THRESHOLD,
)


PLANNER_SYSTEM = """You decompose math problems into atomic steps. Output valid JSON only.

PHASE 1: Extract ALL numeric values with semantic names and UNITS.
PHASE 2: Build atomic steps that reference Phase 1 values.

CRITICAL: Unit Awareness
- Each value name MUST match its semantic meaning
- "distance_miles" for distances, "time_hours" for durations, "price_dollars" for costs
- NEVER mix units: don't assign a distance value to a time variable!
- Read the problem carefully to identify WHAT each number represents

OUTPUT FORMAT (JSON):
{
  "values": {
    "semantic_name_unit": number,
    ...
  },
  "steps": [
    {
      "id": "step_1",
      "task": "ONE atomic operation - verb + what",
      "operation": "add|subtract|multiply|divide",
      "values": {"param_name": "$phase1_name OR {step_N}"},
      "dsl_hint": "+|-|*|/",
      "depends_on": []
    }
  ]
}

RULES:
1. ONE OPERATION PER STEP
2. Reference Phase 1 values with $name (e.g., "$purchase_price_dollars")
3. Reference prior step results with {step_N} (e.g., "{step_1}")
4. NEVER use raw numbers in steps - always reference $names
5. For "increased by X%": extract multiplier (1 + X/100) in Phase 1
6. VALUE NAMES MUST MATCH WHAT THE NUMBER REPRESENTS - read context carefully!

PERCENTAGE HANDLING:
- "X increased by Y%" → base is X, multiply by (1 + Y/100)
- "Y% of X" → base is X, multiply by Y/100

BACKWARDS PROBLEMS (finding unknown starting values):
When possible, reframe as forward computation:
- "X - 3 = 5, find X" → compute 5 + 3 (reverse the operation)
- "X / 2 = 10, find X" → compute 10 * 2

If you cannot determine all values, use null and mark requires_algebra: true.

EXAMPLE 1 - MONEY:
Problem: "Josh buys a house for $80,000, puts in $50,000 repairs. This increased the value by 150%. How much profit?"

{
  "values": {
    "purchase_price_dollars": 80000,
    "repair_cost_dollars": 50000,
    "increase_multiplier": 2.5
  },
  "steps": [
    {
      "id": "step_1",
      "task": "Calculate total investment",
      "operation": "add",
      "values": {"a": "$purchase_price_dollars", "b": "$repair_cost_dollars"},
      "dsl_hint": "+",
      "depends_on": []
    },
    {
      "id": "step_2",
      "task": "Calculate new house value",
      "operation": "multiply",
      "values": {"base": "$purchase_price_dollars", "multiplier": "$increase_multiplier"},
      "dsl_hint": "*",
      "depends_on": []
    },
    {
      "id": "step_3",
      "task": "Calculate profit",
      "operation": "subtract",
      "values": {"new_value": "{step_2}", "investment": "{step_1}"},
      "dsl_hint": "-",
      "depends_on": ["step_1", "step_2"]
    }
  ]
}

EXAMPLE 2 - TIME AND DISTANCE (CAREFUL!):
Problem: "A car drives 180 miles. The trip takes 4 hours normally, but traffic adds 2 hours and slow zones add 0.5 hours. How much time is left?"

{
  "values": {
    "distance_miles": 180,
    "normal_time_hours": 4,
    "traffic_delay_hours": 2,
    "slow_zone_hours": 0.5
  },
  "steps": [
    {
      "id": "step_1",
      "task": "Calculate total extra time from delays",
      "operation": "add",
      "values": {"a": "$traffic_delay_hours", "b": "$slow_zone_hours"},
      "dsl_hint": "+",
      "depends_on": []
    },
    {
      "id": "step_2",
      "task": "Calculate remaining time after delays",
      "operation": "subtract",
      "values": {"total": "$normal_time_hours", "used": "{step_1}"},
      "dsl_hint": "-",
      "depends_on": ["step_1"]
    }
  ]
}

NOTE: In Example 2, "180 miles" is DISTANCE, NOT time! Always read what each number represents.

Output ONLY valid JSON. No explanation."""

SIGNATURE_HINTS_TEMPLATE = """
Known operations (extract values to match):
{hints}
"""



@dataclass
class SignatureHint:
    """Hint about an available signature for the decomposer.

    Communicates what the signature does and what parameters it needs,
    so the decomposer can create steps with appropriate extracted_values.

    For umbrella signatures (clusters), `children` contains the specific
    operations available under this category.
    """
    step_type: str
    description: str
    param_names: list[str] = field(default_factory=list)
    param_descriptions: dict[str, str] = field(default_factory=dict)
    clarifying_questions: list[str] = field(default_factory=list)
    is_cluster: bool = False  # True if this is an umbrella (category) signature
    children: list["SignatureHint"] = field(default_factory=list)  # Child operations for clusters

    def to_hint_text(self) -> str:
        """Format as hint text for the planner prompt.

        Includes clarifying_questions and param_descriptions to guide parameter extraction.
        For nested clusters, shows grandchildren inline: parent[child1[gc1,gc2], child2]
        """
        if self.is_cluster and self.children:
            # Compact cluster format with nested children support
            def format_child(c):
                if c.is_cluster and c.children:
                    gc_str = ",".join(gc.step_type for gc in c.children[:2])
                    return f"{c.step_type}[{gc_str}]"
                return c.step_type
            children_str = ", ".join(format_child(c) for c in self.children[:3])
            return f"- {self.step_type}: {self.description[:50]}... [{children_str}]"

        # Single signature format with extraction guidance
        lines = []
        params_str = f" ({', '.join(self.param_names)})" if self.param_names else ""
        lines.append(f"- {self.step_type}{params_str}: {self.description[:60]}")

        # Add clarifying questions to guide parameter extraction
        if self.clarifying_questions:
            lines.append("  Extract:")
            for q in self.clarifying_questions:
                lines.append(f"    - {q}")

        # Add parameter descriptions for context
        if self.param_descriptions:
            lines.append("  Parameters:")
            for param, desc in self.param_descriptions.items():
                lines.append(f"    - {param}: {desc}")

        return "\n".join(lines)


# Per CLAUDE.md "New Favorite Pattern": Consolidated operation → dsl_hint mapping
# This is the single source of truth for inferring dsl_hint from operation
OPERATION_TO_DSL_HINT = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    # Common variations
    "sum": "+",
    "difference": "-",
    "product": "*",
    "quotient": "/",
    "plus": "+",
    "minus": "-",
    "times": "*",
}


@dataclass
class Step:
    """A single step in the decomposition DAG.

    A step can be either:
    - ATOMIC: Executed directly (sub_plan is None)
    - COMPOSITE: Contains a sub-DAG that must be executed first (sub_plan is not None)

    This enables unlimited recursive nesting: DAG of DAGs of DAGs...
    """
    id: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    result: Optional[str] = None
    success: bool = False
    # Value extraction: semantic_name -> value (numeric or "{step_N}" reference)
    extracted_values: dict[str, Any] = field(default_factory=dict)
    # Optional DSL hint from planner
    dsl_hint: Optional[str] = None
    # Operation description for graph-based routing (per CLAUDE.md)
    # Describes WHAT computation is needed, not the specific values
    # Example: "multiply two numbers" or "divide then add"
    # Note: Operation embeddings stored separately in solver (not on Step) for memory efficiency
    operation: Optional[str] = None
    # Flag for algebra problems that couldn't be reframed as forward computation
    # When True, step will be routed to SymPy solver for backwards solving
    requires_algebra: bool = False
    # Recursive nesting: sub-plan for composite steps
    sub_plan: Optional["DAGPlan"] = None

    def __post_init__(self):
        """Infer dsl_hint from operation if missing (per CLAUDE.md New Favorite Pattern)."""
        # First try: infer from explicit operation field
        if self.dsl_hint is None and self.operation:
            op_lower = self.operation.lower()
            if op_lower in OPERATION_TO_DSL_HINT:
                self.dsl_hint = OPERATION_TO_DSL_HINT[op_lower]
                logger.debug("[planner] Inferred dsl_hint='%s' from operation='%s'",
                           self.dsl_hint, self.operation)

        # Second try: infer from task description (fallback for LLM omissions)
        if self.dsl_hint is None and self.task:
            task_lower = self.task.lower()
            # Check for operation keywords in task description
            # Per CLAUDE.md: route by what operations DO
            if any(kw in task_lower for kw in ["subtract", "difference", "minus", "remaining", "left"]):
                self.dsl_hint = "-"
                self.operation = self.operation or "subtract"
                logger.info("[planner] Inferred dsl_hint='-' from task='%s'", self.task[:40])
            elif any(kw in task_lower for kw in ["add", "sum", "total", "combine", "plus"]):
                self.dsl_hint = "+"
                self.operation = self.operation or "add"
                logger.info("[planner] Inferred dsl_hint='+' from task='%s'", self.task[:40])
            elif any(kw in task_lower for kw in ["multiply", "product", "times", "per "]):
                self.dsl_hint = "*"
                self.operation = self.operation or "multiply"
                logger.info("[planner] Inferred dsl_hint='*' from task='%s'", self.task[:40])
            elif any(kw in task_lower for kw in ["divide", "quotient", "ratio", "split", "per unit"]):
                self.dsl_hint = "/"
                self.operation = self.operation or "divide"
                logger.info("[planner] Inferred dsl_hint='/' from task='%s'", self.task[:40])

    @property
    def is_composite(self) -> bool:
        """True if this step contains a sub-DAG."""
        return self.sub_plan is not None

    @property
    def is_atomic(self) -> bool:
        """True if this step is executed directly (no sub-plan)."""
        return self.sub_plan is None

    def max_depth(self) -> int:
        """Calculate maximum nesting depth from this step.

        Returns:
            0 for atomic steps, 1+ for composite steps
        """
        if self.sub_plan is None:
            return 0
        return 1 + self.sub_plan.max_depth()

    def total_steps(self) -> int:
        """Count total steps including all nested sub-plans.

        Returns:
            1 for atomic steps, 1 + sub_plan steps for composite
        """
        if self.sub_plan is None:
            return 1
        return 1 + self.sub_plan.total_steps()

    def flatten(self, prefix: str = "") -> list[tuple[str, "Step"]]:
        """Flatten step and all nested steps into a list.

        Args:
            prefix: Path prefix for nested steps

        Returns:
            List of (path, step) tuples
        """
        path = f"{prefix}/{self.id}" if prefix else self.id
        result = [(path, self)]
        if self.sub_plan:
            for step in self.sub_plan.steps:
                result.extend(step.flatten(prefix=path))
        return result


class PlanValidationError(Exception):
    """Raised when a plan has invalid structure (cycles, missing deps, etc.)."""
    pass


@dataclass
class DAGPlan:
    """A decomposition plan as a DAG of steps.

    Supports recursive nesting: steps can contain sub-plans,
    creating a DAG of DAGs of unlimited depth.
    """
    steps: list[Step]
    problem: str
    # Depth in the recursive hierarchy (0 = root plan)
    depth: int = 0
    # Optional parent step ID (for sub-plans)
    parent_step_id: Optional[str] = None
    # Phase 1 extracted values: name -> numeric value (for provenance tracking)
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
        """Resolve $name and {step_N} references to actual values.

        Args:
            step_results: Mapping of step_id -> result for {step_N} resolution

        Returns:
            Dict mapping step_id -> resolved extracted_values
        """
        import re
        step_results = step_results or {}
        resolved = {}
        phase1_ref_pattern = re.compile(r'^\$([a-zA-Z_][a-zA-Z0-9_]*)$')
        step_ref_pattern = re.compile(r'^\{(step_?\d+)\}$', re.IGNORECASE)

        for step in self.steps:
            step_resolved = {}
            for key, value in step.extracted_values.items():
                if isinstance(value, (int, float)):
                    # Already a number
                    step_resolved[key] = value
                elif isinstance(value, str):
                    # Check for Phase 1 reference ($name)
                    phase1_match = phase1_ref_pattern.match(value)
                    if phase1_match:
                        ref_name = phase1_match.group(1)
                        if ref_name in self.phase1_values:
                            step_resolved[key] = self.phase1_values[ref_name]
                        else:
                            logger.warning(
                                "[planner] Step '%s' references unknown Phase 1 value: $%s",
                                step.id, ref_name
                            )
                            step_resolved[key] = value  # Keep as-is for error handling
                        continue

                    # Check for step reference ({step_N})
                    step_match = step_ref_pattern.match(value)
                    if step_match:
                        ref_step = step_match.group(1)
                        if ref_step in step_results:
                            step_resolved[key] = step_results[ref_step]
                        else:
                            # Step result not yet available - keep as reference
                            step_resolved[key] = value
                        continue

                    # Try to parse as number (backwards compatibility)
                    try:
                        if "." in value:
                            step_resolved[key] = float(value)
                        else:
                            step_resolved[key] = int(value)
                    except ValueError:
                        # Keep as-is (might be a string value)
                        step_resolved[key] = value
                else:
                    step_resolved[key] = value

            resolved[step.id] = step_resolved

        return resolved

    def validate_value_references(self) -> tuple[bool, list[str]]:
        """Validate that all $name references exist in Phase 1 values.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        import re
        errors = []
        phase1_ref_pattern = re.compile(r'^\$([a-zA-Z_][a-zA-Z0-9_]*)$')

        for step in self.steps:
            for key, value in step.extracted_values.items():
                if isinstance(value, str):
                    match = phase1_ref_pattern.match(value)
                    if match:
                        ref_name = match.group(1)
                        if ref_name not in self.phase1_values:
                            errors.append(
                                f"Step '{step.id}' references undefined value ${ref_name}. "
                                f"Available Phase 1 values: {list(self.phase1_values.keys())}"
                            )

        return len(errors) == 0, errors

    def validate(self, recursive: bool = True) -> tuple[bool, list[str]]:
        """Validate the DAG structure.

        Checks:
        1. Missing dependencies (step depends on non-existent step)
        2. Cyclic dependencies
        3. Invalid step references in extracted_values (e.g., {step_1} pointing to unknown step)
        4. Step references not in depends_on (can't use result from step you don't depend on)
        5. DATA FLOW: Undefined variables in extracted_values (neither numbers nor step refs)

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        import re
        errors = []
        warnings = []  # Non-fatal issues
        step_ids = {s.id for s in self.steps}

        # Pattern to match step references like {step_1}, {step_2}, etc.
        step_ref_pattern = re.compile(r'\{(step_?\d+)\}', re.IGNORECASE)

        # Pattern to detect variable names (not numbers, not step refs)
        # These indicate undefined values that should have been extracted in Phase 1
        variable_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

        # Check for missing dependencies
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on unknown step '{dep}'")

        # Check step references in extracted_values
        for step in self.steps:
            if not step.extracted_values:
                continue
            for key, value in step.extracted_values.items():
                # Only check string values that could be references
                if not isinstance(value, str):
                    continue
                # Find all step references in this value
                matches = step_ref_pattern.findall(value)
                for ref_id in matches:
                    # Normalize: handle both "step1" and "step_1" formats
                    normalized_ref = ref_id.lower()
                    # Check if referenced step exists
                    if normalized_ref not in step_ids and ref_id not in step_ids:
                        errors.append(
                            f"Step '{step.id}' has extracted_value '{key}' referencing "
                            f"unknown step '{{{ref_id}}}'"
                        )
                    # Check if referenced step is in depends_on
                    elif normalized_ref not in step.depends_on and ref_id not in step.depends_on:
                        # Try both formats
                        found_in_deps = any(
                            d.lower() == normalized_ref or d == ref_id
                            for d in step.depends_on
                        )
                        if not found_in_deps:
                            # Auto-fix: add missing dependency instead of error
                            # LLM often forgets to add depends_on when referencing step values
                            step.depends_on.append(normalized_ref)
                            logger.debug(
                                f"[planner] Auto-fixed: added '{normalized_ref}' to "
                                f"depends_on for step '{step.id}' (referenced in extracted_values)"
                            )

        # DATA FLOW VALIDATION: Check for undefined variables in extracted_values
        # Per CLAUDE.md: Every step's inputs must be defined - either from problem text
        # (as literal numbers) or from prior step outputs (as {step_N} references).
        # String values that are neither numbers nor step references indicate undefined
        # variables - these need algebra/backwards solving via SymPy.
        for step in self.steps:
            if not step.extracted_values:
                continue
            for key, value in step.extracted_values.items():
                # Check for null values (LLM couldn't extract)
                if value is None:
                    step.requires_algebra = True
                    logger.info(
                        "[planner] ALGEBRA: Step '%s' has null value for '%s' - will use SymPy",
                        step.id, key
                    )
                    continue
                if not isinstance(value, str):
                    continue  # Numbers are valid
                # Check if it's a step reference - those are valid
                if step_ref_pattern.search(value):
                    continue  # Step references like {step_1} are valid
                # Check if it's a Phase 1 reference ($name) - those are valid
                if value.startswith('$'):
                    continue
                # Check if it looks like a variable name (not a number)
                # This catches cases like "total_vacuums" when we don't know the total
                if variable_pattern.match(value):
                    # This is a variable name that wasn't resolved - mark for algebra
                    step.requires_algebra = True
                    logger.info(
                        "[planner] ALGEBRA: Step '%s' has undefined variable '%s' for '%s' - will use SymPy",
                        step.id, value, key
                    )

        # Check for cycles using DFS with forward traversal
        # Build adjacency list: step -> list of steps that depend on it
        dependents: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep in dependents:  # Only add if dep exists (missing deps handled above)
                    dependents[dep].append(step.id)

        # Track all steps involved in cycles
        cycle_steps: set[str] = set()
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def find_cycles(step_id: str) -> None:
            """DFS to find all steps involved in cycles."""
            visited.add(step_id)
            rec_stack.add(step_id)

            for dependent in dependents.get(step_id, []):
                if dependent not in visited:
                    find_cycles(dependent)
                elif dependent in rec_stack:
                    # Found a cycle - mark all steps in current path
                    cycle_steps.add(dependent)
                    cycle_steps.add(step_id)

            rec_stack.remove(step_id)

        # Run DFS from all unvisited nodes to find all cycles
        for step in self.steps:
            if step.id not in visited:
                find_cycles(step.id)

        if cycle_steps:
            errors.append(f"Cyclic dependencies detected involving steps: {sorted(cycle_steps)}")

        # Recursively validate sub-plans
        if recursive:
            for step in self.steps:
                if step.sub_plan is not None:
                    sub_valid, sub_errors = step.sub_plan.validate(recursive=True)
                    if not sub_valid:
                        prefixed = [f"[{step.id}] {e}" for e in sub_errors]
                        errors.extend(prefixed)

        return (len(errors) == 0, errors)

    def get_execution_order(self, strict: bool = False) -> list[list[Step]]:
        """Get steps grouped by execution level (parallelizable within level).

        Args:
            strict: If True, raise PlanValidationError on invalid DAG.
                   If False (default), log warning and return partial order.

        Returns:
            List of step groups, where each group can be executed in parallel.

        Raises:
            PlanValidationError: If strict=True and DAG is invalid.
        """
        # Validate first
        is_valid, errors = self.validate()
        if not is_valid:
            error_msg = f"Invalid DAG: {'; '.join(errors)}"
            if strict:
                raise PlanValidationError(error_msg)
            else:
                logger.warning(f"{error_msg}. Returning partial execution order.")

        completed = set()
        levels = []
        remaining = {s.id: s for s in self.steps}

        while remaining:
            # Find steps whose dependencies are all satisfied
            ready = [
                s for s in remaining.values()
                if all(d in completed for d in s.depends_on)
            ]
            if not ready:
                # Circular dependency or missing step - log dropped steps
                dropped = list(remaining.keys())
                logger.warning(
                    f"Cannot schedule {len(dropped)} steps due to unresolved dependencies: {dropped}"
                )
                break

            levels.append(ready)
            for s in ready:
                completed.add(s.id)
                del remaining[s.id]

        return levels

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None


def _create_planner_without_warning(client=None, temperature: float = PLANNER_DEFAULT_TEMPERATURE):
    """Create a Planner instance without triggering deprecation warning.

    Used internally by TreeGuidedPlanner for composition/fallback.
    Per CLAUDE.md "New Favorite Pattern": Reuse existing logic, don't duplicate.
    """
    planner = object.__new__(Planner)
    planner.client = client or get_client()
    planner.temperature = temperature
    return planner


class Planner:
    """Decompose problems into DAG of subtasks.

    DEPRECATED: Use TreeGuidedPlanner instead for vocabulary-guided decomposition.
    This class is kept for backwards compatibility but will be removed in a future release.
    """

    def __init__(
        self,
        model: str = PLANNER_DEFAULT_MODEL,
        temperature: float = PLANNER_DEFAULT_TEMPERATURE,
    ):
        import warnings
        warnings.warn(
            "Planner is deprecated. Use TreeGuidedPlanner for vocabulary-guided decomposition.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.client = get_client(model=model)
        self.temperature = temperature

    async def decompose(
        self,
        problem: str,
        signature_hints: Optional[list[SignatureHint]] = None,
        context: Optional[str] = None,
        skip_validation: bool = False,
    ) -> DAGPlan:
        """Decompose a problem into a DAG of steps.

        Args:
            problem: The problem to decompose
            signature_hints: Optional list of SignatureHint objects that tell the
                            decomposer what operations are available and what
                            parameters each needs (from NL interface)
            context: Optional additional context (e.g., original problem when
                    decomposing a sub-step, complexity hints)
            skip_validation: If True, skip data flow validation. Use this for
                            template decomposition (e.g., umbrella creation)
                            where we're creating generic patterns, not concrete plans.

        Returns:
            DAGPlan with steps and dependencies
        """
        logger.debug("[planner] Starting decomposition: problem='%s...'", problem[:80])

        # Build system prompt with optional signature hints
        system_prompt = PLANNER_SYSTEM
        if signature_hints:
            hints_text = "\n\n".join(hint.to_hint_text() for hint in signature_hints)
            system_prompt += SIGNATURE_HINTS_TEMPLATE.format(hints=hints_text)
            logger.debug("[planner] Added %d signature hints with NL interface", len(signature_hints))

        # Build user message with optional context
        user_content = problem
        if context:
            user_content = f"{context}\n\nDecompose this step: {problem}"
            logger.debug("[planner] Added context to decomposition request")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self.client.generate(
            messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        steps, phase1_values = self._parse_steps(response)

        plan = DAGPlan(steps=steps, problem=problem, phase1_values=phase1_values)

        # Skip validation for template decomposition (e.g., umbrella creation)
        if skip_validation:
            logger.info(
                "[planner] Decomposition complete: steps=%d (validation skipped - template mode)",
                len(steps)
            )
        else:
            is_valid, errors = plan.validate()
            logger.info(
                "[planner] Decomposition complete: steps=%d valid=%r",
                len(steps), is_valid
            )
            if errors:
                logger.warning("[planner] Validation errors: %s", errors)

        return plan

    def _parse_steps(self, response: str) -> tuple[list[Step], dict[str, Any]]:
        """Parse steps from planner JSON response.

        Expects JSON format:
        {
            "values": {"name": number, ...},
            "steps": [{"id": "step_1", "task": "...", ...}, ...]
        }

        Returns:
            Tuple of (steps, phase1_values) for provenance tracking
        """
        import json

        steps = []
        phase1_values = {}

        try:
            # Parse JSON response
            data = json.loads(response.strip())

            # Extract Phase 1 values
            phase1_values = data.get("values", {})
            if phase1_values:
                logger.info(
                    "[planner] Phase 1 values extracted: %s",
                    ", ".join(f"{k}={v}" for k, v in phase1_values.items())
                )

            # Parse steps
            for step_data in data.get("steps", []):
                step_id = step_data.get("id", f"step_{len(steps) + 1}")
                task = step_data.get("task", "")

                if not task:
                    logger.warning(f"Step '{step_id}' has empty task")

                step = Step(
                    id=step_id,
                    task=task,
                    depends_on=step_data.get("depends_on", []),
                    extracted_values=step_data.get("values", {}),
                    dsl_hint=step_data.get("dsl_hint"),
                    operation=step_data.get("operation"),
                    requires_algebra=step_data.get("requires_algebra", False),
                )
                steps.append(step)

        except json.JSONDecodeError as e:
            logger.warning(f"[planner] JSON parse failed: {e}")
            logger.warning(f"[planner] Raw response:\n{response[:500]}")
            # Fallback: try to extract from malformed response
            steps, phase1_values = self._parse_steps_fallback(response)

        # Ensure we have at least one step
        if not steps:
            logger.warning("No steps parsed from planner response, using fallback single step")
            steps = [Step(id="solve", task="Solve the problem directly", depends_on=[])]

        # Validate the resulting plan structure
        plan = DAGPlan(steps=steps, problem="", phase1_values=phase1_values)
        is_valid, errors = plan.validate()
        if not is_valid:
            logger.warning(f"Parsed plan has validation errors: {'; '.join(errors)}")

        # Validate Phase 1 value references
        ref_valid, ref_errors = plan.validate_value_references()
        if not ref_valid:
            logger.warning(f"Phase 1 reference errors: {'; '.join(ref_errors)}")

        return steps, phase1_values

    def _parse_steps_fallback(self, response: str) -> tuple[list[Step], dict[str, Any]]:
        """Fallback parser for malformed JSON responses.

        Attempts to extract JSON from response that may have extra text.
        """
        import json

        # Try to find JSON object in response
        start = response.find("{")
        end = response.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                data = json.loads(response[start:end])
                phase1_values = data.get("values", {})
                steps = []

                for step_data in data.get("steps", []):
                    step = Step(
                        id=step_data.get("id", f"step_{len(steps) + 1}"),
                        task=step_data.get("task", ""),
                        depends_on=step_data.get("depends_on", []),
                        extracted_values=step_data.get("values", {}),
                        dsl_hint=step_data.get("dsl_hint"),
                        operation=step_data.get("operation"),
                    )
                    steps.append(step)

                logger.info("[planner] Fallback parser extracted %d steps", len(steps))
                return steps, phase1_values

            except json.JSONDecodeError:
                pass

        logger.warning("[planner] Fallback parser also failed")
        return [], {}


# Convenience function (uses TreeGuidedPlanner without vocabulary - fallback mode)
async def decompose(problem: str) -> DAGPlan:
    """Decompose a problem into steps.

    Note: For vocabulary-guided decomposition, instantiate TreeGuidedPlanner
    with step_db and embedder directly.
    """
    planner = TreeGuidedPlanner()  # No step_db = uses fallback mode
    return await planner.decompose(problem)


# =============================================================================
# TREE-GUIDED DECOMPOSITION (Segmentation LLM)
# =============================================================================
# Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE"
#
# Two-phase approach:
# 1. SEGMENT: Break problem into abstract steps (no values, just operations)
# 2. REFINE: Match steps to tree vocabulary, extract values
#
# This uses the signature tree to guide decomposition, ensuring we decompose
# INTO existing vocabulary when possible, only creating novel operations when
# truly needed (detected via Welford's algorithm on similarity distribution).
# =============================================================================

SEGMENT_SYSTEM = """You segment math problems into abstract operation steps. Output valid JSON only.

Your job is to identify WHAT OPERATIONS are needed, not the specific values.

OUTPUT FORMAT (JSON):
{
  "steps": [
    {
      "id": "step_1",
      "description": "brief description of what this step computes",
      "operation_type": "add|subtract|multiply|divide|percentage|compare|other",
      "depends_on": []
    }
  ]
}

RULES:
1. ONE OPERATION PER STEP - keep steps atomic
2. Focus on the OPERATION TYPE, not the specific numbers
3. Use depends_on to show which steps feed into others
4. description should be generic (e.g., "compute total cost" not "compute 80000 + 50000")

EXAMPLE:
Problem: "Josh buys a house for $80,000, puts in $50,000 repairs. This increased the value by 150%. How much profit?"

{
  "steps": [
    {"id": "step_1", "description": "compute total investment", "operation_type": "add", "depends_on": []},
    {"id": "step_2", "description": "compute new value after increase", "operation_type": "multiply", "depends_on": []},
    {"id": "step_3", "description": "compute profit from sale", "operation_type": "subtract", "depends_on": ["step_1", "step_2"]}
  ]
}

Output ONLY valid JSON. No explanation."""


REFINE_SYSTEM = """You refine abstract steps into concrete operations with values. Output valid JSON only.

You are given:
1. The original problem
2. Abstract steps (what operations are needed)
3. VOCABULARY: Suggested operations from our knowledge base for each step

PHASE 1: Extract ALL numeric values from the problem with semantic names AND UNITS.
PHASE 2: For each step, pick the best matching operation from VOCABULARY (or mark as "novel" if none fit).
PHASE 3: Fill in the parameter values using $name references.

CRITICAL: Unit Awareness
- Each value name MUST match its semantic meaning
- Include units in names: "distance_miles", "time_hours", "price_dollars", "weight_kg"
- NEVER mix units: don't assign a distance value to a time variable!
- Read the problem context carefully: "180 miles" is distance, "4 hours" is time

OUTPUT FORMAT (JSON):
{
  "values": {
    "semantic_name_unit": number,
    ...
  },
  "steps": [
    {
      "id": "step_1",
      "task": "concrete task description",
      "matched_operation": "operation_name from vocabulary OR null if novel",
      "is_novel": false,
      "operation": "add|subtract|multiply|divide",
      "values": {"param": "$semantic_name_unit OR {step_N}"},
      "dsl_hint": "+|-|*|/",
      "depends_on": []
    }
  ]
}

RULES:
1. PREFER vocabulary operations - only mark is_novel=true if NO vocabulary option fits
2. Use $name to reference Phase 1 values (e.g., "$purchase_price_dollars")
3. Use {step_N} to reference prior step results (e.g., "{step_1}")
4. matched_operation should be the exact name from VOCABULARY suggestions
5. VALUE NAMES MUST REFLECT WHAT THE NUMBER REPRESENTS - check units carefully!

Output ONLY valid JSON. No explanation."""


@dataclass
class AbstractStep:
    """An abstract step from segmentation (no values, just operation type)."""
    id: str
    description: str
    operation_type: str  # add, subtract, multiply, divide, percentage, compare, other
    depends_on: list[str] = field(default_factory=list)


@dataclass
class OperationSuggestion:
    """A vocabulary suggestion for an abstract step."""
    operation_name: str  # e.g., "compute_sum"
    similarity: float    # cosine similarity to step description
    signature_id: int    # ID in signature tree
    is_novel: bool = False  # True if below novelty threshold


@dataclass
class SegmentationResult:
    """Result of tree-guided segmentation."""
    abstract_steps: list[AbstractStep]
    suggestions: dict[str, list[OperationSuggestion]]  # step_id -> suggestions
    plan: Optional[DAGPlan] = None  # Final refined plan


class TreeGuidedPlanner:
    """Planner that uses signature tree vocabulary to guide decomposition.

    Per CLAUDE.md "New Favorite Pattern": This is the consolidated entry point
    for tree-guided decomposition. All decomposition should flow through here.

    Flow:
    1. segment(problem) -> abstract steps (1 LLM call)
    2. suggest_operations(steps) -> vocabulary matches (0 LLM calls, uses tree)
    3. refine(problem, steps, suggestions) -> concrete plan (1 LLM call)

    Total: 2 LLM calls regardless of step count (batched by design).
    """

    def __init__(
        self,
        step_db=None,  # StepSignatureDB for vocabulary lookup
        embedder=None,  # For embedding step descriptions
        model: str = PLANNER_DEFAULT_MODEL,
        temperature: float = PLANNER_DEFAULT_TEMPERATURE,
    ):
        self.step_db = step_db
        self.embedder = embedder
        self.client = get_client(model=model)
        self.temperature = temperature

        # Welford's stats for novelty detection (loaded from db)
        self._novelty_count = 0
        self._novelty_mean = 0.0
        self._novelty_m2 = 0.0
        self._novelty_k = TREE_GUIDED_NOVELTY_K  # Threshold = mean - k * stddev

    def _load_novelty_stats(self):
        """Load Welford's stats for novelty detection via data layer."""
        from mycelium.data_layer import get_segmentation_novelty_stats
        try:
            stats = get_segmentation_novelty_stats()
            if stats:
                self._novelty_count = stats.get('count', 0)
                self._novelty_mean = stats.get('mean', 0.0)
                self._novelty_m2 = stats.get('m2', 0.0)
                logger.debug(
                    "[planner] Loaded novelty stats: count=%d, mean=%.3f, stddev=%.3f",
                    self._novelty_count, self._novelty_mean, self._novelty_stddev
                )
        except Exception as e:
            logger.debug("[planner] Could not load novelty stats: %s", e)

    def _save_novelty_stats(self):
        """Save Welford's stats for novelty detection via data layer."""
        from mycelium.data_layer import save_segmentation_novelty_stats
        try:
            stats = {
                'count': self._novelty_count,
                'mean': self._novelty_mean,
                'm2': self._novelty_m2,
            }
            save_segmentation_novelty_stats(stats)
        except Exception as e:
            logger.debug("[planner] Could not save novelty stats: %s", e)

    @property
    def _novelty_stddev(self) -> float:
        """Compute standard deviation from Welford's M2."""
        if self._novelty_count < 2:
            return 0.3  # Default stddev during cold start
        return (self._novelty_m2 / self._novelty_count) ** 0.5

    @property
    def novelty_threshold(self) -> float:
        """Threshold below which a match is considered 'novel'.

        Uses Welford's algorithm: threshold = mean - k * stddev
        Per CLAUDE.md: No arbitrary magic numbers.
        """
        if self._novelty_count < TREE_GUIDED_NOVELTY_MIN_SAMPLES:
            # Cold start: use permissive default from config
            return TREE_GUIDED_NOVELTY_DEFAULT_THRESHOLD
        return max(0.3, self._novelty_mean - self._novelty_k * self._novelty_stddev)

    def _update_novelty_stats(self, similarity: float):
        """Update Welford's running stats with a new similarity observation."""
        self._novelty_count += 1
        delta = similarity - self._novelty_mean
        self._novelty_mean += delta / self._novelty_count
        delta2 = similarity - self._novelty_mean
        self._novelty_m2 += delta * delta2

    async def segment(self, problem: str) -> list[AbstractStep]:
        """Segment problem into abstract steps (Phase 1).

        This is a single LLM call that identifies WHAT operations are needed,
        without extracting specific values. The tree will guide value extraction.

        Args:
            problem: The math problem to segment

        Returns:
            List of AbstractStep objects describing needed operations
        """
        logger.debug("[planner] Segmenting problem: '%s...'", problem[:80])

        messages = [
            {"role": "system", "content": SEGMENT_SYSTEM},
            {"role": "user", "content": problem},
        ]

        response = await self.client.generate(
            messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        steps = self._parse_abstract_steps(response)
        logger.info("[planner] Segmentation complete: %d abstract steps", len(steps))
        return steps

    def _parse_abstract_steps(self, response: str) -> list[AbstractStep]:
        """Parse abstract steps from segmentation response."""
        import json
        steps = []

        try:
            data = json.loads(response.strip())
            for step_data in data.get("steps", []):
                step = AbstractStep(
                    id=step_data.get("id", f"step_{len(steps) + 1}"),
                    description=step_data.get("description", ""),
                    operation_type=step_data.get("operation_type", "other"),
                    depends_on=step_data.get("depends_on", []),
                )
                steps.append(step)
        except json.JSONDecodeError as e:
            logger.warning("[planner] Segment JSON parse failed: %s", e)
            # Fallback: single step
            steps = [AbstractStep(id="step_1", description="solve problem", operation_type="other")]

        return steps

    # Canonical computation graphs for each operation type
    # Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE"
    CANONICAL_GRAPHS = {
        "add": "ADD(param_0, param_1)",
        "subtract": "SUB(param_0, param_1)",
        "multiply": "MUL(param_0, param_1)",
        "divide": "DIV(param_0, param_1)",
        "percentage": "MUL(param_0, DIV(param_1, CONST(100)))",
        "compare": "COMPARE(param_0, param_1)",
        "other": None,  # Fall back to description embedding
    }

    async def suggest_operations_batch(
        self,
        abstract_steps: list[AbstractStep],
        top_k: int = TREE_GUIDED_TOP_K_SUGGESTIONS,
    ) -> dict[str, list[OperationSuggestion]]:
        """Find vocabulary matches for all abstract steps (batched).

        This uses the signature tree to suggest matching operations.
        No LLM calls - just embedding + tree routing.

        Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE"
        Uses GRAPH embeddings (canonical computation graphs) for operational similarity,
        NOT text embeddings of descriptions.

        Args:
            abstract_steps: List of abstract steps from segmentation
            top_k: Number of suggestions per step

        Returns:
            Dict mapping step_id -> list of OperationSuggestion
        """
        from mycelium.embedding_cache import cached_embed_batch
        from mycelium.step_signatures.graph_extractor import graph_to_natural_language

        if self.step_db is None or self.embedder is None:
            logger.warning("[planner] No step_db or embedder - skipping suggestions")
            return {step.id: [] for step in abstract_steps}

        # Load novelty stats for threshold calculation
        self._load_novelty_stats()

        suggestions = {}

        # Convert operation_types to canonical graph natural language descriptions
        # This is what we embed - the OPERATION, not the description
        graph_texts = []
        for step in abstract_steps:
            canonical_graph = self.CANONICAL_GRAPHS.get(step.operation_type)
            if canonical_graph:
                # Convert graph to natural language for embedding
                # e.g., "ADD(param_0, param_1)" -> "add first value and second value"
                nl_text = graph_to_natural_language(canonical_graph)
                graph_texts.append(nl_text)
                logger.debug(
                    "[planner] Step '%s' (%s) -> graph '%s' -> '%s'",
                    step.id, step.operation_type, canonical_graph, nl_text
                )
            else:
                # Unknown operation type - fall back to description
                graph_texts.append(step.description)
                logger.debug(
                    "[planner] Step '%s' (%s) -> fallback to description",
                    step.id, step.operation_type
                )

        # Batch embed all graph texts (with caching)
        embeddings_dict = cached_embed_batch(graph_texts, self.embedder)
        embeddings = [embeddings_dict[text] for text in graph_texts]

        # For each step, route through tree to find matching leaves by GRAPH embedding
        for step, embedding in zip(abstract_steps, embeddings):
            step_suggestions = []

            # Use tree routing to find matching operations
            # Per CLAUDE.md: route by graph_embedding (operational), not text centroid
            matches = self.step_db.match_step_to_leaves_mcts(
                operation_embedding=embedding,
                dag_step_type=step.operation_type,  # Use operation type, not description
                top_k=top_k,
                min_similarity=HINT_ALTERNATIVES_MIN_SIMILARITY,  # Permissive - we'll filter by novelty threshold
            )

            for sig, ucb1_score, similarity in matches:
                is_novel = similarity < self.novelty_threshold
                suggestion = OperationSuggestion(
                    operation_name=sig.step_type,
                    similarity=similarity,
                    signature_id=sig.id,
                    is_novel=is_novel,
                )
                step_suggestions.append(suggestion)

                # Update Welford's stats with this similarity (for learning threshold)
                if not is_novel:
                    self._update_novelty_stats(similarity)

            # If no matches found, mark as novel
            if not step_suggestions:
                step_suggestions.append(OperationSuggestion(
                    operation_name="novel_operation",
                    similarity=0.0,
                    signature_id=-1,
                    is_novel=True,
                ))

            suggestions[step.id] = step_suggestions
            logger.debug(
                "[planner] Step '%s' suggestions: %s",
                step.id,
                [(s.operation_name, f"{s.similarity:.2f}") for s in step_suggestions[:3]]
            )

        # Save updated novelty stats
        self._save_novelty_stats()

        return suggestions

    async def refine(
        self,
        problem: str,
        abstract_steps: list[AbstractStep],
        suggestions: dict[str, list[OperationSuggestion]],
    ) -> DAGPlan:
        """Refine abstract steps into concrete plan with values (Phase 2).

        This is a single LLM call that:
        1. Extracts Phase 1 values from the problem
        2. Matches each step to a vocabulary operation (or marks as novel)
        3. Fills in parameter values

        Args:
            problem: The original problem text
            abstract_steps: Abstract steps from segmentation
            suggestions: Vocabulary suggestions per step from tree routing

        Returns:
            Concrete DAGPlan ready for execution
        """
        logger.debug("[planner] Refining %d steps with vocabulary suggestions", len(abstract_steps))

        # Build vocabulary section for prompt
        vocabulary_text = self._format_vocabulary(abstract_steps, suggestions)

        # Build user message with problem + steps + vocabulary
        user_content = f"""PROBLEM:
{problem}

ABSTRACT STEPS:
{self._format_abstract_steps(abstract_steps)}

VOCABULARY (suggested operations for each step):
{vocabulary_text}

Refine these steps into concrete operations with values."""

        messages = [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        response = await self.client.generate(
            messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        steps, phase1_values = self._parse_refined_steps(response, suggestions)

        plan = DAGPlan(steps=steps, problem=problem, phase1_values=phase1_values)

        # Validate
        is_valid, errors = plan.validate()
        logger.info(
            "[planner] Refinement complete: %d steps, valid=%r, novel=%d",
            len(steps),
            is_valid,
            sum(1 for s in steps if getattr(s, '_is_novel', False))
        )
        if errors:
            logger.warning("[planner] Validation errors: %s", errors)

        return plan

    def _format_vocabulary(
        self,
        abstract_steps: list[AbstractStep],
        suggestions: dict[str, list[OperationSuggestion]],
    ) -> str:
        """Format vocabulary suggestions for the refine prompt."""
        lines = []
        for step in abstract_steps:
            step_suggestions = suggestions.get(step.id, [])
            if step_suggestions:
                suggestion_strs = [
                    f"{s.operation_name} (sim={s.similarity:.2f}{'*' if s.is_novel else ''})"
                    for s in step_suggestions[:3]
                ]
                lines.append(f"{step.id}: {', '.join(suggestion_strs)}")
            else:
                lines.append(f"{step.id}: [no suggestions - create novel operation]")
        return "\n".join(lines)

    def _format_abstract_steps(self, abstract_steps: list[AbstractStep]) -> str:
        """Format abstract steps for the refine prompt."""
        lines = []
        for step in abstract_steps:
            deps = f" (depends: {step.depends_on})" if step.depends_on else ""
            lines.append(f"{step.id}: {step.description} [{step.operation_type}]{deps}")
        return "\n".join(lines)

    def _parse_refined_steps(
        self,
        response: str,
        suggestions: dict[str, list[OperationSuggestion]],
    ) -> tuple[list[Step], dict[str, Any]]:
        """Parse refined steps from LLM response."""
        import json
        steps = []
        phase1_values = {}

        try:
            data = json.loads(response.strip())
            phase1_values = data.get("values", {})

            if phase1_values:
                logger.info(
                    "[planner] Phase 1 values extracted: %s",
                    ", ".join(f"{k}={v}" for k, v in phase1_values.items())
                )

            for step_data in data.get("steps", []):
                step_id = step_data.get("id", f"step_{len(steps) + 1}")

                # Track if this step matched vocabulary or is novel
                matched_op = step_data.get("matched_operation")
                is_novel = step_data.get("is_novel", matched_op is None)

                # Find signature_id from suggestions if matched
                signature_id = None
                if matched_op and step_id in suggestions:
                    for s in suggestions[step_id]:
                        if s.operation_name == matched_op:
                            signature_id = s.signature_id
                            break

                step = Step(
                    id=step_id,
                    task=step_data.get("task", ""),
                    depends_on=step_data.get("depends_on", []),
                    extracted_values=step_data.get("values", {}),
                    dsl_hint=step_data.get("dsl_hint"),
                    operation=step_data.get("operation"),
                )
                # Store metadata for routing
                step._is_novel = is_novel
                step._matched_operation = matched_op
                step._suggested_signature_id = signature_id

                steps.append(step)

        except json.JSONDecodeError as e:
            logger.warning("[planner] Refine JSON parse failed: %s", e)
            steps = [Step(id="solve", task="Solve the problem directly", depends_on=[])]

        return steps, phase1_values

    async def decompose(
        self,
        problem: str,
        signature_hints: Optional[list[SignatureHint]] = None,
        context: Optional[str] = None,
        skip_validation: bool = False,
    ) -> DAGPlan:
        """Decompose a problem into a DAG of steps (API-compatible with Planner).

        Uses tree-guided decomposition when step_db/embedder are available
        AND no context/signature_hints are provided. Otherwise falls back to
        standard single-LLM-call approach via composition with Planner.

        Args:
            problem: The problem to decompose
            signature_hints: Optional list of SignatureHint objects (triggers fallback)
            context: Optional additional context (triggers fallback)
            skip_validation: If True, skip data flow validation

        Returns:
            DAGPlan with steps and dependencies
        """
        # Use tree-guided when we have step_db/embedder AND no special params
        # signature_hints and context require the old Planner's prompt format
        use_tree_guided = (
            self.step_db is not None
            and self.embedder is not None
            and not context
            and not signature_hints
        )

        if use_tree_guided:
            result = await self.decompose_guided(problem)
            return result.plan

        # Fallback: compose with Planner (per CLAUDE.md "New Favorite Pattern")
        # Avoid code duplication by reusing existing Planner logic
        logger.info("[planner] Using fallback decomposition (context=%s, hints=%s)",
                   bool(context), bool(signature_hints))
        fallback = _create_planner_without_warning(self.client, self.temperature)
        return await fallback.decompose(problem, signature_hints, context, skip_validation)

    async def decompose_guided(self, problem: str) -> SegmentationResult:
        """Full tree-guided decomposition with intermediate results.

        Per CLAUDE.md "New Favorite Pattern": This is the single entry point
        for decomposition that uses tree vocabulary to guide the process.

        Per CLAUDE.md "Negotiation between Tree and Planner":
        When enabled, uses iterative negotiation to refine dag_steps based
        on tree vocabulary, biasing towards dag_step decomposition (cheap)
        over leaf_node decomposition (permanent tree change).

        Flow (standard):
        1. segment(problem) -> abstract steps (1 LLM call)
        2. suggest_operations_batch(steps) -> vocabulary matches (0 LLM calls)
        3. refine(problem, steps, suggestions) -> concrete plan (1 LLM call)

        Flow (negotiated):
        1. segment(problem) -> abstract steps (1 LLM call)
        2. Tree evaluates matches, provides decomposition hints
        3. Planner refines poor-matching steps using hints (0-1 LLM calls)
        4. Iterate until good matches or max rounds
        5. Final refine (1 LLM call)

        Args:
            problem: The math problem to decompose

        Returns:
            SegmentationResult with abstract steps, suggestions, and final plan
        """
        from mycelium.config import (
            TREE_PLANNER_NEGOTIATION_ENABLED,
            TREE_PLANNER_NEGOTIATION_MAX_ROUNDS,
            TREE_PLANNER_NEGOTIATION_SIMILARITY_THRESHOLD,
        )

        # Use negotiation when enabled and we have step_db
        if TREE_PLANNER_NEGOTIATION_ENABLED and self.step_db is not None:
            logger.info("[planner] Using Tree-Planner negotiation")
            return await self.decompose_negotiated(
                problem,
                max_rounds=TREE_PLANNER_NEGOTIATION_MAX_ROUNDS,
                similarity_threshold=TREE_PLANNER_NEGOTIATION_SIMILARITY_THRESHOLD,
            )

        # Standard flow (no negotiation)
        logger.info("[planner] Tree-guided decomposition starting (no negotiation)")

        # Phase 1: Segment into abstract steps
        abstract_steps = await self.segment(problem)

        # Phase 2: Get vocabulary suggestions from tree (no LLM)
        suggestions = await self.suggest_operations_batch(abstract_steps)

        # Phase 3: Refine with vocabulary guidance
        plan = await self.refine(problem, abstract_steps, suggestions)

        # Log novelty metrics
        novel_count = sum(1 for s in plan.steps if getattr(s, '_is_novel', False))
        vocab_count = len(plan.steps) - novel_count
        logger.info(
            "[planner] Tree-guided decomposition complete: %d steps (%d vocabulary, %d novel)",
            len(plan.steps), vocab_count, novel_count
        )

        return SegmentationResult(
            abstract_steps=abstract_steps,
            suggestions=suggestions,
            plan=plan,
        )

    async def decompose_negotiated(
        self,
        problem: str,
        max_rounds: int = 2,
        similarity_threshold: float = TREE_PLANNER_NEGOTIATION_SIMILARITY_THRESHOLD,
    ) -> SegmentationResult:
        """Tree-Planner negotiation for dag_step refinement.

        Per CLAUDE.md "Negotiation between Tree and Planner":
        Back and forth negotiation with bias towards decomposing dag_steps
        (cheap, per-problem) over leaf_nodes (permanent tree change).

        Flow:
        1. Segment problem into abstract steps
        2. Tree evaluates matches
        3. For poor matches: Tree provides decomposition hints
        4. Planner refines those dag_steps using hints
        5. Iterate until good matches or max_rounds

        Args:
            problem: The math problem to decompose
            max_rounds: Maximum negotiation rounds (default 2)
            similarity_threshold: Below this, step needs decomposition

        Returns:
            SegmentationResult with final plan
        """
        from mycelium.embedding_cache import cached_embed_batch
        from mycelium.step_signatures.graph_extractor import graph_to_natural_language

        logger.info("[planner] Starting Tree-Planner negotiation (max_rounds=%d)", max_rounds)

        # Phase 1: Initial segmentation
        abstract_steps = await self.segment(problem)

        for round_num in range(max_rounds):
            # Phase 2: Get operation embeddings
            graph_texts = []
            for step in abstract_steps:
                canonical_graph = self.CANONICAL_GRAPHS.get(step.operation_type)
                if canonical_graph:
                    graph_texts.append(graph_to_natural_language(canonical_graph))
                else:
                    graph_texts.append(step.description)

            embeddings_dict = cached_embed_batch(graph_texts, self.embedder)
            embeddings = [embeddings_dict[text] for text in graph_texts]

            # Validate embedding dimensions
            from mycelium.config import EMBEDDING_DIM
            for i, (text, emb) in enumerate(zip(graph_texts, embeddings)):
                if emb is None or emb.shape[0] != EMBEDDING_DIM:
                    logger.warning(
                        "[planner] Bad embedding for step %d: shape=%s, text='%s...'",
                        i, emb.shape if emb is not None else None, text[:50]
                    )

            # Phase 3: Tree evaluates matches and provides hints
            # Per CLAUDE.md "Cluster Boundaries": Welford stats guide accept/reject
            poor_matches = []
            all_hints = {}

            for step_idx, (step, embedding) in enumerate(zip(abstract_steps, embeddings)):
                # step_position is 1-indexed (step_1, step_2, etc.)
                step_position = step_idx + 1
                hints = self.step_db.get_decomposition_hints(
                    step_description=step.description,
                    operation_embedding=embedding,
                    similarity_threshold=similarity_threshold,
                    step_position=step_position,
                )
                all_hints[step.id] = hints

                if hints['needs_decomposition']:
                    poor_matches.append((step, hints))
                    logger.debug(
                        "[planner] Step %d '%s' needs decomposition: %s",
                        step_position, step.description[:30], hints.get('rejection_reason')
                    )

            # If all matches are good, we're done negotiating
            if not poor_matches:
                logger.info(
                    "[planner] Negotiation round %d: All %d steps have good matches",
                    round_num + 1, len(abstract_steps)
                )
                break

            logger.info(
                "[planner] Negotiation round %d: %d/%d steps need decomposition",
                round_num + 1, len(poor_matches), len(abstract_steps)
            )

            # Phase 4: Re-segment poor matches with vocabulary hints
            if round_num < max_rounds - 1:  # Don't refine on last round
                abstract_steps = await self._refine_abstract_steps_with_hints(
                    problem, abstract_steps, poor_matches
                )

        # Phase 5: Stage proposals for steps that couldn't be refined
        # Per CLAUDE.md: When refinement fails, stage for periodic tree review
        if poor_matches and self.step_db is not None:
            logger.info(
                "[planner] Staging %d proposals (refinement couldn't simplify further)",
                len(poor_matches)
            )
            for step, hints in poor_matches:
                # Get embedding for this step
                canonical_graph = self.CANONICAL_GRAPHS.get(step.operation_type)
                if canonical_graph:
                    from mycelium.step_signatures.graph_extractor import graph_to_natural_language
                    graph_text = graph_to_natural_language(canonical_graph)
                else:
                    graph_text = step.description

                graph_embedding = cached_embed_batch([graph_text], self.embedder).get(graph_text)
                if graph_embedding is not None:
                    graph_embedding = np.array(graph_embedding)

                # Extract best match info from hints
                best_match = hints.get('best_match')
                best_match_id = best_match[2] if best_match and len(best_match) > 2 else None
                best_match_sim = best_match[1] if best_match and len(best_match) > 1 else None

                # Stage the proposal
                self.step_db.stage_proposal(
                    step_text=step.description,
                    graph_embedding=graph_embedding,
                    proposed_parent_id=None,  # Let tree review decide
                    best_match_id=best_match_id,
                    best_match_sim=best_match_sim,
                    dsl_hint=step.operation_type,
                    problem_context=problem[:200] if problem else None,
                    rejection_reason=hints.get('rejection_reason'),
                )

        # Final phase: Get suggestions and refine to concrete plan
        suggestions = await self.suggest_operations_batch(abstract_steps)
        plan = await self.refine(problem, abstract_steps, suggestions)

        # Log negotiation outcome
        novel_count = sum(1 for s in plan.steps if getattr(s, '_is_novel', False))
        logger.info(
            "[planner] Negotiation complete: %d steps (%d vocabulary, %d novel)",
            len(plan.steps), len(plan.steps) - novel_count, novel_count
        )

        return SegmentationResult(
            abstract_steps=abstract_steps,
            suggestions=suggestions,
            plan=plan,
        )

    async def _refine_abstract_steps_with_hints(
        self,
        problem: str,
        abstract_steps: list[AbstractStep],
        poor_matches: list[tuple[AbstractStep, dict]],
    ) -> list[AbstractStep]:
        """Re-segment steps that had poor matches using tree vocabulary hints.

        Per CLAUDE.md: Bias towards decomposing dag_steps (cheap) over
        leaf_nodes (permanent). Uses existing vocabulary to guide decomposition.

        Args:
            problem: Original problem text
            abstract_steps: Current abstract steps
            poor_matches: List of (step, hints) for steps needing decomposition

        Returns:
            Updated list of abstract steps
        """
        if not poor_matches:
            return abstract_steps

        # Build refinement prompt with vocabulary hints
        poor_step_ids = {step.id for step, _ in poor_matches}

        # Collect vocabulary hints
        vocab_hints = []
        for step, hints in poor_matches:
            vocab_str = ", ".join(hints.get('suggested_decomposition', []))
            if vocab_str:
                vocab_hints.append(f"- '{step.description}' -> consider decomposing into: {vocab_str}")

        if not vocab_hints:
            return abstract_steps

        # Ask LLM to refine the problematic steps
        refine_prompt = f"""You previously segmented this problem:

PROBLEM:
{problem}

CURRENT STEPS:
{chr(10).join(f"- {s.id}: {s.description} ({s.operation_type})" for s in abstract_steps)}

The following steps are too complex and don't match our vocabulary well:
{chr(10).join(vocab_hints)}

Please re-segment ONLY the problematic steps into smaller, atomic operations.
Keep the good steps as-is. Return the updated step list.

Respond in JSON:
{{"steps": [{{"id": "step_1", "description": "...", "operation_type": "add|subtract|multiply|divide|other", "depends_on": []}}]}}
"""

        messages = [
            {"role": "system", "content": "You help decompose complex steps into simpler operations. Keep atomic steps that work well."},
            {"role": "user", "content": refine_prompt},
        ]

        try:
            response = await self.client.generate(
                messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            refined_steps = self._parse_abstract_steps(response)

            if refined_steps and len(refined_steps) >= len(abstract_steps):
                logger.info(
                    "[planner] Refined %d steps -> %d steps using vocabulary hints",
                    len(abstract_steps), len(refined_steps)
                )
                return refined_steps

        except Exception as e:
            logger.warning("[planner] Refinement with hints failed: %s", e)

        return abstract_steps
