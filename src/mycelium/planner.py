"""Planner: Decompose problems into DAG of subtasks.

The planner breaks down complex problems into simpler steps,
creating a directed acyclic graph (DAG) of subtasks.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .client import get_client
from mycelium.config import PLANNER_DEFAULT_MODEL, PLANNER_DEFAULT_TEMPERATURE


PLANNER_SYSTEM = """You are a mathematical problem decomposer. Your ONLY job is to break down problems into steps and extract numeric values. DO NOT solve the problem.

CRITICAL: Output ONLY in this exact format. No markdown headers, no explanations, no answers.

- id: step_1
  task: [what to do, not how to do it]
  values:
    semantic_name: numeric_value
  depends_on: []

- id: step_2
  task: [what to do]
  values:
    input_value: "{step_1}"
    other_value: 123
  depends_on: [step_1]

- id: final
  task: Combine results to get final answer
  values: {}
  depends_on: [step_2]

RULES:
1. Output ONLY the step list above. Nothing else.
2. DO NOT use "## Step 1:" format. Use "- id: step_1" format.
3. DO NOT solve the problem or compute answers.
4. DO NOT include "The final answer is" or any boxed answers.
5. Each task describes WHAT to do, not the solution.
6. Keep to 3-6 steps.
7. EXTRACT numeric values from the problem and assign semantic names.
8. Use "{step_N}" to reference the result of a previous step.

EXAMPLE for "Earth's circumference is 40,000 km. How many trips for 1 billion meters?":

- id: step_1
  task: Convert 1 billion meters to kilometers
  values:
    distance_meters: 1000000000
    meters_per_km: 1000
  depends_on: []

- id: step_2
  task: Divide total distance by circumference
  values:
    total_km: "{step_1}"
    circumference_km: 40000
  depends_on: [step_1]

GOOD task: "Solve the quadratic equation 2x^2 - 7x + 2 = 0 for its roots"
BAD task: "The roots are x = 1/2 and x = 2" (this is solving, not decomposing)
"""

SIGNATURE_HINTS_TEMPLATE = """
## Available Atomic Operations

The system has learned these reusable step patterns. When possible, decompose into steps that match these operations:

{hints}

Prefer these known patterns when they fit the problem. Use "general reasoning" only when no pattern applies.
"""


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
    # Recursive nesting: sub-plan for composite steps
    sub_plan: Optional["DAGPlan"] = None

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

    def validate(self, recursive: bool = True) -> tuple[bool, list[str]]:
        """Validate the DAG structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        step_ids = {s.id for s in self.steps}

        # Check for missing dependencies
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on unknown step '{dep}'")

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


class Planner:
    """Decompose problems into DAG of subtasks."""

    def __init__(
        self,
        model: str = PLANNER_DEFAULT_MODEL,
        temperature: float = PLANNER_DEFAULT_TEMPERATURE,
    ):
        self.client = get_client(model=model)
        self.temperature = temperature

    async def decompose(
        self,
        problem: str,
        signature_hints: Optional[list[tuple[str, str]]] = None
    ) -> DAGPlan:
        """Decompose a problem into a DAG of steps.

        Args:
            problem: The problem to decompose
            signature_hints: Optional list of (step_type, description) tuples
                            to guide decomposition toward known patterns

        Returns:
            DAGPlan with steps and dependencies
        """
        logger.debug("[planner] Starting decomposition: problem='%s...'", problem[:80])

        # Build system prompt with optional signature hints
        system_prompt = PLANNER_SYSTEM
        if signature_hints:
            hints_text = "\n".join(
                f"- **{step_type}**: {desc}"
                for step_type, desc in signature_hints
            )
            system_prompt += SIGNATURE_HINTS_TEMPLATE.format(hints=hints_text)
            logger.debug("[planner] Added %d signature hints", len(signature_hints))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]

        response = await self.client.generate(messages, temperature=self.temperature)
        steps = self._parse_steps(response)

        plan = DAGPlan(steps=steps, problem=problem)
        is_valid, errors = plan.validate()
        logger.info(
            "[planner] Decomposition complete: steps=%d valid=%r",
            len(steps), is_valid
        )
        if errors:
            logger.warning("[planner] Validation errors: %s", errors)

        return plan

    def _parse_steps(self, response: str) -> list[Step]:
        """Parse steps from planner response.

        Logs warnings when parsing issues are detected (empty tasks, etc.)
        """
        steps = []
        current_step = None
        parse_warnings = []
        parsing_values = False  # Track if we're inside a values: block

        lines = response.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # New step - flexible matching for various formats
            # Matches: "- id: step_1", "-id: step_1", "id: step_1", "- id:step_1"
            id_match = re.match(r'^-?\s*id\s*:\s*(.+)$', stripped, re.IGNORECASE)
            if id_match:
                parsing_values = False
                if current_step:
                    # Validate before appending
                    if not current_step.task:
                        parse_warnings.append(f"Step '{current_step.id}' has empty task")
                    steps.append(current_step)
                step_id = id_match.group(1).strip()
                if not step_id:
                    parse_warnings.append("Found step with empty ID, using 'unnamed'")
                    step_id = f"unnamed_{len(steps)}"
                elif not step_id.replace("_", "").isalnum():
                    parse_warnings.append(f"Step ID '{step_id}' contains invalid characters, sanitizing")
                    step_id = "".join(c if c.isalnum() or c == "_" else "_" for c in step_id)
                current_step = Step(id=step_id, task="", depends_on=[])
                i += 1
                continue

            # Task description - flexible matching
            if re.match(r'^\s*task\s*:', stripped, re.IGNORECASE) and current_step:
                parsing_values = False
                current_step.task = stripped.split(":", 1)[1].strip()
                i += 1
                continue

            # Values block start
            if re.match(r'^\s*values\s*:', stripped, re.IGNORECASE) and current_step:
                parsing_values = True
                # Check for inline empty dict: "values: {}"
                rest = stripped.split(":", 1)[1].strip()
                if rest == "{}" or rest == "":
                    current_step.extracted_values = {}
                i += 1
                continue

            # Inside values block - parse indented key: value pairs
            if parsing_values and current_step and stripped and not re.match(r'^-?\s*(id|task|depends_on)\s*:', stripped, re.IGNORECASE):
                # Parse key: value
                if ":" in stripped:
                    key, val = stripped.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Try to parse as number
                    try:
                        if "." in val:
                            current_step.extracted_values[key] = float(val)
                        else:
                            current_step.extracted_values[key] = int(val)
                    except ValueError:
                        # Keep as string (could be "{step_1}" reference)
                        current_step.extracted_values[key] = val
                i += 1
                continue

            # Dependencies - flexible matching
            if re.match(r'^\s*depends_on\s*:', stripped, re.IGNORECASE) and current_step:
                parsing_values = False
                deps_str = stripped.split(":", 1)[1].strip()
                # Parse list like [step_1, step_2] or []
                deps_str = deps_str.strip("[]")
                if deps_str:
                    deps = [d.strip() for d in deps_str.split(",")]
                    current_step.depends_on = [d for d in deps if d]
                i += 1
                continue

            # DSL hint - optional
            if re.match(r'^\s*dsl_hint\s*:', stripped, re.IGNORECASE) and current_step:
                parsing_values = False
                current_step.dsl_hint = stripped.split(":", 1)[1].strip()
                i += 1
                continue

            i += 1

        if current_step:
            if not current_step.task:
                parse_warnings.append(f"Step '{current_step.id}' has empty task")
            steps.append(current_step)

        # Log any parsing warnings
        if parse_warnings:
            logger.warning(f"Planner parsing issues: {'; '.join(parse_warnings)}")

        # Ensure we have at least one step
        if not steps:
            logger.warning("No steps parsed from planner response, using fallback single step")
            logger.warning(f"[planner] Raw LLM response that failed parsing:\n{response}")
            steps = [Step(id="solve", task="Solve the problem directly", depends_on=[])]

        # Validate the resulting plan structure
        plan = DAGPlan(steps=steps, problem="")
        is_valid, errors = plan.validate()
        if not is_valid:
            logger.warning(f"Parsed plan has validation errors: {'; '.join(errors)}")

        return steps


# Convenience function
async def decompose(problem: str) -> DAGPlan:
    """Decompose a problem into steps."""
    planner = Planner()
    return await planner.decompose(problem)
