"""Planner: Decompose problems into DAG of subtasks.

The planner breaks down complex problems into simpler steps,
creating a directed acyclic graph (DAG) of subtasks.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

from .client import GroqClient
from mycelium.config import PLANNER_DEFAULT_MODEL, PLANNER_DEFAULT_TEMPERATURE


PLANNER_SYSTEM = """You are a mathematical problem decomposer. Break down the given problem into clear, sequential steps.

Each step should be:
1. A single, focused operation
2. Dependent only on previous steps (no circular dependencies)
3. Small enough to solve directly

Output format (YAML-like):
STEPS:
- id: step_1
  task: [describe what this step does]
  depends_on: []  # or list of step IDs like [step_1, step_2]

- id: step_2
  task: [describe what this step does]
  depends_on: [step_1]

...

- id: final
  task: [combine results to get final answer]
  depends_on: [step_N]

Keep it to 2-6 steps for most problems. Only use more for genuinely complex multi-part problems.
"""

SIGNATURE_HINTS_TEMPLATE = """
## Available Atomic Operations

The system has learned these reusable step patterns. When possible, decompose into steps that match these operations:

{hints}

Prefer these known patterns when they fit the problem. Use "general reasoning" only when no pattern applies.
"""


@dataclass
class Step:
    """A single step in the decomposition DAG."""
    id: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    result: Optional[str] = None
    success: bool = False


class PlanValidationError(Exception):
    """Raised when a plan has invalid structure (cycles, missing deps, etc.)."""
    pass


@dataclass
class DAGPlan:
    """A decomposition plan as a DAG of steps."""
    steps: list[Step]
    problem: str

    def validate(self) -> tuple[bool, list[str]]:
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
        self.client = GroqClient(model=model)
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

        for line in response.split("\n"):
            line = line.strip()

            # New step
            if line.startswith("- id:"):
                if current_step:
                    # Validate before appending
                    if not current_step.task:
                        parse_warnings.append(f"Step '{current_step.id}' has empty task")
                    steps.append(current_step)
                step_id = line.split(":", 1)[1].strip()
                if not step_id:
                    parse_warnings.append("Found step with empty ID, using 'unnamed'")
                    step_id = f"unnamed_{len(steps)}"
                elif not step_id.replace("_", "").isalnum():
                    parse_warnings.append(f"Step ID '{step_id}' contains invalid characters, sanitizing")
                    step_id = "".join(c if c.isalnum() or c == "_" else "_" for c in step_id)
                current_step = Step(id=step_id, task="", depends_on=[])

            # Task description
            elif line.startswith("task:") and current_step:
                current_step.task = line.split(":", 1)[1].strip()

            # Dependencies
            elif line.startswith("depends_on:") and current_step:
                deps_str = line.split(":", 1)[1].strip()
                # Parse list like [step_1, step_2] or []
                deps_str = deps_str.strip("[]")
                if deps_str:
                    deps = [d.strip() for d in deps_str.split(",")]
                    current_step.depends_on = [d for d in deps if d]

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
