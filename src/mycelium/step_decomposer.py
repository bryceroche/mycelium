"""Step Decomposer: Break complex steps into atomic sub-steps.

When a DSL has low confidence for a step, we decompose it further
until we reach truly atomic operations that DSLs can handle reliably.

This enables recursive decomposition:
1. Problem → DAG of steps (planner)
2. Step has low DSL confidence → decompose into sub-steps
3. Repeat until atomic (high DSL confidence or max depth)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .client import GroqClient
from .planner import Step, DAGPlan
from .config import (
    PLANNER_DEFAULT_MODEL,
    PLANNER_DEFAULT_TEMPERATURE,
    RECURSIVE_MAX_DEPTH,
    RECURSIVE_CONFIDENCE_THRESHOLD,
    RECURSIVE_DECOMPOSITION_ENABLED,
)

logger = logging.getLogger(__name__)

# Use config values for consistency
# If decomposition is disabled (aggressive injection mode), set depth to 0 to prevent any decomposition
MAX_DECOMPOSITION_DEPTH = RECURSIVE_MAX_DEPTH if RECURSIVE_DECOMPOSITION_ENABLED else 0
DECOMPOSITION_CONFIDENCE_THRESHOLD = RECURSIVE_CONFIDENCE_THRESHOLD

STEP_DECOMPOSER_SYSTEM = """You are a mathematical step decomposer. Given a complex step that couldn't be executed directly, break it into simpler atomic sub-steps.

CRITICAL: Each sub-step should be simple enough that it can be computed with basic operations (arithmetic, simple algebra, applying a formula).

Output format - ONLY output steps in this exact format:

- id: sub_1
  task: [simple atomic operation]
  depends_on: []

- id: sub_2
  task: [simple atomic operation]
  depends_on: [sub_1]

- id: final
  task: Combine results
  depends_on: [sub_2]

RULES:
1. Break into 2-4 sub-steps MAXIMUM
2. Each sub-step should be ONE atomic operation (add, multiply, solve simple equation, etc.)
3. Sub-steps should be simpler than the original step
4. Use clear, specific task descriptions
5. DO NOT solve - only decompose

GOOD atomic tasks:
- "Calculate 5 * 3"
- "Solve x + 2 = 7 for x"
- "Compute 2^10"
- "Add the results from sub_1 and sub_2"

BAD tasks (too complex):
- "Solve the quadratic equation and find all roots"
- "Set up and solve the system of equations"
- "Apply the formula and simplify"
"""

STEP_DECOMPOSER_USER = """The following step could not be executed directly (DSL confidence too low).
Break it into simpler atomic sub-steps.

Original step: {step_task}

Context from previous steps:
{context}

Decompose into 2-4 atomic sub-steps:"""


@dataclass
class DecomposedStep:
    """Result of decomposing a complex step."""
    original_step: Step
    sub_steps: list[Step]
    depth: int  # Current decomposition depth

    @property
    def is_atomic(self) -> bool:
        """A step is atomic if it couldn't be decomposed further."""
        return len(self.sub_steps) == 0


class StepDecomposer:
    """Decompose complex steps into atomic sub-steps."""

    def __init__(
        self,
        model: str = PLANNER_DEFAULT_MODEL,
        temperature: float = PLANNER_DEFAULT_TEMPERATURE,
    ):
        self.client = GroqClient(model=model)
        self.temperature = temperature

    async def decompose_step(
        self,
        step: Step,
        context: dict[str, str],
        depth: int = 0,
    ) -> DecomposedStep:
        """Decompose a single step into sub-steps.

        Args:
            step: The step to decompose
            context: Results from previous steps
            depth: Current decomposition depth

        Returns:
            DecomposedStep with sub-steps (empty if step is atomic/max depth)
        """
        # Check max depth
        if depth >= MAX_DECOMPOSITION_DEPTH:
            logger.debug("[decomposer] Max depth reached for step=%s", step.id)
            return DecomposedStep(original_step=step, sub_steps=[], depth=depth)

        # Format context
        ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items()) if context else "None"

        # Call LLM to decompose
        messages = [
            {"role": "system", "content": STEP_DECOMPOSER_SYSTEM},
            {"role": "user", "content": STEP_DECOMPOSER_USER.format(
                step_task=step.task,
                context=ctx_str,
            )},
        ]

        try:
            response = await self.client.generate(messages, temperature=self.temperature)
            sub_steps = self._parse_sub_steps(response, step.id)

            if len(sub_steps) <= 1:
                # Couldn't decompose further - this is atomic
                logger.debug("[decomposer] Step is atomic: step=%s", step.id)
                return DecomposedStep(original_step=step, sub_steps=[], depth=depth)

            logger.info(
                "[decomposer] Decomposed step=%s into %d sub-steps at depth=%d",
                step.id, len(sub_steps), depth
            )
            return DecomposedStep(original_step=step, sub_steps=sub_steps, depth=depth)

        except Exception as e:
            logger.warning("[decomposer] Failed to decompose step=%s: %s", step.id, e)
            return DecomposedStep(original_step=step, sub_steps=[], depth=depth)

    def _parse_sub_steps(self, response: str, parent_id: str) -> list[Step]:
        """Parse sub-steps from LLM response."""
        steps = []
        current_step = None

        for line in response.split("\n"):
            line = line.strip()

            # New step
            id_match = re.match(r'^-?\s*id\s*:\s*(.+)$', line, re.IGNORECASE)
            if id_match:
                if current_step:
                    steps.append(current_step)
                step_id = id_match.group(1).strip()
                # Prefix with parent ID to avoid collisions
                full_id = f"{parent_id}_{step_id}"
                current_step = Step(id=full_id, task="", depends_on=[])

            # Task description
            elif re.match(r'^\s*task\s*:', line, re.IGNORECASE) and current_step:
                current_step.task = line.split(":", 1)[1].strip()

            # Dependencies
            elif re.match(r'^\s*depends_on\s*:', line, re.IGNORECASE) and current_step:
                deps_str = line.split(":", 1)[1].strip().strip("[]")
                if deps_str:
                    # Prefix dependencies with parent ID
                    deps = [f"{parent_id}_{d.strip()}" for d in deps_str.split(",") if d.strip()]
                    current_step.depends_on = deps

        if current_step:
            steps.append(current_step)

        # Filter out steps with empty tasks
        steps = [s for s in steps if s.task]

        return steps


# Convenience function
async def decompose_step(
    step: Step,
    context: dict[str, str],
    depth: int = 0,
) -> DecomposedStep:
    """Decompose a step into sub-steps."""
    decomposer = StepDecomposer()
    return await decomposer.decompose_step(step, context, depth)
