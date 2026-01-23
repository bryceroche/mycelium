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


PLANNER_SYSTEM = """You decompose math problems into atomic steps. Output valid JSON only.

PHASE 1: Extract ALL numeric values with semantic names.
PHASE 2: Build atomic steps that reference Phase 1 values.

OUTPUT FORMAT (JSON):
{
  "values": {
    "name": number,
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
2. Reference Phase 1 values with $name (e.g., "$purchase_price")
3. Reference prior step results with {step_N} (e.g., "{step_1}")
4. NEVER use raw numbers in steps - always reference $names
5. For "increased by X%": extract multiplier (1 + X/100) in Phase 1

PERCENTAGE HANDLING:
- "X increased by Y%" → base is X, multiply by (1 + Y/100)
- "Y% of X" → base is X, multiply by Y/100

ALGEBRA (work backwards):
- If given a RESULT and asked for INPUT, reverse operations
- add↔subtract, multiply↔divide

EXAMPLE:
Problem: "Josh buys a house for $80,000, puts in $50,000 repairs. This increased the value by 150%. How much profit?"

{
  "values": {
    "purchase_price": 80000,
    "repair_cost": 50000,
    "increase_multiplier": 2.5
  },
  "steps": [
    {
      "id": "step_1",
      "task": "Calculate total investment",
      "operation": "add",
      "values": {"a": "$purchase_price", "b": "$repair_cost"},
      "dsl_hint": "+",
      "depends_on": []
    },
    {
      "id": "step_2",
      "task": "Calculate new house value",
      "operation": "multiply",
      "values": {"base": "$purchase_price", "multiplier": "$increase_multiplier"},
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
        # variables - these are values the LLM assumed exist but weren't extracted.
        for step in self.steps:
            if not step.extracted_values:
                continue
            for key, value in step.extracted_values.items():
                if not isinstance(value, str):
                    continue  # Numbers are valid
                # Check if it's a step reference - those are valid
                if step_ref_pattern.search(value):
                    continue  # Step references like {step_1} are valid
                # Check if it looks like a variable name (not a number)
                # This catches cases like "total_vacuums" when we don't know the total
                if variable_pattern.match(value):
                    # This is a variable name that wasn't resolved to a number
                    # or step reference - likely an undefined value
                    errors.append(
                        f"Step '{step.id}' has undefined variable '{value}' for input '{key}'. "
                        f"This value must be either a number from the problem or a {{step_N}} reference."
                    )
                    logger.warning(
                        "[planner] DATA FLOW ERROR: Step '%s' uses undefined variable '%s' for '%s'. "
                        "The problem likely requires algebra (working backwards) or the value wasn't extracted in Phase 1.",
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


# Convenience function
async def decompose(problem: str) -> DAGPlan:
    """Decompose a problem into steps."""
    planner = Planner()
    return await planner.decompose(problem)
