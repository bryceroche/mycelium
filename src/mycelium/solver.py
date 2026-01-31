"""Solver: Minimal implementation for local decomposition architecture.

Core loop:
1. Decompose problem into atomic steps (depth=1) via GTS
2. For each step: find existing signature or create new one
3. Execute DSL with resolved values
4. Record success/failure for Welford stats

Per CLAUDE.md Big 3:
- System Independence: Let tree self-organize via stats
- New Favorite Pattern: Single pathway via find_or_create()
- The Flow: DB Stats → Welford → Tree Structure
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from mycelium.config import DB_PATH
from mycelium.plan_models import Step, DAGPlan
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.function_registry import call_function, get_function_info
from mycelium.embedding_cache import cached_embed
from mycelium.answer_norm import normalize_answer
from mycelium.data_layer.mcts import (
    create_dag,
    create_dag_steps,
    grade_dag,
    run_postmortem,
)

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Result of solving a problem."""
    success: bool
    answer: Optional[str] = None
    expected: Optional[str] = None
    steps_executed: int = 0
    steps_succeeded: int = 0
    error: Optional[str] = None


@dataclass
class StepResult:
    """Result of executing a single step."""
    success: bool
    result: Optional[str] = None
    signature_id: Optional[int] = None
    similarity: float = 0.0
    error: Optional[str] = None


class Solver:
    """Minimal solver for function pointer architecture.

    Routes steps to signatures via embedding similarity.
    Executes functions from registry and records stats for Welford learning.
    """

    def __init__(self, db_path: str = None):
        """Initialize solver with signature database.

        Args:
            db_path: Path to signature database.
        """
        self.db_path = db_path or DB_PATH
        self.step_db = StepSignatureDB(self.db_path)

    async def solve(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
        plan: Optional[DAGPlan] = None,
    ) -> SolveResult:
        """Solve a problem.

        Args:
            problem: The problem text
            expected_answer: Expected answer for grading
            plan: Pre-decomposed plan (if None, uses local decomposition)

        Returns:
            SolveResult with success status and answer
        """
        # Get or create plan
        if plan is None:
            plan = await self._decompose(problem)
            if plan is None:
                return SolveResult(
                    success=False,
                    error="Failed to decompose problem",
                    expected=expected_answer,
                )

        # Create DAG for tracking
        dag_id = create_dag(problem)
        step_ids = create_dag_steps(dag_id, [s.task for s in plan.steps])

        # Execute steps
        context = dict(plan.phase1_values) if plan.phase1_values else {}
        steps_executed = 0
        steps_succeeded = 0

        for i, step in enumerate(plan.steps):
            step_result = await self._execute_step(step, context, plan)
            steps_executed += 1

            if step_result.success:
                steps_succeeded += 1
                context[step.id] = step_result.result
                step.result = step_result.result
                step.success = True
            else:
                logger.warning(
                    "[solver] Step %s failed: %s",
                    step.id, step_result.error
                )
                step.success = False

        # Get final answer
        final_answer = None
        if plan.steps and plan.steps[-1].result:
            final_answer = str(plan.steps[-1].result)

        # Grade result
        success = False
        if final_answer and expected_answer:
            norm_answer = normalize_answer(final_answer)
            norm_expected = normalize_answer(expected_answer)
            success = norm_answer == norm_expected

        # Record outcome
        grade_dag(dag_id, success)
        run_postmortem(dag_id, self.step_db)

        return SolveResult(
            success=success,
            answer=final_answer,
            expected=expected_answer,
            steps_executed=steps_executed,
            steps_succeeded=steps_succeeded,
        )

    async def _decompose(self, problem: str) -> Optional[DAGPlan]:
        """Decompose problem into atomic steps using mathdecomp.

        Uses LLM-based recursive decomposition to break problem into
        atomic function calls that can be executed via function_registry.
        """
        try:
            from mycelium.mathdecomp import decompose_with_api

            decomp = decompose_with_api(problem, max_retries=2)

            if not decomp.verified and decomp.error:
                logger.warning("[solver] Decomposition failed: %s", decomp.error)
                return None

            # Convert mathdecomp.Decomposition to DAGPlan
            return self._convert_decomp_to_dag(decomp, problem)

        except Exception as e:
            logger.warning("[solver] Decomposition error: %s", e)
            return None

    def _convert_decomp_to_dag(self, decomp, problem: str) -> DAGPlan:
        """Convert mathdecomp.Decomposition to DAGPlan format.

        Maps the atomic steps to our standard Step/DAGPlan format
        for compatibility with the execution pipeline.
        """
        steps: list[Step] = []

        # Build extraction value map
        extraction_values = {
            ext["id"]: ext["value"] for ext in decomp.extractions
        }

        for md_step in decomp.steps:
            # Build dependency list from inputs
            depends_on = []
            extracted_values = {}

            for i, inp in enumerate(md_step.inputs):
                if inp.type.value == "step":
                    depends_on.append(inp.id)
                elif inp.type.value == "extraction":
                    # Map extraction to positional param
                    if inp.id in extraction_values:
                        extracted_values[f"arg_{i}"] = extraction_values[inp.id]

            step = Step(
                id=md_step.id,
                task=md_step.semantic or f"{md_step.func} operation",
                depends_on=depends_on,
                extracted_values=extracted_values,
                operation=md_step.func,
            )
            steps.append(step)

        return DAGPlan(
            steps=steps,
            problem=problem,
            phase1_values=extraction_values,
        )

    async def _execute_step(
        self,
        step: Step,
        context: dict,
        plan: DAGPlan,
    ) -> StepResult:
        """Execute a single step with hybrid GTS + Embedding + Welford routing.

        Per CLAUDE.md "The Flow": DB Stats → Welford → Tree Structure.

        Hybrid routing:
        1. Embed step, route via cosine similarity to best match
        2. Get adaptive threshold from Welford stats
        3. If similarity >= threshold → trust tree (use existing signature)
        4. If similarity < threshold → trust GTS (create new signature)
        5. Execute DSL, record outcome for Welford learning
        """
        # Embed step (synchronous)
        step_embedding = cached_embed(step.task)
        if step_embedding is None:
            return StepResult(success=False, error="Failed to embed step")

        embedding_list = step_embedding.tolist()

        # Route to best matching signature
        routing = self.step_db.route_to_best(embedding_list)

        # Get adaptive threshold from Welford stats
        threshold = self.step_db.get_adaptive_threshold(fallback=0.85)

        # Hybrid decision: trust tree or trust GTS?
        if routing.signature and routing.similarity >= threshold:
            # HIGH similarity: trust tree, use existing signature
            signature = routing.signature
            created = False
            logger.debug(
                "[solver] Trust tree: sim=%.3f >= threshold=%.3f, using sig=%s",
                routing.similarity, threshold, signature.id
            )
        else:
            # LOW similarity: trust GTS, create new signature
            func_hint = self._get_func_hint(step)
            signature, created = self.step_db.find_or_create(
                step_text=step.task,
                embedding=embedding_list,
                dsl_hint=func_hint,
            )
            if created:
                logger.info(
                    "[solver] Trust GTS: sim=%.3f < threshold=%.3f, created sig=%s for: %s",
                    routing.similarity, threshold, signature.id, step.task[:50]
                )

        # Record similarity for Welford learning (only for existing signatures)
        if not created and routing.similarity > 0:
            self.step_db.record_similarity(signature.id, routing.similarity)

        # Resolve values from context
        resolved_values = self._resolve_values(step, context, plan)

        # Execute function from registry
        func_name = signature.func_name or step.operation
        if func_name:
            try:
                # Get function info for arity
                func_info = get_function_info(func_name)
                if func_info is None:
                    raise ValueError(f"Unknown function: {func_name}")

                # Collect arguments in order
                args = []
                arity = func_info.get("arity", 2)

                # First add dependency results, then extracted values
                for i in range(arity):
                    key = f"arg_{i}"
                    if key in resolved_values:
                        args.append(resolved_values[key])
                    elif 'a' in resolved_values and i == 0:
                        args.append(resolved_values['a'])
                    elif 'b' in resolved_values and i == 1:
                        args.append(resolved_values['b'])

                if len(args) < arity:
                    raise ValueError(f"Not enough arguments: got {len(args)}, need {arity}")

                result = call_function(func_name, *args)

                # Record success with similarity for Welford
                self.step_db.record_success(signature.id, routing.similarity)
                return StepResult(
                    success=True,
                    result=str(result),
                    signature_id=signature.id,
                    similarity=routing.similarity,
                )

            except Exception as e:
                logger.warning("[solver] Function execution failed: %s", e)
                self.step_db.record_failure(signature.id)
                return StepResult(
                    success=False,
                    error=f"Function execution failed: {e}",
                    signature_id=signature.id,
                    similarity=routing.similarity,
                )

        # No function available
        self.step_db.record_failure(signature.id)
        return StepResult(
            success=False,
            error="No function available",
            signature_id=signature.id,
            similarity=routing.similarity,
        )

    def _get_func_hint(self, step: Step) -> Optional[str]:
        """Extract function name hint from step for signature creation.

        Per CLAUDE.md New Favorite Pattern: consolidate to single entry point.
        This extracts the function name so find_or_create() can set proper func_name.
        """
        # Map operation names to function registry keys
        func_map = {
            'add': 'add',
            'sub': 'sub',
            'subtract': 'sub',
            'mul': 'mul',
            'multiply': 'mul',
            'truediv': 'truediv',
            'div': 'truediv',
            'divide': 'truediv',
            'pow': 'pow',
            'power': 'pow',
            'sqrt': 'sqrt',
            'abs': 'abs',
            'floor': 'floor',
            'ceil': 'ceil',
        }

        # Check step.operation field first (set during decomposition)
        if step.operation and step.operation in func_map:
            return func_map[step.operation]

        # Fallback: extract from task text
        task_lower = step.task.lower()
        for op_name, func_name in func_map.items():
            if op_name in task_lower:
                return func_name

        return None

    def _resolve_values(
        self,
        step: Step,
        context: dict,
        plan: DAGPlan,
    ) -> dict:
        """Resolve step values from context and plan.

        Maps extracted values and step references to positional params.
        Function calls use 'a', 'b' or 'arg_0', 'arg_1' as operands.
        """
        # Collect values in order: step dependencies first, then NUM_X values
        operands = []

        # 1. Add results from dependent steps (step_N references)
        for dep_id in step.depends_on:
            if dep_id in context:
                operands.append(context[dep_id])

        # 2. Add extracted NUM_X values in order
        num_values = []
        for key, value in sorted(step.extracted_values.items()):
            if isinstance(value, (int, float)):
                num_values.append(value)
            elif isinstance(value, str):
                # Check for $reference to phase1 values
                if value.startswith("$"):
                    ref = value[1:]
                    if ref in plan.phase1_values:
                        num_values.append(plan.phase1_values[ref])
                    elif ref in context:
                        num_values.append(context[ref])
                # Check for {step_N} reference
                elif value.startswith("{") and value.endswith("}"):
                    ref = value[1:-1]
                    if ref in context:
                        num_values.append(context[ref])
                else:
                    # Try to parse as number
                    try:
                        num_values.append(float(value))
                    except ValueError:
                        pass

        operands.extend(num_values)

        # 3. Map to positional params a, b (for function calls)
        resolved = {}
        if len(operands) >= 1:
            resolved['a'] = operands[0]
        if len(operands) >= 2:
            resolved['b'] = operands[1]

        # Also include original keys for compatibility
        for key, value in step.extracted_values.items():
            if isinstance(value, (int, float)):
                resolved[key] = value

        return resolved


# Convenience function
async def solve(
    problem: str,
    expected_answer: Optional[str] = None,
    plan: Optional[DAGPlan] = None,
    db_path: str = None,
) -> SolveResult:
    """Solve a problem using the default solver."""
    solver = Solver(db_path=db_path)
    return await solver.solve(problem, expected_answer, plan)
