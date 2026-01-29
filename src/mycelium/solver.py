"""Solver: Minimal implementation for local decomposition architecture.

Core loop:
1. Decompose problem into steps (local, no LLM)
2. For each step: route to best signature via embeddings
3. Execute DSL
4. Record success/failure for Welford stats

Per CLAUDE.md Big 4:
- System Independence: Let tree self-organize via stats
- New Favorite Pattern: Single pathway for routing
- The Flow: DB Stats → Welford → Tree Structure
- True Atomic Decomposition: Bias towards more steps
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from mycelium.config import DB_PATH, USE_GTS_DECOMPOSITION, GTS_MODEL_PATH
from mycelium.plan_models import Step, DAGPlan
from mycelium.gts_decomposer import GTSDecomposer, DecomposedStep
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.step_signatures.dsl_executor import try_execute_dsl_math
from mycelium.step_signatures.utils import cosine_similarity
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
    """Minimal solver for local decomposition architecture.

    Routes steps to signatures via embedding similarity.
    Executes DSL and records stats for Welford learning.
    """

    def __init__(self, db_path: str = None, use_gts: bool = None):
        """Initialize solver with signature database.

        Args:
            db_path: Path to signature database.
            use_gts: Whether to use GTS decomposition. If None, uses config.
        """
        self.db_path = db_path or DB_PATH
        self.step_db = StepSignatureDB(self.db_path)
        self._use_gts = use_gts if use_gts is not None else USE_GTS_DECOMPOSITION
        self._gts_decomposer: Optional[GTSDecomposer] = None

    @property
    def gts_decomposer(self) -> GTSDecomposer:
        """Lazy-load GTSDecomposer when first accessed."""
        if self._gts_decomposer is None:
            self._gts_decomposer = GTSDecomposer(model_path=GTS_MODEL_PATH)
        return self._gts_decomposer

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
        """Decompose problem into steps.

        If USE_GTS_DECOMPOSITION is enabled, uses GTS model.
        Otherwise falls back to None (caller should provide plan).
        """
        if self._use_gts:
            try:
                return await self._decompose_with_gts(problem)
            except NotImplementedError as e:
                logger.warning(
                    "[solver] GTS beam search not implemented, falling back: %s", e
                )
                return None
            except Exception as e:
                logger.warning(
                    "[solver] GTS decomposition failed, falling back: %s", e
                )
                return None

        logger.warning("[solver] Local decomposition not yet implemented")
        return None

    async def _decompose_with_gts(self, problem: str) -> Optional[DAGPlan]:
        """Decompose problem using GTS model.

        Per CLAUDE.md Big 5 #4 (True Atomic Decomposition):
        GTS decomposes problems into atomic steps for routing.

        Args:
            problem: The problem text.

        Returns:
            DAGPlan with atomic steps, or None if decomposition fails.

        Raises:
            NotImplementedError: If GTS beam search not yet implemented.
        """
        # Run GTS decomposition (may raise NotImplementedError)
        decomposed_steps = self.gts_decomposer.decompose(problem)

        if not decomposed_steps:
            logger.warning("[solver] GTS returned no steps for problem")
            return None

        # Convert DecomposedStep list to DAGPlan
        return self._convert_gts_to_dag(decomposed_steps, problem)

    def _convert_gts_to_dag(
        self,
        decomposed_steps: list[DecomposedStep],
        problem: str,
    ) -> DAGPlan:
        """Convert GTS DecomposedStep list to DAGPlan format.

        Maps the GTS atomic steps to our standard Step/DAGPlan format
        for compatibility with the existing execution pipeline.

        Args:
            decomposed_steps: List of DecomposedStep from GTSDecomposer.
            problem: The original problem text.

        Returns:
            DAGPlan with Step objects ready for execution.
        """
        steps: list[Step] = []

        for ds in decomposed_steps:
            # Build dependency list in our format
            depends_on = [f"step_{dep}" for dep in ds.depends_on]

            # Extract operation type from operation string (e.g., "add two numbers" -> "add")
            operation = ds.operation.split()[0] if ds.operation else None

            step = Step(
                id=f"step_{ds.step_number}",
                task=ds.operation,
                depends_on=depends_on,
                extracted_values=ds.extracted_values,
                operation=operation,
            )
            steps.append(step)

        return DAGPlan(
            steps=steps,
            problem=problem,
        )

    async def _execute_step(
        self,
        step: Step,
        context: dict,
        plan: DAGPlan,
    ) -> StepResult:
        """Execute a single step.

        1. Embed step text
        2. Find best matching signature
        3. Execute DSL with resolved values
        4. Return result
        """
        # Embed step (synchronous)
        step_embedding = cached_embed(step.task)
        if step_embedding is None:
            return StepResult(success=False, error="Failed to embed step")

        # Route to best signature
        signature, similarity = self._route_to_signature(step_embedding, step)
        if signature is None:
            return StepResult(
                success=False,
                error="No matching signature found",
                similarity=similarity,
            )

        # Resolve values from context
        resolved_values = self._resolve_values(step, context, plan)

        # Execute DSL
        if signature.dsl_code:
            result = try_execute_dsl_math(signature.dsl_code, resolved_values)
            if result is not None:
                # Record success
                self.step_db.record_success(signature.id)
                return StepResult(
                    success=True,
                    result=str(result),
                    signature_id=signature.id,
                    similarity=similarity,
                )

        # DSL failed or no DSL
        self.step_db.record_failure(signature.id)
        return StepResult(
            success=False,
            error="DSL execution failed",
            signature_id=signature.id,
            similarity=similarity,
        )

    def _route_to_signature(
        self,
        step_embedding: list[float],
        step: Step,
    ) -> tuple[Optional[StepSignature], float]:
        """Route step to best matching signature.

        Simple cosine similarity routing.
        Returns (signature, similarity) or (None, 0.0) if no match.
        """
        # Get all leaf signatures
        leaves = self.step_db.get_all_leaves()
        if not leaves:
            return None, 0.0

        best_sig = None
        best_sim = 0.0

        for sig in leaves:
            if sig.embedding is None:
                continue

            sim = cosine_similarity(step_embedding, sig.embedding)
            if sim > best_sim:
                best_sim = sim
                best_sig = sig

        return best_sig, best_sim

    def _resolve_values(
        self,
        step: Step,
        context: dict,
        plan: DAGPlan,
    ) -> dict:
        """Resolve step values from context and plan."""
        resolved = {}

        for key, value in step.extracted_values.items():
            if isinstance(value, (int, float)):
                resolved[key] = value
            elif isinstance(value, str):
                # Check for $reference to phase1 values
                if value.startswith("$"):
                    ref = value[1:]
                    if ref in plan.phase1_values:
                        resolved[key] = plan.phase1_values[ref]
                    elif ref in context:
                        resolved[key] = context[ref]
                    else:
                        resolved[key] = value
                # Check for {step_N} reference
                elif value.startswith("{") and value.endswith("}"):
                    ref = value[1:-1]
                    if ref in context:
                        resolved[key] = context[ref]
                    else:
                        resolved[key] = value
                else:
                    # Try to parse as number
                    try:
                        resolved[key] = float(value)
                    except ValueError:
                        resolved[key] = value
            else:
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
