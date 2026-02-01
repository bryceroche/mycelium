"""Solver: Minimal implementation for flat prototype architecture.

Uses LLM decomposition with step chaining to solve math problems.
Steps can reference previous step results for multi-step calculations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, List

import numpy as np

from mycelium.config import DB_PATH
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.function_registry import execute, REGISTRY
from mycelium.embedding_cache import cached_embed
from mycelium.llm_decomposer import LLMDecomposer, DecomposedStep, Decomposition

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single step."""
    success: bool
    result: Optional[str] = None
    signature_id: Optional[int] = None
    similarity: float = 0.0
    error: Optional[str] = None
    func_name: Optional[str] = None
    embedding: Optional[list] = None


@dataclass
class SolveContext:
    """Context for tracking solve progress."""
    successful_steps: list = field(default_factory=list)
    failed_steps: list = field(default_factory=list)
    max_depth: int = 5


class Solver:
    """Solver using LLM decomposition with step chaining.

    Routes steps to signatures via embedding similarity.
    Executes functions from registry and records stats for learning.
    """

    def __init__(self, db_path: str = None, model: str = "gpt-4o-mini"):
        """Initialize solver with signature database.

        Args:
            db_path: Path to signature database.
            model: LLM model for decomposition
        """
        self.db_path = db_path or DB_PATH
        self.step_db = StepSignatureDB(self.db_path)
        self.decomposer = LLMDecomposer(model=model)
        self.model = model

    def solve_with_trend(self, problem: str, context: SolveContext = None) -> Any:
        """Solve a math problem.

        Args:
            problem: The problem text
            context: Optional solve context

        Returns:
            The final answer
        """
        answer, _ = self.solve_with_trend_and_results(problem, context)
        return answer

    def solve_with_trend_and_results(
        self, problem: str, context: SolveContext = None
    ) -> tuple[Any, List[StepResult]]:
        """Solve a math problem and return both answer and step results.

        Uses step chaining: steps can reference previous step results.

        Args:
            problem: The problem text
            context: Optional solve context

        Returns:
            (answer, step_results) tuple
        """
        if context is None:
            context = SolveContext()

        # Get signature menu for decomposition
        menu = self.step_db.format_signature_menu(max_examples_per_func=10)

        # Get full decomposition with extractions and step references
        decomposition = self.decomposer.decompose_full(problem, menu)
        logger.info(f"Decomposed into {len(decomposition.steps)} steps with {len(decomposition.extractions)} extractions")

        # Build value context from extractions
        value_context: dict[str, float] = {}
        for ext in decomposition.extractions:
            value_context[ext.id] = ext.value
            logger.debug(f"Extraction: {ext.id} = {ext.value}")

        # Solve each step in order, resolving references
        results = []
        for step in decomposition.steps:
            # Resolve inputs from context
            resolved_params = []
            for inp in step.inputs:
                try:
                    resolved_value = inp.resolve(value_context)
                    resolved_params.append(resolved_value)
                except ValueError as e:
                    logger.warning(f"Could not resolve input {inp.ref}: {e}")
                    if step.params:
                        resolved_params = step.params
                        break

            # Execute with resolved parameters
            result = self._execute_step(step, resolved_params, context)
            results.append(result)

            # Store result in context for subsequent steps
            if result.success and result.result is not None:
                try:
                    value_context[step.id] = float(result.result)
                    logger.debug(f"Step {step.id}: {result.result}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert result to float: {result.result}")

        # Get answer from the answer_step
        answer = None
        if decomposition.answer_step and decomposition.answer_step in value_context:
            answer = value_context[decomposition.answer_step]
        elif results:
            answer = results[-1].result

        return answer, results

    def _execute_step(
        self,
        step: DecomposedStep,
        resolved_params: List[float],
        context: SolveContext
    ) -> StepResult:
        """Execute a single step with resolved parameters."""
        # Embed and classify
        embedding = cached_embed(step.description)
        func_name, similarity, sig = self.step_db.classify(embedding)

        # Record coverage observation for Welford tracking
        if sig is not None:
            threshold = self.step_db.get_adaptive_threshold()
            self.step_db.record_coverage(sig.id, similarity, threshold)

            # Warn if signature has high outcome variance (decomposition candidate)
            if sig.outcome_count >= 10 and sig.outcome_variance >= 0.20:
                logger.warning(
                    f"[decomp-candidate] sig={sig.id} has high outcome variance "
                    f"(var={sig.outcome_variance:.3f}, n={sig.outcome_count}): "
                    f"'{sig.description[:40]}...'"
                )

        logger.debug(
            f"Step: '{step.description[:50]}...' -> {func_name} (sim={similarity:.3f})"
        )

        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        try:
            if func_name and func_name in REGISTRY:
                result = execute(func_name, *resolved_params)
                success = True
                logger.debug(f"Executed {func_name}({resolved_params}) = {result}")

                if sig:
                    self.step_db.record_success(sig.id, similarity)
            else:
                result = None
                success = False
                logger.warning(f"No function '{func_name}' in registry")

                if sig:
                    self.step_db.record_failure(sig.id)

            step_result = StepResult(
                success=success,
                result=str(result) if result is not None else None,
                func_name=func_name,
                similarity=similarity,
                embedding=embedding_list,
                signature_id=sig.id if sig else None
            )

            if success:
                context.successful_steps.append(step_result)

            return step_result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            if sig:
                self.step_db.record_failure(sig.id)
            return StepResult(
                success=False,
                error=str(e),
                func_name=func_name,
                similarity=similarity,
                embedding=embedding_list,
                signature_id=sig.id if sig else None
            )

    def record_learning(
        self, step_results: List[StepResult], problem_correct: bool
    ) -> dict:
        """Record learning from step results based on problem outcome.

        Args:
            step_results: List of StepResult from solving
            problem_correct: Whether the overall problem was solved correctly

        Returns:
            dict with learning stats
        """
        success_recorded = 0
        failure_recorded = 0

        for result in step_results:
            if result.signature_id is None:
                continue

            if problem_correct:
                # Use embedding-aware success recording if embedding available
                if result.embedding is not None:
                    self.step_db.record_success_with_embedding(
                        result.signature_id,
                        np.array(result.embedding, dtype=np.float32),
                        result.similarity,
                    )
                else:
                    self.step_db.record_success(result.signature_id, result.similarity)
                success_recorded += 1
            else:
                self.step_db.record_failure(result.signature_id)
                failure_recorded += 1

        return {
            "problem_correct": problem_correct,
            "success_recorded": success_recorded,
            "failure_recorded": failure_recorded,
        }


def solve_problem(problem: str, model: str = "gpt-4o-mini") -> Any:
    """Convenience function to solve a problem.

    Args:
        problem: The problem text
        model: LLM model for decomposition

    Returns:
        The final answer
    """
    solver = Solver(model=model)
    return solver.solve_with_trend(problem)
