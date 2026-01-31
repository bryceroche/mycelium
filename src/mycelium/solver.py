"""Solver: Minimal implementation for local decomposition architecture.

Core loop:
1. Decompose problem into atomic steps (depth=1) via GTS
2. For each step: find existing signature or create new one
3. Execute DSL with resolved values
4. Record success/failure for Welford stats

Per CLAUDE.md Big 3:
- System Independence: Let tree self-organize via stats
- New Favorite Pattern: Single pathway via find_or_create()
- The Flow: DB Stats -> Welford -> Tree Structure

Similarity-trend recursion:
- Keep decomposing while similarity improves
- Stop when similarity plateaus or max depth reached
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, List

from mycelium.config import DB_PATH
from mycelium.plan_models import Step, DAGPlan
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.function_registry import call_function, get_function_info, execute, REGISTRY
from mycelium.embedding_cache import cached_embed, cached_embed_async
from mycelium.answer_norm import normalize_answer
from mycelium.data_layer.mcts import (
    create_dag,
    create_dag_steps,
    grade_dag,
    run_postmortem,
)
from mycelium.llm_decomposer import LLMDecomposer, DecomposedStep, Decomposition, StepInput

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
    func_name: Optional[str] = None  # Function used for execution
    embedding: Optional[list] = None  # Step embedding for learning
    step: Optional["Step"] = None  # Reference to original step


@dataclass
class SolveContext:
    """Context for tracking solve progress and successful steps.

    Used for post-mortem learning from successful executions.
    Also contains similarity-trend recursion settings.
    """
    successful_steps: list = field(default_factory=list)
    failed_steps: list = field(default_factory=list)
    all_steps: list = field(default_factory=list)
    # Similarity-trend recursion settings
    max_depth: int = 5
    min_similarity_improvement: float = 0.02  # Stop if improvement < this


class Solver:
    """Minimal solver for function pointer architecture.

    Routes steps to signatures via embedding similarity.
    Executes functions from registry and records stats for Welford learning.

    Supports two modes:
    1. DAGPlan mode: Uses pre-decomposed plans from mathdecomp (async solve())
    2. Similarity-trend mode: Uses LLMDecomposer with recursive decomposition (solve_with_trend())
    """

    def __init__(self, db_path: str = None, model: str = "gpt-4o-mini"):
        """Initialize solver with signature database.

        Args:
            db_path: Path to signature database.
            model: LLM model for decomposition (default: gpt-4o-mini)
        """
        self.db_path = db_path or DB_PATH
        self.step_db = StepSignatureDB(self.db_path)
        self.decomposer = LLMDecomposer(model=model)
        self.model = model

    # =========================================================================
    # SIMILARITY-TREND BASED SOLVING (Sync API)
    # =========================================================================

    def solve_with_trend(self, problem: str, context: SolveContext = None) -> Any:
        """Solve a math problem using similarity-trend based recursion.

        Key insight: Keep decomposing while similarity improves. Stop when it plateaus.

        Args:
            problem: The problem text
            context: Optional solve context with settings

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
            context: Optional solve context with settings

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
                    # Fall back to literal params if available
                    if step.params:
                        resolved_params = step.params
                        break

            # Execute with resolved parameters
            result = self._solve_step_with_trend_chained(
                step, resolved_params, menu, context, prev_similarity=0, depth=0
            )
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

    def _solve_step_with_trend_chained(
        self,
        step: DecomposedStep,
        resolved_params: List[float],
        menu: str,
        context: SolveContext,
        prev_similarity: float,
        depth: int
    ) -> StepResult:
        """Solve a single step with resolved parameters (step chaining support)."""

        # Embed and classify
        embedding = cached_embed(step.description)
        func_name, similarity, sig = self.step_db.classify(embedding)
        threshold = self.step_db.get_adaptive_threshold()

        # Record coverage observation for Welford tracking
        if sig is not None:
            self.step_db.record_coverage(sig.id, similarity, threshold)

        logger.debug(
            f"[depth={depth}] Step: '{step.description[:50]}...' "
            f"-> {func_name} (sim={similarity:.3f}, thresh={threshold:.3f})"
        )

        # Execute with resolved params
        return self._execute_with_resolved_params(
            step, func_name, resolved_params, similarity, embedding, sig, context
        )

    def _execute_with_resolved_params(
        self,
        step: DecomposedStep,
        func_name: str,
        resolved_params: List[float],
        similarity: float,
        embedding: Any,
        sig: Optional[StepSignature],
        context: SolveContext
    ) -> StepResult:
        """Execute a step with pre-resolved parameters."""
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        try:
            if func_name and func_name in REGISTRY:
                result = execute(func_name, *resolved_params)
                success = True
                logger.debug(f"Executed {func_name}({resolved_params}) = {result}")

                # Record success for Welford learning
                if sig:
                    self.step_db.record_success(sig.id, similarity)
            else:
                # No matching function
                result = None
                success = False
                logger.warning(f"No function '{func_name}' in registry")

                # Record failure
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

            # Track for post-mortem
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

        If problem was solved correctly, record success embeddings.
        If problem was wrong, record failure embeddings.

        Args:
            step_results: List of StepResult from solving
            problem_correct: Whether the overall problem was solved correctly

        Returns:
            dict with learning stats
        """
        import numpy as np

        success_recorded = 0
        failure_recorded = 0

        for result in step_results:
            if result.signature_id is None or result.embedding is None:
                continue

            embedding = np.array(result.embedding, dtype=np.float32)

            if problem_correct:
                self.step_db.record_success_with_embedding(
                    result.signature_id, embedding, result.similarity
                )
                success_recorded += 1
            else:
                self.step_db.record_failure_with_embedding(
                    result.signature_id, embedding
                )
                failure_recorded += 1

        return {
            "problem_correct": problem_correct,
            "success_recorded": success_recorded,
            "failure_recorded": failure_recorded,
        }

    def _solve_step_with_trend(
        self,
        step: DecomposedStep,
        menu: str,
        context: SolveContext,
        prev_similarity: float,
        depth: int
    ) -> StepResult:
        """Solve a single step with similarity-trend recursion."""

        # Embed and classify
        embedding = cached_embed(step.description)
        func_name, similarity, sig = self.step_db.classify(embedding)
        threshold = self.step_db.get_adaptive_threshold()

        # Record coverage observation for Welford tracking
        if sig is not None:
            self.step_db.record_coverage(sig.id, similarity, threshold)

        logger.debug(
            f"[depth={depth}] Step: '{step.description[:50]}...' "
            f"-> {func_name} (sim={similarity:.3f}, thresh={threshold:.3f})"
        )

        # Good match - execute
        if similarity >= threshold:
            return self._execute_decomposed_step(step, func_name, similarity, embedding, sig, context)

        # Similarity not improving - stop decomposing
        improvement = similarity - prev_similarity
        if depth > 0 and improvement < context.min_similarity_improvement:
            logger.debug(f"Similarity not improving ({improvement:.3f}), executing anyway")
            return self._execute_decomposed_step(step, func_name, similarity, embedding, sig, context)

        # Hit max depth - execute anyway
        if depth >= context.max_depth:
            logger.debug(f"Max depth {context.max_depth} reached, executing anyway")
            return self._execute_decomposed_step(step, func_name, similarity, embedding, sig, context)

        # Decompose further
        logger.debug(f"Low similarity ({similarity:.3f}), decomposing further")
        sub_steps = self.decomposer.decompose(step.description, menu)

        if not sub_steps or len(sub_steps) == 1:
            # Can't decompose further - execute anyway
            return self._execute_decomposed_step(step, func_name, similarity, embedding, sig, context)

        # Recurse on sub-steps
        sub_results = []
        for sub_step in sub_steps:
            sub_result = self._solve_step_with_trend(
                sub_step, menu, context,
                prev_similarity=similarity,
                depth=depth + 1
            )
            sub_results.append(sub_result)

        # Combine sub-results (simple: sum for now)
        combined_result = sum(
            float(r.result) for r in sub_results
            if r.result is not None and r.success
        )

        return StepResult(
            success=all(r.success for r in sub_results),
            result=str(combined_result),
            func_name="combined",
            similarity=similarity,
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        )

    def _execute_decomposed_step(
        self,
        step: DecomposedStep,
        func_name: str,
        similarity: float,
        embedding: Any,
        sig: Optional[StepSignature],
        context: SolveContext
    ) -> StepResult:
        """Execute a decomposed step using the function registry."""
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        try:
            if func_name and func_name in REGISTRY:
                result = execute(func_name, *step.params)
                success = True
                logger.debug(f"Executed {func_name}({step.params}) = {result}")

                # Record success for Welford learning
                if sig:
                    self.step_db.record_success(sig.id, similarity)
            else:
                # No matching function
                result = None
                success = False
                logger.warning(f"No function '{func_name}' in registry")

                # Record failure
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

            # Track for post-mortem
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

    # =========================================================================
    # ORIGINAL DAGPlan-BASED SOLVING (Async API)
    # =========================================================================

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
        # Embed step (async to avoid blocking event loop)
        step_embedding = await cached_embed_async(step.task)
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
                func_name=func_hint,
            )
            if created:
                logger.info(
                    "[solver] Trust GTS: sim=%.3f < threshold=%.3f, created sig=%s for: %s",
                    routing.similarity, threshold, signature.id, step.task[:50]
                )

        # Record similarity for Welford learning (only for existing signatures)
        if not created and routing.similarity > 0:
            self.step_db.record_similarity(signature.id, routing.similarity)

        # Record coverage observation for all signature matches
        if routing.signature is not None:
            self.step_db.record_coverage(routing.signature.id, routing.similarity, threshold)

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
                    func_name=func_name,
                    embedding=embedding_list,
                    step=step,
                )

            except Exception as e:
                logger.warning("[solver] Function execution failed: %s", e)
                self.step_db.record_failure(signature.id)
                return StepResult(
                    success=False,
                    error=f"Function execution failed: {e}",
                    signature_id=signature.id,
                    similarity=routing.similarity,
                    func_name=func_name,
                    embedding=embedding_list,
                    step=step,
                )

        # No function available
        self.step_db.record_failure(signature.id)
        return StepResult(
            success=False,
            error="No function available",
            signature_id=signature.id,
            similarity=routing.similarity,
            embedding=embedding_list,
            step=step,
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

    # =========================================================================
    # POST-MORTEM LEARNING
    # =========================================================================

    def learn_from_success(self, context: SolveContext) -> dict:
        """Post-mortem: create/merge signatures from successful steps.

        Called AFTER verifying the answer is correct.
        Per CLAUDE.md: Signatures are created from proven successes.

        Args:
            context: SolveContext with successful_steps list

        Returns:
            Stats dict: {"merged": count, "created": count}
        """
        stats = {"merged": 0, "created": 0}

        for step_result in context.successful_steps:
            if not step_result.success or step_result.func_name is None:
                continue

            if step_result.func_name == "combined":
                # Skip combined results (from recursion)
                continue

            embedding = step_result.embedding
            if embedding is None:
                continue

            func_name = step_result.func_name
            description = step_result.step.task if step_result.step else "unknown"

            # Decide: merge or create?
            action, sig_id = self.step_db.should_merge_or_create(embedding, func_name)

            if action == "merge":
                self.step_db.merge_into_signature(sig_id, embedding, description)
                logger.info(f"Merged '{description[:30]}...' into signature {sig_id}")
                stats["merged"] += 1
            else:
                # Create new signature
                sig, created = self.step_db.find_or_create(
                    step_text=description,
                    embedding=embedding,
                    func_name=func_name,
                )
                if created:
                    logger.info(f"Created new signature {sig.id} for '{description[:30]}...'")
                    stats["created"] += 1

        return stats

    async def solve_and_learn(
        self,
        problem: str,
        expected_answer: Any = None,
    ) -> tuple[Any, dict]:
        """Solve a problem and learn from success if answer matches.

        Args:
            problem: The problem text
            expected_answer: Optional expected answer for verification

        Returns:
            (result, learn_stats) tuple
        """
        context = SolveContext()
        result = await self.solve_with_context(problem, expected_answer, context=context)

        learn_stats = {"merged": 0, "created": 0, "learned": False}

        # If we have expected answer, verify and learn
        if expected_answer is not None:
            # Simple comparison (could use answer_norm for better matching)
            if self._answers_match(result.answer, expected_answer):
                learn_stats = self.learn_from_success(context)
                learn_stats["learned"] = True
                logger.info(f"Answer correct! Learned: {learn_stats}")
            else:
                logger.info(f"Answer {result.answer} != expected {expected_answer}, not learning")

        return result, learn_stats

    def _answers_match(self, result: Any, expected: Any, tolerance: float = 1e-6) -> bool:
        """Check if result matches expected answer."""
        if result is None:
            return False

        try:
            # Try numeric comparison
            r = float(result)
            e = float(expected)
            return abs(r - e) < tolerance or abs(r - e) / max(abs(e), 1) < tolerance
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(result).strip().lower() == str(expected).strip().lower()

    async def solve_with_context(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
        plan: Optional[DAGPlan] = None,
        context: Optional[SolveContext] = None,
    ) -> SolveResult:
        """Solve a problem with context tracking for post-mortem learning.

        Args:
            problem: The problem text
            expected_answer: Expected answer for grading
            plan: Pre-decomposed plan (if None, uses local decomposition)
            context: SolveContext to track steps (created if not provided)

        Returns:
            SolveResult with success status and answer
        """
        if context is None:
            context = SolveContext()

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
        exec_context = dict(plan.phase1_values) if plan.phase1_values else {}
        steps_executed = 0
        steps_succeeded = 0

        for i, step in enumerate(plan.steps):
            step_result = await self._execute_step(step, exec_context, plan)
            steps_executed += 1
            context.all_steps.append(step_result)

            if step_result.success:
                steps_succeeded += 1
                exec_context[step.id] = step_result.result
                step.result = step_result.result
                step.success = True
                context.successful_steps.append(step_result)
            else:
                logger.warning(
                    "[solver] Step %s failed: %s",
                    step.id, step_result.error
                )
                step.success = False
                context.failed_steps.append(step_result)

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


# Convenience functions
async def solve(
    problem: str,
    expected_answer: Optional[str] = None,
    plan: Optional[DAGPlan] = None,
    db_path: str = None,
) -> SolveResult:
    """Solve a problem using the default solver."""
    solver = Solver(db_path=db_path)
    return await solver.solve(problem, expected_answer, plan)


async def solve_and_learn_problem(
    problem: str,
    expected_answer: Any = None,
    db_path: str = None,
) -> tuple[SolveResult, dict]:
    """Convenience function to solve and learn from a problem.

    Args:
        problem: The problem text
        expected_answer: Expected answer for verification and learning
        db_path: Optional database path

    Returns:
        (SolveResult, learn_stats) tuple
    """
    solver = Solver(db_path=db_path)
    return await solver.solve_and_learn(problem, expected_answer)


def solve_problem(problem: str, model: str = "gpt-4o-mini") -> Any:
    """Convenience function to solve a problem (sync, similarity-trend mode).

    This uses the LLMDecomposer + similarity-trend recursion approach.

    Args:
        problem: The problem text
        model: LLM model for decomposition

    Returns:
        The final answer
    """
    solver = Solver(model=model)
    return solver.solve_with_trend(problem)
