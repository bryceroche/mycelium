"""Solver: Step-level signature matching with cluster-based retrieval.

Enhanced Flow:
    Problem → Planner → DAG steps → For each step:
                                      ↓
                          Step → embed → find matching StepSignature cluster
                                      ↓
                          [Match?] → Try DSL execution
                                      ↓
                          [DSL low confidence?] → DECOMPOSE FURTHER (recursive)
                                      ↓
                          [DSL high confidence?] → Execute deterministically
                          [No match?] → Solve step, create new signature cluster
                                      ↓
                          Track success → update cluster stats

Key improvement: Signatures are stored at the STEP level, not whole-problem level.
This enables reuse of solution methods across different problems that share step types.

Recursive Decomposition:
    When a DSL has low confidence (can't map parameters reliably), we decompose
    the step into smaller sub-steps until we reach truly atomic operations that
    DSLs can handle. This is the key to self-improvement - complex steps that
    fail get broken down into simpler patterns that can be learned.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .step_decomposer import (
    StepDecomposer,
    DecomposedStep,
    MAX_DECOMPOSITION_DEPTH,
    DECOMPOSITION_CONFIDENCE_THRESHOLD,
)
from .semantic_extractor import extract_semantic_info
from .step_output import StepOutput, detect_output_type

logger = logging.getLogger(__name__)

# =============================================================================
# Dynamic DSL Avoidance (Learned from Lift Data)
# =============================================================================
# Instead of hardcoded exemplars, we learn which signatures to avoid injecting
# based on lift data. Signatures with negative lift have their centroids added
# to the "avoid space" dynamically.
#
# Key insight: The same DSL can help in some contexts but hurt in others.
# We track lift per-signature and avoid injection when lift is negative.
#
# DSL Improvement Loop:
# 1. Signature created → cold start injection to sample success
# 2. After N uses → calculate lift (injected vs baseline success rate)
# 3. Negative lift → add to avoid space, mark for DSL improvement
# 4. DSL improved → reset lift stats (new dsl_version), re-enter probation
# 5. Positive lift after improvement → remove from avoid space

# Minimum uses before calculating lift (cold start period)
MIN_USES_FOR_LIFT = 5

# Lift threshold: below this, signature is considered "harmful" for DSL
NEGATIVE_LIFT_THRESHOLD = -0.05  # -5% lift = avoid
PROBATION_USES = 10  # Uses needed to exit probation

# Cache for negative-lift signature embeddings (rebuilt periodically)
_avoid_embeddings_cache: Optional[np.ndarray] = None
_avoid_signature_ids: list[str] = []
_avoid_cache_time: float = 0.0


def get_signature_lift(signature: "StepSignature") -> tuple[float, bool]:
    """Calculate lift for a signature.

    Returns:
        (lift, has_enough_data): Lift value and whether we have enough data.
        Lift = injected_success_rate - baseline_success_rate
    """
    inj_uses = signature.injected_uses or 0
    base_uses = signature.non_injected_uses or 0

    if inj_uses < MIN_USES_FOR_LIFT or base_uses < MIN_USES_FOR_LIFT:
        return 0.0, False

    inj_rate = (signature.injected_successes or 0) / inj_uses
    base_rate = (signature.non_injected_successes or 0) / base_uses

    return inj_rate - base_rate, True


def should_avoid_dsl_for_signature(
    signature: "StepSignature",
    step_embedding: np.ndarray,
    step_db: "StepSignatureDB",
) -> tuple[bool, str]:
    """Determine if DSL should be avoided for this signature.

    Returns:
        (should_avoid, reason): Whether to skip DSL and why.
    """
    # In training mode, never avoid - we want all signal (success AND failure)
    # Failed injections teach us to avoid bad DSLs in benchmark mode
    if TRAINING_MODE:
        return False, "training_mode"

    # Check lift for this specific signature
    lift, has_data = get_signature_lift(signature)

    if has_data and lift < NEGATIVE_LIFT_THRESHOLD:
        return True, f"negative_lift({lift:.2f})"

    # Check if in probation (recently improved DSL)
    # Skip probation sampling when DSL_PROBATION_ENABLED=False (benchmark mode)
    if DSL_PROBATION_ENABLED:
        dsl_version = getattr(signature, 'dsl_version', 1) or 1
        version_uses = getattr(signature, 'dsl_version_uses', 0) or 0

        if dsl_version > 1 and version_uses < PROBATION_USES:
            # In probation - randomly decide whether to inject
            import random
            if random.random() > PROBATION_INJECTION_RATE:
                return True, f"probation(v{dsl_version}, {version_uses}/{PROBATION_USES} uses)"

    # Check similarity to other negative-lift signatures (avoid similar contexts)
    avoid_embeddings = _get_avoid_embeddings(step_db)
    if avoid_embeddings is not None and len(avoid_embeddings) > 0:
        step_norm = step_embedding / (np.linalg.norm(step_embedding) + 1e-9)
        avoid_norms = avoid_embeddings / (np.linalg.norm(avoid_embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(avoid_norms, step_norm)
        max_sim = float(np.max(similarities))

        if max_sim >= NEGATIVE_LIFT_SIMILARITY:
            return True, f"similar_to_negative_lift(sim={max_sim:.2f})"

    return False, "ok"


def _get_avoid_embeddings(step_db: "StepSignatureDB") -> Optional[np.ndarray]:
    """Get embeddings of signatures with negative lift (cached)."""
    global _avoid_embeddings_cache, _avoid_signature_ids, _avoid_cache_time

    now = time.time()
    if _avoid_embeddings_cache is not None and (now - _avoid_cache_time) < AVOID_CACHE_TTL:
        return _avoid_embeddings_cache

    # Rebuild cache from database
    negative_lift_sigs = step_db.get_negative_lift_signatures(
        min_uses=MIN_USES_FOR_LIFT,
        lift_threshold=NEGATIVE_LIFT_THRESHOLD
    )

    if not negative_lift_sigs:
        _avoid_embeddings_cache = None
        _avoid_signature_ids = []
        _avoid_cache_time = now
        return None

    _avoid_signature_ids = [s.signature_id for s in negative_lift_sigs]
    _avoid_embeddings_cache = np.array([s.centroid for s in negative_lift_sigs])
    _avoid_cache_time = now

    logger.info("[council] Rebuilt avoid cache: %d negative-lift signatures", len(negative_lift_sigs))
    return _avoid_embeddings_cache


def clear_avoid_cache():
    """Clear the avoid embeddings cache (call after DSL improvements)."""
    global _avoid_embeddings_cache, _avoid_signature_ids, _avoid_cache_time
    _avoid_embeddings_cache = None
    _avoid_signature_ids = []
    _avoid_cache_time = 0.0


class SynthesisStrategy(Enum):
    """Strategy for synthesizing final answer from step results."""
    ONCE_AT_END = "once_at_end"  # Simple: collect all, synthesize once
    INCREMENTAL = "incremental"  # Complex: merge at synthesis points progressively


# Thresholds for selecting incremental synthesis strategy
# Based on design decision: "depth>=4, width>=3" triggers incremental
INCREMENTAL_DEPTH_THRESHOLD = 4
INCREMENTAL_WIDTH_THRESHOLD = 3

# String truncation limits for logging and context
MAX_STEP_TASK_LOG_LENGTH = 50  # Max chars of step task to show in logs
MAX_PARENT_PROBLEM_LENGTH = 200  # Max chars of parent problem in step context

from .client import GroqClient
from .embedder import Embedder
from .planner import Planner, DAGPlan, Step
from mycelium.config import (
    SOLVER_DEFAULT_MODEL,
    PLANNER_DEFAULT_MODEL,
    CLIENT_DEFAULT_TEMPERATURE,
    PIPELINE_MIN_SIMILARITY,
    DSL_PROBATION_ENABLED,
    DSL_MIN_CONFIDENCE,
    DSL_LLM_THRESHOLD,
    RECURSIVE_DECOMPOSITION_ENABLED,
    RECURSIVE_MAX_DEPTH,
    RECURSIVE_CONFIDENCE_THRESHOLD,
    TRAINING_MODE,
    PROBATION_INJECTION_RATE,
    AVOID_CACHE_TTL,
    NEGATIVE_LIFT_SIMILARITY,
)
from .step_signatures import (
    StepSignatureDB,
    StepSignature,
    StepIOSchema,
    try_execute_formula,
    execute_dsl_with_confidence,
    execute_dsl_with_llm_matching,
)
from .answer_norm import answers_equivalent_llm
from .phase_constraints import (
    assign_phases,
    compute_execution_score,
    infer_execution_phase,
    PhaseAssignment,
    PhaseScore,
)
from .prompt_templates import get_registry


@dataclass
class DAGMetrics:
    """Metrics for analyzing DAG complexity to select synthesis strategy."""
    depth: int  # Number of levels in execution order
    max_width: int  # Maximum number of parallel steps at any level
    total_steps: int
    synthesis_points: list[str]  # Step IDs that have multiple dependencies (natural merge points)

    def should_use_incremental(self) -> bool:
        """Determine if incremental synthesis is warranted."""
        return (
            self.depth >= INCREMENTAL_DEPTH_THRESHOLD
            and self.max_width >= INCREMENTAL_WIDTH_THRESHOLD
        )


@dataclass
class StepResult:
    """Result from executing a single step."""
    step_id: str
    task: str
    result: str
    success: bool = True
    embedding: Optional[np.ndarray] = None

    # Signature info
    signature_matched: Optional[StepSignature] = None
    signature_used: Optional[StepSignature] = None
    signature_similarity: float = 0.0
    is_new_signature: bool = False
    was_injected: bool = False

    # I/O schema validation
    output_valid: bool = True
    output_validation_msg: str = ""
    used_io_schema: bool = False

    # Decomposition tracking
    decomposed: bool = False
    decomposition_depth: int = 0
    sub_step_results: list["StepResult"] = field(default_factory=list)

    # Semantic context (for DSL parameter mapping)
    semantic_meaning: str = ""  # What result represents: "area of triangle ABC"
    semantic_type: str = ""     # Category: "area", "length", "count", "ratio"
    numeric_value: Optional[float] = None  # Parsed numeric value


@dataclass
class SolverResult:
    """Result from the council solver."""
    answer: str
    reasoning: str
    success: bool = False

    # Decomposition info
    plan: Optional[DAGPlan] = None
    step_results: list[StepResult] = field(default_factory=list)
    used_decomposition: bool = False

    # Step signature stats
    total_steps: int = 0
    signatures_matched: int = 0  # Steps that matched existing signatures
    signatures_new: int = 0  # Steps that created new signatures
    steps_with_injection: int = 0
    new_signatures_created: int = 0  # Deprecated, use signatures_new

    # Phase constraint stats
    phase_assignment: Optional[PhaseAssignment] = None
    phase_score: Optional[PhaseScore] = None
    phase_coherence: float = 1.0  # 0-1, how well execution respected phase ordering

    # Synthesis strategy info
    synthesis_strategy: SynthesisStrategy = SynthesisStrategy.ONCE_AT_END
    dag_metrics: Optional[DAGMetrics] = None
    intermediate_merges: int = 0  # Number of incremental merges performed

    # Error aggregation for debugging
    errors: list[str] = field(default_factory=list)  # Collected error messages from failed steps

    # Metrics
    elapsed_ms: float = 0.0

    @property
    def failed_steps(self) -> list[StepResult]:
        """Get all steps that failed during execution."""
        return [r for r in self.step_results if not r.success]

    @property
    def has_errors(self) -> bool:
        """Check if any steps failed (either via exception or success=False)."""
        return len(self.errors) > 0 or any(not r.success for r in self.step_results)


def extract_result(text: str, pattern: str = "RESULT") -> str:
    """Extract result from step response."""
    match = re.search(rf"{pattern}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Pattern not found, fall back to last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        logger.debug(f"extract_result: empty response, pattern='{pattern}'")
        return ""
    logger.debug(f"extract_result: pattern '{pattern}' not found, using last line")
    return lines[-1]


def extract_answer(text: str) -> str:
    """Extract final answer."""
    patterns = [
        r"ANSWER:\s*(.+?)(?:\n|$)",
        r"answer is[:\s]+(.+?)(?:\.|$)",
        r"\\boxed{([^}]+)}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    # No patterns matched, fall back to last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        logger.debug("extract_answer: empty response, no patterns matched")
        return ""
    logger.debug("extract_answer: no answer patterns matched, using last line")
    return lines[-1]


class Solver:
    """Enhanced solver with step-level signature matching.

    Always decomposes problems into atomic steps, matches each step against
    known signatures, and injects proven methods when matches are found.
    """

    def __init__(
        self,
        step_db: Optional[StepSignatureDB] = None,
        solver_model: str = SOLVER_DEFAULT_MODEL,
        planner_model: str = PLANNER_DEFAULT_MODEL,
        temperature: float = CLIENT_DEFAULT_TEMPERATURE,
        min_step_similarity: float = PIPELINE_MIN_SIMILARITY,
        match_mode: str = "auto",  # Signature matching mode
        injection_mode: str = "all",  # Injection strategy: none, dsl, formula, procedure, guidance, all
        use_hints: bool = True,  # Enable signature-guided decomposition hints
    ):
        self.step_db = step_db or StepSignatureDB()
        self.solver_client = GroqClient(model=solver_model)
        self.planner = Planner(model=planner_model)
        self.embedder = Embedder.get_instance()
        self._solver_model = solver_model
        self._planner_model = planner_model

        self.temperature = temperature
        self.min_step_similarity = min_step_similarity
        self.match_mode = match_mode
        self.injection_mode = injection_mode  # none, dsl, formula, procedure, guidance, all
        self.use_hints = use_hints

    async def solve(
        self,
        problem: str,
        ground_truth: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> SolverResult:
        """Solve a problem using step-level signature matching.

        Always decomposes problems into atomic steps - like factoring integers
        into primes, we factor problems into irreducible solution patterns.

        Flow:
        1. Decompose problem into steps via DAG planner
        2. For each step:
           - Embed step, find matching signature cluster
           - If match: inject proven method
           - If no match: solve and create new signature
        3. Track outcomes to update signature stats

        Args:
            problem: The math problem
            ground_truth: Optional expected answer
            problem_id: Optional problem identifier

        Returns:
            SolverResult with answer and step-level signature info
        """
        start = time.time()

        logger.debug(f"[council] Starting solve, problem_id={problem_id}")

        # Always decompose - signatures are the primes, decomposition is factorization
        result = await self._solve_with_decomposition(problem)

        elapsed = (time.time() - start) * 1000

        # Check success using LLM judge
        success = await self._check_success(result.answer, ground_truth, problem)
        result.success = success
        result.elapsed_ms = elapsed

        logger.info(
            f"[council] Solved in {elapsed:.0f}ms: success={success} "
            f"steps={len(result.step_results)} injections={result.steps_with_injection}"
        )

        # Update signature stats based on outcome
        if ground_truth:
            await self._update_step_signatures(result.step_results, success)

        return result

    async def _solve_with_decomposition(
        self,
        problem: str,
    ) -> SolverResult:
        """Solve by decomposing into steps with signature injection.

        Always decomposes - like factoring integers into primes,
        we factor problems into irreducible solution signatures.

        Args:
            problem: The problem to solve
        """
        # Get signature hints to guide decomposition (if enabled)
        if self.use_hints:
            signature_hints = self.step_db.get_signature_hints(limit=15)
            hinted_step_types = {hint[0] for hint in signature_hints}
            logger.info("[council] Provided %d signature hints to planner", len(signature_hints))
        else:
            signature_hints = None
            hinted_step_types = set()
            logger.info("[council] Signature hints disabled")

        # Decompose problem into atomic steps (with signature-guided hints)
        plan = await self.planner.decompose(problem, signature_hints=signature_hints)

        # Compute phase assignment for the DAG
        phase_assignment = assign_phases(plan)

        # Execute steps in order
        step_results = []
        context = {}
        typed_context: dict[str, StepOutput] = {}  # Rich typed outputs for DSL
        step_descriptions = {}  # Track what each step computes for better DSL param matching
        rich_context = {}  # Semantic context for DSL parameter mapping
        completed_steps: set[str] = set()
        execution_phases: dict[str, float] = {}
        steps_with_injection = 0
        new_signatures = 0
        errors: list[str] = []  # Aggregate errors for debugging

        for level in plan.get_execution_order():
            # Infer execution phases for steps at this level before execution
            for step in level:
                exec_phase = infer_execution_phase(
                    step.id, phase_assignment, completed_steps
                )
                execution_phases[step.id] = exec_phase

            # Execute steps with error handling - return_exceptions=True prevents
            # one failing step from cancelling others in the same level
            # Use recursive execution to decompose-until-confident
            level_results = await asyncio.gather(
                *[self._execute_step_recursive(step, context, step_descriptions, problem, depth=0) for step in level],
                return_exceptions=True,
            )

            for step, result in zip(level, level_results):
                # Handle exceptions from individual steps
                if isinstance(result, BaseException):
                    error_msg = f"[{step.id}] {type(result).__name__}: {result}"
                    errors.append(error_msg)
                    logger.error(
                        "Step execution failed: step_id=%s, task='%s', error=%s",
                        step.id,
                        step.task[:MAX_STEP_TASK_LOG_LENGTH],
                        result,
                        exc_info=result,
                    )
                    result = StepResult(
                        step_id=step.id,
                        task=step.task,
                        result=f"Step failed: {error_msg}",
                        success=False,
                    )

                step_results.append(result)
                context[step.id] = result.result
                # Create rich typed output for DSL execution
                typed_context[step.id] = detect_output_type(result.result)
                step_descriptions[step.id] = step.task  # Track what this step computed
                # Store semantic info for DSL parameter mapping
                rich_context[step.id] = {
                    "value": result.numeric_value,
                    "meaning": result.semantic_meaning,
                    "type": result.semantic_type,
                    "raw": result.result,
                }
                completed_steps.add(step.id)

                if result.signature_used:
                    steps_with_injection += 1
                if result.is_new_signature:
                    new_signatures += 1

        # Compute phase score
        phase_score = compute_execution_score(
            base_score=1.0,
            assignment=phase_assignment,
            execution_phases=execution_phases,
        )

        # Synthesize final answer using DAG-aware strategy
        answer, synthesis_strategy, dag_metrics, intermediate_merges = await self._synthesize_answer(
            problem, step_results, plan
        )

        # Count signature matching stats
        signatures_matched = sum(
            1 for r in step_results if r.signature_matched is not None
        )

        # Count hint matches - steps that matched a hinted signature
        hint_matches = sum(
            1 for r in step_results
            if r.signature_matched is not None
            and r.signature_matched.step_type in hinted_step_types
        )

        # Log hint effectiveness
        total_steps = len(step_results)
        if signature_hints:
            logger.info(
                "[council] Hint effectiveness: %d/%d steps matched hinted signatures (%.0f%%), "
                "%d/%d matched any signature (%.0f%%)",
                hint_matches, total_steps, 100 * hint_matches / total_steps if total_steps else 0,
                signatures_matched, total_steps, 100 * signatures_matched / total_steps if total_steps else 0,
            )

        return SolverResult(
            answer=answer,
            reasoning=self._format_reasoning(step_results),
            plan=plan,
            step_results=step_results,
            used_decomposition=True,
            total_steps=len(plan.steps),
            signatures_matched=signatures_matched,
            signatures_new=new_signatures,
            steps_with_injection=steps_with_injection,
            new_signatures_created=new_signatures,
            phase_assignment=phase_assignment,
            phase_score=phase_score,
            phase_coherence=phase_score.coherence,
            synthesis_strategy=synthesis_strategy,
            dag_metrics=dag_metrics,
            intermediate_merges=intermediate_merges,
            errors=errors,
        )

    async def _execute_step_recursive(
        self,
        step: Step,
        context: dict,
        step_descriptions: dict,
        problem: str,
        depth: int = 0,
    ) -> StepResult:
        """Execute a step, recursively decomposing if confidence is low.

        This implements the "decompose until confident" strategy:
        1. Try to match step against signature library
        2. If confidence < threshold, decompose step into sub-steps
        3. Execute sub-steps and combine results
        4. Recursion bounded by RECURSIVE_MAX_DEPTH

        Args:
            step: The step to execute
            context: Results from prior steps
            step_descriptions: Map of step_id -> task description for DSL param matching
            problem: The original problem (for context)
            depth: Current recursion depth
        """
        # First, check signature confidence before executing
        embedding = self.embedder.embed(step.task)

        # Pass depth as origin_depth so new signatures know their decomposition level
        signature, is_new = self.step_db.find_or_create(
            step_text=step.task,
            embedding=embedding,
            min_similarity=self.min_step_similarity,
            parent_problem=problem[:MAX_PARENT_PROBLEM_LENGTH],
            match_mode=self.match_mode if self.match_mode != "baseline" else "cosine",
            origin_depth=depth,
        )

        # Compute confidence (signature similarity)
        confidence = self._get_similarity(embedding, signature) if signature.centroid is not None else 0.0

        # Should we re-decompose?
        should_redecompose = (
            RECURSIVE_DECOMPOSITION_ENABLED
            and depth < RECURSIVE_MAX_DEPTH
            and confidence < RECURSIVE_CONFIDENCE_THRESHOLD
            and not is_new  # Don't re-decompose brand new signatures
            and signature.step_type not in ("general_step", "solve_problem")  # Avoid infinite loops
        )

        if should_redecompose:
            logger.info(
                "[recursive] Re-decomposing step '%s' (confidence=%.2f < %.2f, depth=%d)",
                step.task[:50], confidence, RECURSIVE_CONFIDENCE_THRESHOLD, depth
            )

            # Decompose this step as a sub-problem
            sub_plan = await self.planner.decompose(
                f"Solve this step: {step.task}\n\nContext from original problem:\n{problem[:500]}",
                signature_hints=self.step_db.get_signature_hints(limit=10) if self.use_hints else None,
            )

            # Execute sub-steps recursively
            sub_context = dict(context)  # Copy parent context
            sub_step_descriptions = {}  # Track sub-step descriptions
            sub_results = []

            for level in sub_plan.get_execution_order():
                level_results = await asyncio.gather(
                    *[self._execute_step_recursive(sub_step, sub_context, sub_step_descriptions, problem, depth + 1)
                      for sub_step in level],
                    return_exceptions=True,
                )

                for sub_step, result in zip(level, level_results):
                    if isinstance(result, BaseException):
                        result = StepResult(
                            step_id=sub_step.id,
                            task=sub_step.task,
                            result=f"Sub-step failed: {result}",
                            success=False,
                        )
                    sub_results.append(result)
                    sub_context[sub_step.id] = result.result
                    sub_step_descriptions[sub_step.id] = sub_step.task  # Track description

            # Combine sub-results into single result for this step
            combined_result = await self._combine_sub_results(step.task, sub_results)

            return StepResult(
                step_id=step.id,
                task=step.task,
                result=combined_result,
                embedding=embedding,
                signature_matched=signature,
                signature_similarity=confidence,
                is_new_signature=is_new,
                was_injected=False,  # Re-decomposed, not injected
            )

        # Confidence is good enough, execute normally
        return await self._execute_step_with_signature(step, context, step_descriptions, problem)

    async def _combine_sub_results(self, task: str, sub_results: list[StepResult]) -> str:
        """Combine sub-step results into a single result."""
        if not sub_results:
            return "No sub-results"

        # If only one sub-result, return it directly
        if len(sub_results) == 1:
            return sub_results[0].result

        # Combine multiple sub-results
        results_str = "\n".join(
            f"- {r.step_id}: {r.result}" for r in sub_results
        )

        prompt = f"""Combine these sub-step results into a single answer for the task: "{task}"

Sub-step results:
{results_str}

Provide the combined result. End with RESULT: <your answer>"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Combine the results."},
        ]

        response = await self.solver_client.generate(messages, temperature=self.temperature)
        return extract_result(response)

    async def _execute_step_with_signature(
        self,
        step: Step,
        context: dict,
        step_descriptions: dict,
        problem: str,
        decomposition_depth: int = 0,
    ) -> StepResult:
        """Execute a step with signature lookup and potential injection.

        Supports recursive decomposition: when DSL confidence is low, the step
        is broken into sub-steps and solved recursively until atomic operations
        are reached.

        Args:
            step: The step to execute
            context: Results from previous steps
            step_descriptions: Map of step_id -> task description for DSL param matching
            problem: The original problem text
            decomposition_depth: Current recursion depth (for preventing infinite loops)

        Supports three modes:
        - I/O schema mode: Use structured inputs and output format from schema
        - Hard matching (default): Collapse to single best signature method
        - Soft matching: Present superposition of methods weighted by similarity
        """
        # Embed the step task
        embedding = self.embedder.embed(step.task)

        # Find or create signature for this step type
        # Pass decomposition_depth so new signatures know their origin level
        signature, is_new = self.step_db.find_or_create(
            step_text=step.task,
            embedding=embedding,
            min_similarity=self.min_step_similarity,
            parent_problem=problem[:MAX_PARENT_PROBLEM_LENGTH],
            match_mode=self.match_mode if self.match_mode != "baseline" else "cosine",
            origin_depth=decomposition_depth,
        )

        logger.debug(
            f"[council] step={step.id} sig={signature.step_type} "
            f"new={is_new} reliable={signature.is_reliable}"
        )

        # Build context string - ALWAYS include original problem to prevent context loss
        ctx_str = f"Original problem:\n{problem}\n\n"

        if step.depends_on:
            ctx_str += "Results from previous steps:\n"
            for dep_id in step.depends_on:
                if dep_id in context:
                    ctx_str += f"- {dep_id}: {context[dep_id]}\n"
        else:
            ctx_str += "This is the first step."

        # Check if this is a semantic umbrella that should route to child
        if signature.is_semantic_umbrella and signature.child_signatures:
            routed_child = await self._route_to_child_signature(
                parent=signature,
                step=step,
                context=context,
                step_descriptions=step_descriptions,
                problem=problem,
                embedding=embedding,
            )
            if routed_child is not None:
                return routed_child

        # Get I/O schema and determine execution mode
        io_schema = signature.get_io_schema()
        similarity = 0.0
        was_injected = False
        direct_execution = False
        result = ""

        # Should we inject? Based on injection_mode and signature reliability
        # injection_mode: none, dsl, formula, procedure, guidance, all
        # Cold start: always inject for first 10 uses to solve catch-22
        if self.injection_mode == "none" or self.match_mode == "baseline":
            should_inject = False
        else:
            # should_inject() handles cold start internally - always True for first 10 uses
            should_inject = signature.should_inject()

        formatted_inputs = io_schema.format_inputs(context) if io_schema.inputs else ctx_str

        # Extract inputs for DSL/formula execution using rich typed detection
        # If io_schema exists, use its structured extraction
        # Otherwise, use StepOutput type detection for better parsing
        if io_schema and io_schema.inputs:
            numeric_inputs = io_schema.extract_numeric_inputs(context)
        else:
            # Use StepOutput type detection for reliable value extraction
            numeric_inputs = {}
            sympy_inputs = {}  # Store symbolic expressions for SYMPY layer

            # Extract from prior step results using typed detection
            for key, value in context.items():
                typed = detect_output_type(str(value))
                if typed.is_numeric():
                    numeric_inputs[key] = typed.numeric
                if typed.is_symbolic():
                    # Store sympy-ready expressions for symbolic DSL operations
                    sympy_inputs[key] = typed.sympy_expr
                    # Also store variables found (e.g., ['x', 'y'])
                    if typed.variables:
                        sympy_inputs[f"{key}_vars"] = typed.variables

            # Merge sympy_inputs into numeric_inputs (DSL executor will pick appropriate)
            # Use suffix to distinguish: step_1_sympy for symbolic, step_1 for numeric
            for key, expr in sympy_inputs.items():
                if not key.endswith("_vars"):
                    numeric_inputs[f"{key}_sympy"] = expr
                else:
                    numeric_inputs[key] = expr

            # Also extract numbers from the step task itself
            # This handles cases like "Calculate 2^10" -> [2, 10]
            task_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', step.task)
            for i, num_str in enumerate(task_numbers):
                try:
                    numeric_inputs[f"task_num_{i}"] = float(num_str)
                except ValueError:
                    pass

            # For first steps (no context), also extract from original problem
            # This enables DSL injection even when no prior steps exist
            if not context:
                problem_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', problem)
                for i, num_str in enumerate(problem_numbers):
                    try:
                        # Use problem_num_X prefix to distinguish from task numbers
                        numeric_inputs[f"problem_num_{i}"] = float(num_str)
                    except ValueError:
                        pass

        # =================================================================
        # Route by injection_mode: controls which strategies are enabled
        # Priority when all enabled: dsl → formula → procedure → guidance → plain
        # =================================================================

        # Helper to check if a mode is enabled
        def mode_enabled(mode: str) -> bool:
            return self.injection_mode == "all" or self.injection_mode == mode

        # DSL MODE: Deterministic execution (safe even for new signatures)
        # Skip DSL based on learned lift data (dynamic avoidance)
        dsl_skipped_reason = None
        if mode_enabled("dsl") and signature.dsl_script and self.match_mode != "baseline":
            # Check if DSL should be avoided based on lift data
            should_avoid, avoid_reason = should_avoid_dsl_for_signature(
                signature, embedding, self.step_db
            )
            if should_avoid:
                dsl_skipped_reason = avoid_reason
                logger.debug("[council] DSL skipped (%s): step=%s", avoid_reason, step.id)
            else:
                # Use LLM-assisted DSL execution with param matching
                # Config controls aggressive (data collection) vs conservative (benchmark) mode
                dsl_result, dsl_success, dsl_confidence = await execute_dsl_with_llm_matching(
                    signature.dsl_script, numeric_inputs,
                    client=self.solver_client,
                    min_confidence=DSL_MIN_CONFIDENCE,
                    llm_threshold=DSL_LLM_THRESHOLD,
                    step_descriptions=step_descriptions,
                    step_task=step.task,
                )
                if dsl_success:
                    result = str(dsl_result)
                    direct_execution = True
                    was_injected = True
                    similarity = self._get_similarity(embedding, signature)
                    logger.debug("[council] DSL executed: step=%s result=%s confidence=%.2f", step.id, result, dsl_confidence)
                elif dsl_confidence < DECOMPOSITION_CONFIDENCE_THRESHOLD:
                    # LOW CONFIDENCE: Trigger recursive decomposition
                    # Instead of LLM fallback, break step into smaller sub-steps
                    if decomposition_depth < MAX_DECOMPOSITION_DEPTH:
                        logger.info(
                            "[council] DSL low confidence (%.2f) - decomposing step=%s at depth=%d",
                            dsl_confidence, step.id, decomposition_depth
                        )
                        decomposed_result = await self._decompose_and_solve_step(
                            step=step,
                            context=context,
                            problem=problem,
                            embedding=embedding,
                            signature=signature,
                            is_new=is_new,
                            decomposition_depth=decomposition_depth,
                        )
                        if decomposed_result is not None:
                            return decomposed_result
                        # If decomposition failed, fall through to LLM
                        logger.debug("[council] Decomposition failed, falling back to LLM: step=%s", step.id)
                    else:
                        logger.debug(
                            "[council] DSL skipped (low confidence %.2f, max depth): step=%s",
                            dsl_confidence, step.id
                        )

        # FORMULA MODE: AST-based formula execution
        if mode_enabled("formula") and not direct_execution and io_schema.formula and numeric_inputs:
            direct_result = try_execute_formula(io_schema.formula, numeric_inputs)
            if direct_result is not None:
                result = str(direct_result)
                direct_execution = True
                was_injected = True
                similarity = self._get_similarity(embedding, signature)
                logger.debug("[council] Formula executed: step=%s formula=%s result=%s", step.id, io_schema.formula, result)

        # For LLM-based modes, need should_inject=True (lift-gated)
        if not direct_execution:
            if mode_enabled("procedure") and should_inject and io_schema.procedure:
                # PROCEDURE MODE: LLM follows ordered steps
                prompt = get_registry().format(
                    "step_solver_with_procedure",
                    formatted_inputs=formatted_inputs,
                    task=step.task,
                    procedure_steps=io_schema.format_procedure(),
                )
                similarity = self._get_similarity(embedding, signature)
                was_injected = True
                logger.debug("[council] Procedure injection: step=%s", step.id)

            elif mode_enabled("guidance") and should_inject and io_schema.method_template_v2:
                # GUIDANCE MODE: LLM gets natural language hint
                prompt = get_registry().format(
                    "step_solver_with_guidance",
                    formatted_inputs=formatted_inputs,
                    task=step.task,
                    guidance=io_schema.method_template_v2,
                )
                similarity = self._get_similarity(embedding, signature)
                was_injected = True
                logger.debug("[council] Guidance injection: step=%s", step.id)

            else:
                # PLAIN MODE: Solve from scratch
                prompt = get_registry().format(
                    "step_solver_plain",
                    context=ctx_str,
                    task=step.task,
                )

        # Execute via LLM if not already computed directly
        if not direct_execution:
            exec_mode = io_schema.execution_mode if should_inject and io_schema else "plain"
            logger.debug(
                "[council] LLM execution: step=%s mode=%s injected=%s",
                step.id, exec_mode, was_injected
            )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Execute: {step.task}"},
            ]
            try:
                response = await self.solver_client.generate(messages, temperature=self.temperature)
                result = extract_result(response)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "API call failed for step: step_id=%s, task='%s', error=%s",
                    step.id,
                    step.task[:MAX_STEP_TASK_LOG_LENGTH],
                    e,
                    exc_info=True,
                )
                return StepResult(
                    step_id=step.id,
                    task=step.task,
                    result=f"API error: {type(e).__name__}: {e}",
                    success=False,
                    embedding=embedding,
                    signature_matched=signature,
                    is_new_signature=is_new,
                )

        # Validate output
        output_valid, output_validation_msg = (
            io_schema.validate_output(result) if io_schema else (True, "")
        )

        # Extract semantic info for DSL parameter mapping
        semantic_info = extract_semantic_info(step.task, result)

        return StepResult(
            step_id=step.id,
            task=step.task,
            result=result,
            embedding=embedding,
            signature_matched=signature,
            signature_used=signature if was_injected else None,
            signature_similarity=similarity,
            is_new_signature=is_new,
            was_injected=was_injected,
            output_valid=output_valid,
            output_validation_msg=output_validation_msg,
            used_io_schema=bool(io_schema.inputs),
            # Semantic context for downstream DSL mapping
            semantic_meaning=semantic_info.meaning,
            semantic_type=semantic_info.semantic_type,
            numeric_value=semantic_info.value,
        )

    async def _decompose_and_solve_step(
        self,
        step: Step,
        context: dict,
        problem: str,
        embedding: np.ndarray,
        signature: "StepSignature",
        is_new: bool,
        decomposition_depth: int,
    ) -> Optional[StepResult]:
        """Decompose a complex step into sub-steps and solve recursively.

        This is triggered when DSL confidence is too low, indicating the step
        is too complex for deterministic execution. We break it into smaller
        sub-steps until we reach atomic operations.

        Args:
            step: The step to decompose
            context: Results from previous steps
            problem: The original problem text
            embedding: The step's embedding vector
            signature: The matched signature (for stats tracking)
            is_new: Whether this is a new signature
            decomposition_depth: Current recursion depth

        Returns:
            StepResult if decomposition succeeded, None otherwise
        """
        from .planner import Step as PlannerStep

        # Create decomposer (lazy - only when needed)
        if not hasattr(self, '_step_decomposer'):
            self._step_decomposer = StepDecomposer()

        # Decompose the step
        decomposed = await self._step_decomposer.decompose_step(
            step=step,
            context=context,
            depth=decomposition_depth,
        )

        # If couldn't decompose (atomic or error), return None to fall back to LLM
        if decomposed.is_atomic or len(decomposed.sub_steps) == 0:
            logger.debug("[council] Step is atomic, cannot decompose further: step=%s", step.id)
            return None

        logger.info(
            "[council] Decomposed step=%s into %d sub-steps at depth=%d",
            step.id, len(decomposed.sub_steps), decomposition_depth
        )

        # Execute sub-steps in dependency order
        sub_context = dict(context)  # Copy context
        sub_step_descriptions = {}  # Track sub-step descriptions for DSL param matching
        sub_results = []

        # Clean up dependencies - only allow references to sibling sub-steps
        valid_ids = {s.id for s in decomposed.sub_steps}
        for sub_step in decomposed.sub_steps:
            # Filter dependencies to only valid sibling IDs
            sub_step.depends_on = [d for d in sub_step.depends_on if d in valid_ids]

        # Simple topological sort based on depends_on
        executed = set()
        remaining = list(decomposed.sub_steps)

        while remaining:
            # Find steps whose dependencies are satisfied
            ready = [
                s for s in remaining
                if all(d in executed or d in context for d in s.depends_on)
            ]

            if not ready:
                # Fallback: if first iteration and nothing ready, force first sub-step
                # (LLM may have output bad dependencies)
                if not executed and remaining:
                    first_step = remaining[0]
                    first_step.depends_on = []  # Clear bad dependencies
                    ready = [first_step]
                    logger.debug("[council] Forcing first sub-step with no deps: %s", first_step.id)
                else:
                    # Circular dependency or missing dep
                    logger.warning("[council] Cannot schedule remaining sub-steps: %s",
                                 [s.id for s in remaining])
                    break

            # Execute ready steps (could parallelize, but sequential for simplicity)
            for sub_step in ready:
                sub_result = await self._execute_step_with_signature(
                    step=sub_step,
                    context=sub_context,
                    step_descriptions=sub_step_descriptions,
                    problem=problem,
                    decomposition_depth=decomposition_depth + 1,
                )
                sub_results.append(sub_result)
                sub_context[sub_step.id] = sub_result.result
                sub_step_descriptions[sub_step.id] = sub_step.task  # Track description
                executed.add(sub_step.id)
                remaining.remove(sub_step)

        if not sub_results:
            logger.warning("[council] No sub-steps executed for step=%s", step.id)
            return None

        # Aggregate sub-results
        if len(sub_results) == 1:
            aggregated_result = sub_results[-1].result
        else:
            # Use the result of the last step (usually "combine results")
            aggregated_result = sub_results[-1].result

        # Track that we used decomposition
        logger.info(
            "[council] Decomposition complete: step=%s sub_steps=%d result=%s",
            step.id, len(sub_results), aggregated_result[:50] if aggregated_result else "None"
        )

        # Return aggregated result
        return StepResult(
            step_id=step.id,
            task=step.task,
            result=aggregated_result,
            embedding=embedding,
            signature_matched=signature,
            signature_used=None,  # Not directly injected
            signature_similarity=0.0,
            is_new_signature=is_new,
            was_injected=False,  # Decomposition, not injection
            decomposed=True,  # Mark as decomposed
            decomposition_depth=decomposition_depth + 1,
            sub_step_results=sub_results,  # Track all sub-step results
        )

    async def _route_to_child_signature(
        self,
        parent: StepSignature,
        step: Step,
        context: dict,
        step_descriptions: dict,
        problem: str,
        embedding: np.ndarray,
    ) -> Optional[StepResult]:
        """Route a semantic umbrella to the appropriate child signature.

        When a step matches a semantic umbrella (e.g., compute_probability),
        we use LLM to select which specific child pattern applies (e.g.,
        prob_single_draw, prob_conditional, etc.) and execute that child's DSL.

        Args:
            parent: The semantic umbrella signature that was matched
            step: The step being executed
            context: Results from previous steps
            step_descriptions: Map of step_id -> task description
            problem: The original problem text
            embedding: The step's embedding vector

        Returns:
            StepResult if routing succeeded, None to fall back to parent execution
        """
        import json

        try:
            child_specs = json.loads(parent.child_signatures)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[routing] Invalid child_signatures JSON for sig=%d", parent.id)
            return None

        if not child_specs:
            return None

        # Build routing prompt with child options
        options = "\n".join(
            f"{i+1}. {spec['step_type']}: {spec['condition']}"
            for i, spec in enumerate(child_specs)
        )

        ctx_str = f"Problem: {problem[:200]}\n\n"
        if context:
            ctx_str += "Previous results:\n"
            for dep_id, result in list(context.items())[-3:]:  # Last 3 results
                ctx_str += f"- {dep_id}: {result}\n"

        routing_prompt = f"""You are routing a math step to the correct solution pattern.

Step to solve: {step.task}

Context:
{ctx_str}

Available patterns:
{options}

Which pattern number (1-{len(child_specs)}) best matches this step? Reply with ONLY the number."""

        messages = [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": "Select pattern number:"},
        ]

        try:
            response = await self.solver_client.generate(messages, temperature=0.0)
            # Extract number from response
            match = re.search(r'(\d+)', response.strip())
            if not match:
                logger.debug("[routing] Could not parse pattern number from: %s", response)
                return None

            choice = int(match.group(1)) - 1
            if choice < 0 or choice >= len(child_specs):
                logger.debug("[routing] Invalid pattern choice: %d", choice + 1)
                return None

            selected_spec = child_specs[choice]
            child_sig = self.step_db.get_signature(selected_spec['id'])

            if child_sig is None:
                logger.warning("[routing] Child signature not found: id=%d", selected_spec['id'])
                return None

            logger.info(
                "[routing] Umbrella '%s' -> child '%s' for step='%s'",
                parent.step_type, child_sig.step_type, step.task[:40]
            )

            # Execute the child's DSL
            if child_sig.dsl_script:
                # Extract inputs using StepOutput type detection
                numeric_inputs = {}
                sympy_inputs = {}
                for key, value in context.items():
                    typed = detect_output_type(str(value))
                    if typed.is_numeric():
                        numeric_inputs[key] = typed.numeric
                    if typed.is_symbolic():
                        sympy_inputs[key] = typed.sympy_expr

                # Merge sympy inputs with suffix
                for key, expr in sympy_inputs.items():
                    numeric_inputs[f"{key}_sympy"] = expr

                # Extract from step task
                task_numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', step.task)
                for i, num_str in enumerate(task_numbers):
                    try:
                        numeric_inputs[f"task_num_{i}"] = float(num_str)
                    except ValueError:
                        pass

                # Execute child DSL
                dsl_result, dsl_success, dsl_confidence = await execute_dsl_with_llm_matching(
                    child_sig.dsl_script,
                    numeric_inputs,
                    client=self.solver_client,
                    min_confidence=DSL_MIN_CONFIDENCE,
                    llm_threshold=DSL_LLM_THRESHOLD,
                    step_descriptions=step_descriptions,
                    step_task=step.task,
                )

                if dsl_success:
                    logger.debug("[routing] Child DSL executed: result=%s conf=%.2f", dsl_result, dsl_confidence)
                    return StepResult(
                        step_id=step.id,
                        task=step.task,
                        result=str(dsl_result),
                        embedding=embedding,
                        signature_matched=parent,
                        signature_used=child_sig,
                        signature_similarity=self._get_similarity(embedding, parent),
                        is_new_signature=False,
                        was_injected=True,
                    )
                else:
                    logger.debug("[routing] Child DSL failed (conf=%.2f), falling back", dsl_confidence)

        except Exception as e:
            logger.error("[routing] Failed to route: %s", e)

        return None

    def _get_similarity(self, embedding: np.ndarray, signature: StepSignature) -> float:
        """Compute cosine similarity between an embedding and a signature's centroid.

        Args:
            embedding: The embedding vector to compare (typically from the current step).
            signature: The signature containing a centroid to compare against.

        Returns:
            Cosine similarity score in range [0, 1]. Returns 0.0 if the signature
            has no centroid (e.g., newly created signature without examples).
        """
        if signature.centroid is None:
            return 0.0
        from .step_signatures import cosine_similarity
        return cosine_similarity(embedding, signature.centroid)

    def _analyze_dag(self, plan: DAGPlan) -> DAGMetrics:
        """Analyze DAG structure to determine synthesis strategy.

        Extracts structural metrics from the execution plan to inform
        how intermediate results should be merged into a final answer.

        Args:
            plan: The DAGPlan containing steps with dependency relationships.

        Returns:
            DAGMetrics with:
            - depth: Number of execution levels (longest dependency chain)
            - max_width: Maximum parallel steps at any level
            - total_steps: Total number of steps in the plan
            - synthesis_points: Step IDs with multiple dependencies (merge points)
        """
        levels = plan.get_execution_order()
        depth = len(levels)
        max_width = max(len(level) for level in levels) if levels else 0

        # Find synthesis points: steps that depend on multiple other steps
        # These are natural places to merge intermediate results
        synthesis_points = []
        for step in plan.steps:
            if len(step.depends_on) >= 2:
                synthesis_points.append(step.id)

        return DAGMetrics(
            depth=depth,
            max_width=max_width,
            total_steps=len(plan.steps),
            synthesis_points=synthesis_points,
        )

    def _select_synthesis_strategy(self, dag_metrics: DAGMetrics) -> SynthesisStrategy:
        """Select synthesis strategy based on DAG complexity.

        Chooses between single-pass and incremental synthesis based on
        the plan's structural complexity.

        Args:
            dag_metrics: Metrics from _analyze_dag() describing plan structure.

        Returns:
            SynthesisStrategy.INCREMENTAL for complex DAGs (depth >= 4 AND width >= 3),
            SynthesisStrategy.ONCE_AT_END otherwise (single synthesis at completion).
        """
        if dag_metrics.should_use_incremental():
            return SynthesisStrategy.INCREMENTAL
        return SynthesisStrategy.ONCE_AT_END

    async def _synthesize_answer(
        self,
        problem: str,
        step_results: list[StepResult],
        plan: Optional[DAGPlan] = None,
    ) -> tuple[str, SynthesisStrategy, DAGMetrics | None, int]:
        """Synthesize final answer from step results.

        Uses DAG-aware strategy selection:
        - Simple DAGs: single synthesis pass at the end
        - Complex DAGs: incremental merges at synthesis points

        Returns:
            Tuple of (answer, strategy_used, dag_metrics, intermediate_merges)
        """
        # Analyze DAG if available
        dag_metrics = None
        strategy = SynthesisStrategy.ONCE_AT_END
        intermediate_merges = 0

        if plan:
            dag_metrics = self._analyze_dag(plan)
            strategy = self._select_synthesis_strategy(dag_metrics)

        if strategy == SynthesisStrategy.INCREMENTAL and dag_metrics:
            answer, intermediate_merges = await self._incremental_synthesize(
                problem, step_results, plan, dag_metrics
            )
        else:
            answer = await self._once_at_end_synthesize(problem, step_results)

        return answer, strategy, dag_metrics, intermediate_merges

    async def _once_at_end_synthesize(
        self,
        problem: str,
        step_results: list[StepResult],
    ) -> str:
        """Simple synthesis: collect all results and synthesize once."""
        results_str = "\n".join(
            f"- {r.step_id}: {r.task}\n  Result: {r.result}"
            for r in step_results
        )

        prompt = get_registry().format("final_synthesizer", problem=problem, step_results=results_str)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Give the final answer."},
        ]

        response = await self.solver_client.generate(messages, temperature=self.temperature)
        return extract_answer(response)

    async def _incremental_synthesize(
        self,
        problem: str,
        step_results: list[StepResult],
        plan: DAGPlan,
        dag_metrics: DAGMetrics,
    ) -> tuple[str, int]:
        """Incremental synthesis: merge at synthesis points progressively.

        Steps with multiple dependencies are natural synthesis points.
        We merge their dependent results before the final synthesis.

        Returns:
            Tuple of (answer, number_of_intermediate_merges)
        """
        # Build lookup for quick access to results
        result_by_id = {r.step_id: r for r in step_results}
        merged_results: dict[str, str] = {}  # synthesis_point -> merged result
        merge_count = 0

        # Process synthesis points (steps with multiple deps)
        for synthesis_point_id in dag_metrics.synthesis_points:
            step = plan.get_step(synthesis_point_id)
            if not step or len(step.depends_on) < 2:
                continue

            # Get results from dependencies
            dep_results = []
            for dep_id in step.depends_on:
                if dep_id in result_by_id:
                    r = result_by_id[dep_id]
                    dep_results.append(f"- {r.step_id}: {r.task}\n  Result: {r.result}")
                elif dep_id in merged_results:
                    dep_results.append(f"- {dep_id} (merged): {merged_results[dep_id]}")

            if len(dep_results) >= 2:
                # Merge these intermediate results
                merged = await self._merge_intermediate(dep_results)
                merged_results[synthesis_point_id] = merged
                merge_count += 1

        # Final synthesis with all results (original + merged)
        # Replace results at synthesis points with merged versions where available
        final_results = []
        processed_deps: set[str] = set()

        for r in step_results:
            if r.step_id in dag_metrics.synthesis_points and r.step_id in merged_results:
                # This is a synthesis point - include the merged result
                final_results.append(
                    f"- {r.step_id}: {r.task}\n  Result: {r.result}\n  (Merged from deps: {merged_results[r.step_id]})"
                )
                # Track which deps we've incorporated via merge
                step = plan.get_step(r.step_id)
                if step:
                    processed_deps.update(step.depends_on)
            elif r.step_id not in processed_deps:
                # Regular step not already covered by a merge
                final_results.append(f"- {r.step_id}: {r.task}\n  Result: {r.result}")

        results_str = "\n".join(final_results)
        prompt = get_registry().format("final_synthesizer", problem=problem, step_results=results_str)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Give the final answer."},
        ]

        response = await self.solver_client.generate(messages, temperature=self.temperature)
        return extract_answer(response), merge_count

    async def _merge_intermediate(self, dep_results: list[str]) -> str:
        """Merge intermediate results from dependent steps."""
        results_str = "\n".join(dep_results)
        prompt = get_registry().format("incremental_synthesizer", step_results=results_str)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Merge these results."},
        ]

        response = await self.solver_client.generate(messages, temperature=self.temperature)
        return extract_result(response, "MERGED")

    async def _check_success(self, predicted: str, ground_truth: Optional[str], problem: Optional[str] = None) -> bool:
        """Check if prediction matches ground truth using LLM judge."""
        if not ground_truth:
            return False
        return await answers_equivalent_llm(predicted, ground_truth, problem)

    async def _update_step_signatures(self, step_results: list[StepResult], overall_success: bool) -> None:
        """Update step signature statistics and trigger DSL generation if ready.

        Tracks lift data: whether method was injected or not, to enable
        lift-gated injection (only inject when it demonstrably helps).

        Also triggers DSL generation when a signature becomes reliable
        (enough uses + high success rate).
        """
        from .step_signatures import maybe_generate_dsl

        for step_result in step_results:
            if step_result.signature_matched:
                sig_id = step_result.signature_matched.id

                # Record usage
                self.step_db.record_usage(
                    signature_id=sig_id,
                    step_text=step_result.task,
                    success=overall_success,
                    was_injected=step_result.was_injected,
                    match_mode=self.match_mode,
                )

                # Try to generate DSL if signature is now reliable
                # Generate DSL after 3+ matches to enable deterministic execution
                # DSL generation only happens once per signature
                try:
                    await maybe_generate_dsl(
                        db=self.step_db,
                        client=self.solver_client,
                        signature_id=sig_id,
                        min_uses=3,  # Generate DSL after 3 matches
                        min_success_rate=0.8,
                    )
                except Exception as e:
                    logger.debug("DSL generation check failed: %s", e)

    def _format_reasoning(self, step_results: list[StepResult]) -> str:
        """Format step results into readable reasoning."""
        lines = ["Decomposition approach:"]
        for r in step_results:
            lines.append(f"\n**{r.step_id}**: {r.task}")
            if r.signature_used:
                lines.append(f"  [Used signature: {r.signature_used.step_type}]")
            elif r.is_new_signature:
                lines.append(f"  [New signature discovered]")
            lines.append(f"  Result: {r.result}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get statistics from the step signature database."""
        return self.step_db.get_stats()
