"""Solver V2: Simplified step-level execution with Natural Language signatures.

Flow:
    Problem → Planner → DAG steps → For each step:
        1. Embed step text
        2. Find matching signature (or create new)
        3. Execute: DSL if available, else LLM
        4. Record success/failure

Key difference from V1: Signatures speak natural language.
- clarifying_questions help extract parameters
- param_descriptions explain what each DSL param means
- Lazy NL: new signatures start empty, get filled in as we learn
"""

import asyncio
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import random

from mycelium.config import (
    MIN_MATCH_THRESHOLD,
    MIN_MATCH_THRESHOLD_COLD_START,
    MIN_MATCH_RAMP_SIGNATURES,
    RECURSIVE_DECOMPOSITION_ENABLED,
    RECURSIVE_MAX_DEPTH,
    RECURSIVE_CONFIDENCE_THRESHOLD,
    UMBRELLA_MAX_DEPTH,
    UMBRELLA_ROUTING_THRESHOLD,
    DEPTH_FORCE_DECOMPOSE_DEPTH,
    DEPTH_DECOMPOSE_DECAY_BASE,
    DEPTH_DECOMPOSE_MIN_PROB,
    ZERO_LLM_ROUTING_ENABLED,
    ZERO_LLM_MIN_SIMILARITY,
    ZERO_LLM_MIN_SUCCESS_RATE,
    ZERO_LLM_MIN_USES,
    ZERO_LLM_REQUIRE_DSL,
    DSL_EXPR_CACHE_MAX_SIZE,
    HINT_LIMIT,
    HINT_MIN_SIMILARITY,
)
from mycelium.planner import Planner, Step, DAGPlan
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.step_signatures.db import normalize_step_text
from mycelium.step_signatures.dsl_executor import DSLSpec, try_execute_dsl, try_execute_dsl_math
from mycelium.step_signatures.dsl_generator import regenerate_dsl
from mycelium.embedder import Embedder
from mycelium.embedding_cache import cached_embed

logger = logging.getLogger(__name__)


# =============================================================================
# SMOOTH EXPANSION RATE
# =============================================================================
# Per CLAUDE.md: "A SMOOTH and CONTINUOUS learning process is key"
#
# Formula: expansion_rate = (1 - accuracy) * (1 + k * exp(-sig_count / threshold))
#
# - Failure-driven: low accuracy → high expansion
# - Cold-start boost: few signatures → extra multiplier
# - Smooth taper: as system matures, expansion naturally decreases

# Caches for smooth expansion calculation
_signature_count_cache = {"count": 0, "last_check": 0}
_accuracy_cache = {"accuracy": 0.0, "successes": 0, "total": 0, "last_update": 0}
_reuse_cache = {"rate": 0.0, "matched": 0, "total_steps": 0}


def get_signature_count() -> int:
    """Get current signature count (cached for performance).

    Uses singleton ConnectionManager to avoid creating fresh connections.
    """
    import time
    from mycelium.data_layer import get_db

    now = time.time()
    # Cache for 1 second to avoid DB hits on every call
    if now - _signature_count_cache["last_check"] > 1.0:
        try:
            db = get_db()
            with db.connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()[0]
            _signature_count_cache["count"] = count
            _signature_count_cache["last_check"] = now
        except Exception as e:
            logger.warning("[solver] Failed to get signature count: %s", e)
    return _signature_count_cache["count"]


def get_adaptive_match_threshold() -> float:
    """Get cold-start aware match threshold.

    During cold start (few signatures), use HIGHER threshold to create more signatures.
    As DB matures, lower threshold to reduce fragmentation.

    This implements the "aggressive branching during cold start" principle from CLAUDE.md.
    """
    sig_count = get_signature_count()

    if sig_count >= MIN_MATCH_RAMP_SIGNATURES:
        return MIN_MATCH_THRESHOLD  # Mature: use lower threshold

    # Linear interpolation from cold start to mature threshold
    progress = sig_count / MIN_MATCH_RAMP_SIGNATURES  # 0.0 to 1.0
    threshold = MIN_MATCH_THRESHOLD_COLD_START - (progress * (MIN_MATCH_THRESHOLD_COLD_START - MIN_MATCH_THRESHOLD))
    return threshold


def update_accuracy(success: bool) -> float:
    """Update rolling accuracy with a new result.

    Uses exponential moving average for smooth tracking.
    Also records to AdaptiveExploration for MCTS parameter adaptation.

    Args:
        success: Whether the problem was solved correctly

    Returns:
        Current accuracy estimate
    """
    import time
    now = time.time()

    # Record to AdaptiveExploration for MCTS parameter adaptation
    from mycelium.mcts.adaptive import AdaptiveExploration
    AdaptiveExploration.get_instance().record_result(success)

    # Update counts
    _accuracy_cache["total"] += 1
    if success:
        _accuracy_cache["successes"] += 1

    # Calculate accuracy with smoothing to prevent wild swings early on
    # Blend with 20% prior baseline, decaying over first 10 problems
    total = _accuracy_cache["total"]
    if total > 0:
        raw_accuracy = _accuracy_cache["successes"] / total
        prior_weight = max(0, 10 - total) / 10
        _accuracy_cache["accuracy"] = prior_weight * 0.2 + (1 - prior_weight) * raw_accuracy

    _accuracy_cache["last_update"] = now
    return _accuracy_cache["accuracy"]


def get_accuracy() -> float:
    """Get current accuracy estimate."""
    return _accuracy_cache["accuracy"]


def update_reuse_rate(matched: int, total_steps: int) -> float:
    """Update reuse rate with results from a solved problem.

    Reuse rate = signatures_matched / total_steps
    Tracks how efficiently we're reusing existing signatures.

    Args:
        matched: Number of signatures matched in this problem
        total_steps: Total steps in this problem

    Returns:
        Current reuse rate estimate
    """
    _reuse_cache["matched"] += matched
    _reuse_cache["total_steps"] += total_steps

    if _reuse_cache["total_steps"] > 0:
        _reuse_cache["rate"] = _reuse_cache["matched"] / _reuse_cache["total_steps"]

    return _reuse_cache["rate"]


def get_reuse_rate() -> float:
    """Get current reuse rate (signatures_matched / total_steps)."""
    return _reuse_cache["rate"]


def get_expansion_rate() -> float:
    """Self-tuning expansion based on accuracy AND reuse efficiency.

    | Accuracy | Reuse | Action |
    |----------|-------|--------|
    | Low      | Low   | Slow down - fragmenting, not learning |
    | Low      | High  | Expand - existing sigs aren't enough |
    | High     | Any   | Minimal - we're doing well |

    Returns:
        Expansion rate in [0.05, 1.0]
    """
    import math

    accuracy = get_accuracy()
    reuse_rate = get_reuse_rate()
    sig_count = get_signature_count()

    # 1. Accuracy-driven sigmoid: base expansion from performance
    # At accuracy=0: ~1.0, at 0.7: 0.5, at 1.0: ~0
    accuracy_factor = 1.0 / (1.0 + math.exp((accuracy - 0.7) / 0.15))

    # 2. Reuse modulation: low reuse = fragmenting, slow down
    # At cold start (few sigs), ignore reuse (give it time to build up)
    cold_floor = math.exp(-sig_count / 100)
    effective_reuse = max(reuse_rate, cold_floor)

    # 3. Combine: accuracy determines desire, reuse gates it
    expansion = accuracy_factor * effective_reuse

    # 4. Cold-start boost for very few signatures
    cold_boost = 1.0 + math.exp(-sig_count / 3000)
    expansion = expansion * cold_boost

    # Clamp to bounds
    expansion = max(0.05, min(1.0, expansion))

    logger.debug(
        "[expansion] rate=%.2f (accuracy=%.2f, reuse=%.2f, sigs=%d)",
        expansion, accuracy, reuse_rate, sig_count
    )

    return expansion


def should_force_decompose(depth: int) -> bool:
    """Smooth expansion-based decomposition strategy.

    Uses continuous expansion rate (no toggle) to decide decomposition.
    Per CLAUDE.md: "A SMOOTH and CONTINUOUS learning process is key"

    The expansion rate is driven by:
    - Accuracy: failing → expand more
    - Signature count: cold start → extra boost

    Depth also factors in: shallow depths always decompose (routing layer),
    deeper depths respect the expansion rate (execution layer).

    Args:
        depth: Current signature depth in the hierarchy

    Returns:
        True if should force decompose, False if should try DSL execution
    """
    # Smooth expansion is always enabled per CLAUDE.md (no toggle)
    # Get smooth expansion rate
    expansion_rate = get_expansion_rate()

    # Apply depth decay uniformly from depth 0
    # This prevents cascade of decomposition at shallow depths
    # depth_factor: 1.0 at depth 0, decays by DECAY_BASE per depth
    depth_factor = DEPTH_DECOMPOSE_DECAY_BASE ** depth

    # Combined probability: expansion_rate * depth_factor
    # High expansion + shallow depth = higher prob
    # Low expansion OR deep depth = lower prob
    prob = max(DEPTH_DECOMPOSE_MIN_PROB, expansion_rate * depth_factor)

    if random.random() < prob:
        logger.debug(
            "[expansion] Decomposing: depth=%d prob=%.2f (expansion=%.2f, depth_factor=%.2f)",
            depth, prob, expansion_rate, depth_factor
        )
        return True

    return False


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    task: str
    result: str
    success: bool
    signature_id: Optional[int] = None
    signature_type: Optional[str] = None
    is_new_signature: bool = False
    was_injected: bool = False  # True if DSL was used
    was_routed: bool = False  # True if routed through umbrella
    elapsed_ms: float = 0.0


@dataclass
class SolverResult:
    """Result of solving a complete problem."""
    problem: str
    answer: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    total_steps: int = 0
    signatures_matched: int = 0
    signatures_new: int = 0
    steps_with_injection: int = 0
    steps_with_routing: int = 0  # Routed through umbrella (also counts as reuse)
    matched_and_reused: int = 0  # Matched AND (DSL succeeded OR routed)
    error: Optional[str] = None


# =============================================================================
# Solver
# =============================================================================

class Solver:
    """V2 Solver with Natural Language signatures.

    Simple flow:
    1. Plan: Decompose problem into steps
    2. Execute: For each step, find signature and execute
    3. Synthesize: Combine results into final answer
    """

    def __init__(
        self,
        solver_client=None,
        db_path: str = None,
        min_similarity: float = MIN_MATCH_THRESHOLD,
    ):
        """Initialize the solver.

        Args:
            solver_client: LLM client for step execution (optional, creates default if None)
            db_path: Path to signature database
            min_similarity: Minimum cosine similarity for signature matching
        """
        from mycelium.client import get_client

        self.planner = Planner()  # Uses its own client internally
        self.solver_client = solver_client or get_client()
        self.step_db = StepSignatureDB(db_path=db_path)
        self.embedder = Embedder.get_instance()
        self.min_similarity = min_similarity
        self._background_tasks: set[asyncio.Task] = set()  # Track background tasks
        # LRU cache for DSL expressions: (operation, param_names) -> (expr, used_params)
        # Bounded to DSL_EXPR_CACHE_MAX_SIZE to prevent memory growth
        self._dsl_expr_cache: OrderedDict[tuple[str, frozenset[str]], tuple[str, list[str]]] = OrderedDict()

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task with proper lifecycle management.

        Tasks are tracked to prevent garbage collection and enable clean shutdown.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> int:
        """Wait for pending background tasks to complete.

        Call this for clean shutdown. Returns count of tasks that were pending.
        """
        if not self._background_tasks:
            return 0
        pending = len(self._background_tasks)
        logger.debug("[solver] Waiting for %d background tasks", pending)
        done, not_done = await asyncio.wait(
            self._background_tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )
        if not_done:
            logger.warning("[solver] %d background tasks did not complete in time", len(not_done))
        return pending

    def _try_zero_llm_solve(
        self,
        problem: str,
        problem_embedding: np.ndarray,
    ) -> Optional[SolverResult]:
        """Attempt to solve without any LLM calls using mature signature tree.

        This is the "fast path" for problems that match mature signatures.
        Routes the problem embedding through the hierarchy and executes
        DSL directly if a high-confidence match is found.

        Args:
            problem: The problem text
            problem_embedding: Pre-computed embedding of the problem

        Returns:
            SolverResult if successful, None to fall back to planner
        """
        import time
        start_time = time.time()

        if not ZERO_LLM_ROUTING_ENABLED:
            return None

        # Route problem through signature hierarchy
        matched_sig, path = self.step_db.route_through_hierarchy(
            embedding=problem_embedding,
            min_similarity=ZERO_LLM_MIN_SIMILARITY,
        )

        if matched_sig is None:
            logger.debug("[zero-llm] No signature matched at threshold %.2f", ZERO_LLM_MIN_SIMILARITY)
            return None

        # Check if signature is mature enough
        if matched_sig.uses < ZERO_LLM_MIN_USES:
            logger.debug(
                "[zero-llm] Signature %s has only %d uses (need %d)",
                matched_sig.step_type, matched_sig.uses, ZERO_LLM_MIN_USES
            )
            return None

        success_rate = matched_sig.successes / matched_sig.uses if matched_sig.uses > 0 else 0
        if success_rate < ZERO_LLM_MIN_SUCCESS_RATE:
            logger.debug(
                "[zero-llm] Signature %s has %.1f%% success rate (need %.1f%%)",
                matched_sig.step_type, success_rate * 100, ZERO_LLM_MIN_SUCCESS_RATE * 100
            )
            return None

        # Check if signature is a leaf with DSL (not an umbrella that needs decomposition)
        if matched_sig.is_semantic_umbrella:
            logger.debug("[zero-llm] Matched signature %s is umbrella, need to decompose", matched_sig.step_type)
            return None

        if ZERO_LLM_REQUIRE_DSL and not matched_sig.dsl_script:
            logger.debug("[zero-llm] Signature %s has no DSL script", matched_sig.step_type)
            return None

        # Extract numeric values from problem text for DSL execution
        values = self._extract_values_from_problem(problem)
        if not values:
            logger.debug("[zero-llm] Could not extract values from problem")
            return None

        # Try to execute DSL with extracted values
        try:
            dsl_spec = DSLSpec.from_json(f'{{"type":"{matched_sig.dsl_type or "math"}","script":"{matched_sig.dsl_script}"}}')
            result, success = try_execute_dsl(dsl_spec, values, step_task=problem)

            if success and result is not None:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    "[zero-llm] SUCCESS: '%s' → %s (sig=%s, path_len=%d, %.1fms)",
                    problem[:50], result, matched_sig.step_type, len(path), elapsed_ms
                )

                # Record usage for learning
                self.step_db.record_usage(
                    matched_sig.id,
                    step_text=problem,
                    step_completed=True,
                    was_injected=True,
                )

                return SolverResult(
                    problem=problem,
                    answer=str(result),
                    success=True,
                    steps=[StepResult(
                        step_id="zero_llm",
                        task=problem,
                        result=str(result),
                        success=True,
                        signature_id=matched_sig.id,
                        signature_type=matched_sig.step_type,
                        is_new_signature=False,
                        was_injected=True,
                        elapsed_ms=elapsed_ms,
                    )],
                    elapsed_ms=elapsed_ms,
                    total_steps=1,
                    signatures_matched=1,
                    steps_with_injection=1,
                    matched_and_reused=1,
                )

        except Exception as e:
            logger.debug("[zero-llm] DSL execution failed: %s", e)

        return None

    def _extract_values_from_problem(self, problem: str) -> dict:
        """Extract numeric values from problem text for DSL execution.

        Uses simple heuristics to find numbers and assign them as params.
        For more complex extraction, falls back to planner.

        Args:
            problem: The problem text

        Returns:
            Dict of param names to values (e.g., {"value_1": 10, "value_2": 5})
        """
        import re

        # Find all numbers in the problem
        # Match integers and decimals, including negative numbers
        numbers = re.findall(r'-?\d+\.?\d*', problem)

        if not numbers:
            return {}

        # Convert to floats and assign generic param names
        values = {}
        for i, num_str in enumerate(numbers):
            try:
                num = float(num_str)
                # Use int if it's a whole number
                if num == int(num):
                    num = int(num)
                values[f"value_{i+1}"] = num
                # Also add step_N alias for compatibility
                values[f"step_{i+1}"] = num
            except ValueError:
                logger.debug("[solver] Non-numeric result at index %d: %s", i, str(result)[:50])
                continue

        return values

    async def solve(self, problem: str) -> SolverResult:
        """Solve a problem end-to-end.

        Args:
            problem: The problem text

        Returns:
            SolverResult with answer and step details
        """
        import time
        start_time = time.time()

        try:
            # 0. Embed problem (used for both zero-LLM and planner hints)
            # Use cached_embed to avoid redundant computation
            problem_embedding = cached_embed(problem, self.embedder)

            # 0.5. Try zero-LLM solve first (skip planner for mature signatures)
            zero_llm_result = self._try_zero_llm_solve(problem, problem_embedding)
            if zero_llm_result is not None:
                return zero_llm_result

            # 1. Plan: Decompose into steps (with signature hints for NL interface)
            signature_hints = self.step_db.get_signature_hints(
                limit=HINT_LIMIT,
                problem_embedding=problem_embedding,
                min_similarity=HINT_MIN_SIMILARITY,
            )

            # Decompose problem into steps
            plan = await self.planner.decompose(
                problem,
                signature_hints=signature_hints,
            )

            # Validate DAG structure before execution
            # Skip validation for single-step plans (no dependencies to check, no cycles possible)
            if len(plan.steps) <= 1:
                is_valid, errors = True, []
            else:
                is_valid, errors = plan.validate()
            if not is_valid:
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error=f"Invalid DAG: {'; '.join(errors)}",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            if not plan.steps:
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error="Planning failed: no steps generated",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            # Pre-warm DSL expression cache in parallel for independent steps
            await self._prewarm_dsl_cache(plan.steps)

            # 2. Execute steps in dependency order (parallel where possible)
            step_results = []
            step_results_by_id = {}  # step_id → StepResult for ordering
            context = {}  # step_id → result
            step_descriptions = {}  # step_id → task description (for NL param matching)
            signatures_new = 0
            signatures_matched = 0
            steps_with_injection = 0
            steps_with_routing = 0
            matched_and_reused = 0  # Matched AND (DSL succeeded OR routed)

            completed_ids = set()
            remaining_steps = list(plan.steps)

            while remaining_steps:
                # Find steps with all dependencies satisfied (ready to run)
                ready_steps = [
                    s for s in remaining_steps
                    if all(dep in completed_ids for dep in s.depends_on)
                ]

                if not ready_steps:
                    # No progress - cycle or missing dependency
                    logger.warning("[solver] DAG stuck: %d steps remaining with unmet deps", len(remaining_steps))
                    break

                # Execute ready steps in parallel
                async def execute_one(step):
                    step_context = {
                        dep: context[dep]
                        for dep in step.depends_on
                        if dep in context
                    }
                    step_desc_context = {
                        dep: step_descriptions[dep]
                        for dep in step.depends_on
                        if dep in step_descriptions
                    }
                    return step, await self._execute_step(step, problem, step_context, step_desc_context)

                if len(ready_steps) > 1:
                    logger.debug("[solver] Executing %d steps in parallel", len(ready_steps))

                results = await asyncio.gather(*[execute_one(s) for s in ready_steps])

                # Process results
                failed_step = None
                for step, result in results:
                    step_results_by_id[step.id] = result
                    completed_ids.add(step.id)
                    remaining_steps.remove(step)

                    # Track stats
                    if result.is_new_signature:
                        signatures_new += 1
                    else:
                        signatures_matched += 1
                        if result.was_injected or result.was_routed:
                            matched_and_reused += 1
                    if result.was_injected:
                        steps_with_injection += 1
                    if result.was_routed:
                        steps_with_routing += 1

                    # Store result and description for dependent steps
                    context[step.id] = result.result
                    step_descriptions[step.id] = step.task

                    # Track first failure
                    if not result.success and failed_step is None:
                        failed_step = (step, result)

                # Abort DAG on step failure (prevent cascading empty strings)
                if failed_step:
                    step, result = failed_step
                    logger.warning(
                        "[solver] Step failed, aborting DAG: step=%s task='%s'",
                        step.id, step.task[:50]
                    )
                    # Build step_results in original order
                    for s in plan.steps:
                        if s.id in step_results_by_id:
                            step_results.append(step_results_by_id[s.id])
                    elapsed_ms = (time.time() - start_time) * 1000
                    return SolverResult(
                        problem=problem,
                        answer="",
                        success=False,
                        error=f"Step {step.id} failed: {step.task[:100]}",
                        steps=step_results,
                        elapsed_ms=elapsed_ms,
                        total_steps=len(step_results),
                        signatures_matched=signatures_matched,
                        signatures_new=signatures_new,
                        steps_with_injection=steps_with_injection,
                        steps_with_routing=steps_with_routing,
                        matched_and_reused=matched_and_reused,
                    )

            # Build step_results in original order
            for s in plan.steps:
                if s.id in step_results_by_id:
                    step_results.append(step_results_by_id[s.id])

            # 3. Synthesize final answer
            final_answer = await self._synthesize(problem, step_results, context)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "[solver] Solved in %.0fms: steps=%d new=%d matched=%d reused=%d (dsl=%d, routed=%d)",
                elapsed_ms, len(step_results), signatures_new, signatures_matched,
                matched_and_reused, steps_with_injection, steps_with_routing
            )

            return SolverResult(
                problem=problem,
                answer=final_answer,
                success=True,
                steps=step_results,
                elapsed_ms=elapsed_ms,
                total_steps=len(step_results),
                signatures_matched=signatures_matched,
                signatures_new=signatures_new,
                steps_with_injection=steps_with_injection,
                steps_with_routing=steps_with_routing,
                matched_and_reused=matched_and_reused,
            )

        except Exception as e:
            logger.exception("[solver] Error solving problem")
            return SolverResult(
                problem=problem,
                answer="",
                success=False,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        depth: int = 0,
    ) -> StepResult:
        """Execute a single step.

        Flow:
        0. If composite (has sub_plan), recursively execute sub-plan first
        1. Embed step text
        2. Find or create signature
        3. If umbrella, route to child signature
        4. Try DSL if available (using NL interface for param matching)
        5. Fall back to LLM if needed
        6. Record usage

        Args:
            step: The step to execute
            problem: Original problem text
            context: step_id → result from previous steps
            step_descriptions: step_id → task description (for NL param matching)
            depth: Recursion depth for composite steps
        """
        step_descriptions = step_descriptions or {}
        import time
        start_time = time.time()

        # 0. Handle composite steps (recursive DAG of DAGs)
        if step.is_composite:
            return await self._execute_composite_step(step, problem, context, step_descriptions, depth)

        # 1. Normalize and embed step (strip numbers for better matching)
        # Use cached_embed to avoid redundant computation for repeated steps
        normalized_task = normalize_step_text(step.task)
        embedding = cached_embed(normalized_task, self.embedder)

        # 2. Find or create signature (use original text for description)
        # Pass extracted_values and dsl_hint from planner for bidirectional LLM-signature communication
        # Use adaptive threshold: higher during cold start (more signatures), lower when mature
        adaptive_threshold = get_adaptive_match_threshold()
        signature, is_new = await self.step_db.find_or_create_async(
            step_text=step.task,  # Keep original for description
            embedding=embedding,   # Use normalized embedding for matching
            min_similarity=adaptive_threshold,
            parent_problem=problem,
            origin_depth=depth,  # Track decomposition depth
            extracted_values=getattr(step, 'extracted_values', None),
            dsl_hint=getattr(step, 'dsl_hint', None),  # LLM → signature communication
        )

        logger.debug(
            "[solver] Step '%s' → signature '%s' (new=%s, umbrella=%s, dsl_type=%s, threshold=%.2f)",
            step.task[:40], signature.step_type, is_new, signature.is_semantic_umbrella,
            signature.dsl_type, adaptive_threshold
        )

        # 2.5. Auto-decompose if signature needs children
        # Case 1: decompose-type that isn't umbrella yet
        # Case 2: umbrella (possibly auto-demoted) with no children
        needs_decompose = False
        children = None  # Track fetched children to avoid redundant DB query
        if signature.dsl_type == "decompose" and not signature.is_semantic_umbrella:
            needs_decompose = True
            reason = "decompose type needs children"
        elif signature.is_semantic_umbrella:
            children = self.step_db.get_children(signature.id, for_routing=True)
            if not children:
                needs_decompose = True
                reason = "umbrella has no children (auto-demoted?)"

        if needs_decompose:
            logger.info(
                "[solver] Auto-decomposing '%s' (%s)",
                signature.step_type, reason
            )
            await self._auto_decompose_signature(signature)
            # Refresh signature and children after decomposition
            signature = self.step_db.get_signature(signature.id)
            children = None  # Will be re-fetched in _try_umbrella_routing

        # 3. If umbrella, try routing to child signature
        result = None
        was_injected = False
        was_routed = False  # Track if we routed through umbrella
        routed_signature = signature

        if signature.is_semantic_umbrella:
            child_result = await self._try_umbrella_routing(signature, step, problem, context, step_descriptions, embedding=embedding, children=children)
            if child_result is not None:
                result, routed_signature, was_injected = child_result
                was_routed = True  # Successfully routed through umbrella
                logger.info(
                    "[solver] Umbrella routed: '%s' → '%s'",
                    signature.step_type, routed_signature.step_type
                )

        # 4. Try DSL execution if not already routed
        # Key: Use dsl_hint from planner (LLM writes the expression)
        # Don't check routed_signature.dsl_script - umbrellas are pure routers with no DSL
        # Also handles extraction-only steps (no dsl_hint but has extracted_values)
        has_dsl_hint = getattr(step, 'dsl_hint', None) is not None
        has_extracted_values = bool(getattr(step, 'extracted_values', None))
        sig_depth = routed_signature.depth or 0
        at_shallow_depth = should_force_decompose(sig_depth)

        # Try DSL at all depths - if it works, use it
        # Also try for extraction-only steps (no hint but has values)
        if result is None and (has_dsl_hint or has_extracted_values):
            dsl_result = await self._try_dsl(routed_signature, step, context, step_descriptions)
            if dsl_result is not None:
                result = dsl_result
                was_injected = True
                logger.debug("[solver] DSL executed: %s", result[:50] if result else "")

        # 4.5. COLD START: Decompose at shallow depths ONLY ON FAILURE
        # If DSL succeeded, we have a working signature - no need to decompose
        # Only decompose when DSL failed to explore alternative paths
        if result is None and at_shallow_depth and not routed_signature.is_semantic_umbrella:
            logger.info(
                "[solver] Depth %d: decomposing '%s' (DSL failed)",
                sig_depth, routed_signature.step_type
            )
            await self._auto_decompose_signature(routed_signature)
            routed_signature = self.step_db.get_signature(routed_signature.id)

        # 4.6. If we don't have a result yet, try routing through umbrella
        if result is None and routed_signature.is_semantic_umbrella:
            child_result = await self._try_umbrella_routing(
                routed_signature, step, problem, context, step_descriptions, embedding=embedding
            )
            if child_result is not None:
                result, routed_signature, was_injected = child_result
                was_routed = True
                logger.info(
                    "[solver] Routed through umbrella to: '%s'",
                    routed_signature.step_type
                )

        # 4.7. Fallback: try DSL on umbrella itself (it may still have DSL from before promotion)
        if result is None and signature.is_semantic_umbrella and (has_dsl_hint or has_extracted_values):
            logger.debug("[solver] Trying umbrella's own DSL as fallback")
            dsl_result = await self._try_dsl(signature, step, context, step_descriptions)
            if dsl_result is not None:
                result = dsl_result
                was_injected = True
                routed_signature = signature  # Use original umbrella signature
                logger.info("[solver] Umbrella fallback DSL succeeded: %s", result[:30] if result else "")

        # 4.8. CREATE NEW CHILD ON ROUTING FAILURE (per CLAUDE.md: failing signatures decompose)
        # If umbrella routing failed (no matching child), create new child for current step
        # This grows the tree by adding specialized children to handle novel steps
        if result is None and routed_signature.is_semantic_umbrella:
            logger.info(
                "[solver] Router umbrella '%s' failed to route, creating new child for step '%s'",
                routed_signature.step_type, step.task[:40]
            )
            # Create new child signature directly under this umbrella
            new_child = self.step_db.create_signature(
                step_text=step.task,
                embedding=embedding,
                parent_id=routed_signature.id,
                origin_depth=depth + 1,
                extracted_values=getattr(step, 'extracted_values', None),
                dsl_hint=getattr(step, 'dsl_hint', None),
            )
            logger.info(
                "[solver] Created new child '%s' (id=%d) under umbrella '%s'",
                new_child.step_type, new_child.id, routed_signature.step_type
            )

            # Execute the new child's DSL
            dsl_result = await self._try_dsl(new_child, step, context, step_descriptions)
            if dsl_result is not None:
                result = dsl_result
                was_injected = True
                routed_signature = new_child
                logger.info("[solver] New child DSL succeeded: %s", result[:30] if result else "")

        # 5. No LLM fallback - strict DAG execution
        # Three outcomes: route to child, create child, or fail
        if result is None:
            logger.warning(
                "[solver] DSL failed, step failed (no LLM fallback): %s",
                step.task[:50]
            )
            # Record failure for pattern learning (per CLAUDE.md: failures are valuable data)
            self.step_db.record_failure(
                step_text=step.task,
                failure_type="dsl_error",
                error_message="DSL execution returned None",
                signature_id=routed_signature.id if routed_signature else None,
                context={
                    "problem": problem[:200] if problem else None,
                    "was_routed": was_routed,
                    "is_new": is_new,
                },
            )
            result = ""  # Empty result = failure

        # 6. Record usage (step_completed = returned result, not problem correctness)
        # Problem correctness is tracked separately via update_problem_outcome()
        step_completed = bool(result)
        uses = self.step_db.record_usage(
            signature_id=routed_signature.id,
            step_text=step.task,
            step_completed=step_completed,
            was_injected=was_injected,
        )

        # 7. Regenerate DSL on mod 10 uses (continuous learning)
        # Background task: don't block the hot path
        if uses > 0 and uses % 10 == 0:
            self._create_background_task(
                self._regenerate_dsl_background(routed_signature.id, uses)
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return StepResult(
            step_id=step.id,
            task=step.task,
            result=result or "",
            success=step_completed,
            signature_id=routed_signature.id,
            signature_type=routed_signature.step_type,
            is_new_signature=is_new,
            was_injected=was_injected,
            was_routed=was_routed,
            elapsed_ms=elapsed_ms,
        )

    async def _execute_composite_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        depth: int = 0,
    ) -> StepResult:
        """Execute a composite step by recursively executing its sub-plan.

        A composite step contains a sub-DAG. We:
        1. Execute all steps in the sub-plan (respecting dependencies)
        2. Use the sub-plan's final result as this step's result

        This enables unlimited recursive nesting: DAG of DAGs of DAGs...

        Args:
            step: The composite step to execute
            problem: Original problem text
            context: step_id → result from parent/sibling steps
            step_descriptions: step_id → task description (for NL param matching)
            depth: Recursion depth for nested composites
        """
        step_descriptions = step_descriptions or {}
        import time
        start_time = time.time()

        sub_plan = step.sub_plan
        logger.info(
            "[solver] Executing composite step '%s' with %d sub-steps (depth=%d)",
            step.id, len(sub_plan.steps), depth
        )

        # Execute sub-plan steps in dependency order
        sub_context = dict(context)  # Inherit parent context
        sub_step_descriptions = dict(step_descriptions)  # Inherit parent descriptions
        sub_results = []

        sub_execution_order = self._get_execution_order(sub_plan)

        for sub_step in sub_execution_order:
            # Build context from sub-plan dependencies
            step_context = {
                dep: sub_context[dep]
                for dep in sub_step.depends_on
                if dep in sub_context
            }
            # Also include parent context
            step_context.update({
                k: v for k, v in context.items()
                if k not in step_context
            })

            # Build step descriptions for semantic param matching
            step_desc_context = {
                dep: sub_step_descriptions[dep]
                for dep in sub_step.depends_on
                if dep in sub_step_descriptions
            }
            # Also include parent descriptions
            step_desc_context.update({
                k: v for k, v in step_descriptions.items()
                if k not in step_desc_context
            })

            # Recursively execute (handles nested composites)
            sub_result = await self._execute_step(
                sub_step, problem, step_context, step_desc_context, depth=depth + 1
            )
            sub_results.append(sub_result)

            # Abort composite on sub-step failure (prevent cascading empty strings)
            if not sub_result.success:
                logger.warning(
                    "[solver] Composite sub-step failed, aborting: sub_step=%s task='%s'",
                    sub_step.id, sub_step.task[:50]
                )
                elapsed_ms = (time.time() - start_time) * 1000
                return StepResult(
                    step_id=step.id,
                    task=step.task,
                    result="",  # Empty result = failure
                    success=False,
                    signature_id=None,
                    signature_type=f"composite[{len(sub_results)}]",
                    is_new_signature=False,
                    was_injected=False,
                    elapsed_ms=elapsed_ms,
                )

            # Store result and description for dependent sub-steps
            sub_context[sub_step.id] = sub_result.result
            sub_step_descriptions[sub_step.id] = sub_step.task

        # Aggregate sub-results into composite result
        # The final sub-step's result becomes this step's result
        final_result = sub_results[-1].result if sub_results else ""
        # Empty sub_results should be considered failure (not vacuous truth)
        all_success = bool(sub_results) and all(r.success for r in sub_results)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "[solver] Composite step '%s' completed: sub_steps=%d success=%s (%.0fms)",
            step.id, len(sub_results), all_success, elapsed_ms
        )

        return StepResult(
            step_id=step.id,
            task=step.task,
            result=final_result,
            success=all_success,
            signature_id=None,  # Composite steps don't have their own signature
            signature_type=f"composite[{len(sub_results)}]",
            is_new_signature=False,
            was_injected=False,
            elapsed_ms=elapsed_ms,
        )

    async def _try_umbrella_routing(
        self,
        umbrella: StepSignature,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        visited: Optional[set[int]] = None,
        embedding: Optional[np.ndarray] = None,
        depth: int = 0,
        children: Optional[list] = None,
    ) -> Optional[tuple[str, StepSignature, bool]]:
        """Try to route through umbrella to a child signature.

        Uses embedding similarity for fast routing (~0ms) instead of LLM calls.

        Returns (result, child_signature, was_injected) or None if no match.

        Args:
            umbrella: The umbrella signature to route through
            step: The step being executed
            problem: The original problem text
            context: Results from previous steps
            visited: Set of already-visited umbrella IDs (cycle detection)
            embedding: Step embedding for similarity-based routing
            depth: Current recursion depth (for limiting chain length)
            children: Pre-fetched children (avoids redundant DB query)
        """
        from mycelium.step_signatures.utils import cosine_similarity

        # Depth limit: prevent unbounded recursion through long umbrella chains
        if depth >= UMBRELLA_MAX_DEPTH:
            logger.warning(
                "[solver] Umbrella routing depth limit reached: depth=%d, umbrella=%s",
                depth, umbrella.step_type
            )
            return None

        # Cycle detection: prevent infinite recursion on malformed DAG
        if visited is None:
            visited = set()
        if umbrella.id in visited:
            logger.warning(
                "[solver] Cycle detected in umbrella routing: %d already visited",
                umbrella.id
            )
            return None
        visited.add(umbrella.id)

        # Use pre-fetched children if available, else fetch (avoids redundant query)
        # Per CLAUDE.md: "Umbrella routing should not require LLM call" - fast mode
        if children is None:
            children = self.step_db.get_children(umbrella.id, for_routing=True)
        if not children:
            return None

        # Embedding-based routing: compare step embedding to child centroids
        # This is ~0ms vs ~500ms for LLM routing
        if embedding is not None:
            best_child = None
            best_sim = 0.0
            best_condition = ""

            for child_sig, condition in children:
                # Capture centroid once to avoid TOCTOU race condition
                centroid = child_sig.centroid
                if centroid is not None:
                    sim = cosine_similarity(embedding, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_child = child_sig
                        best_condition = condition

            # Use embedding match if similarity is reasonable
            if best_child and best_sim > UMBRELLA_ROUTING_THRESHOLD:
                logger.debug(
                    "[solver] Umbrella routing (embedding): '%s' → '%s' (sim=%.3f)",
                    umbrella.step_type, best_child.step_type, best_sim
                )
                child_sig = best_child
            else:
                logger.debug(
                    "[solver] Umbrella routing: no good embedding match (best=%.3f)",
                    best_sim
                )
                return None
        else:
            # No embedding available - cannot route without LLM
            # Per CLAUDE.md: "Only call LLM on leaf nodes" + "Umbrella = Router"
            # Return None to trigger decomposition/failure (failures are valuable data)
            logger.debug(
                "[solver] Umbrella routing: no embedding available, cannot route"
            )
            return None

        # Recurse if child is also an umbrella (pass visited set, embedding, and depth)
        if child_sig.is_semantic_umbrella:
            return await self._try_umbrella_routing(
                child_sig, step, problem, context, step_descriptions, visited, embedding,
                depth=depth + 1
            )

        # Try child's DSL at all depths - if it works, use it
        # Use dsl_hint from step, not child_sig.dsl_script (umbrellas have no DSL)
        # Also handles extraction-only steps (no hint but has values)
        has_dsl_hint = getattr(step, 'dsl_hint', None) is not None
        has_extracted_values = bool(getattr(step, 'extracted_values', None))
        if has_dsl_hint or has_extracted_values:
            dsl_result = await self._try_dsl(child_sig, step, context, step_descriptions)
            if dsl_result is not None:
                return (dsl_result, child_sig, True)

        # Return child for further processing (may need decomposition on failure)
        return (None, child_sig, False)

    async def _try_dsl(
        self,
        signature: StepSignature,
        step: Step,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
    ) -> Optional[str]:
        """Try to execute a DSL script.

        LLM writes the arithmetic expression using available param names.
        No heuristic mapping - LLM always picks the right params for the task.

        Also handles extraction-only steps (no dsl_hint, just extracted_values).
        """
        # Get operation hint from planner
        dsl_hint = getattr(step, 'dsl_hint', None)
        extracted_values = getattr(step, 'extracted_values', {}) or {}

        # Handle extraction-only steps: no dsl_hint but has single extracted value
        # These steps just extract a constant from the problem (e.g., "eggs per day = 16")
        if not dsl_hint and extracted_values:
            # Find first non-reference value
            for key, val in extracted_values.items():
                if isinstance(val, (int, float)):
                    logger.info("[solver] Extraction-only step: %s = %s", key, val)
                    return str(val)
                elif isinstance(val, str) and val and not (val.startswith('{') and val.endswith('}')):
                    # Non-empty string value that's not a reference
                    try:
                        num_val = float(val)
                        logger.info("[solver] Extraction-only step: %s = %s", key, num_val)
                        return str(num_val)
                    except ValueError:
                        logger.debug("[solver] Non-numeric string value for %s: %s", key, val[:50])
            logger.debug("[solver] No extractable value found in extracted_values")
            return None

        if not dsl_hint:
            # No hint from planner - can't execute DSL
            logger.debug("[solver] No dsl_hint for step, skipping DSL")
            return None

        # Build available params from context + extracted values
        params = {}

        # Add validated extracted values (resolve references)
        # Note: extracted_values was already validated above using signature's param_descriptions
        if extracted_values:
            for key, val in extracted_values.items():
                if isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                    ref_key = val[1:-1]
                    if ref_key in context:
                        try:
                            params[key] = float(context[ref_key])
                        except (ValueError, TypeError):
                            logger.debug("[solver] Non-numeric ref %s=%s, keeping as-is", ref_key, str(context[ref_key])[:30])
                            params[key] = context[ref_key]
                else:
                    params[key] = val

        # Add context values (results from previous steps)
        for key, value in context.items():
            if key not in params:
                try:
                    params[key] = float(value)
                except (ValueError, TypeError):
                    logger.debug("[solver] Non-numeric context %s=%s, keeping as-is", key, str(value)[:30])
                    params[key] = value

        if len(params) < 2:
            # Single param = extraction step, just return the value
            # This handles cases where planner provides dsl_hint but only 1 value
            if len(params) == 1:
                val = list(params.values())[0]
                logger.info("[solver] Single-param extraction: %s", val)
                return str(val)
            logger.debug("[solver] Need at least 2 params for DSL, got %d", len(params))
            return None

        logger.debug("[solver] _try_dsl: hint=%s, params=%s", dsl_hint, list(params.keys()))

        # LLM writes the expression with correct params
        try:
            expr_result = await self._llm_write_expression(dsl_hint, params, step.task)
            if expr_result:
                script, used_params = expr_result
                logger.debug("[solver] LLM wrote: %s (used: %s)", script, used_params)

                dsl_spec = DSLSpec(
                    layer=DSLSpec.from_json('{"type":"math"}').layer,
                    script=script,
                    params=used_params,
                )

                result, success = try_execute_dsl(dsl_spec, params, step_task=step.task)
                logger.debug("[solver] DSL exec: result=%s, success=%s", result, success)

                if success and result is not None:
                    logger.info("[solver] DSL success: %s → %s", step.task[:30], result)
                    return str(result)
            else:
                logger.debug("[solver] LLM returned no expression")

        except Exception as e:
            logger.debug("[solver] DSL execution failed: %s", e)

        return None

    async def _prewarm_dsl_cache(self, steps: list) -> None:
        """Pre-warm DSL expression cache by parallelizing LLM calls for independent steps.

        Steps whose extracted_values don't reference {step_N} can have their
        expressions pre-computed in parallel, saving sequential LLM latency.
        """
        prewarm_tasks = []

        for step in steps:
            # Skip steps without dsl_hint
            if not step.dsl_hint:
                continue

            # Check if step has independent params (no {step_N} references)
            params = step.extracted_values or {}
            has_step_refs = any(
                isinstance(v, str) and v.startswith('{step_')
                for v in params.values()
            )

            if has_step_refs:
                continue  # Can't pre-compute, depends on prior results

            # Check if already cached
            param_names = frozenset(k for k in params.keys() if not k.startswith('{'))
            cache_key = (step.dsl_hint.strip().lower(), param_names)
            if cache_key in self._dsl_expr_cache:
                continue  # Already cached

            # Queue for parallel pre-warming
            prewarm_tasks.append(
                self._llm_write_expression(step.dsl_hint, params, step.task)
            )

        if prewarm_tasks:
            logger.debug("[solver] Pre-warming DSL cache: %d parallel calls", len(prewarm_tasks))
            await asyncio.gather(*prewarm_tasks, return_exceptions=True)

    async def _llm_write_expression(
        self,
        operation: str,
        params: dict,
        task: str,
    ) -> Optional[tuple[str, list[str]]]:
        """Ask LLM to write arithmetic expression using available params.

        Args:
            operation: The operation hint (+, -, *, /)
            params: Available param names and values
            task: The step task description

        Returns:
            (script, param_list) or None if failed
        """
        # Cache key: (operation, frozenset of param names)
        param_names = frozenset(k for k in params.keys() if not k.startswith('{'))
        cache_key = (operation.strip().lower(), param_names)

        # Check cache first - saves ~1-2s LLM call
        if cache_key in self._dsl_expr_cache:
            expr, used_params = self._dsl_expr_cache[cache_key]
            self._dsl_expr_cache.move_to_end(cache_key)  # LRU: mark as recently used
            logger.debug("[solver] DSL cache hit: %s -> %s", cache_key[0], expr)
            return expr, used_params

        # Format available params
        param_info = ", ".join(f"{k}={v}" for k, v in params.items() if not k.startswith('{'))

        prompt = f"""Write a simple arithmetic expression for this task.

Task: {task}
Operation: {operation}
Available values: {param_info}

Rules:
- Use EXACTLY the variable names provided (e.g., step_1, eggs_per_day)
- Write ONLY the expression, nothing else
- Example: step_1 + step_2
- Example: eggs_per_day * days

Expression:"""

        try:
            from mycelium.client import get_client
            client = get_client()
            response = await client.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )

            # Parse response - should be just the expression
            expr = response.strip().split('\n')[0].strip()

            # Extract param names used in expression
            used_params = [k for k in params.keys() if k in expr and not k.startswith('{')]

            if len(used_params) >= 2:
                logger.debug("[solver] LLM expression: %s (params: %s)", expr, used_params)
                # Cache for future use (bounded LRU)
                self._dsl_expr_cache[cache_key] = (expr, used_params)
                # Evict oldest entries if over max size
                while len(self._dsl_expr_cache) > DSL_EXPR_CACHE_MAX_SIZE:
                    self._dsl_expr_cache.popitem(last=False)
                return expr, used_params

        except Exception as e:
            logger.debug("[solver] LLM expression failed: %s", e)

        return None

    def _extract_json_result(self, response: str) -> str:
        """Extract result from JSON response (may be embedded in text)."""
        if not response:
            return ""

        import json

        def format_result(value):
            """Format a result value for output."""
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value == int(value):
                    return str(int(value))
                return str(value)
            return str(value).strip()

        # Try parsing entire response as JSON first
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                if "result" in data:
                    return format_result(data["result"])
                if "answer" in data:
                    return format_result(data["answer"])
        except json.JSONDecodeError:
            logger.debug("[solver] Direct JSON parse failed, trying brace matching")

        # Find JSON objects with balanced braces (handles nested objects)
        for key in ("result", "answer"):
            pattern = f'"{key}"'
            if pattern not in response:
                continue

            # Find all { positions and try parsing from each
            for i, char in enumerate(response):
                if char != '{':
                    continue
                # Try to find balanced closing brace
                depth = 0
                for j in range(i, len(response)):
                    if response[j] == '{':
                        depth += 1
                    elif response[j] == '}':
                        depth -= 1
                        if depth == 0:
                            # Found balanced braces, try parsing
                            candidate = response[i:j+1]
                            try:
                                data = json.loads(candidate)
                                if isinstance(data, dict) and key in data:
                                    return format_result(data[key])
                            except json.JSONDecodeError:
                                logger.debug("[solver] Brace-matched JSON parse failed at pos %d", i)
                            break

        # Fallback to regex extraction
        logger.debug("[solver] JSON extraction failed, using regex")
        return self._extract_result(response)

    def _extract_result(self, response: str) -> str:
        """Extract numeric/symbolic result from LLM response."""
        if not response:
            return ""

        # Clean up response
        text = response.strip()

        # Try to find boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try to find "= X" pattern
        equals_match = re.search(r'=\s*([^\n,;]+)$', text, re.MULTILINE)
        if equals_match:
            return equals_match.group(1).strip()

        # Try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        # Return cleaned text
        return text.split('\n')[0].strip()

    async def _synthesize(
        self,
        problem: str,
        step_results: list[StepResult],
        context: dict[str, str],
    ) -> str:
        """Synthesize final answer from step results using JSON mode."""
        # If only one step, return its result
        if len(step_results) == 1:
            return step_results[0].result

        # Build synthesis prompt
        prompt_parts = [
            "Based on these step results, provide the FINAL answer to the problem.",
            "",
            f"Problem: {problem}",
            "",
            "Step results:",
        ]

        for result in step_results:
            prompt_parts.append(f"  {result.step_id}: {result.task}")
            prompt_parts.append(f"    Result: {result.result}")

        # Request JSON output (include "JSON" for Groq compatibility)
        prompt_parts.append("")
        prompt_parts.append('Respond with valid JSON: {"result": <final_answer>}')

        prompt = "\n".join(prompt_parts)
        messages = [{"role": "user", "content": prompt}]

        response = await self.solver_client.generate(
            messages,
            response_format={"type": "json_object"},
        )
        return self._extract_json_result(response)

    def _get_execution_order(self, plan: DAGPlan) -> list[Step]:
        """Get steps in dependency-respecting execution order."""
        # Topological sort
        completed = set()
        order = []

        while len(order) < len(plan.steps):
            for step in plan.steps:
                if step.id in completed:
                    continue
                if all(dep in completed for dep in step.depends_on):
                    order.append(step)
                    completed.add(step.id)
                    break
            else:
                # No progress - cycle or missing dep
                remaining = [s for s in plan.steps if s.id not in completed]
                order.extend(remaining)
                break

        return order

    def record_problem_outcome(
        self,
        result: SolverResult,
        correct: bool,
    ) -> list[int]:
        """Propagate problem correctness to all signatures used.

        Call this after grading a problem to track real success rates.
        This enables negative lift detection for umbrella learning.

        Args:
            result: The SolverResult from solve()
            correct: Whether the final answer was correct

        Returns:
            List of signature IDs that may need decomposition (low confidence)
        """
        signature_ids = [
            step.signature_id
            for step in result.steps
            if step.signature_id is not None
        ]
        self.step_db.update_problem_outcome(signature_ids, correct)

        # Update rolling accuracy and reuse rate for self-tuning expansion
        current_accuracy = update_accuracy(correct)
        current_reuse = update_reuse_rate(result.signatures_matched, result.total_steps)
        expansion_rate = get_expansion_rate()
        logger.info(
            "[solver] Problem %s - accuracy=%.1f%%, reuse=%.1f%%, expansion=%.2f",
            "correct" if correct else "failed", current_accuracy * 100, current_reuse * 100, expansion_rate
        )

        # Log step-level details on failure for debugging
        if not correct:
            logger.warning(
                "[solver] Problem failed - steps involved: %s",
                [(s.step_id, s.signature_type, s.result[:30] if s.result else "None")
                 for s in result.steps]
            )

        # Check which signatures might need decomposition
        # Use same thresholds as umbrella_learner for consistency
        from mycelium.config import (
            UMBRELLA_MIN_USES_FOR_EVALUATION,
            UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION,
        )
        candidates = []
        for sig_id in signature_ids:
            sig = self.step_db.get_signature(sig_id)
            if sig and sig.dsl_type == "decompose" and sig.uses >= UMBRELLA_MIN_USES_FOR_EVALUATION:
                if sig.success_rate <= UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION and not sig.is_semantic_umbrella:
                    candidates.append(sig_id)
                    logger.info(
                        "[solver] Signature '%s' (id=%d) needs decomposition: "
                        "uses=%d, success_rate=%.1f%%",
                        sig.step_type, sig_id, sig.uses, sig.success_rate * 100
                    )

        return candidates

    async def _auto_decompose_signature(self, signature, recursion_depth: int = 0) -> bool:
        """Auto-decompose a decompose-type signature into computable children.

        Called when we encounter a decompose-type signature that needs children.
        Creates children with actual DSLs and promotes parent to umbrella.

        During BIG BANG phase, recursively decomposes children to explode tree structure.

        Args:
            signature: The decompose-type signature to decompose
            recursion_depth: Current recursion depth (to prevent runaway)

        Returns:
            True if decomposition succeeded, False otherwise
        """
        from mycelium.step_signatures.umbrella_learner import UmbrellaLearner

        # Hard limit on recursion to prevent runaway
        MAX_DECOMPOSE_RECURSION = 5
        if recursion_depth >= MAX_DECOMPOSE_RECURSION:
            logger.debug(
                "[solver] Decomposition recursion limit reached: depth=%d",
                recursion_depth
            )
            return False

        learner = UmbrellaLearner(self.step_db)
        try:
            child_ids = await learner.decompose_signature(signature)
            if child_ids:
                logger.info(
                    "[solver] Auto-decomposed '%s' into %d children (recursion=%d)",
                    signature.step_type, len(child_ids), recursion_depth
                )

                # Recursive decomposition: controlled by smooth expansion rate
                # High expansion (cold start/failing) = more recursive decomposition
                # Low expansion (mature/succeeding) = less recursive decomposition
                for child_id in child_ids:
                    child_sig = self.step_db.get_signature(child_id)
                    if child_sig and not child_sig.is_semantic_umbrella:
                        child_depth = child_sig.depth or 0
                        if should_force_decompose(child_depth):
                            logger.debug(
                                "[expansion] Recursive decompose: '%s' at depth %d",
                                child_sig.step_type, child_depth
                            )
                            await self._auto_decompose_signature(
                                child_sig, recursion_depth + 1
                            )

                return True
            else:
                logger.warning(
                    "[solver] Could not auto-decompose '%s' (no children created)",
                    signature.step_type
                )
                return False
        except Exception as e:
            logger.error("[solver] Auto-decomposition failed: %s", e)
            return False

    async def maybe_learn_umbrellas(self, candidates: list[int]) -> dict:
        """Trigger umbrella learning if there are candidates.

        Call this after record_problem_outcome() with its return value.

        Args:
            candidates: Signature IDs that may need decomposition

        Returns:
            Dict with learning statistics (empty if no candidates)
        """
        if not candidates:
            return {"candidates": 0, "decomposed": 0, "children_created": 0}

        logger.info("[solver] Auto-triggering umbrella learning for %d candidates", len(candidates))
        return await self.learn_umbrellas()

    async def _regenerate_dsl_background(self, signature_id: int, uses: int) -> None:
        """Background task to regenerate DSL without blocking hot path."""
        try:
            regenerated = await regenerate_dsl(
                db=self.step_db,
                client=self.solver_client,
                signature_id=signature_id,
            )
            if regenerated:
                logger.info(
                    "[solver] Regenerated DSL for signature %d at %d uses",
                    signature_id, uses
                )
        except Exception as e:
            logger.warning("[solver] DSL regeneration failed: %s", e)

    async def learn_umbrellas(self) -> dict:
        """Learn umbrella structure from failing guidance signatures.

        Call this periodically or after a batch of solves to:
        1. Find guidance signatures that are failing
        2. Decompose them into specialized children
        3. Future solves will route through umbrellas to children

        Returns:
            Dict with learning statistics
        """
        from mycelium.step_signatures.umbrella_learner import UmbrellaLearner

        learner = UmbrellaLearner(self.step_db)
        return await learner.learn_from_failures()

    def run_decay_cycle(self, force: bool = False) -> dict:
        """Run signature decay lifecycle management.

        Per CLAUDE.md: "slow decay: sig_uses / total_problems"

        This analyzes all signatures and:
        - Warns about signatures with declining traffic
        - Demotes umbrellas with no healthy children
        - Archives signatures that have been critical for too long
        - Tracks recovery when archived signatures revive

        Args:
            force: Run even if not enough time has passed since last run

        Returns:
            Dict with decay statistics (healthy, warning, critical, archived counts)
        """
        from mycelium.step_signatures.decay import run_decay_cycle

        report = run_decay_cycle(force=force)
        return {
            "total_signatures": report.total_signatures,
            "healthy": report.healthy_count,
            "warning": report.warning_count,
            "critical": report.critical_count,
            "archived": report.archived_count,
            "recovering": report.recovering_count,
            "actions_taken": len(report.actions_taken),
        }
