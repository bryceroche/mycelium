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
import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import random

from mycelium.config import (
    DB_PATH,
    MIN_MATCH_THRESHOLD,
    MIN_MATCH_THRESHOLD_COLD_START,
    MIN_MATCH_RAMP_SIGNATURES,
    UMBRELLA_MAX_DEPTH,
    UMBRELLA_ROUTING_THRESHOLD,
    DEPTH_DECOMPOSE_DECAY_BASE,
    DEPTH_DECOMPOSE_MIN_PROB,
    ZERO_LLM_ROUTING_ENABLED,
    ZERO_LLM_MIN_SIMILARITY,
    ZERO_LLM_MIN_SUCCESS_RATE,
    ZERO_LLM_MIN_USES,
    ZERO_LLM_REQUIRE_DSL,
    DSL_EXPR_CACHE_MAX_SIZE,
    COLD_START_HALFLIFE,
    HINT_LIMIT,
    HINT_MIN_SIMILARITY,
    THREAD_TRACKING_ENABLED,
    THREAD_MAX_FORKS_PER_STEP,
    THREAD_CREDIT_DECAY_PER_FORK,
    THREAD_MIN_CREDIT,
    GRAPH_ROUTING_ENABLED,
    GRAPH_ROUTING_MIN_SIMILARITY,
    GRAPH_ROUTING_BOOST_FACTOR,
    # Maturity sigmoid (decompose vs create new)
    MATURITY_DECOMPOSE_ENABLED,
    MATURITY_SIGMOID_MIDPOINT,
    MATURITY_SIGMOID_STEEPNESS,
    MATURITY_ACCURACY_WEIGHT,
    MATURITY_MIN_DECOMPOSE_PROB,
    MATURITY_MAX_DECOMPOSE_PROB,
    MATURITY_ESCAPE_MIN_SUBSTEPS,
    MATURITY_ESCAPE_MAX_MISSES,
)
from mycelium.planner import TreeGuidedPlanner, Step, DAGPlan
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.step_signatures.db import normalize_step_text
from mycelium.step_signatures.dsl_executor import DSLSpec, try_execute_dsl, try_execute_dsl_math
from mycelium.step_signatures.dsl_generator import regenerate_dsl
from mycelium.step_signatures.stats import record_step_stats
from mycelium.step_signatures.utils import cosine_similarity
from mycelium.embedder import Embedder
from mycelium.embedding_cache import cached_embed, cached_embed_batch
from mycelium.difficulty import estimate_difficulty
from mycelium.answer_norm import normalize_answer
from mycelium.data_layer.mcts import (
    create_dag,
    create_dag_steps,
    grade_dag,
    create_thread,
    complete_thread,
    log_thread_step,
    run_postmortem,  # Single pathway for all post-mortem analysis
    store_dag_step_embedding,
)

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

    # Defensive: ensure success is a bool (not string/int)
    success = bool(success)

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
    from mycelium.config import EXPANSION_SIGMOID_MIDPOINT, EXPANSION_SIGMOID_STEEPNESS

    accuracy = get_accuracy()
    reuse_rate = get_reuse_rate()
    sig_count = get_signature_count()

    # 1. Accuracy-driven sigmoid: base expansion from performance
    # Per mycelium-7khj: use config instead of hardcoded threshold
    # At accuracy=midpoint: 0.5, at accuracy=1.0: ~0
    accuracy_factor = 1.0 / (1.0 + math.exp((accuracy - EXPANSION_SIGMOID_MIDPOINT) / EXPANSION_SIGMOID_STEEPNESS))

    # 2. Reuse modulation: low reuse = fragmenting, slow down
    # At cold start (few sigs), ignore reuse (give it time to build up)
    cold_floor = math.exp(-sig_count / 100)
    effective_reuse = max(reuse_rate, cold_floor)

    # 3. Combine: accuracy determines desire, reuse gates it
    expansion = accuracy_factor * effective_reuse

    # 4. Cold-start boost for very few signatures
    cold_boost = 1.0 + math.exp(-sig_count / COLD_START_HALFLIFE)
    expansion = expansion * cold_boost

    # Clamp to bounds
    expansion = max(0.05, min(1.0, expansion))

    logger.debug(
        "[expansion] rate=%.2f (accuracy=%.2f, reuse=%.2f, sigs=%d)",
        expansion, accuracy, reuse_rate, sig_count
    )

    return expansion


def should_force_decompose(depth: int, step_db=None) -> bool:
    """Smooth expansion-based decomposition strategy.

    Thin wrapper around get_decomposition_decision() for signature context.

    Uses continuous expansion rate (no toggle) to decide decomposition.
    Per CLAUDE.md: "A SMOOTH and CONTINUOUS learning process is key"

    Args:
        depth: Current signature depth in the hierarchy
        step_db: Optional StepSignatureDB (unused, kept for API compatibility)

    Returns:
        True if should force decompose, False if should try DSL execution
    """
    # Delegate to unified decision interface
    decision = get_decomposition_decision(step_db, depth=depth, context="signature")

    if decision.should_decompose:
        logger.debug(
            "[expansion] Decomposing: depth=%d prob=%.2f reason=%s",
            depth, decision.probability, decision.reason
        )

    return decision.should_decompose


def compute_maturity_decompose_prob(step_db) -> float:
    """Compute probability of decomposing vs creating new when routing fails.

    Per mycelium-jaq9: Uses sigmoid based on system maturity:
        P(decompose) = sigmoid(maturity_score)
        maturity_score = (num_sigs - midpoint) / steepness + accuracy_weight * accuracy

    Early (few signatures): Low P(decompose) → create new signatures to bootstrap
    Mature (many signatures): High P(decompose) → reuse existing via decomposition

    Args:
        step_db: StepSignatureDB for signature count

    Returns:
        Probability of decomposing (0.0 to 1.0)
    """
    import math

    if not MATURITY_DECOMPOSE_ENABLED:
        return 0.0  # Disabled → always create new

    # Get system state
    sig_count = step_db.count_signatures() if step_db else 0
    # TODO: Wire up actual accuracy tracking from mcts/adaptive.py
    # For now, default to 0.0 (no accuracy boost - rely on signature count)
    accuracy = 0.0

    # Compute maturity score
    # Base: how far past midpoint in signature count
    # Boost: accuracy contribution (good accuracy → more likely to decompose)
    base_score = (sig_count - MATURITY_SIGMOID_MIDPOINT) / MATURITY_SIGMOID_STEEPNESS
    accuracy_boost = MATURITY_ACCURACY_WEIGHT * accuracy
    maturity_score = base_score + accuracy_boost

    # Apply sigmoid: 1 / (1 + exp(-x))
    try:
        raw_prob = 1.0 / (1.0 + math.exp(-maturity_score))
    except OverflowError:
        # exp overflow → maturity_score very negative → prob ≈ 0
        raw_prob = 0.0

    # Clamp to configured bounds
    prob = max(MATURITY_MIN_DECOMPOSE_PROB, min(MATURITY_MAX_DECOMPOSE_PROB, raw_prob))

    logger.debug(
        "[maturity] P(decompose)=%.2f (sigs=%d, accuracy=%.2f, score=%.2f)",
        prob, sig_count, accuracy, maturity_score
    )

    return prob


def should_try_decompose_first(step_db) -> bool:
    """Sample whether to try decomposition before creating new signature.

    Thin wrapper around get_decomposition_decision() for routing context.
    Per mycelium-jaq9: When routing fails, decide based on maturity sigmoid.

    Args:
        step_db: StepSignatureDB for maturity calculation

    Returns:
        True if should try decomposition, False if should create new
    """
    decision = get_decomposition_decision(step_db, context="routing")

    if decision.should_decompose:
        logger.debug(
            "[maturity] Decomposing first: prob=%.2f reason=%s",
            decision.probability, decision.reason
        )

    return decision.should_decompose


# =============================================================================
# DECOMPOSITION HIERARCHY
# =============================================================================
# All decomposition flows through planner.decompose() as the core LLM call.
#
# DECISION Functions (when/if to decompose):
#   - should_force_decompose(depth) → expansion-based for signature building
#   - should_try_decompose_first(step_db) → maturity-based for routing
#   - compute_maturity_decompose_prob(step_db) → raw probability calculation
#   - compute_decompose_score() [mcts.py] → continuous diagnostic score
#
# EXECUTION Functions (how to decompose):
#   - planner.decompose() → THE CORE: breaks problem into DAG via LLM
#   - _decompose_complex_step() → inline decomposition + execute sub-steps
#   - _try_maturity_decomposition() → decompose to reuse existing signatures
#   - _auto_decompose_signature() → build tree by decomposing signatures
#   - UmbrellaLearner.decompose_signature() → async signature decomposition
#
# Call hierarchy:
#   planner.decompose()
#   ├── _decompose_complex_step() → execute sub-steps recursively
#   ├── _try_maturity_decomposition() → route sub-steps through existing
#   └── UmbrellaLearner.decompose_signature() → create child signatures
#       └── _auto_decompose_signature() calls UmbrellaLearner


@dataclass
class DecompositionDecision:
    """Unified decomposition decision result."""
    should_decompose: bool
    reason: str  # Why we decided to decompose or not
    probability: float = 0.0  # Computed probability (for logging/debugging)
    depth: int = 0  # Depth context if applicable


def get_decomposition_decision(
    step_db,
    depth: int = 0,
    context: str = "routing",
) -> DecompositionDecision:
    """Unified decomposition decision interface.

    Consolidates all decomposition decision logic into a single entry point.
    Different contexts use different strategies:

    - "routing": Use maturity sigmoid (more signatures → prefer decompose)
    - "signature": Use expansion rate + depth decay (tree building)
    - "diagnostic": Use continuous score (failure analysis)

    Args:
        step_db: StepSignatureDB for computing metrics
        depth: Current depth in hierarchy (for signature context)
        context: Decision context ("routing", "signature", "diagnostic")

    Returns:
        DecompositionDecision with should_decompose and reason
    """
    if context == "routing":
        # Maturity-based: more signatures → prefer decomposing to reuse
        prob = compute_maturity_decompose_prob(step_db)
        should = random.random() < prob
        return DecompositionDecision(
            should_decompose=should,
            reason=f"maturity_sigmoid (prob={prob:.2f})",
            probability=prob,
            depth=depth,
        )

    elif context == "signature":
        # Expansion-based: use smooth expansion rate with depth decay
        expansion_rate = get_expansion_rate()
        depth_factor = DEPTH_DECOMPOSE_DECAY_BASE ** depth
        prob = max(DEPTH_DECOMPOSE_MIN_PROB, expansion_rate * depth_factor)
        should = random.random() < prob
        return DecompositionDecision(
            should_decompose=should,
            reason=f"expansion_rate (exp={expansion_rate:.2f}, depth={depth})",
            probability=prob,
            depth=depth,
        )

    else:
        # Default: don't decompose
        return DecompositionDecision(
            should_decompose=False,
            reason=f"unknown_context:{context}",
            probability=0.0,
            depth=depth,
        )


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
    # Preserve step data for reactive exploration retries
    dsl_hint: Optional[str] = None
    extracted_values: Optional[dict] = None
    depends_on: Optional[list[str]] = None


@dataclass
class DSLResult:
    """Result of DSL execution with expression/inputs for learning.

    Per mycelium-nvc9: DSL examples need to include the actual expression
    and inputs used, not just the result, so DSL regeneration can learn
    what worked.
    """
    result: Optional[str]  # The computed result (e.g., "15")
    expression: Optional[str] = None  # The DSL expression (e.g., "a * b")
    inputs: Optional[str] = None  # JSON: input values (e.g., '{"a": 5, "b": 3}')


@dataclass
class PathOutcome:
    """Outcome of a single path during multi-path MCTS exploration.

    Used to track which signatures produced which answers during exploration,
    enabling ground truth comparison for operational equivalence learning.

    Key insight: During training, ground truth lets us LABEL paths as operationally
    equivalent or different. Paths that produce the same correct answer are
    operationally equivalent even if their vocab differs.
    """
    signature_id: int
    answer: Optional[str]  # What this path produced (None if failed)
    step_id: str  # Which step this was for
    embedding_similarity: float = 0.0  # Cosine similarity to signature graph_embedding
    dsl_type: str = "unknown"  # DSL type for operational alignment tracking
    thread_id: str = ""  # Thread that explored this path (for multi-path credit)
    # Note: embedding field removed - centroid updates are no longer used (graph-only routing)


@dataclass
class ThreadContext:
    """Tracks a complete execution path through the DAG for multi-path credit/blame.

    A "thread" = one complete execution path through all DAG steps.
    When multi-path exploration forks at any step, each alternative becomes a child thread.
    Only one thread's answer becomes the final result (winner).

    After grading against ground truth, we know which threads were correct vs incorrect,
    enabling:
    - Positive credit to entire winning thread (not just final step)
    - Negative credit to losing threads (UCB1 stats update)
    - Per-signature thread win/loss tracking for cluster analysis
    """
    thread_id: str  # UUID
    parent_thread_id: Optional[str] = None  # Parent thread (if forked)
    fork_step_id: Optional[str] = None  # Step where this thread forked from parent
    fork_depth: int = 0  # How many forks from root thread
    # (sig_id, step_id, was_primary) - was_primary=True if this was the best/first choice at that step
    signature_steps: list[tuple[int, str, bool]] = field(default_factory=list)
    step_results: dict[str, str] = field(default_factory=dict)  # step_id -> result
    is_winner: bool = False  # Whether this thread produced the final answer
    final_answer: Optional[str] = None  # Answer produced by this thread
    created_at: str = ""  # ISO timestamp

    def add_signature(self, sig_id: int, step_id: str, was_primary: bool = True) -> None:
        """Record that a signature was used at a step in this thread."""
        self.signature_steps.append((sig_id, step_id, was_primary))

    @property
    def signature_ids(self) -> list[int]:
        """Get just the signature IDs (for backward compatibility)."""
        return [sig_id for sig_id, _, _ in self.signature_steps]

    def fork(self, step_id: str, new_signature_id: int) -> "ThreadContext":
        """Create a child thread at a fork point.

        The new signature at the fork is marked as non-primary (it's an alternative).

        Args:
            step_id: The step where the fork occurs
            new_signature_id: Signature ID for the new path

        Returns:
            New ThreadContext that branches from this thread
        """
        import uuid
        from datetime import datetime, timezone

        # Copy existing signature_steps and add the new one (marked as non-primary since it's a fork)
        new_sig_steps = self.signature_steps.copy()
        new_sig_steps.append((new_signature_id, step_id, False))  # False = not primary choice

        return ThreadContext(
            thread_id=str(uuid.uuid4()),
            parent_thread_id=self.thread_id,
            fork_step_id=step_id,
            fork_depth=self.fork_depth + 1,
            signature_steps=new_sig_steps,
            step_results=self.step_results.copy(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )


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

        self.solver_client = solver_client or get_client()
        self.step_db = StepSignatureDB(db_path=db_path)
        self.embedder = Embedder.get_instance()
        self.min_similarity = min_similarity
        # TreeGuidedPlanner uses step_db + embedder for vocabulary-guided decomposition
        self.planner = TreeGuidedPlanner(step_db=self.step_db, embedder=self.embedder)
        self._background_tasks: set[asyncio.Task] = set()  # Track background tasks
        # LRU cache for DSL expressions: (operation, param_names) -> (expr, used_params)
        # Bounded to DSL_EXPR_CACHE_MAX_SIZE to prevent memory growth
        self._dsl_expr_cache: OrderedDict[tuple[str, frozenset[str]], tuple[str, list[str]]] = OrderedDict()
        # MCTS multi-path outcomes: step_id -> list[PathOutcome]
        # Used for ground truth comparison to determine operational equivalence
        # Cleared after each problem is graded via record_problem_outcome()
        self._pending_path_outcomes: dict[str, list[PathOutcome]] = {}

        # Thread tracking state for multi-path credit/blame backpropagation
        # Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"
        # thread_id -> ThreadContext (tracks complete execution paths)
        self._active_threads: dict[str, ThreadContext] = {}
        # List of all thread IDs for current problem (for iteration during grading)
        self._problem_threads: list[str] = []
        # Root thread ID for current problem (parent of all forks)
        self._root_thread_id: Optional[str] = None

        # Routing context tracking for MCTS amplitude logging
        # Set during route_with_confidence, cleared at start of each step
        self._routing_confidence: float = 1.0
        self._routing_similarity: Optional[float] = None
        self._routing_ucb1_gap: Optional[float] = None
        self._routing_was_undecided: bool = False

        # Position-aware routing context (per plan_step_stats)
        # Set during step execution, used in route_with_confidence for position penalties
        self._current_step_position: Optional[int] = None  # 1, 2, 3... (None = unknown)

        # MCTS DAG tracking (set per-problem in solve())
        self._current_dag_id: Optional[str] = None
        self._dag_step_ids: dict[str, str] = {}  # step.id -> dag_step_id

        # Batch expressions: step_id -> (expression, params_used)
        # Set per-problem in solve() via _batch_write_expressions()
        self._batch_expressions: dict[str, tuple[str, list[str]]] = {}

        # Operation embeddings for graph-based routing (set per-problem in solve())
        # Stored separately from Step objects for memory efficiency (~24KB per embedding)
        self._operation_embeddings: dict[str, list[float]] = {}  # step.id -> embedding

        # DSL regeneration flag (per beads mycelium-flbq)
        # Set to True when post-mortem batch threshold reached
        self._pending_dsl_regen: bool = False

        # Nodes flagged for decomposition by post-mortem interference detection
        # These have operational_failures > 0 already set
        self._postmortem_flagged_nodes: list[int] = []

        # Reactive exploration context (per CLAUDE.md: explore alternatives on failure)
        # Stored after record_problem_outcome() for async processing
        self._pending_reactive_exploration: Optional[dict] = None

        # Reactive exploration mode: when True, use adaptive multipliers for exploration
        # Per mycelium-02nn: Replaces _force_exploration with Welford-guided thresholds
        # This multiplies gap threshold (more lenient) and budget (explore more paths)
        self._reactive_exploration_mode: bool = False

        # Current reactive exploration multipliers (Welford-adaptive, set when entering mode)
        # Per CLAUDE.md "The Flow": These come from DB Statistics → Welford
        self._reactive_gap_mult: float = 2.0
        self._reactive_budget_mult: float = 1.5

        # Phase 1 values for provenance tracking (set per-problem in solve())
        # Maps value name -> numeric value for resolving $name references
        self._current_phase1_values: dict[str, Any] = {}

        # DSL execution tracking for example storage (per beads mycelium-nvc9)
        # Stores DSLResult from last successful DSL execution for learning
        self._last_dsl_info: Optional[DSLResult] = None

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

    def _get_adaptive_gap_threshold(self) -> float:
        """Get adaptive UCB1 gap threshold for branching decisions.

        Per mycelium-02nn: Uses Welford stats to compute adaptive threshold.
        In reactive exploration mode, multiplies threshold using Welford-adaptive multipliers.

        Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure

        Returns:
            Gap threshold (branch when min_gap < threshold)
        """
        # Get base threshold from Welford stats
        base_threshold = self.step_db.get_adaptive_gap_threshold()

        # In reactive exploration, use Welford-adaptive multiplier
        if self._reactive_exploration_mode:
            threshold = base_threshold * self._reactive_gap_mult
            logger.debug(
                "[solver] Reactive exploration: gap threshold %.3f -> %.3f (mult=%.2f)",
                base_threshold, threshold, self._reactive_gap_mult
            )
            return threshold

        return base_threshold

    def _get_effective_budget(self, base_budget: float) -> float:
        """Get effective compute budget, boosted during reactive exploration.

        Per mycelium-02nn: In reactive exploration mode, boost budget using
        Welford-adaptive multipliers.

        Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure

        Args:
            base_budget: The base compute budget

        Returns:
            Effective budget (possibly boosted)
        """
        from mycelium.config import REACTIVE_EXPLORATION_MIN_BUDGET

        if self._reactive_exploration_mode:
            boosted = max(base_budget * self._reactive_budget_mult, REACTIVE_EXPLORATION_MIN_BUDGET)
            logger.debug(
                "[solver] Reactive exploration: budget %.1f -> %.1f (mult=%.2f, min=%.1f)",
                base_budget, boosted, self._reactive_budget_mult, REACTIVE_EXPLORATION_MIN_BUDGET
            )
            return boosted

        return base_budget

    def _try_zero_llm_solve(
        self,
        problem: str,
        problem_embedding: np.ndarray,
        difficulty: float = None,
    ) -> Optional[SolverResult]:
        """Attempt to solve without any LLM calls using mature signature tree.

        This is the "fast path" for problems that match mature signatures.
        Routes the problem embedding through the hierarchy and executes
        DSL directly if a high-confidence match is found.

        Args:
            problem: The problem text
            problem_embedding: Pre-computed embedding of the problem
            difficulty: Problem difficulty for tracking (0.0-1.0)

        Returns:
            SolverResult if successful, None to fall back to planner
        """
        import time
        start_time = time.time()

        if not ZERO_LLM_ROUTING_ENABLED:
            return None

        # Route problem through signature hierarchy
        matched_sig, path = self.step_db.route_through_hierarchy(
            operation_embedding=problem_embedding,
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

                # Record usage for learning (with difficulty for stats tracking)
                self.step_db.record_usage(
                    matched_sig.id,
                    step_text=problem,
                    step_completed=True,
                    was_injected=True,
                    difficulty=difficulty,
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
            # Record failure for pattern learning (per CLAUDE.md: failures are valuable data)
            self._record_failure(
                step_text=problem,
                failure_type="dsl_error",
                error_message=str(e),
                signature=sig,
                source="zero_llm",
                problem=problem,
            )

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

    async def solve(
        self,
        problem: str,
        compute_budget: float = None,
        benchmark: str = None,
        ground_truth: str = None,
    ) -> SolverResult:
        """Solve a problem end-to-end.

        Args:
            problem: The problem text
            compute_budget: MCTS exploration budget (None = adaptive based on difficulty)
                - None = adaptive budget (harder problems get more exploration)
                - 1.0 = single best path (backward compatible)
                - 2.0+ = explore multiple paths at low-confidence nodes
            benchmark: Dataset name (e.g., "gsm8k", "math500_L1") for MCTS tracking
            ground_truth: Correct answer for MCTS DAG grading

        Returns:
            SolverResult with answer and step details
        """
        from mycelium.config import COMPUTE_BUDGET_BASE, TRAINING_MODE
        from mycelium.difficulty import get_exploration_budget

        # Track if caller explicitly set budget (vs adaptive)
        explicit_budget = compute_budget is not None

        import time
        import uuid
        from datetime import datetime, timezone
        start_time = time.time()

        try:
            # Initialize thread tracking for this problem (if enabled and multi-path)
            # Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"
            if THREAD_TRACKING_ENABLED and TRAINING_MODE:
                self._root_thread_id = str(uuid.uuid4())
                root_thread = ThreadContext(
                    thread_id=self._root_thread_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                self._active_threads = {self._root_thread_id: root_thread}
                self._problem_threads = [self._root_thread_id]
            else:
                self._root_thread_id = None
                self._active_threads = {}
                self._problem_threads = []

            # 0. Embed problem (used for both zero-LLM and planner hints)
            # Use cached_embed to avoid redundant computation
            problem_embedding = cached_embed(problem, self.embedder)

            # 0.1. Estimate problem difficulty for adaptive behavior
            # Difficulty affects: depth, credit multiplier, routing preferences
            difficulty = estimate_difficulty(problem)
            logger.debug("[solver] Estimated difficulty: %.2f for problem: %s", difficulty, problem[:50])

            # 0.15. Create MCTS DAG record for this problem (training mode only)
            # Per CLAUDE.md: Track problem_id, problem_desc, benchmark, difficulty_level
            problem_id = hashlib.sha256(problem.encode()).hexdigest()[:16]
            if TRAINING_MODE:
                self._current_dag_id = create_dag(
                    problem_id=problem_id,
                    problem_desc=problem,
                    benchmark=benchmark,
                    difficulty_level=difficulty,
                    ground_truth=ground_truth,
                )
                logger.debug("[solver] Created MCTS DAG %s for problem %s", self._current_dag_id, problem_id)

                # Create root thread record for MCTS tracking
                # Per CLAUDE.md: "Create mcts_thread record for root thread at problem start"
                if THREAD_TRACKING_ENABLED and self._root_thread_id:
                    create_thread(
                        dag_id=self._current_dag_id,
                        parent_thread_id=None,
                        fork_at_step=None,
                        fork_reason=None,
                        thread_id=self._root_thread_id,
                    )
                    logger.debug("[solver] Created root thread %s for DAG %s", self._root_thread_id[:12], self._current_dag_id)
            else:
                self._current_dag_id = None

            # 0.2. Adaptive compute budget: scale by difficulty (if not explicitly set)
            # Per CLAUDE.md: "Multi step simulated mcts rollouts"
            # Both training AND inference use adaptive budget for accuracy
            # Difference: training records path outcomes for learning, inference doesn't
            if explicit_budget:
                pass  # Use caller's value
            else:
                compute_budget = get_exploration_budget(difficulty, base_budget=COMPUTE_BUDGET_BASE)
                logger.debug(
                    "[solver] Adaptive budget: %.1f (difficulty=%.2f, mode=%s)",
                    compute_budget, difficulty, "training" if TRAINING_MODE else "inference"
                )

            # 0.5. Try zero-LLM solve first (skip planner for mature signatures)
            zero_llm_result = self._try_zero_llm_solve(problem, problem_embedding, difficulty)
            if zero_llm_result is not None:
                return zero_llm_result

            # 1. Plan: Decompose into steps
            # Per CLAUDE.md "Negotiation between Tree and Planner":
            # TreeGuidedPlanner handles vocabulary internally through negotiation.
            # Don't pass signature_hints - let the planner negotiate with the tree.
            plan = await self.planner.decompose(problem)

            # Store Phase 1 values for provenance tracking during execution
            # These are resolved when $name references appear in extracted_values
            self._current_phase1_values = plan.phase1_values or {}
            if self._current_phase1_values:
                logger.debug(
                    "[solver] Phase 1 values for provenance: %s",
                    list(self._current_phase1_values.keys())
                )

            # Validate DAG structure before execution
            # Skip validation for single-step plans (no dependencies to check, no cycles possible)
            if len(plan.steps) <= 1:
                is_valid, errors = True, []
            else:
                is_valid, errors = plan.validate()
            if not is_valid:
                # Record validation failure (per CLAUDE.md: failures are valuable data)
                self._record_failure(
                    step_text=problem,
                    failure_type="validation",
                    error_message=f"Invalid DAG: {'; '.join(errors)}",
                    source="planner",
                    problem=problem,
                )
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error=f"Invalid DAG: {'; '.join(errors)}",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            if not plan.steps:
                # Record planning failure (per CLAUDE.md: failures are valuable data)
                self._record_failure(
                    step_text=problem,
                    failure_type="validation",
                    error_message="Planning failed: no steps generated",
                    source="planner",
                    problem=problem,
                )
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error="Planning failed: no steps generated",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            # Log DAG steps to mcts_dag_steps table (training mode only)
            # Per beads issue: wire up step logging when DAG plan is generated
            # Use execution order to get proper step_num (level) and branch_num (position in level)
            self._step_positions = {}  # step.id -> level_num (for position-aware routing)
            if TRAINING_MODE and self._current_dag_id:
                execution_levels = plan.get_execution_order()
                dag_step_tuples = []
                step_order = []  # Track step objects in same order as tuples
                for level_num, level_steps in enumerate(execution_levels, start=1):
                    for branch_num, step in enumerate(level_steps, start=1):
                        dag_step_tuples.append(
                            (step.task, level_num, branch_num, step.is_atomic, step.dsl_hint)
                        )
                        step_order.append(step)
                        # Track position for position-aware routing
                        self._step_positions[step.id] = level_num
                dag_step_ids = create_dag_steps(self._current_dag_id, dag_step_tuples)
                # Store mapping for thread step logging
                self._dag_step_ids = {
                    step.id: dag_step_id
                    for step, dag_step_id in zip(step_order, dag_step_ids)
                }
                logger.debug(
                    "[solver] Logged %d DAG steps for %s (%d levels)",
                    len(dag_step_ids), self._current_dag_id, len(execution_levels)
                )
            else:
                self._dag_step_ids = {}

            # Pre-warm embedding cache with batch call (avoids N sequential embed calls)
            self._prewarm_step_embeddings(plan.steps)

            # Store dag_step embeddings for decomposition decisions
            # Per CLAUDE.md: Track (dag_step, leaf_node) pairs to decide what to decompose
            if TRAINING_MODE and self._dag_step_ids:
                self._store_dag_step_embeddings(plan.steps)

            # Pre-warm operation embeddings for graph-based routing
            # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
            # Operations are extracted during decomposition, batch embedded here
            self._prewarm_operation_embeddings(plan.steps)

            # 1.4. BLOCKING DECOMPOSITION: Check for complex steps and decompose before execution
            # Per beads mycelium-mm08: Complex steps block until decomposed into atomic operations
            # This batches across multiple problems - steps queue up, decomposition fires when threshold met
            if TRAINING_MODE:
                from mycelium.client import LLMClient
                from mycelium.config import DECOMP_MIN_BATCH_SIZE, DECOMP_MAX_QUEUE_AGE_SEC
                async with LLMClient() as client:
                    queue_ids, decomp_results = await self._blocking_decompose_complex_steps(
                        plan=plan,
                        problem=problem,
                        client=client,
                        min_batch_size=DECOMP_MIN_BATCH_SIZE,
                        max_queue_age_sec=DECOMP_MAX_QUEUE_AGE_SEC,
                    )
                    if queue_ids:
                        plan = await self._expand_plan_with_decompositions(
                            plan, queue_ids, decomp_results
                        )

            # 1.5. BATCH EXPRESSION WRITING (single LLM call for all steps)
            # Collect step info for batch expression writing
            # Phase 1 values ($name) are resolved NOW, step refs ({step_N}) stay as placeholders
            step_infos = []
            for step in plan.steps:
                if not step.dsl_hint:
                    continue
                # Build params from extracted_values with Phase 1 resolution
                params = {}
                for key, val in (step.extracted_values or {}).items():
                    if isinstance(val, str):
                        if val.startswith('{') and val.endswith('}'):
                            # Reference to previous step - use step_N as placeholder
                            params[key] = val[1:-1]  # {step_1} -> step_1
                        elif val.startswith('$'):
                            # Phase 1 reference - resolve to actual numeric value
                            ref_name = val[1:]
                            if ref_name in self._current_phase1_values:
                                params[key] = self._current_phase1_values[ref_name]
                            else:
                                # Partial match fallback
                                matched_key = None
                                for p1_key in self._current_phase1_values:
                                    if p1_key in ref_name or ref_name in p1_key:
                                        matched_key = p1_key
                                        break
                                if matched_key:
                                    params[key] = self._current_phase1_values[matched_key]
                                else:
                                    logger.warning(
                                        "[solver] Batch expr: unknown Phase 1 ref $%s (available: %s)",
                                        ref_name, list(self._current_phase1_values.keys())
                                    )
                                    params[key] = val  # Keep as-is for debugging
                        else:
                            params[key] = val
                    else:
                        params[key] = val
                if params:
                    step_infos.append({
                        "step_id": step.id,
                        "task": step.task,
                        "operation": step.dsl_hint,
                        "params": params,
                    })

            # Single LLM call to write all expressions
            batch_expressions = {}
            if step_infos:
                batch_expressions = await self._batch_write_expressions(step_infos)
                logger.info(
                    "[solver] Batch expressions: %d/%d steps",
                    len(batch_expressions), len(step_infos)
                )

            # Store batch expressions for use during execution
            self._batch_expressions = batch_expressions

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
                    # Set current step position for position-aware routing
                    self._current_step_position = self._step_positions.get(step.id)
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
                    return step, await self._execute_step(
                        step, problem, step_context, step_desc_context,
                        compute_budget=compute_budget,
                        difficulty=difficulty,
                        thread_id=self._root_thread_id,
                    )

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

            # Update ALL threads' final_answer for thread tracking
            # Per CLAUDE.md: consolidate methods - all threads get the same final answer
            # so grading in _record_thread_outcomes compares correctly against ground_truth
            # (Bug fix: fork threads previously had step results, not problem final answers)
            for thread_id in self._problem_threads:
                thread = self._active_threads.get(thread_id)
                if thread:
                    thread.final_answer = final_answer

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "[solver] Solved in %.0fms: steps=%d new=%d matched=%d reused=%d (dsl=%d, routed=%d)",
                elapsed_ms, len(step_results), signatures_new, signatures_matched,
                matched_and_reused, steps_with_injection, steps_with_routing
            )

            # Auto-restructure check (per mycelium-heh3, CLAUDE.md System Independence)
            # Runs every N problems after cold start to reorganize tree structure
            if TRAINING_MODE:
                problem_count = self.step_db.get_total_problems_solved()
                restructure_result = self.step_db.maybe_restructure(problem_count)
                if restructure_result.get("ran"):
                    logger.info(
                        "[solver] Restructure: %d clusters, %d orphans cleaned",
                        restructure_result.get("clusters_created", 0),
                        restructure_result.get("orphans_cleaned", 0)
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
            # Record exception (per CLAUDE.md: failures are valuable data)
            self._record_failure(
                step_text=problem,
                failure_type="llm_error",
                error_message=str(e),
                source="solver_exception",
                problem=problem,
            )
            # Grade DAG as failed on exception (training mode only)
            if self._current_dag_id:
                grade_dag(self._current_dag_id, success=False)
                logger.debug("[solver] Graded MCTS DAG %s as failed (exception)", self._current_dag_id)
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
        compute_budget: float = 1.0,
        difficulty: float = None,
        thread_id: str = None,
        decomp_depth: int = 0,
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
            compute_budget: MCTS exploration budget (>1 enables multi-path)
            thread_id: Thread ID for multi-path credit tracking (None if not tracking)
            decomp_depth: Inline decomposition depth (to prevent infinite loops)
        """
        step_descriptions = step_descriptions or {}
        import time
        from mycelium.config import TRAINING_MODE
        start_time = time.time()

        # Reset routing context for MCTS amplitude logging
        # Updated during routing in _explore_multiple_paths, read at end for log_thread_step
        self._routing_confidence = 1.0
        self._routing_similarity = None
        self._routing_ucb1_gap = None
        self._routing_was_undecided = False

        # Clear DSL tracking from previous step (per beads mycelium-nvc9)
        self._last_dsl_info = None

        # 0. Handle composite steps (recursive DAG of DAGs)
        if step.is_composite:
            return await self._execute_composite_step(
                step, problem, context, step_descriptions, depth,
                compute_budget=compute_budget,
                thread_id=thread_id,
            )

        # 1. Get operation embedding for graph-based routing
        # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
        # Operation embeddings are pre-computed during decomposition (batch embedded)
        operation_embedding = None
        if step.id in self._operation_embeddings:
            operation_embedding = np.array(self._operation_embeddings[step.id])

        # 2. GRAPH-FIRST ROUTING (per mycelium-pl5c)
        # Check graph routing FIRST - route by what operations DO, not what they SOUND LIKE
        # If graph routing finds a high-confidence operational match, use it directly
        # This prevents creating new signatures for operationally identical steps
        signature = None
        is_new = False
        graph_matched = False

        if GRAPH_ROUTING_ENABLED and operation_embedding is not None:
            graph_results = self.step_db.route_by_graph_embedding(
                operation_embedding,
                min_similarity=GRAPH_ROUTING_MIN_SIMILARITY,
                top_k=3,
            )

            if graph_results:
                best_sig, best_sim = graph_results[0]
                # High-confidence graph match - use this signature directly
                # Per mycelium-7khj: use config instead of hardcoded threshold
                from mycelium.config import GRAPH_ROUTING_HIGH_CONFIDENCE
                if best_sim >= GRAPH_ROUTING_HIGH_CONFIDENCE:
                    # Check rejection threshold for leaf signatures using adaptive Welford threshold
                    # Per CLAUDE.md "New Favorite Pattern": use consolidated check_rejection
                    from mycelium.step_signatures.rejection_utils import check_rejection
                    from mycelium.config import COLD_START_SIGNATURE_THRESHOLD

                    # Cold start check: skip rejection while building vocabulary
                    sig_count = self.step_db.count_signatures()
                    is_cold_start = sig_count < COLD_START_SIGNATURE_THRESHOLD

                    # Check rejection using unified utility (uses config for k, min_samples, default_threshold)
                    rejection_result = check_rejection(
                        signature=best_sig,
                        similarity=best_sim,
                        is_cold_start=is_cold_start,
                        step_text=step.task,
                        problem_context=problem[:500] if problem else None,
                    )

                    if not best_sig.is_semantic_umbrella and rejection_result.rejected:
                        # Leaf rejects this step - try MCTS alternatives before decomposing
                        logger.info(
                            "[solver] GRAPH-FIRST REJECTED: '%s' by sig %d (%s) sim=%.3f < adaptive_threshold=%.3f (n=%d, mean=%.3f) - trying MCTS alternatives",
                            step.task[:40], best_sig.id, best_sig.step_type, best_sim, rejection_result.threshold,
                            best_sig.success_sim_count, best_sig.success_sim_mean
                        )

                        # MCTS fallback: get top-3 alternative leaves
                        # Per brainstorm: try re-routing sideways before decomposing depth-wise
                        # Use a slightly lower min_similarity to find alternatives
                        mcts_candidates = self.step_db.match_step_to_leaves_mcts(
                            operation_embedding=operation_embedding,
                            dag_step_type=getattr(step, 'dsl_hint', None) or step.task[:40],
                            top_k=3,
                            min_similarity=max(0.5, rejection_result.threshold - 0.1),  # Allow slightly lower alternatives
                        )

                        # Filter out the already-rejected signature
                        # Each alternative uses its OWN adaptive threshold via check_rejection
                        alt_candidates = []
                        for sig, ucb1, sim in mcts_candidates:
                            if sig.id == best_sig.id:
                                continue
                            # Check alternative without recording (just threshold check)
                            alt_result = check_rejection(
                                signature=sig,
                                similarity=sim,
                                is_cold_start=is_cold_start,
                                record=False,  # Don't record - just checking threshold
                            )
                            if not alt_result.rejected:
                                alt_candidates.append((sig, ucb1, sim))

                        if alt_candidates:
                            # Try best alternative (sideways re-routing)
                            alt_sig, alt_ucb1, alt_sim = alt_candidates[0]
                            logger.info(
                                "[solver] MCTS re-route: trying alternative sig %d (%s) ucb1=%.3f sim=%.3f",
                                alt_sig.id, alt_sig.step_type, alt_ucb1, alt_sim
                            )
                            # Use alternative signature instead of decomposing
                            signature = alt_sig
                            is_new = False
                            graph_matched = True
                        else:
                            # No viable alternatives - all leaves reject, decompose (depth)
                            logger.info(
                                "[solver] No MCTS alternatives above threshold - inline decomposition"
                            )
                            decomp_result = await self._decompose_complex_step(
                                step=step,
                                problem=problem,
                                context=context,
                                step_descriptions=step_descriptions or {},
                                hint=f"Rejected by all leaves (best sim={best_sim:.3f}). Break into atomic operations.",
                                log_tag="inline_decomp",
                                signature_type="decomposed",
                                difficulty=difficulty,
                                thread_id=thread_id,
                                decomp_depth=decomp_depth,
                            )
                            if decomp_result is not None:
                                logger.info("[solver] Inline decomposition succeeded for '%s'", step.task[:40])
                                return decomp_result
                            else:
                                logger.warning("[solver] Inline decomposition failed for '%s'", step.task[:40])
                                return StepResult(
                                    step_id=step.id,
                                    task=step.task,
                                    result="[decomposition failed]",
                                    success=False,
                                    signature_id=best_sig.id,
                                    signature_type=best_sig.step_type,
                                    is_new_signature=False,
                                    was_injected=False,
                                    elapsed_ms=(time.time() - start_time) * 1000,
                                )
                    else:
                        # Accept match (cold start or above threshold)
                        if is_cold_start and best_sim < adaptive_threshold:
                            logger.debug(
                                "[solver] Cold start: accepting low-sim match (sig_count=%d < %d, sim=%.3f < threshold=%.3f)",
                                sig_count, COLD_START_SIGNATURE_THRESHOLD, best_sim, adaptive_threshold
                            )
                        signature = best_sig
                        is_new = False
                        graph_matched = True
                        logger.info(
                            "[solver] GRAPH-FIRST match: '%s' → sig %d (%s) sim=%.3f",
                            step.task[:40], signature.id, signature.step_type, best_sim
                        )

        # 3. COLD START / SIGNATURE CREATION
        # If GRAPH-FIRST didn't match, try to find/create via graph-only routing in db.py
        # Per CLAUDE.md: routing uses graph_embedding exclusively (no text centroid fallback)
        if signature is None:
            adaptive_threshold = get_adaptive_match_threshold()
            signature, is_new = await self.step_db.find_or_create_async(
                step_text=step.task,  # Keep original for description
                min_similarity=adaptive_threshold,
                parent_problem=problem,
                origin_depth=depth,  # Track decomposition depth
                extracted_values=getattr(step, 'extracted_values', None),
                dsl_hint=getattr(step, 'dsl_hint', None),  # For graph routing
                embedder=self.embedder,  # For cold start graph embedding
            )

        # Handle rejection from routing (signature is None means step was rejected)
        if signature is None:
            logger.info(
                "[solver] Step '%s' rejected by routing - trying MCTS alternatives",
                step.task[:40]
            )

            # MCTS fallback: get top-3 alternative leaves before decomposing
            # Only try if we have an operation embedding
            # Use permissive min_similarity - each candidate uses its own adaptive threshold
            mcts_candidates = []
            if operation_embedding is not None:
                mcts_candidates = self.step_db.match_step_to_leaves_mcts(
                    operation_embedding=operation_embedding,
                    dag_step_type=getattr(step, 'dsl_hint', None) or step.task[:40],
                    top_k=3,
                    min_similarity=0.5,  # Permissive - adaptive threshold applied per-candidate
                )

            if mcts_candidates:
                # Found viable alternative - use it (sideways re-routing)
                alt_sig, alt_ucb1, alt_sim = mcts_candidates[0]
                logger.info(
                    "[solver] MCTS re-route from routing rejection: sig %d (%s) ucb1=%.3f sim=%.3f",
                    alt_sig.id, alt_sig.step_type, alt_ucb1, alt_sim
                )
                signature = alt_sig
                is_new = False
            else:
                # No alternatives - decompose (depth)
                logger.info("[solver] No MCTS alternatives - inline decomposition")
                decomp_result = await self._decompose_complex_step(
                    step=step,
                    problem=problem,
                    context=context,
                    step_descriptions=step_descriptions or {},
                    hint="Rejected by routing (no match above threshold). Break into atomic operations.",
                    log_tag="inline_decomp",
                    signature_type="decomposed",
                    difficulty=difficulty,
                    thread_id=thread_id,
                    decomp_depth=decomp_depth,
                )
                if decomp_result is not None:
                    logger.info("[solver] Inline decomposition succeeded for '%s'", step.task[:40])
                    return decomp_result
                else:
                    logger.warning("[solver] Inline decomposition failed for '%s'", step.task[:40])
                    return StepResult(
                        step_id=step.id,
                        task=step.task,
                        result="[decomposition failed]",
                        success=False,
                        signature_id=None,
                        signature_type=None,
                        is_new_signature=False,
                        was_injected=False,
                        elapsed_ms=(time.time() - start_time) * 1000,
                    )

        logger.debug(
            "[solver] Step '%s' → signature '%s' (new=%s, umbrella=%s, dsl_type=%s, graph_matched=%s)",
            step.task[:40], signature.step_type, is_new, signature.is_semantic_umbrella,
            signature.dsl_type, graph_matched
        )

        # 2.3. Maturity-based decompose vs create (per mycelium-jaq9)
        # When we create a NEW signature, maturity sigmoid decides whether to:
        # - Keep as atomic (early/cold start: build vocabulary)
        # - Try decomposition first (mature: reuse existing signatures)
        maturity_triggered_decompose = False
        if is_new and signature.dsl_type != "decompose" and should_try_decompose_first(self.step_db):
            logger.info(
                "[maturity] New signature '%s' created, maturity suggests trying decomposition first",
                signature.step_type
            )
            # Try to decompose and see if sub-steps match existing signatures
            decompose_result = await self._try_maturity_decomposition(
                step, signature, problem, context, step_descriptions or {}, difficulty
            )
            if decompose_result is not None:
                # Decomposition succeeded - all sub-steps matched existing signatures
                logger.info(
                    "[maturity] Decomposition succeeded - reusing existing signatures"
                )
                maturity_triggered_decompose = True
                # Return the decomposed result
                return StepResult(
                    step_id=step.id,
                    task=step.task,
                    result=decompose_result,
                    success=True,
                    signature_id=signature.id,
                    signature_type=signature.step_type,
                    is_new_signature=is_new,
                    was_injected=False,
                    was_routed=True,  # Decomposition is a form of routing
                )
            else:
                # Decomposition failed (escape hatch) - keep the new atomic signature
                logger.info(
                    "[maturity] Decomposition failed (escape hatch) - keeping atomic signature '%s'",
                    signature.step_type
                )

        # 2.5. Auto-decompose if signature needs children
        # Case 1: decompose-type that isn't umbrella yet
        # Case 2: umbrella (possibly auto-demoted) with no children
        # EXCEPTION: Brand new umbrellas (uses=0) should NOT auto-decompose
        #            Let them get their first child organically (cold-start protection)
        needs_decompose = False
        children = None  # Track fetched children to avoid redundant DB query
        if signature.dsl_type == "decompose" and not signature.is_semantic_umbrella:
            needs_decompose = True
            reason = "decompose type needs children"
        elif signature.is_semantic_umbrella:
            # Skip cache for this critical check - in multiprocess environments,
            # another process may have created children that our cache doesn't know about
            children = self.step_db.get_children(signature.id, for_routing=True, skip_cache=True)
            if not children:
                # Cold-start protection: Don't decompose brand new umbrellas
                # They need a chance to get children organically first
                if signature.uses == 0:
                    logger.info(
                        "[solver] Skipping auto-decompose for new umbrella '%s' (uses=0, cold-start)",
                        signature.step_type
                    )
                else:
                    needs_decompose = True
                    reason = "umbrella has no children (auto-demoted?)"

        if needs_decompose:
            logger.info(
                "[solver] Auto-decomposing '%s' (id=%d) (%s, difficulty=%.2f)",
                signature.step_type, signature.id, reason, difficulty or 0.0
            )
            await self._auto_decompose_signature(signature, difficulty=difficulty)
            # Refresh signature and children after decomposition
            signature = self.step_db.get_signature(signature.id)
            children = None  # Will be re-fetched in _try_umbrella_routing

        # 3. If umbrella, try routing to child signature
        result = None
        was_injected = False
        was_routed = False  # Track if we routed through umbrella
        routed_signature = signature

        if signature.is_semantic_umbrella:
            child_result = await self._try_umbrella_routing(signature, step, problem, context, step_descriptions, embedding=operation_embedding, children=children)
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
        # IMPORTANT: Skip DSL execution for umbrellas - they are routers, not executors
        explored_sigs = []  # Track all signatures explored (for backpropagation)
        if result is None and (has_dsl_hint or has_extracted_values) and not routed_signature.is_semantic_umbrella:
            # Multi-path exploration when budget > 1.0
            # Confidence check happens inside _explore_multiple_paths
            if compute_budget > 1.0:
                mp_result, mp_sig, explored_sigs, mp_injected = await self._explore_multiple_paths(
                    step, problem, context, step_descriptions or {},
                    operation_embedding, compute_budget, thread_id,
                )
                if mp_result is not None:
                    result = mp_result
                    routed_signature = mp_sig
                    was_injected = mp_injected
                    was_routed = True
                    logger.info(
                        "[solver] Multi-path found result via '%s' (explored %d sigs)",
                        mp_sig.step_type, len(explored_sigs)
                    )
            else:
                # Single-path: just try DSL on routed signature
                # Compute similarity for amplitude logging (not available from multi-path routing)
                # Use graph_embedding for routing (per CLAUDE.md: route by what operations DO)
                if routed_signature.graph_embedding is not None and operation_embedding is not None:
                    self._routing_similarity = cosine_similarity(operation_embedding, routed_signature.graph_embedding)
                    self._routing_confidence = self._routing_similarity  # Use similarity as confidence proxy
                result = await self._execute_dsl_and_record(routed_signature, step, context, step_descriptions)
                if result is not None:
                    was_injected = True
                    logger.debug("[solver] DSL executed: %s", result[:50] if result else "")

        # 4.5. COLD START: Decompose at shallow depths ONLY ON FAILURE
        # If DSL succeeded, we have a working signature - no need to decompose
        # Only decompose when DSL failed to explore alternative paths
        if result is None and at_shallow_depth and not routed_signature.is_semantic_umbrella:
            logger.info(
                "[solver] Depth %d: decomposing '%s' (DSL failed, difficulty=%.2f)",
                sig_depth, routed_signature.step_type, difficulty or 0.0
            )
            await self._auto_decompose_signature(routed_signature, difficulty=difficulty)
            routed_signature = self.step_db.get_signature(routed_signature.id)

        # 4.6. If we don't have a result yet, try routing through umbrella
        if result is None and routed_signature.is_semantic_umbrella:
            child_result = await self._try_umbrella_routing(
                routed_signature, step, problem, context, step_descriptions, embedding=operation_embedding
            )
            if child_result is not None:
                result, routed_signature, was_injected = child_result
                was_routed = True
                logger.info(
                    "[solver] Routed through umbrella to: '%s'",
                    routed_signature.step_type
                )

        # 4.7. CREATE NEW CHILD ON ROUTING FAILURE (per CLAUDE.md: failing signatures decompose)
        # If umbrella routing failed (no matching child), create new child for current step
        # This grows the tree by adding specialized children to handle novel steps
        if result is None and routed_signature.is_semantic_umbrella:
            from mycelium.config import NEW_CHILD_SIMILARITY_THRESHOLD

            # First check if there's an existing child that's "close enough"
            # Routing failed (similarity < UMBRELLA_ROUTING_THRESHOLD), but maybe
            # there's a child above NEW_CHILD_SIMILARITY_THRESHOLD we should use
            # instead of creating a duplicate
            existing_children = self.step_db.get_children(routed_signature.id, for_routing=True)
            best_existing_child = None
            best_existing_sim = 0.0

            for child_sig, _condition in existing_children:
                # Use graph_embedding for routing (per CLAUDE.md: route by what operations DO)
                if child_sig.graph_embedding is not None and operation_embedding is not None:
                    sim = cosine_similarity(operation_embedding, child_sig.graph_embedding)
                    if sim > best_existing_sim:
                        best_existing_sim = sim
                        best_existing_child = child_sig

            # Use existing child if similarity is above threshold (avoid duplicates)
            if best_existing_child and best_existing_sim >= NEW_CHILD_SIMILARITY_THRESHOLD:
                logger.info(
                    "[solver] Using existing child '%s' (sim=%.3f >= %.3f threshold) instead of creating new",
                    best_existing_child.step_type, best_existing_sim, NEW_CHILD_SIMILARITY_THRESHOLD
                )
                routed_signature = best_existing_child
                was_routed = True
            else:
                # No close-enough child exists, create new one
                logger.info(
                    "[solver] Router umbrella '%s' failed to route (best_sim=%.3f < %.3f), creating new child",
                    routed_signature.step_type, best_existing_sim, NEW_CHILD_SIMILARITY_THRESHOLD
                )
                new_child = self.step_db.create_signature(
                    step_text=step.task,
                    embedding=None,  # Graph-only routing, no text embedding needed
                    parent_id=routed_signature.id,
                    origin_depth=depth + 1,
                    extracted_values=getattr(step, 'extracted_values', None),
                    dsl_hint=getattr(step, 'dsl_hint', None),
                    embedder=self.embedder,  # For graph embedding
                )
                logger.info(
                    "[solver] Created new child '%s' (id=%d) under umbrella '%s'",
                    new_child.step_type, new_child.id, routed_signature.step_type
                )
                routed_signature = new_child

            # Execute the child's DSL (either existing or newly created)
            result = await self._execute_dsl_and_record(routed_signature, step, context, step_descriptions)
            if result is not None:
                was_injected = True
                logger.info("[solver] Child DSL succeeded: %s", result[:30] if result else "")

        # 5. No LLM fallback - strict DAG execution
        # Three outcomes: route to child, create child, or fail
        if result is None:
            logger.warning(
                "[solver] DSL failed, step failed (no LLM fallback): %s",
                step.task[:50]
            )
            # Record failure for pattern learning (per CLAUDE.md: failures are valuable data)
            self._record_failure(
                step_text=step.task,
                failure_type="dsl_error",
                error_message="DSL execution returned None",
                signature=routed_signature,
                problem=problem,
                extra_context={"was_routed": was_routed, "is_new": is_new},
            )
            result = ""  # Empty result = failure

        # 6. Record usage (step_completed = returned result, not problem correctness)
        # Problem correctness is tracked separately via update_problem_outcome()
        step_completed = bool(result)

        # 6.0. Update example with result (for DSL regeneration)
        # This records successful DSL outputs so regenerate_dsl can learn patterns
        # Per beads mycelium-nvc9: include expression/inputs for better learning
        if routed_signature and step_completed and result:
            dsl_info = getattr(self, '_last_dsl_info', None)
            self.step_db.update_example_result(
                signature_id=routed_signature.id,
                step_text=step.task,
                result=str(result),
                success=True,
                expression=dsl_info.expression if dsl_info else None,
                inputs=dsl_info.inputs if dsl_info else None,
            )
            # Clear the stored DSL info for next step
            self._last_dsl_info = None

        # 6.1. MCTS backpropagation: record usage for ALL explored signatures
        # Key insight: This is how multi-path exploration teaches cluster splitting
        # - Winning path signatures get step_completed=True
        # - Losing path signatures get step_completed=False
        if explored_sigs and len(explored_sigs) > 1:
            winning_id = routed_signature.id if routed_signature else None
            for sig in explored_sigs:
                sig_completed = (sig.id == winning_id) and step_completed
                self.step_db.record_usage(
                    signature_id=sig.id,
                    step_text=step.task,
                    step_completed=sig_completed,
                    was_injected=was_injected if sig.id == winning_id else False,
                    difficulty=difficulty,
                )
            logger.debug(
                "[solver] MCTS backprop: recorded %d explored sigs, winner=%s",
                len(explored_sigs), winning_id
            )
            uses = routed_signature.uses + 1 if routed_signature else 0
        else:
            # Single-path: record normally
            uses = self.step_db.record_usage(
                signature_id=routed_signature.id,
                step_text=step.task,
                step_completed=step_completed,
                was_injected=was_injected,
                difficulty=difficulty,
            )

        # Record execution outcome in Welford stats (per periodic tree review plan)
        # This tracks the execution success rate for each signature
        if routed_signature:
            self.step_db.update_welford_exec(routed_signature.id, success=step_completed)

        # 7. Regenerate DSL on mod 10 uses (continuous learning)
        # Background task: don't block the hot path
        if uses > 0 and uses % 10 == 0:
            self._create_background_task(
                self._regenerate_dsl_background(routed_signature.id, uses)
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # 8. Record step stats (feature-flagged, non-blocking)
        record_step_stats(
            db_path=DB_PATH,
            step_text=step.task,
            latency_ms=elapsed_ms,
            signature_id=routed_signature.id if routed_signature else None,
            routing_depth=routed_signature.depth if routed_signature else 0,
            was_routed=was_routed,
            success=step_completed,
            used_dsl=was_injected,
        )

        # 9. Track signature in thread (for single-path and multi-path credit tracking)
        # This ensures ALL routing is tracked, not just multi-path exploration
        if thread_id and routed_signature:
            thread = self._active_threads.get(thread_id)
            if thread:
                # Only add if not already tracked (multi-path adds in _explore_multiple_paths)
                if not any(sig_id == routed_signature.id and s_id == step.id
                          for sig_id, s_id, _ in thread.signature_steps):
                    thread.add_signature(routed_signature.id, step.id, was_primary=True)
                thread.step_results[step.id] = result or ""

        # 10. Log MCTS thread step with amplitude (training mode only)
        # Per beads issue: Wire up mcts_thread_steps amplitude logging
        # Key fields: amplitude (prior confidence), was_undecided, ucb1_gap, similarity_score, node_id
        if TRAINING_MODE and thread_id and routed_signature and self._current_dag_id:
            dag_step_id = self._dag_step_ids.get(step.id)
            if dag_step_id:
                log_thread_step(
                    thread_id=thread_id,
                    dag_id=self._current_dag_id,
                    dag_step_id=dag_step_id,
                    node_id=routed_signature.id,
                    amplitude=self._routing_confidence,
                    similarity_score=self._routing_similarity,
                    was_undecided=self._routing_was_undecided,
                    ucb1_gap=self._routing_ucb1_gap,
                    alternatives_considered=len(explored_sigs) if explored_sigs else 1,
                    step_result=result[:500] if result else None,
                    step_success=step_completed,
                    node_depth=routed_signature.depth,
                )
                logger.debug(
                    "[solver] Logged thread step: thread=%s dag_step=%s node=%d amp=%.2f undecided=%s success=%s",
                    thread_id[:12] if thread_id else None,
                    dag_step_id[:12] if dag_step_id else None,
                    routed_signature.id,
                    self._routing_confidence,
                    self._routing_was_undecided,
                    step_completed,
                )

                # Update dag_step embedding with node_id (for decomposition decisions)
                from mycelium.data_layer.mcts import update_dag_step_embedding_outcome
                try:
                    update_dag_step_embedding_outcome(
                        dag_step_id=dag_step_id,
                        node_id=routed_signature.id,
                        success=step_completed,
                    )
                except Exception as e:
                    logger.debug("[solver] Failed to update dag_step embedding: %s", e)

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
            # Preserve step data for reactive exploration retries
            dsl_hint=getattr(step, 'dsl_hint', None),
            extracted_values=getattr(step, 'extracted_values', None),
            depends_on=step.depends_on,
        )

    async def _execute_composite_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        depth: int = 0,
        compute_budget: float = 1.0,
        thread_id: str = None,
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
            compute_budget: MCTS exploration budget (>1 enables multi-path)
            thread_id: Thread ID for multi-path credit tracking (None if not tracking)
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
                sub_step, problem, step_context, step_desc_context, depth=depth + 1,
                compute_budget=compute_budget,
                thread_id=thread_id,
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

        # Graph-embedding routing: compare operation embedding to child graph_embeddings
        # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
        # This is ~0ms vs ~500ms for LLM routing
        if embedding is not None:
            child_scores = []
            for child_sig, condition in children:
                # Use graph_embedding for routing (captures operational semantics)
                graph_emb = child_sig.graph_embedding
                if graph_emb is not None:
                    sim = cosine_similarity(embedding, graph_emb)
                    child_scores.append((child_sig, sim, condition))

            # Sort by similarity descending
            child_scores.sort(key=lambda x: x[1], reverse=True)

            # Use best match if available
            best_child = child_scores[0][0] if child_scores else None
            best_sim = child_scores[0][1] if child_scores else 0.0

            # Use embedding match if similarity is reasonable
            if best_child and best_sim > UMBRELLA_ROUTING_THRESHOLD:
                logger.debug(
                    "[solver] Umbrella routing (embedding): '%s' → '%s' (sim=%.3f)",
                    umbrella.step_type, best_child.step_type, best_sim
                )
                # Per CLAUDE.md "System Independence": Update Welford route stats
                # This tracks the umbrella's routing similarity distribution for adaptive thresholds
                self.step_db.update_welford_route(umbrella.id, best_sim)
                child_sig = best_child
            else:
                logger.debug(
                    "[solver] Umbrella routing: no good embedding match (best=%.3f)",
                    best_sim
                )
                # Record routing failure (per CLAUDE.md: failures are valuable data)
                self._record_failure(
                    step_text=step.task,
                    failure_type="routing",
                    error_message=f"No good embedding match (best={best_sim:.3f})",
                    signature=umbrella,
                    source="umbrella_routing",
                    extra_context={"umbrella": umbrella.step_type, "best_sim": best_sim},
                )
                return None
        else:
            # No embedding available - cannot route without LLM
            # Per CLAUDE.md: "Only call LLM on leaf nodes" + "Umbrella = Router"
            # Return None to trigger decomposition/failure (failures are valuable data)
            logger.debug(
                "[solver] Umbrella routing: no embedding available, cannot route"
            )
            # Record routing failure (per CLAUDE.md: failures are valuable data)
            self._record_failure(
                step_text=step.task,
                failure_type="routing",
                error_message="No embedding available for routing",
                signature=umbrella,
                source="umbrella_routing",
                extra_context={"umbrella": umbrella.step_type},
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
            result = await self._execute_dsl_and_record(child_sig, step, context, step_descriptions)
            if result is not None:
                return (result, child_sig, True)

        # Return child for further processing (may need decomposition on failure)
        return (None, child_sig, False)

    async def _explore_multiple_paths(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str],
        operation_embedding: Optional[np.ndarray],
        compute_budget: float = 1.0,
        thread_id: str = None,
    ) -> tuple[Optional[str], StepSignature, list[StepSignature], bool]:
        """MCTS multi-path exploration when confidence is low.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Uses operation_embedding (graph-based) for routing, not text embedding.

        Uses route_with_confidence() to get alternatives, then explores
        up to `compute_budget` paths in parallel. Returns best result.

        Key for training: Exploring multiple paths reveals which routes
        lead to different outcomes, helping split vocab-based clusters
        into operation-based clusters.

        When thread tracking is enabled, creates fork threads for each
        alternative path to enable per-signature thread win/loss tracking.

        Args:
            step: The step being executed
            problem: Original problem text
            context: Results from previous steps
            step_descriptions: Task descriptions for NL param matching
            operation_embedding: Operation embedding for graph-based routing (may be None)
            compute_budget: Max paths to explore (1.0 = single path)
            thread_id: Thread ID for multi-path credit tracking (None if not tracking)

        Returns:
            Tuple of (result, best_signature, explored_sigs, was_injected)
            - result: DSL execution result (or None if all failed)
            - best_signature: The signature that produced the result
            - explored_sigs: All signatures explored (for backpropagation)
            - was_injected: Whether result came from DSL execution
        """
        from mycelium.step_signatures.db import RoutingResult

        # Handle case where no operation embedding available
        if operation_embedding is None:
            # Can't route without operation embedding - return empty result
            logger.debug(
                "[solver] No operation embedding for multi-path - returning empty result"
            )
            return (None, None, [], False)

        # Get routing result with confidence and alternatives
        # Pass dag_step_type to enable step-node stats lookup for routing decisions
        # Per CLAUDE.md: "(dag_step_id, node_id) is what we're learning"
        dag_step_type = getattr(step, 'dsl_hint', None) or getattr(step, 'task', None)
        routing_result = self.step_db.route_with_confidence(
            operation_embedding,
            min_similarity=get_adaptive_match_threshold(),
            top_k=int(compute_budget) + 1,  # Get enough alternatives
            dag_step_type=dag_step_type,
            step_position=self._current_step_position,  # Position-aware routing
        )

        # Store routing context for MCTS amplitude logging
        self._routing_confidence = routing_result.confidence
        self._routing_ucb1_gap = routing_result.min_gap
        self._routing_similarity = routing_result.best_similarity  # Actual cosine similarity

        explored_sigs = list(routing_result.path)  # Start with best path

        # Graph-based routing: Route by what operations DO, not what they SOUND LIKE
        # Per CLAUDE.md: "embedding clusters by vocab not operational semantics" (problem to solve)
        # Graph routing addresses this by comparing operation embedding to graph embeddings
        graph_matches = {}  # sig_id -> similarity for graph routing boost
        graph_signatures = {}  # sig_id -> StepSignature for graph-only candidates
        if GRAPH_ROUTING_ENABLED and operation_embedding is not None:
            try:
                # Compare to graph embeddings (what DSLs actually compute)
                graph_routing_results = self.step_db.route_by_graph_embedding(
                    operation_embedding,
                    min_similarity=GRAPH_ROUTING_MIN_SIMILARITY,
                    top_k=5,
                )
                # Build lookup for boosting centroid candidates
                for sig, sim in graph_routing_results:
                    graph_matches[sig.id] = sim
                    graph_signatures[sig.id] = sig  # Store signature for graph-only adds
                    logger.debug(
                        "[solver] Graph routing match: sig %d (%s) sim=%.3f",
                        sig.id, sig.computation_graph[:30] if sig.computation_graph else "?", sim
                    )
            except Exception as e:
                # Graph routing is optional - don't fail on errors
                logger.debug("[solver] Graph routing failed (continuing): %s", e)

        # Selective branching: only branch when undecided (per CLAUDE.md)
        # Use UCB1 gap to detect uncertainty: high gap = confident, low gap = undecided
        # Also respect single-path mode (compute_budget <= 1.0)
        # Per mycelium-02nn: Use adaptive threshold (Welford-guided) instead of static
        adaptive_gap_threshold = self._get_adaptive_gap_threshold()
        effective_budget = self._get_effective_budget(compute_budget)
        is_undecided = routing_result.min_gap < adaptive_gap_threshold
        if not is_undecided or effective_budget <= 1.0:
            if routing_result.signature is not None:
                result = await self._execute_dsl_and_record(
                    routing_result.signature, step, context, step_descriptions
                )
                if result is not None:
                    return (result, routing_result.signature, explored_sigs, True)
                return (None, routing_result.signature, explored_sigs, False)
            return (None, routing_result.signature, explored_sigs, False)

        # Undecided (low UCB1 gap) + multi-path mode: explore alternatives
        # Mark as undecided for MCTS amplitude tracking
        self._routing_was_undecided = True
        num_paths = min(int(effective_budget), len(routing_result.alternatives) + 1)
        reactive_tag = " [REACTIVE]" if self._reactive_exploration_mode else ""
        logger.info(
            "[solver] Selective branching%s: gap=%.3f (threshold=%.3f), exploring %d paths",
            reactive_tag, routing_result.min_gap, adaptive_gap_threshold, num_paths
        )

        # Collect alternative leaf signatures to try (with similarity scores)
        candidates = []  # List of (signature, similarity_score, graph_boost_applied)

        def _apply_graph_boost(sig_id: int, base_score: float) -> tuple[float, bool]:
            """Apply graph routing boost if signature appears in graph matches."""
            if sig_id in graph_matches:
                graph_sim = graph_matches[sig_id]
                boosted = base_score + (GRAPH_ROUTING_BOOST_FACTOR * graph_sim)
                logger.debug(
                    "[solver] Graph boost: sig %d score %.3f -> %.3f (graph_sim=%.3f)",
                    sig_id, base_score, boosted, graph_sim
                )
                return (boosted, True)
            return (base_score, False)

        # Add best path's leaf if it's not an umbrella
        if routing_result.signature and not routing_result.signature.is_semantic_umbrella:
            # Use confidence as proxy for similarity for best path, apply graph boost
            base_score = routing_result.confidence
            boosted_score, had_boost = _apply_graph_boost(
                routing_result.signature.id, base_score
            )
            candidates.append((routing_result.signature, boosted_score))

        # Add alternatives from each level (flatten)
        for level_alts in routing_result.alternatives:
            for alt_sig, score in level_alts:
                # Check if already added (by signature id)
                already_added = any(sig.id == alt_sig.id for sig, _ in candidates)
                if not alt_sig.is_semantic_umbrella and not already_added:
                    # Apply graph boost if this signature appears in graph routing
                    boosted_score, _ = _apply_graph_boost(alt_sig.id, score)
                    candidates.append((alt_sig, boosted_score))
                    explored_sigs.append(alt_sig)
                    if len(candidates) >= num_paths:
                        break
            if len(candidates) >= num_paths:
                break

        # Re-sort candidates by boosted score (graph matches rise to top)
        if graph_matches:
            candidates.sort(key=lambda x: x[1], reverse=True)
            logger.debug(
                "[solver] After graph boost re-sort: %s",
                [(sig.id, score) for sig, score in candidates[:3]]
            )

            # Add graph-only candidates (found by graph routing but not centroid)
            # This is key for "route by what operations DO, not what they SOUND LIKE"
            candidate_ids = {sig.id for sig, _ in candidates}
            for graph_sig_id, graph_sim in graph_matches.items():
                if graph_sig_id not in candidate_ids and len(candidates) < num_paths:
                    graph_sig = graph_signatures.get(graph_sig_id)
                    if graph_sig:
                        # Graph-only candidates get their graph similarity as score
                        candidates.append((graph_sig, graph_sim))
                        explored_sigs.append(graph_sig)
                        logger.debug(
                            "[solver] Added graph-only candidate: sig %d (%s) sim=%.3f",
                            graph_sig.id,
                            graph_sig.computation_graph[:30] if graph_sig.computation_graph else "?",
                            graph_sim,
                        )

        if not candidates:
            # No leaf candidates found
            return (None, routing_result.signature, explored_sigs, False)

        # Try DSL on each candidate in parallel
        async def try_candidate(sig_with_score):
            sig, score = sig_with_score
            dsl_result = await self._try_dsl(sig, step, context, step_descriptions)
            result = dsl_result.result if dsl_result else None
            return (sig, score, result, dsl_result)

        results = await asyncio.gather(*[try_candidate(c) for c in candidates])

        # Store DSL info from first successful result (primary candidate)
        # Per CLAUDE.md "New Favorite Pattern": consolidated _last_dsl_info assignment
        for _, _, result, dsl_result in results:
            if dsl_result is not None:
                self._last_dsl_info = dsl_result
                break

        # Store path outcomes for ground truth comparison (operational equivalence learning)
        # Key insight: After problem is graded, we can determine which paths are
        # operationally equivalent (produce correct answer) vs different (produce wrong answer)
        #
        # Thread tracking: Create fork threads for each alternative path
        # Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"
        path_outcomes = []
        parent_thread = self._active_threads.get(thread_id) if thread_id else None

        # Limit forks per step to prevent thread explosion
        max_forks = min(len(results), THREAD_MAX_FORKS_PER_STEP)

        for i, (sig, score, result, _dsl_result) in enumerate(results):
            # Create fork thread for each alternative (if thread tracking enabled)
            fork_thread_id = ""
            if parent_thread and THREAD_TRACKING_ENABLED and len(results) > 1:
                # First candidate stays on parent thread, others fork (up to max)
                if i == 0:
                    fork_thread_id = thread_id
                    # Update parent thread with this signature (primary choice)
                    parent_thread.add_signature(sig.id, step.id, was_primary=True)
                    parent_thread.step_results[step.id] = result or ""
                elif i < max_forks:
                    # Create fork thread for alternative (respecting limit)
                    fork_thread = parent_thread.fork(step.id, sig.id)
                    fork_thread.step_results[step.id] = result or ""
                    fork_thread.final_answer = result
                    self._active_threads[fork_thread.thread_id] = fork_thread
                    self._problem_threads.append(fork_thread.thread_id)
                    fork_thread_id = fork_thread.thread_id

                    # Create mcts_thread record for fork
                    # Per CLAUDE.md: "Create fork records when MCTS branches"
                    if self._current_dag_id:
                        dag_step_id = self._dag_step_ids.get(step.id)
                        create_thread(
                            dag_id=self._current_dag_id,
                            parent_thread_id=thread_id,
                            fork_at_step=dag_step_id,
                            fork_reason="undecided",  # Only branch when UCB1 gap is low
                            thread_id=fork_thread_id,
                        )

                        # Log thread step for fork with amplitude data
                        if dag_step_id:
                            log_thread_step(
                                thread_id=fork_thread_id,
                                dag_id=self._current_dag_id,
                                dag_step_id=dag_step_id,
                                node_id=sig.id,
                                amplitude=self._routing_confidence,
                                similarity_score=score,
                                was_undecided=True,  # Always undecided in multi-path
                                ucb1_gap=self._routing_ucb1_gap,
                                alternatives_considered=len(candidates),
                                step_result=result[:500] if result else None,
                                step_success=result is not None,
                                node_depth=sig.depth,
                            )

                    logger.debug(
                        "[solver] Created fork thread %s for signature %s at step %s",
                        fork_thread_id[:8], sig.step_type, step.id
                    )
                # else: skip this alternative (over max_forks limit)

            path_outcomes.append(PathOutcome(
                signature_id=sig.id,
                answer=result,
                step_id=step.id,
                embedding_similarity=score,
                dsl_type=sig.dsl_type or "unknown",
                thread_id=fork_thread_id,
            ))

        if path_outcomes:
            self._pending_path_outcomes[step.id] = path_outcomes
            logger.debug(
                "[solver] Stored %d path outcomes for step '%s' (for ground truth comparison)",
                len(path_outcomes), step.id
            )

        # Find first success (or None if all failed)
        best_sig = candidates[0][0]  # First candidate's signature
        best_result = None
        for sig, _score, result, _dsl in results:
            if result is not None:
                best_sig = sig
                best_result = result
                logger.info(
                    "[solver] Multi-path: found success via '%s'",
                    sig.step_type
                )
                break

        if best_result is None:
            logger.debug("[solver] Multi-path: all %d paths failed DSL", len(candidates))
            # Record failure for all explored paths (per CLAUDE.md: failures are valuable data)
            for sig in explored_sigs:
                self._record_failure(
                    step_text=step.task,
                    failure_type="dsl_error",
                    error_message=f"Multi-path DSL failed ({len(candidates)} paths explored)",
                    signature=sig,
                    source="multi_path",
                    extra_context={"paths_explored": len(candidates)},
                )

        return (best_result, best_sig, explored_sigs, best_result is not None)

    def _evaluate_formula_reference(self, formula: str, context: dict[str, str]) -> Optional[float]:
        """Evaluate a formula with $references like '$a + $b'.

        Per CLAUDE.md "New Favorite Pattern": consolidated formula evaluation.
        Handles cases where planner generates formulas in extracted_values.

        Args:
            formula: Formula string like "$traffic_time + $slow_drive_time"
            context: Previous step results

        Returns:
            Computed result, or None on failure
        """
        import re

        try:
            # Replace $references with actual values
            def replace_ref(match):
                ref_name = match.group(1)
                # Try Phase 1 values
                if ref_name in self._current_phase1_values:
                    return str(self._current_phase1_values[ref_name])
                # Try partial match
                for p1_key in self._current_phase1_values:
                    if p1_key in ref_name or ref_name in p1_key:
                        return str(self._current_phase1_values[p1_key])
                logger.debug("[solver] Formula ref not found: $%s", ref_name)
                return "0"  # Default to 0 if not found

            # Replace all $name patterns
            evaluated = re.sub(r'\$(\w+)', replace_ref, formula)

            # Replace {step_N} references with context values
            def replace_step_ref(match):
                step_key = match.group(1)
                if step_key in context:
                    return str(context[step_key])
                logger.debug("[solver] Formula step ref not found: {%s}", step_key)
                return "0"

            evaluated = re.sub(r'\{(\w+)\}', replace_step_ref, evaluated)

            # Safely evaluate the expression
            # Only allow basic math operations
            allowed = set('0123456789.+-*/() ')
            if not all(c in allowed for c in evaluated):
                logger.warning("[solver] Formula contains invalid chars: %s", evaluated[:50])
                return None

            result = eval(evaluated)  # Safe: only numeric chars and operators
            return float(result)

        except Exception as e:
            logger.debug("[solver] Formula evaluation failed: %s (%s)", formula[:50], e)
            return None

    async def _execute_dsl_and_record(
        self,
        signature: StepSignature,
        step: Step,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
    ) -> Optional[str]:
        """Execute DSL and store info for example recording.

        Per CLAUDE.md "New Favorite Pattern": Single entry point for DSL execution
        that consolidates the _last_dsl_info storage instead of scattering it
        across multiple call sites.

        Args:
            signature: The signature to execute DSL for
            step: The step being executed
            context: Results from previous steps
            step_descriptions: Descriptions of all steps

        Returns:
            The result string if DSL succeeded, None otherwise.
            Side effect: stores DSL info in self._last_dsl_info for later recording.
        """
        dsl_result = await self._try_dsl(signature, step, context, step_descriptions)
        if dsl_result is not None:
            self._last_dsl_info = dsl_result
            return dsl_result.result
        return None

    async def _try_dsl(
        self,
        signature: StepSignature,
        step: Step,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
    ) -> Optional[DSLResult]:
        """Try to execute a DSL script.

        LLM writes the arithmetic expression using available param names.
        No heuristic mapping - LLM always picks the right params for the task.

        Also handles extraction-only steps (no dsl_hint, just extracted_values).

        Returns:
            DSLResult with result, expression, and inputs (for DSL learning).
            None if DSL execution failed or was not applicable.
        """
        # Get operation hint from planner
        dsl_hint = getattr(step, 'dsl_hint', None)
        extracted_values = getattr(step, 'extracted_values', {}) or {}

        # Debug logging for DSL execution
        logger.debug(
            "[solver] _try_dsl: task='%s' dsl_hint=%s extracted_values=%s context_keys=%s",
            step.task[:40] if step.task else "None",
            dsl_hint,
            extracted_values,
            list(context.keys()) if context else []
        )

        # Check if step requires algebra (backwards solving)
        requires_algebra = getattr(step, 'requires_algebra', False)
        if requires_algebra:
            logger.info("[solver] Step '%s' requires algebra - trying SymPy solver", step.task[:40])
            algebra_result = await self._try_algebra_solve(step, context)
            if algebra_result is not None:
                logger.info("[solver] Algebra solved: %s → %s", step.task[:40], algebra_result)
                return DSLResult(result=algebra_result, expression="algebra_solve", inputs=None)
            logger.debug("[solver] Algebra solve failed, falling back to regular DSL")

        # Handle extraction-only steps: no dsl_hint but has single extracted value
        # These steps just extract a constant from the problem (e.g., "eggs per day = 16")
        if not dsl_hint and extracted_values:
            # Find first non-reference value
            for key, val in extracted_values.items():
                if isinstance(val, (int, float)):
                    logger.info("[solver] Extraction-only step: %s = %s", key, val)
                    return DSLResult(result=str(val), expression="extract", inputs=json.dumps({key: val}))
                elif isinstance(val, str) and val and not (val.startswith('{') and val.endswith('}')):
                    # Non-empty string value that's not a reference
                    try:
                        num_val = float(val)
                        logger.info("[solver] Extraction-only step: %s = %s", key, num_val)
                        return DSLResult(result=str(num_val), expression="extract", inputs=json.dumps({key: num_val}))
                    except ValueError:
                        logger.debug("[solver] Non-numeric string value for %s: %s", key, val[:50])
            logger.debug("[solver] No extractable value found in extracted_values")
            return None

        if not dsl_hint:
            # No hint from planner - can't execute DSL
            logger.info("[solver] No dsl_hint for step '%s', skipping DSL (extracted_values=%s)",
                       step.task[:40] if step.task else "None", extracted_values)
            return None

        # Build available params from context + extracted values
        params = {}

        # Add validated extracted values (resolve references)
        # Supports two reference types:
        # 1. $name - Phase 1 value reference (e.g., $purchase_price)
        # 2. {step_N} - Prior step result reference (e.g., {step_1})
        if extracted_values:
            for key, val in extracted_values.items():
                if isinstance(val, str):
                    # Check for formula reference (e.g., "$a + $b", "$x * $y")
                    # Per CLAUDE.md "New Favorite Pattern": consolidated formula evaluation
                    if '+' in val or '-' in val or '*' in val or '/' in val:
                        # Formula with operators - evaluate it
                        formula_result = self._evaluate_formula_reference(val, context)
                        if formula_result is not None:
                            params[key] = formula_result
                            logger.info("[solver] Resolved formula '%s' → %s", val[:40], formula_result)
                        else:
                            logger.warning("[solver] Failed to evaluate formula: %s", val[:50])
                    # Check for Phase 1 value reference ($name)
                    elif val.startswith('$'):
                        ref_name = val[1:]  # Remove $ prefix
                        if ref_name in self._current_phase1_values:
                            params[key] = self._current_phase1_values[ref_name]
                            logger.debug(
                                "[solver] Resolved $%s → %s (Phase 1 provenance)",
                                ref_name, params[key]
                            )
                        else:
                            # Try partial match fallback (e.g., single_glass_price -> glass_price)
                            matched_key = None
                            for p1_key in self._current_phase1_values:
                                # Check if either is a suffix/substring of the other
                                if p1_key in ref_name or ref_name in p1_key:
                                    matched_key = p1_key
                                    break
                            if matched_key:
                                params[key] = self._current_phase1_values[matched_key]
                                logger.debug(
                                    "[solver] Resolved $%s → %s (partial match: %s)",
                                    ref_name, params[key], matched_key
                                )
                            else:
                                logger.warning(
                                    "[solver] Unknown Phase 1 reference: $%s (available: %s)",
                                    ref_name, list(self._current_phase1_values.keys())
                                )
                                params[key] = val  # Keep as-is for error handling
                    # Check for step reference ({step_N})
                    elif val.startswith('{') and val.endswith('}'):
                        ref_key = val[1:-1]
                        if ref_key in context:
                            try:
                                params[key] = float(context[ref_key])
                            except (ValueError, TypeError):
                                logger.debug("[solver] Non-numeric ref %s=%s, keeping as-is", ref_key, str(context[ref_key])[:30])
                                params[key] = context[ref_key]
                    else:
                        # Try to parse as number (backwards compatibility)
                        try:
                            params[key] = float(val) if '.' in val else int(val)
                        except ValueError:
                            params[key] = val
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
                key = list(params.keys())[0]
                val = list(params.values())[0]
                logger.info("[solver] Single-param extraction: %s", val)
                return DSLResult(result=str(val), expression="extract", inputs=json.dumps({key: val}))
            # Per CLAUDE.md: failures are valuable data - log why DSL failed
            logger.info("[solver] DSL failed for '%s': need 2+ params, got %d (extracted_values=%s, context_keys=%s)",
                       step.task[:40] if step.task else "None", len(params), extracted_values, list(context.keys()))
            return None

        logger.debug("[solver] _try_dsl: hint=%s, params=%s", dsl_hint, list(params.keys()))

        # Check for batch-written expression first (single LLM call for all steps)
        batch_expressions = getattr(self, '_batch_expressions', {})
        expr_result = None

        if step.id in batch_expressions:
            script, used_params = batch_expressions[step.id]
            logger.debug("[solver] Using batch expression for %s: %s", step.id, script)
            expr_result = (script, used_params)
        else:
            # Fallback: LLM writes the expression (individual call)
            # Inject few-shot examples from signature for better LLM guidance
            few_shot_prompt = signature.get_few_shot_prompt() if signature else ""
            signature_id = signature.id if signature else None
            expr_result = await self._llm_write_expression(dsl_hint, params, step.task, few_shot_prompt, signature_id)

        try:
            if expr_result:
                script, used_params = expr_result
                logger.debug("[solver] Expression: %s (used: %s)", script, used_params)

                dsl_spec = DSLSpec(
                    layer=DSLSpec.from_json('{"type":"math"}').layer,
                    script=script,
                    params=used_params,
                )

                result, success = try_execute_dsl(dsl_spec, params, step_task=step.task)
                logger.debug("[solver] DSL exec: result=%s, success=%s", result, success)

                if success and result is not None:
                    # Build {param: value} from used_params, extracting only used values
                    used_inputs = {p: params.get(p) for p in used_params if p in params}
                    logger.info("[solver] DSL success: %s → %s", step.task[:30], result)
                    return DSLResult(
                        result=str(result),
                        expression=script,
                        inputs=json.dumps(used_inputs),
                    )
            else:
                logger.debug("[solver] LLM returned no expression")

        except Exception as e:
            logger.debug("[solver] DSL execution failed: %s", e)
            # Record failure (per CLAUDE.md: failures are valuable data)
            self._record_failure(
                step_text=step.task,
                failure_type="dsl_error",
                error_message=str(e),
                signature=signature,
                source="try_dsl",
                extra_context={"dsl_hint": getattr(step, "dsl_hint", None)},
            )

        return None

    async def _try_algebra_solve(
        self,
        step,
        context: dict[str, str],
    ) -> Optional[str]:
        """Attempt backwards solving using SymPy.

        Called when a step has undefined variables that require
        working backwards from known results.

        Args:
            step: The step requiring algebra
            context: Results from previous steps

        Returns:
            Solved value as string, or None on failure
        """
        from mycelium.step_signatures.sympy_layer import (
            build_equation_from_values,
            try_execute_dsl_sympy,
        )

        extracted_values = getattr(step, 'extracted_values', {}) or {}
        operation = getattr(step, 'operation', None) or getattr(step, 'dsl_hint', None) or "unknown"

        # Build values dict with None for unknowns, resolved values for knowns
        values = {}
        for key, val in extracted_values.items():
            if val is None:
                # This is the unknown we need to solve for
                values[key] = None
            elif isinstance(val, str):
                if val.startswith('{') and val.endswith('}'):
                    # Step reference - resolve from context
                    ref_step = val[1:-1]
                    if ref_step in context:
                        try:
                            values[key] = float(context[ref_step])
                        except (ValueError, TypeError):
                            values[key] = context[ref_step]
                    else:
                        # Step result not available - can't solve
                        logger.debug("[algebra] Missing step reference: %s", ref_step)
                        return None
                elif val.startswith('$'):
                    # Phase 1 reference - resolve from stored values
                    ref_name = val[1:]
                    phase1_val = self._current_phase1_values.get(ref_name)
                    if phase1_val is not None:
                        values[key] = phase1_val
                    else:
                        # Phase 1 value not found - this is the unknown
                        values[key] = None
                else:
                    # Try to parse as number
                    try:
                        values[key] = float(val) if '.' in val else int(val)
                    except ValueError:
                        # String that's not a number - might be the unknown
                        values[key] = None
            elif isinstance(val, (int, float)):
                values[key] = val
            else:
                values[key] = val

        # Build equation and solve
        script, unknown = build_equation_from_values(values, operation)
        if not script:
            logger.debug("[algebra] Could not build equation for step")
            return None

        result = try_execute_dsl_sympy(script, values, unknown_var=unknown)

        if result is not None:
            return str(result)
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

    def _prewarm_step_embeddings(self, steps: list) -> None:
        """Batch embed all step texts to pre-warm the embedding cache.

        Instead of N sequential embed calls during step execution,
        this makes a single batch API call upfront. The embeddings
        are stored in the cache, so cached_embed() hits during execution.
        """
        if not steps:
            return

        # Collect normalized step texts (same normalization as _execute_step)
        texts = []
        for step in steps:
            if not step.is_composite:
                normalized = normalize_step_text(step.task)
                texts.append(normalized)

        if not texts:
            return

        # Batch embed - populates the cache
        logger.debug("[solver] Pre-warming embeddings: %d steps in single batch call", len(texts))
        cached_embed_batch(texts, self.embedder)

    def _store_dag_step_embeddings(self, steps: list) -> None:
        """Store embeddings for dag_steps to enable similarity-based decomposition decisions.

        Per CLAUDE.md: Track (dag_step, leaf_node) pairs and use statistics to
        decide which to decompose on failure.

        Called after _prewarm_step_embeddings so embeddings are already cached.
        """
        if not self._current_dag_id or not self._dag_step_ids:
            return

        stored = 0
        for step in steps:
            if step.is_composite:
                continue

            dag_step_id = self._dag_step_ids.get(step.id)
            if not dag_step_id:
                continue

            # Get cached embedding
            normalized = normalize_step_text(step.task)
            embedding = cached_embed(normalized, self.embedder)

            # Store for similarity lookups
            try:
                store_dag_step_embedding(
                    dag_id=self._current_dag_id,
                    dag_step_id=dag_step_id,
                    step_desc=step.task,
                    embedding=embedding,
                    node_id=None,  # Will be updated when step executes
                )
                stored += 1
            except Exception as e:
                logger.warning("[solver] Failed to store dag_step embedding: %s", e)

        if stored > 0:
            logger.debug("[solver] Stored %d dag_step embeddings for similarity lookups", stored)

    def _mark_step_for_decomposition(self, dag_step_id: str) -> None:
        """Mark a dag_step pattern for future decomposition.

        When similar steps come in later, we'll decompose them proactively
        based on this signal that the step pattern is too complex.

        Per CLAUDE.md: "Failing signatures get decomposed" - same applies to steps.
        """
        from mycelium.data_layer.mcts import update_dag_step_embedding_outcome

        try:
            # Mark the step as needing decomposition by setting success=0
            # This affects find_similar_dag_steps() which checks success rates
            update_dag_step_embedding_outcome(
                dag_step_id=dag_step_id,
                node_id=0,  # Placeholder - we're marking the step itself
                success=False,
            )
            logger.info(
                "[solver] Marked dag_step %s for decomposition (step pattern too complex)",
                dag_step_id[:20]
            )
        except Exception as e:
            logger.warning("[solver] Failed to mark step for decomposition: %s", e)

    def _trigger_signature_decomposition(self, signature_id: int) -> None:
        """Trigger decomposition of a failing signature.

        This promotes the signature to an umbrella (if not already) and flags it
        for the umbrella learner to create specialized children.

        Per CLAUDE.md: "Failing signatures get decomposed"
        """
        try:
            sig = self.step_db.get_signature(signature_id)
            if sig is None:
                return

            # Skip if already an umbrella with children
            if sig.is_semantic_umbrella:
                children = self.step_db.get_children(signature_id, for_routing=True)
                if children:
                    logger.debug(
                        "[solver] Sig %d already umbrella with %d children, skipping",
                        signature_id, len(children)
                    )
                    return

            # Flag for split/decomposition - increments operational_failures
            # which triggers umbrella_learner consideration
            self.step_db.flag_for_split(signature_id, reason="diagnostic_postmortem")

            logger.info(
                "[solver] Triggered decomposition for sig %d '%s' (accuracy=%.1f%%)",
                signature_id,
                sig.step_type[:30] if sig.step_type else "?",
                sig.success_rate * 100,
            )
        except Exception as e:
            logger.warning("[solver] Failed to trigger sig decomposition: %s", e)

    def _record_routing_miss(self, dag_step_id: str, signature_id: int) -> None:
        """Record a routing miss - this (step, sig) pair should be avoided.

        Similar steps succeeded with other signatures, so this was a bad routing
        decision. We record it to improve future routing.

        Per CLAUDE.md: "Routing: Decompose the search space"
        """
        try:
            # Update the step embedding to mark this node as unsuccessful for this step
            from mycelium.data_layer.mcts import update_dag_step_embedding_outcome

            update_dag_step_embedding_outcome(
                dag_step_id=dag_step_id,
                node_id=signature_id,
                success=False,
            )

            logger.info(
                "[solver] Recorded routing miss: step %s should not route to sig %d",
                dag_step_id[:20], signature_id
            )
        except Exception as e:
            logger.warning("[solver] Failed to record routing miss: %s", e)

    def _record_failure(
        self,
        step_text: str,
        failure_type: str,
        error_message: str = None,
        signature=None,
        source: str = None,
        problem: str = None,
        extra_context: dict = None,
        is_operational: bool = False,
    ) -> int:
        """Record a step failure for pattern learning.

        Consolidates all failure recording to ensure consistent:
        - Text truncation (200 chars for step_text, 500 for problem in context)
        - Context formatting with source identification
        - Signature ID extraction from objects or ints
        - Operational failure stat tracking

        Per CLAUDE.md: "Failures Are Valuable Data Points"
        - Record every failure—it feeds the post-mortem analysis
        - Accumulated failure patterns trigger decomposition
        - Success/failure stats drive routing decisions

        Args:
            step_text: The step/problem text that failed
            failure_type: Category of failure:
                - 'dsl_error': DSL execution failed
                - 'validation': Plan/result validation failed
                - 'llm_error': LLM call failed
                - 'routing': Umbrella routing failed
                - 'operational': Signature produced wrong answer
            error_message: The error text/description
            signature: Optional signature (StepSignature object or int ID) that failed
            source: Identifier for where the failure originated (e.g., "zero_llm", "try_dsl")
            problem: Optional problem text to include in context (truncated to 500 chars)
            extra_context: Additional context dict to merge
            is_operational: If True, increments operational_failures stat on signature

        Returns:
            ID of the failure record
        """
        # Truncate step_text to 200 chars
        step_text_truncated = step_text[:200] if step_text else ""

        # Extract signature ID from object or use int directly
        signature_id = None
        if signature is not None:
            if isinstance(signature, int):
                signature_id = signature
            else:
                signature_id = getattr(signature, "id", None)

        # Build context dict
        context = {}
        if source:
            context["source"] = source
        if problem:
            context["problem"] = problem[:500]
        if extra_context:
            context.update(extra_context)

        return self.step_db.record_failure(
            step_text=step_text_truncated,
            failure_type=failure_type,
            error_message=error_message,
            signature_id=signature_id,
            context=context if context else None,
            increment_operational_failures=is_operational,
        )

    def _prewarm_operation_embeddings(self, steps: list) -> None:
        """Batch embed all step operations for graph-based routing.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Operations are extracted during decomposition (1 LLM call), then batch
        embedded here (1 embedding call). This avoids per-step LLM calls for
        operation extraction during routing.

        Embeddings stored in self._operation_embeddings dict (not on Step objects)
        for memory efficiency - each embedding is ~24KB (3072 floats).
        """
        # Clear previous problem's embeddings
        self._operation_embeddings.clear()

        if not steps:
            return

        # Collect non-None operations with their step IDs
        operations = []
        step_ids = []  # Track which step each operation belongs to
        for step in steps:
            if step.operation and not step.is_composite:
                operations.append(step.operation)
                step_ids.append(step.id)

        if not operations:
            logger.debug("[solver] No operations to embed (steps may lack operation field)")
            return

        # Batch embed all operations
        logger.debug("[solver] Pre-warming operation embeddings: %d operations in single batch call", len(operations))
        embeddings_dict = cached_embed_batch(operations, self.embedder)

        # Store embeddings in instance dict (keyed by step.id)
        for operation, step_id in zip(operations, step_ids):
            if operation in embeddings_dict:
                embedding = embeddings_dict[operation]
                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    self._operation_embeddings[step_id] = embedding.tolist()
                else:
                    self._operation_embeddings[step_id] = list(embedding)

        logger.debug("[solver] Stored operation embeddings for %d steps", len(self._operation_embeddings))

    async def _llm_write_expression(
        self,
        operation: str,
        params: dict,
        task: str,
        few_shot_prompt: str = "",
        signature_id: int = None,
    ) -> Optional[tuple[str, list[str]]]:
        """Ask LLM to write arithmetic expression using available params.

        Args:
            operation: The operation hint (+, -, *, /)
            params: Available param names and values
            task: The step task description
            few_shot_prompt: Optional few-shot examples from matched signature
            signature_id: Optional signature ID for cache isolation

        Returns:
            (script, param_list) or None if failed
        """
        # Cache key: (operation, param names, signature_id)
        # Include signature_id so different signatures with same op+params get separate cache entries
        param_names = frozenset(k for k in params.keys() if not k.startswith('{'))
        cache_key = (operation.strip().lower(), param_names, signature_id)

        # Check cache first - saves ~1-2s LLM call
        if cache_key in self._dsl_expr_cache:
            expr, used_params = self._dsl_expr_cache[cache_key]
            self._dsl_expr_cache.move_to_end(cache_key)  # LRU: mark as recently used
            logger.debug("[solver] DSL cache hit: %s -> %s", cache_key[0], expr)
            return expr, used_params

        # Format available params
        param_info = ", ".join(f"{k}={v}" for k, v in params.items() if not k.startswith('{'))

        # Build prompt with optional few-shot examples
        few_shot_section = ""
        if few_shot_prompt:
            few_shot_section = f"\nSimilar problems solved:\n{few_shot_prompt}\n"

        prompt = f"""Write a simple arithmetic expression for this task.

Task: {task}
Operation: {operation}
Available values: {param_info}
{few_shot_section}
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

    async def _batch_write_expressions(
        self,
        step_infos: list[dict],
    ) -> dict[str, tuple[str, list[str]]]:
        """Write arithmetic expressions for all steps in ONE LLM call.

        This replaces N individual _llm_write_expression calls with a single
        batch call using JSON mode for reliable parsing.

        Args:
            step_infos: List of dicts with keys:
                - step_id: The step identifier
                - task: The step task description
                - operation: The operation hint (+, -, *, /)
                - params: Dict of available param names and values

        Returns:
            Dict mapping step_id -> (expression, params_used)
        """
        import json

        if not step_infos:
            return {}

        # Build the batch prompt
        steps_text = []
        for i, info in enumerate(step_infos, 1):
            param_str = ", ".join(f"{k}={v}" for k, v in info["params"].items())
            steps_text.append(
                f"{i}. step_id: {info['step_id']}\n"
                f"   task: {info['task']}\n"
                f"   operation: {info['operation']}\n"
                f"   available_params: {param_str}"
            )

        prompt = f"""Write arithmetic expressions for these steps.

CRITICAL: Use ONLY the exact parameter names provided in available_params.
Do NOT invent or modify parameter names.

Steps:
{chr(10).join(steps_text)}

Output valid JSON with this exact structure:
{{
  "expressions": [
    {{"step_id": "step_1", "expression": "param_a + param_b", "params_used": ["param_a", "param_b"]}},
    ...
  ]
}}

Rules:
- Use EXACTLY the variable names from available_params
- Write simple arithmetic: +, -, *, /
- Each expression should use the operation specified
- Output ONLY the JSON, no explanation"""

        try:
            from mycelium.client import get_client
            client = get_client()
            response = await client.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            data = json.loads(response.strip())
            expressions = data.get("expressions", [])

            result = {}
            for expr_info in expressions:
                step_id = expr_info.get("step_id")
                expression = expr_info.get("expression")
                params_used = expr_info.get("params_used", [])

                if step_id and expression:
                    result[step_id] = (expression, params_used)
                    logger.debug(
                        "[solver] Batch expr: %s -> %s (params: %s)",
                        step_id, expression, params_used
                    )

            logger.info(
                "[solver] Batch wrote %d/%d expressions in single LLM call",
                len(result), len(step_infos)
            )
            return result

        except Exception as e:
            logger.warning("[solver] Batch expression write failed: %s", e)
            return {}

    def _build_step_params(
        self,
        step,
        context: dict[str, str],
    ) -> dict:
        """Build params dict for a step from extracted_values and context.

        Resolves $name (Phase 1) and {step_N} references to actual values.
        """
        params = {}
        extracted_values = getattr(step, 'extracted_values', {}) or {}

        # Add validated extracted values (resolve references)
        for key, val in extracted_values.items():
            if isinstance(val, str):
                # Check for Phase 1 value reference ($name)
                if val.startswith('$'):
                    ref_name = val[1:]
                    if ref_name in self._current_phase1_values:
                        params[key] = self._current_phase1_values[ref_name]
                    else:
                        # Try partial match fallback
                        matched_key = None
                        for p1_key in self._current_phase1_values:
                            if p1_key in ref_name or ref_name in p1_key:
                                matched_key = p1_key
                                break
                        if matched_key:
                            params[key] = self._current_phase1_values[matched_key]
                        else:
                            params[key] = val  # Keep as-is
                # Check for step reference ({step_N})
                elif val.startswith('{') and val.endswith('}'):
                    ref_key = val[1:-1]
                    if ref_key in context:
                        try:
                            params[key] = float(context[ref_key])
                        except (ValueError, TypeError):
                            params[key] = context[ref_key]
                else:
                    # Try to parse as number
                    try:
                        params[key] = float(val) if '.' in val else int(val)
                    except ValueError:
                        params[key] = val
            else:
                params[key] = val

        # Add context values (results from previous steps)
        for key, value in context.items():
            if key not in params:
                try:
                    params[key] = float(value)
                except (ValueError, TypeError):
                    params[key] = value

        return params

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

    def _answers_match(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are equivalent (for operational equivalence).

        Uses normalize_answer for consistency with pipeline evaluation.

        Args:
            answer1: First answer string
            answer2: Second answer string (typically ground truth)

        Returns:
            True if answers are considered equivalent
        """
        if answer1 is None or answer2 is None:
            return False

        # Use normalize_answer for consistency with pipeline evaluation
        # This handles LaTeX, currency, units, fractions, etc.
        a1 = normalize_answer(answer1)
        a2 = normalize_answer(answer2)

        # Direct match after normalization
        if a1 == a2:
            return True

        # Try numeric comparison as fallback (handles "42" vs "42.0" vs "42.00")
        try:
            n1 = float(a1.replace(",", ""))
            n2 = float(a2.replace(",", ""))
            # Use relative tolerance for floating point
            return abs(n1 - n2) < 1e-9 or (n2 != 0 and abs((n1 - n2) / n2) < 1e-6)
        except (ValueError, TypeError):
            pass

        return False

    async def _retry_with_alternatives(
        self,
        problem: str,
        failed_result: SolverResult,
        ground_truth: str,
        difficulty: float = None,
    ) -> Optional[tuple[SolverResult, str]]:
        """Retry a failed problem exploring alternative nodes at each step.

        Per CLAUDE.md: "If we got the wrong answer that should trigger a larger MCTS
        rollout where we explore the tree more widely searching for the right
        leaf_node, dag_step pairs"

        This function re-runs steps from the failed thread, trying alternative
        signatures at each decision point. If a winning path is found, it returns
        that result for divergence analysis.

        Args:
            problem: Original problem text
            failed_result: The failed SolverResult
            ground_truth: Correct answer to compare against
            difficulty: Problem difficulty (for compute budget)

        Returns:
            Tuple of (winning_result, winning_thread_id) if found, else None
        """
        from mycelium.config import (
            REACTIVE_EXPLORATION_MAX_ALTERNATIVES,
            REACTIVE_EXPLORATION_MAX_RETRIES,
            REACTIVE_EXPLORATION_MIN_SIMILARITY,
        )
        from mycelium.data_layer.mcts import create_thread, log_thread_step, complete_thread, grade_thread
        import uuid

        # Get step embeddings and signatures from the failed run
        failed_steps = failed_result.steps
        if not failed_steps:
            return None

        logger.info(
            "[reactive] Starting reactive exploration for failed problem with %d steps",
            len(failed_steps)
        )

        # For each step, get alternative signatures we could have tried
        step_alternatives: list[list[tuple[int, float]]] = []  # list of (sig_id, similarity) per step
        for step_result in failed_steps:
            if step_result.signature_id is None:
                step_alternatives.append([])
                continue

            # Get step embedding (should be cached from original run)
            step_text = normalize_step_text(step_result.task)
            embedding = cached_embed(step_text, self.embedder)

            # Find similar signatures excluding the one we used
            similar = self.step_db.find_similar(
                embedding,
                threshold=REACTIVE_EXPLORATION_MIN_SIMILARITY,
                limit=REACTIVE_EXPLORATION_MAX_ALTERNATIVES + 1,
            )

            # Filter out the signature we already tried
            alternatives = [
                (sig.id, sim) for sig, sim in similar
                if sig.id != step_result.signature_id and not sig.is_semantic_umbrella
            ][:REACTIVE_EXPLORATION_MAX_ALTERNATIVES]

            step_alternatives.append(alternatives)
            logger.debug(
                "[reactive] Step '%s' has %d alternatives: %s",
                step_result.step_id[:20] if step_result.step_id else "?",
                len(alternatives),
                [(a[0], f"{a[1]:.3f}") for a in alternatives]
            )

        # Try alternative paths (greedy: substitute one step at a time)
        for retry_num in range(min(REACTIVE_EXPLORATION_MAX_RETRIES, len(failed_steps))):
            # Find step with most promising alternatives
            best_step_idx = -1
            best_alt_sim = 0.0
            for i, alts in enumerate(step_alternatives):
                if alts and alts[0][1] > best_alt_sim:
                    best_alt_sim = alts[0][1]
                    best_step_idx = i

            if best_step_idx < 0:
                logger.debug("[reactive] No more alternatives to try")
                break

            # Pop best alternative for this step
            alt_sig_id, alt_sim = step_alternatives[best_step_idx].pop(0)
            failed_step = failed_steps[best_step_idx]

            logger.info(
                "[reactive] Retry %d: trying sig %d (sim=%.3f) for step '%s' (was sig %d)",
                retry_num + 1, alt_sig_id, alt_sim,
                failed_step.step_id[:20] if failed_step.step_id else "?",
                failed_step.signature_id
            )

            # Create a new thread for this exploration
            explore_thread_id = f"explore-{uuid.uuid4().hex[:8]}"
            # Look up the actual dag_step_id for the fork point
            fork_dag_step_id = self._dag_step_ids.get(failed_step.step_id)
            if self._current_dag_id and fork_dag_step_id:
                create_thread(
                    dag_id=self._current_dag_id,
                    parent_thread_id=self._root_thread_id,
                    fork_at_step=fork_dag_step_id,
                    fork_reason="reactive_exploration",
                    thread_id=explore_thread_id,
                )

            # Get alternative signature
            alt_sig = self.step_db.get_signature(alt_sig_id)
            if not alt_sig:
                continue

            # Try executing with the alternative signature
            # Build context from previous steps (use original results up to this point)
            context = {}
            step_descriptions = {}
            for i, prev_step in enumerate(failed_steps[:best_step_idx]):
                if prev_step.result:
                    context[prev_step.step_id] = prev_step.result
                step_descriptions[prev_step.step_id] = prev_step.task

            # Create a Step object for execution (using preserved data from StepResult)
            from mycelium.planner import Step
            step_obj = Step(
                id=failed_step.step_id,
                task=failed_step.task,
                depends_on=failed_step.depends_on or list(context.keys()),
            )
            # Copy dsl_hint and extracted_values from preserved StepResult fields
            if failed_step.dsl_hint:
                step_obj.dsl_hint = failed_step.dsl_hint
            if failed_step.extracted_values:
                step_obj.extracted_values = failed_step.extracted_values

            # Try DSL with alternative signature
            try:
                dsl_result = await self._try_dsl(alt_sig, step_obj, context, step_descriptions)
                result_str = dsl_result.result if dsl_result else None

                # Log the thread step (only if we have a valid dag_step_id)
                if self._current_dag_id:
                    # Look up the actual dag_step_id from our mapping
                    dag_step_id = self._dag_step_ids.get(failed_step.step_id)
                    if dag_step_id:
                        log_thread_step(
                            thread_id=explore_thread_id,
                            dag_id=self._current_dag_id,
                            dag_step_id=dag_step_id,
                            node_id=alt_sig_id,
                            amplitude=alt_sim,
                            similarity_score=alt_sim,
                            was_undecided=1,
                            alternatives_considered=len(step_alternatives[best_step_idx]) + 1,
                            step_result=result_str[:500] if result_str else None,
                            step_success=1 if result_str else 0,
                        )

                if dsl_result is None:
                    logger.debug("[reactive] Alternative sig %d failed DSL execution", alt_sig_id)
                    if self._current_dag_id:
                        complete_thread(explore_thread_id, final_answer=None)
                        grade_thread(explore_thread_id, success=False)
                    continue

                # Execute remaining steps with original signatures
                remaining_context = dict(context)
                remaining_context[failed_step.step_id] = result_str
                all_results = [result_str]

                for remaining_step in failed_steps[best_step_idx + 1:]:
                    step_descriptions[remaining_step.step_id] = remaining_step.task
                    rem_step_obj = Step(
                        id=remaining_step.step_id,
                        task=remaining_step.task,
                        depends_on=[d for d in (remaining_step.depends_on or []) if d in remaining_context],
                    )
                    # Copy dsl_hint and extracted_values from preserved StepResult fields
                    if remaining_step.dsl_hint:
                        rem_step_obj.dsl_hint = remaining_step.dsl_hint
                    if remaining_step.extracted_values:
                        rem_step_obj.extracted_values = remaining_step.extracted_values

                    rem_sig = self.step_db.get_signature(remaining_step.signature_id)
                    if rem_sig:
                        rem_dsl_result = await self._try_dsl(rem_sig, rem_step_obj, remaining_context, step_descriptions)
                        if rem_dsl_result:
                            remaining_context[remaining_step.step_id] = rem_dsl_result.result
                            all_results.append(rem_dsl_result.result)

                # Check if final answer matches ground truth
                final_answer = all_results[-1] if all_results else None
                is_correct = self._answers_match(str(final_answer), ground_truth) if final_answer else False

                if self._current_dag_id:
                    complete_thread(explore_thread_id, final_answer=str(final_answer) if final_answer else None)
                    grade_thread(explore_thread_id, success=is_correct)

                if is_correct:
                    logger.info(
                        "[reactive] Found winning path via sig %d! (answer=%s)",
                        alt_sig_id, str(final_answer)[:30] if final_answer else "None"
                    )
                    # Create a result for the winning path
                    winning_result = SolverResult(
                        problem=problem,
                        answer=str(final_answer) if final_answer else "",
                        success=True,
                        steps=failed_steps,  # Keep original steps for comparison
                    )
                    return (winning_result, explore_thread_id)

            except Exception as e:
                logger.debug("[reactive] Error trying alternative: %s", e)
                if self._current_dag_id:
                    complete_thread(explore_thread_id, final_answer=None)
                    grade_thread(explore_thread_id, success=False)
                continue

        logger.info("[reactive] No winning path found after %d retries", REACTIVE_EXPLORATION_MAX_RETRIES)
        return None

    async def _decompose_and_resolve(
        self,
        failed_result: SolverResult,
        ground_truth: str,
        difficulty: float = None,
        decomposition_depth: int = 0,
    ) -> Optional[tuple[SolverResult, str]]:
        """Decompose failing steps and re-solve the problem.

        Per CLAUDE.md: "The step is likely too complex and needs decomposition"

        When reactive exploration fails to find a winning path by trying alternative
        signatures, this method decomposes the failing steps into smaller sub-steps
        and re-solves the entire problem.

        Args:
            failed_result: The failed SolverResult
            ground_truth: Correct answer to compare against
            difficulty: Problem difficulty
            decomposition_depth: Current decomposition depth (prevents infinite recursion)

        Returns:
            Tuple of (winning_result, thread_id) if decomposition found a solution, else None
        """
        from mycelium.config import STEP_DECOMPOSITION_MAX_DEPTH
        from mycelium.data_layer.mcts import create_thread, complete_thread, grade_thread
        import uuid

        # Prevent infinite recursion
        if decomposition_depth >= STEP_DECOMPOSITION_MAX_DEPTH:
            logger.debug("[decompose] Max decomposition depth reached (%d)", decomposition_depth)
            return None

        problem = failed_result.problem
        failed_steps = failed_result.steps

        if not failed_steps:
            return None

        logger.info(
            "[decompose] Attempting step decomposition for failed problem (depth=%d, steps=%d)",
            decomposition_depth, len(failed_steps)
        )

        # Identify which steps are most likely failing
        # For now, try decomposing all steps that had a result (executed but gave wrong answer)
        steps_to_decompose = []
        for step_result in failed_steps:
            if step_result.result is not None:
                steps_to_decompose.append(step_result)

        if not steps_to_decompose:
            logger.debug("[decompose] No steps with results to decompose")
            return None

        # Create a thread for this decomposition attempt
        decomp_thread_id = f"decomp-{uuid.uuid4().hex[:8]}"
        if self._current_dag_id:
            create_thread(
                dag_id=self._current_dag_id,
                parent_thread_id=self._root_thread_id,
                fork_at_step=None,  # Decomposition doesn't fork from a specific step
                fork_reason="step_decomposition",
                thread_id=decomp_thread_id,
            )

        try:
            # Re-plan the problem with a hint to use finer-grained decomposition
            signature_hints = self.step_db.get_signature_hints(
                limit=HINT_LIMIT,
                problem_embedding=cached_embed(problem, self.embedder),
                min_similarity=HINT_MIN_SIMILARITY,
            )

            # Add context about what failed
            failed_step_descriptions = [
                f"- '{s.task[:60]}' produced '{str(s.result)[:30]}'"
                for s in steps_to_decompose[:3]  # Limit to first 3 for prompt length
            ]
            decomposition_context = (
                "The previous solution was incorrect. "
                "Please break down the problem into MORE GRANULAR steps. "
                "These steps may need finer decomposition:\n" +
                "\n".join(failed_step_descriptions)
            )

            # Re-decompose with the hint
            new_plan = await self.planner.decompose(
                problem,
                signature_hints=signature_hints,
                context=decomposition_context,
            )

            if not new_plan or not new_plan.steps:
                logger.debug("[decompose] Re-decomposition produced no steps")
                if self._current_dag_id:
                    complete_thread(decomp_thread_id, final_answer=None)
                    grade_thread(decomp_thread_id, success=False)
                return None

            # Check if we got more steps (finer decomposition)
            if len(new_plan.steps) <= len(failed_steps):
                logger.debug(
                    "[decompose] Re-decomposition didn't produce finer steps (%d vs %d)",
                    len(new_plan.steps), len(failed_steps)
                )
                if self._current_dag_id:
                    complete_thread(decomp_thread_id, final_answer=None)
                    grade_thread(decomp_thread_id, success=False)
                return None

            logger.info(
                "[decompose] Re-decomposed into %d steps (was %d)",
                len(new_plan.steps), len(failed_steps)
            )

            # Update Phase 1 values for the new plan (for $name resolution)
            self._current_phase1_values = new_plan.phase1_values or {}

            # Execute the new plan
            # Validate DAG structure
            if len(new_plan.steps) > 1:
                is_valid, errors = new_plan.validate()
                if not is_valid:
                    logger.warning("[decompose] New plan has invalid DAG: %s", errors)
                    if self._current_dag_id:
                        complete_thread(decomp_thread_id, final_answer=None)
                        grade_thread(decomp_thread_id, success=False)
                    return None

            # Batch write expressions for all steps
            # Build step_infos dicts from Step objects (same format as main solve())
            step_infos = []
            for step in new_plan.steps:
                params = {}
                for key, val in (step.extracted_values or {}).items():
                    if isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                        params[key] = val[1:-1]  # {step_1} -> step_1
                    else:
                        params[key] = val
                if params:
                    step_infos.append({
                        "step_id": step.id,
                        "task": step.task,
                        "operation": step.dsl_hint,
                        "params": params,
                    })
            if step_infos:
                await self._batch_write_expressions(step_infos)

            # Execute steps in dependency order
            step_order = self._get_execution_order(new_plan)
            context = {}
            step_descriptions = {s.id: s.task for s in new_plan.steps}

            for step in step_order:
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

                step_result = await self._execute_step(
                    step, problem, step_context, step_desc_context,
                    compute_budget=1.0,  # Single path for decomposition
                    difficulty=difficulty,
                    thread_id=decomp_thread_id,
                )

                if step_result.result is None:
                    logger.debug("[decompose] Step '%s' failed in re-execution", step.id)
                    if self._current_dag_id:
                        complete_thread(decomp_thread_id, final_answer=None)
                        grade_thread(decomp_thread_id, success=False)
                    return None

                context[step.id] = step_result.result

            # Get final answer from last step
            final_step = step_order[-1] if step_order else None
            final_answer = context.get(final_step.id) if final_step else None

            # Check if the answer matches ground truth
            is_correct = self._answers_match(str(final_answer), ground_truth) if final_answer else False

            if self._current_dag_id:
                complete_thread(decomp_thread_id, final_answer=str(final_answer) if final_answer else None)
                grade_thread(decomp_thread_id, success=is_correct)

            if is_correct:
                logger.info(
                    "[decompose] Decomposition found correct answer: %s",
                    str(final_answer)[:30] if final_answer else "None"
                )
                winning_result = SolverResult(
                    problem=problem,
                    answer=str(final_answer) if final_answer else "",
                    success=True,
                    steps=failed_result.steps,  # Keep original for comparison
                )
                return (winning_result, decomp_thread_id)

            logger.debug("[decompose] Decomposed solution also incorrect")
            return None

        except Exception as e:
            logger.error("[decompose] Step decomposition failed: %s", e)
            if self._current_dag_id:
                complete_thread(decomp_thread_id, final_answer=None)
                grade_thread(decomp_thread_id, success=False)
            return None

    async def _run_reactive_exploration(
        self,
        result: SolverResult,
        ground_truth: str,
        difficulty: float = None,
    ) -> dict:
        """Run reactive exploration when a problem fails.

        Per CLAUDE.md: Compare winning vs losing threads to find divergence points.
        First divergence is root cause for blame assignment.

        Args:
            result: The failed SolverResult
            ground_truth: Correct answer
            difficulty: Problem difficulty

        Returns:
            Dict with exploration statistics
        """
        from mycelium.data_layer.mcts import find_divergence_points, assign_divergence_blame
        from mycelium.config import (
            REACTIVE_EXPLORATION_FULL_RESOLVE,
            REACTIVE_EXPLORATION_NUM_THREADS,
            REACTIVE_EXPLORATION_TEMPERATURE,
        )
        from mycelium.planner import TreeGuidedPlanner

        stats = {
            "reactive_exploration_triggered": True,
            "winning_path_found": False,
            "divergence_points": 0,
            "blame_assigned": 0,
            "exploration_threads_spawned": 0,
        }

        winning = None
        # Save original DAG ID (contains losing threads) for cross-DAG comparison
        original_dag_id = self._current_dag_id
        exploration_dag_ids = []  # Collect exploration DAG IDs (may contain winning threads)

        # Try N full re-solves with forced exploration (higher temp + epsilon for diversity)
        # Per mycelium-l703: spawn multiple threads for cross-examination
        if REACTIVE_EXPLORATION_FULL_RESOLVE:
            num_threads = REACTIVE_EXPLORATION_NUM_THREADS
            logger.info(
                "[reactive] Spawning %d exploration threads (temp=%.2f)",
                num_threads, REACTIVE_EXPLORATION_TEMPERATURE
            )

            # Swap planner to higher-temp version for exploration diversity
            original_planner = self.planner
            self.planner = TreeGuidedPlanner(
                step_db=self.step_db,
                embedder=self.embedder,
                temperature=REACTIVE_EXPLORATION_TEMPERATURE,
            )

            # Per mycelium-02nn + CLAUDE.md "The Flow": Get Welford-adaptive multipliers
            # DB Statistics → Welford → Tree Structure (via multipliers)
            self._reactive_gap_mult, self._reactive_budget_mult = (
                self.step_db.get_adaptive_reactive_multipliers()
            )
            logger.info(
                "[reactive] Using adaptive multipliers: gap=%.2f, budget=%.2f",
                self._reactive_gap_mult, self._reactive_budget_mult
            )
            self._reactive_exploration_mode = True

            try:
                for thread_idx in range(num_threads):
                    stats["exploration_threads_spawned"] += 1
                    logger.info("[reactive] Running exploration thread %d/%d", thread_idx + 1, num_threads)

                    explore_result = await self.solve(result.problem, ground_truth=ground_truth)
                    if self._current_dag_id:
                        exploration_dag_ids.append(self._current_dag_id)

                    # Check if this thread found the correct answer
                    is_correct = (
                        explore_result.answer and
                        normalize_answer(explore_result.answer) == normalize_answer(ground_truth)
                    )

                    # CRITICAL: Grade exploration threads so cross-DAG comparison works
                    # Without this, threads have success=NULL and divergence detection fails
                    self.record_problem_outcome(explore_result, is_correct, ground_truth=ground_truth)

                    if is_correct:
                        logger.info("[reactive] Thread %d found winning path!", thread_idx + 1)
                        winning = (explore_result, self._root_thread_id)
                        stats["full_resolve_success"] = True
                        stats["winning_thread_idx"] = thread_idx + 1
                        break
                    else:
                        logger.info("[reactive] Thread %d didn't find winning path", thread_idx + 1)

                if winning is None:
                    stats["full_resolve_success"] = False
            finally:
                self._reactive_exploration_mode = False
                self.planner = original_planner  # Restore original planner

            # Per CLAUDE.md "The Flow": Record outcome to update Welford stats
            # This feeds back into DB Statistics → Welford → future multipliers
            found_winner = winning is not None
            self.step_db.update_reactive_exploration_stats(
                found_winner=found_winner,
                gap_mult_used=self._reactive_gap_mult,
                budget_mult_used=self._reactive_budget_mult,
            )
            logger.info(
                "[reactive] Recorded exploration outcome: found_winner=%s (gap=%.2f, budget=%.2f)",
                found_winner, self._reactive_gap_mult, self._reactive_budget_mult
            )

            stats["total_dags_for_crossexam"] = len(exploration_dag_ids) + (1 if original_dag_id else 0)

        # Fallback: try single-step alternatives (original approach)
        if winning is None:
            winning = await self._retry_with_alternatives(
                result.problem, result, ground_truth, difficulty
            )

        if winning is None:
            logger.info("[reactive] No winning alternative found, trying step decomposition")

            # Fallback: try decomposing failing steps into sub-steps
            from mycelium.config import (
                STEP_DECOMPOSITION_FALLBACK_ENABLED,
                STEP_DECOMPOSITION_MIN_STEPS,
            )

            if STEP_DECOMPOSITION_FALLBACK_ENABLED and len(result.steps) >= STEP_DECOMPOSITION_MIN_STEPS:
                decomp_result = await self._decompose_and_resolve(
                    result, ground_truth, difficulty
                )
                if decomp_result is not None:
                    stats["step_decomposition_triggered"] = True
                    stats["winning_path_found"] = True
                    stats["decomposition_succeeded"] = True
                    logger.info("[reactive] Step decomposition found winning path")
                    # Continue to divergence analysis with the decomposed result
                    winning = decomp_result
                else:
                    stats["step_decomposition_triggered"] = True
                    stats["decomposition_succeeded"] = False
                    logger.info("[reactive] Step decomposition also failed")

                    # dag_step decomposition failed → check if leaf nodes need decomposition
                    # Per brainstorm: Use maturity-based decision to flag underperforming leaves
                    await self._evaluate_leaf_decomposition_after_failure(result, stats)

                    return stats
            else:
                return stats

        winning_result, winning_thread_id = winning
        stats["winning_path_found"] = True
        logger.info(
            "[reactive] Found winning thread %s, running cross-examination across %d DAGs",
            winning_thread_id, len(exploration_dag_ids) + (1 if original_dag_id else 0)
        )

        # Cross-examine using cross-DAG comparison
        # Original DAG (losing threads) vs exploration DAGs (potential winning threads)
        # Per mycelium-l703: compare winning vs losing threads ACROSS DAGs
        total_divergence_points = 0
        total_blame_assigned = 0
        all_blame_stats = {}

        if original_dag_id and exploration_dag_ids:
            # Cross-DAG comparison: original (losers) + exploration (potential winners)
            logger.info(
                "[reactive] Cross-DAG comparison: original=%s vs %d exploration DAGs",
                original_dag_id[:8], len(exploration_dag_ids)
            )
            divergence_points = find_divergence_points(
                dag_id=original_dag_id,
                cross_dag_ids=exploration_dag_ids
            )
            total_divergence_points = len(divergence_points)

            if divergence_points:
                # Log divergence details (first 5)
                for dp in divergence_points[:5]:
                    logger.info(
                        "[reactive] Cross-DAG divergence at step %s (idx=%d): "
                        "winning node=%s, losing node=%s",
                        dp.divergence_dag_step_id,
                        dp.divergence_step_idx,
                        dp.winning_node_at_divergence,
                        dp.losing_node_at_divergence,
                    )

                # Assign targeted blame/credit using cross-DAG comparison
                blame_stats = assign_divergence_blame(
                    dag_id=original_dag_id,
                    step_db=self.step_db,
                    cross_dag_ids=exploration_dag_ids
                )
                total_blame_assigned = (
                    blame_stats.get("divergence_blame_assigned", 0) +
                    blame_stats.get("suffix_blame_assigned", 0)
                )
                all_blame_stats = blame_stats
        elif original_dag_id:
            # Fallback: single DAG analysis (within-DAG comparison only)
            logger.info("[reactive] Single-DAG analysis for %s (no exploration DAGs)", original_dag_id[:8])
            divergence_points = find_divergence_points(dag_id=original_dag_id)
            total_divergence_points = len(divergence_points)

            if divergence_points:
                blame_stats = assign_divergence_blame(dag_id=original_dag_id, step_db=self.step_db)
                total_blame_assigned = (
                    blame_stats.get("divergence_blame_assigned", 0) +
                    blame_stats.get("suffix_blame_assigned", 0)
                )
                all_blame_stats = blame_stats

        stats["divergence_points"] = total_divergence_points
        stats["blame_assigned"] = total_blame_assigned
        stats.update(all_blame_stats)

        if total_divergence_points > 0:
            logger.info(
                "[reactive] Cross-exam complete: %d divergence points, %d blame assigned",
                total_divergence_points, total_blame_assigned
            )
            logger.info(
                "[reactive] Divergence blame: %d primary, %d suffix, %d prefix credit",
                all_blame_stats.get("divergence_blame_assigned", 0),
                all_blame_stats.get("suffix_blame_assigned", 0),
                all_blame_stats.get("shared_prefix_credit", 0),
            )
        else:
            logger.info("[reactive] No divergence points found (all threads same outcome?)")

        return stats

    async def _evaluate_leaf_decomposition_after_failure(
        self,
        result: SolverResult,
        stats: dict,
    ) -> None:
        """Evaluate if leaf nodes need splitting based on divergence.

        Natural splitting inspired by nature (nautilus, trees, lungs):
        - Binary split is the atomic operation (like cell division)
        - Split on DIVERGENCE (success vs failure embedding clusters)
        - WIDTH vs DEPTH based on semantic distance:
          - Close embeddings but divergent outcomes -> WIDTH (variants)
          - Distant embeddings with divergent outcomes -> DEPTH (abstraction)
        - The tree structure EMERGES, not designed

        Args:
            result: The failed SolverResult with step information
            stats: Dict to record statistics about decisions made
        """
        from mycelium.step_signatures.divergence import (
            get_signature_outcome_embeddings,
            maybe_split_on_divergence,
        )

        # Get all leaf signatures involved in this problem
        split_results = []
        for step in result.steps:
            if step.signature_id is None:
                continue

            # Get the signature
            sig = self.step_db.get_signature(step.signature_id)
            if sig is None or sig.is_semantic_umbrella:
                continue  # Skip umbrellas (routers don't execute)

            # Get historical success/failure embeddings for this signature
            success_embeddings, failure_embeddings = get_signature_outcome_embeddings(
                step.signature_id
            )

            # Check for divergence and maybe split
            split_result = maybe_split_on_divergence(
                self.step_db,
                sig,
                success_embeddings,
                failure_embeddings,
            )

            if split_result is not None:
                split_results.append({
                    "signature_id": step.signature_id,
                    "step_task": step.task[:50] if step.task else "",
                    "split_type": split_result.split_type,
                    "success": split_result.success,
                    "child_a_id": split_result.child_a_id,
                    "child_b_id": split_result.child_b_id,
                    "reason": split_result.reason,
                })
                logger.info(
                    "[divergence] Split sig %d (%s): %s -> children %s, %s",
                    step.signature_id,
                    split_result.split_type,
                    split_result.reason,
                    split_result.child_a_id,
                    split_result.child_b_id,
                )

        # Record stats
        stats["divergence_splits"] = split_results
        stats["signatures_split"] = len([r for r in split_results if r["success"]])

        if split_results:
            logger.info(
                "[divergence] Evaluated %d leaves: %d split (width=%d, depth=%d)",
                len(result.steps),
                stats["signatures_split"],
                len([r for r in split_results if r["split_type"] == "width"]),
                len([r for r in split_results if r["split_type"] == "depth"]),
            )

    def record_problem_outcome(
        self,
        result: SolverResult,
        correct: bool,
        difficulty: float = None,
        ground_truth: str = None,
    ) -> list[int]:
        """Propagate problem correctness to all signatures used.

        Call this after grading a problem to track real success rates.
        This enables negative lift detection for umbrella learning.

        DIFFICULTY-WEIGHTED CREDIT: Harder problems provide more valuable signal.
        - difficulty=0.0 (trivial) → 1.0x credit
        - difficulty=0.5 (GSM8K) → 3.0x credit
        - difficulty=1.0 (competition) → 5.0x credit

        Args:
            result: The SolverResult from solve()
            correct: Whether the final answer was correct
            difficulty: Problem difficulty for weighted credit (0.0-1.0)
            ground_truth: The correct answer (for MCTS path outcome comparison)

        Returns:
            List of signature IDs that may need decomposition (low confidence)
        """
        signature_ids = [
            step.signature_id
            for step in result.steps
            if step.signature_id is not None
        ]
        self.step_db.update_problem_outcome(signature_ids, correct, difficulty=difficulty)

        # Grade the MCTS DAG (training mode only)
        # Per CLAUDE.md: Set success/graded_at after final answer comparison
        if self._current_dag_id:
            grade_dag(self._current_dag_id, success=correct)
            logger.debug("[solver] Graded MCTS DAG %s: success=%s", self._current_dag_id, correct)
            # NOTE: Postmortem runs AFTER thread outcomes are recorded (see below)

        # MCTS Training: Process path outcomes for operational equivalence learning
        # Only in training mode - inference skips this overhead
        # Key insight: Ground truth lets us determine which paths are operationally
        # equivalent (produce correct answer) vs different (produce wrong answer)
        from mycelium.config import TRAINING_MODE
        from mycelium.step_signatures.operational_alignment import record_routing_outcome

        if TRAINING_MODE and self._pending_path_outcomes and ground_truth is not None:
            # Build step_id -> winning result mapping (for logging only)
            step_results = {step.step_id: step.result for step in result.steps}
            correct_paths = 0
            incorrect_paths = 0

            for step_id, path_outcomes in self._pending_path_outcomes.items():
                winning_result = step_results.get(step_id)

                for outcome in path_outcomes:
                    # A path is operationally correct if it produced the correct answer
                    # Compare directly to ground_truth, not to winning_result
                    # This catches cases where a non-winning path would have been correct
                    path_correct = (
                        outcome.answer is not None and
                        self._answers_match(outcome.answer, ground_truth)
                    )

                    # Note: Centroid updates removed - graph embeddings are structural (not learned)
                    # Learning happens through UCB1 stats (uses, successes) tracked elsewhere

                    # Record for operational alignment validation
                    # This tracks whether MCTS is actually helping distinguish operations
                    try:
                        record_routing_outcome(
                            db_path=DB_PATH,
                            signature_id=outcome.signature_id,
                            step_text=step_id,
                            embedding_similarity=outcome.embedding_similarity,
                            was_correct=path_correct,
                            dsl_type=outcome.dsl_type,
                            problem_id=result.problem[:50] if result.problem else None,
                        )
                    except Exception as e:
                        logger.debug("[solver] Failed to record alignment outcome: %s", e)

                    if path_correct:
                        correct_paths += 1
                        logger.debug(
                            "[solver] Path via sig %d was operationally correct for step '%s'",
                            outcome.signature_id, step_id
                        )
                    else:
                        incorrect_paths += 1
                        # Record failure for potential cluster splitting
                        # Per CLAUDE.md: "Record every failure—it feeds the refinement loop"
                        self._record_failure(
                            step_text=step_id,
                            failure_type="operational",
                            signature=outcome.signature_id,
                            source="thread_outcome",
                            extra_context={
                                "produced_answer": outcome.answer,
                                "expected_answer": ground_truth,
                            },
                            is_operational=True,
                        )
                        logger.debug(
                            "[solver] Path via sig %d was operationally different for step '%s' "
                            "(path_answer=%s, ground_truth=%s, winning=%s)",
                            outcome.signature_id, step_id,
                            outcome.answer[:20] if outcome.answer else "None",
                            ground_truth[:20] if ground_truth else "None",
                            winning_result[:20] if winning_result else "None"
                        )

            # Clear pending path outcomes after processing
            logger.info(
                "[solver] Processed %d path outcomes: %d correct, %d incorrect (ground_truth=%s)",
                correct_paths + incorrect_paths, correct_paths, incorrect_paths,
                ground_truth[:20] if ground_truth else "None"
            )

        # Record thread outcomes for multi-path credit/blame (if enabled and threads exist)
        # Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"
        if THREAD_TRACKING_ENABLED and TRAINING_MODE and self._problem_threads:
            self._record_thread_outcomes(result, correct, ground_truth)

        # Run post-mortem AFTER threads are graded (so we have success values)
        # Per CLAUDE.md: "High confidence + failure = strong negative signal"
        # Single call to run_postmortem() handles both interference and diagnostics
        if self._current_dag_id:
            try:
                postmortem_stats = run_postmortem(
                    self._current_dag_id,
                    step_db=self.step_db,
                    step_embeddings=getattr(self, '_step_embeddings', None),
                    include_interference=True,
                    include_diagnostics=True,
                )
                if postmortem_stats.get("high_conf_wrong", 0) > 0:
                    logger.warning(
                        "[solver] Post-mortem: %d high-confidence wrong in DAG %s",
                        postmortem_stats["high_conf_wrong"], self._current_dag_id
                    )
                if postmortem_stats.get("total_steps", 0) > 0:
                    logger.info(
                        "[solver] Post-mortem: %d steps, %d won, %d lost",
                        postmortem_stats["total_steps"],
                        postmortem_stats.get("threads_won", 0),
                        postmortem_stats.get("threads_lost", 0),
                    )
                # Per beads mycelium-flbq: Check if DSL regen batch is ready
                if postmortem_stats.get("dsl_regen_ready"):
                    self._pending_dsl_regen = True
                    logger.info(
                        "[solver] DSL regeneration ready: %d nodes accumulated",
                        postmortem_stats.get("dsl_regen_nodes_accumulated", 0)
                    )

                # Log nodes/steps needing decomposition from post-mortem analysis
                nodes_needing = postmortem_stats.get("nodes_needing_decomposition", [])
                steps_needing = postmortem_stats.get("steps_needing_decomposition", [])
                # Store nodes flagged by interference for candidate list building
                self._postmortem_flagged_nodes = postmortem_stats.get("nodes_flagged_split", [])
                if nodes_needing:
                    logger.info(
                        "[solver] Post-mortem: %d nodes need decomposition: %s",
                        len(nodes_needing), nodes_needing
                    )
                if steps_needing:
                    logger.info(
                        "[solver] Post-mortem: %d steps need decomposition: %s",
                        len(steps_needing), [s[1][:40] for s in steps_needing]
                    )
                if self._postmortem_flagged_nodes:
                    logger.info(
                        "[solver] Post-mortem: %d nodes flagged for split (destructive interference): %s",
                        len(self._postmortem_flagged_nodes), self._postmortem_flagged_nodes
                    )

                # Handle decomposition decisions from post-mortem
                # Per CLAUDE.md "New Favorite Pattern": Use consolidated list from data layer
                all_steps = postmortem_stats.get("all_steps_to_decompose", [])
                diag_sigs = postmortem_stats.get("signatures_to_decompose", [])
                routing_misses = postmortem_stats.get("routing_misses", [])

                if all_steps or diag_sigs or routing_misses:
                    logger.info(
                        "[solver] Post-mortem decomposition: "
                        "steps=%d, sigs=%d, routing_misses=%d",
                        len(all_steps),
                        len(diag_sigs),
                        len(routing_misses),
                    )

                    # === ACT ON DECOMPOSITION DECISIONS ===

                    # 1. Steps to decompose: Mark step patterns for future decomposition
                    #    (consolidated from both decomposition analysis and diagnostics)
                    for dag_step_id in all_steps:
                        self._mark_step_for_decomposition(dag_step_id)

                    # 2. Signatures to decompose: Promote to umbrella or flag for rewrite
                    for sig_id in diag_sigs:
                        self._trigger_signature_decomposition(sig_id)

                    # 3. Routing misses: Record bad (step, sig) pairs for routing avoidance
                    for dag_step_id, sig_id in routing_misses:
                        self._record_routing_miss(dag_step_id, sig_id)

            except Exception as e:
                logger.error("[solver] Postmortem failed: %s", e)

        # Always clear pending outcomes (memory safety)
        self._pending_path_outcomes.clear()

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

            # Store context for reactive exploration (if enabled)
            # Caller can invoke maybe_run_reactive_exploration() to find winning alternatives
            from mycelium.config import REACTIVE_EXPLORATION_ENABLED
            if REACTIVE_EXPLORATION_ENABLED and ground_truth:
                self._pending_reactive_exploration = {
                    "result": result,
                    "ground_truth": ground_truth,
                    "difficulty": difficulty,
                }
                logger.debug("[solver] Reactive exploration pending for failed problem")
            else:
                self._pending_reactive_exploration = None
        else:
            self._pending_reactive_exploration = None

        # Check which signatures might need decomposition
        # Use same thresholds as umbrella_learner for consistency
        # Two categories:
        # 1. Failing guidance signatures (dsl_type="decompose", not yet umbrella)
        # 2. Auto-demoted router umbrellas with NO children
        from mycelium.config import (
            UMBRELLA_MIN_USES_FOR_EVALUATION,
            UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION,
        )
        candidates = []
        for sig_id in signature_ids:
            sig = self.step_db.get_signature(sig_id)
            if not sig or sig.uses < UMBRELLA_MIN_USES_FOR_EVALUATION:
                continue
            if sig.success_rate > UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION:
                continue

            # Category 1: decompose type not yet promoted to umbrella
            is_decompose_candidate = (
                sig.dsl_type == "decompose"
                and not sig.is_semantic_umbrella
            )

            # Category 2: auto-demoted router umbrellas without children
            is_orphan_umbrella = False
            if sig.is_semantic_umbrella and sig.dsl_type == "router":
                children = self.step_db.get_children(sig_id, for_routing=True)
                is_orphan_umbrella = len(children) == 0

            if is_decompose_candidate or is_orphan_umbrella:
                candidates.append(sig_id)
                logger.info(
                    "[solver] Signature '%s' (id=%d) needs decomposition: "
                    "uses=%d, success_rate=%.1f%%, orphan=%s",
                    sig.step_type, sig_id, sig.uses, sig.success_rate * 100, is_orphan_umbrella
                )

        # Also add nodes identified by post-mortem decomposition analysis
        # These are nodes with low win rates from mcts_thread_steps
        # CRITICAL: Flag them with operational_failures so get_decomposition_candidates picks them up
        from mycelium.data_layer.mcts import get_failing_nodes_for_decomposition
        from mycelium.config import DECOMP_MIN_ATTEMPTS_COLD, DECOMP_MIN_ATTEMPTS_MATURE, DECOMP_MAX_WIN_RATE
        from mycelium.mcts.adaptive import AdaptiveExploration

        # Adaptive min_attempts: lower during cold start, higher when mature
        # This ensures we flag failing signatures quickly during early learning
        adaptive = AdaptiveExploration.get_instance()
        accuracy = adaptive.global_accuracy
        # Interpolate: cold (0% accuracy) → COLD threshold, mature (100%) → MATURE threshold
        adaptive_min_attempts = int(
            DECOMP_MIN_ATTEMPTS_COLD + accuracy * (DECOMP_MIN_ATTEMPTS_MATURE - DECOMP_MIN_ATTEMPTS_COLD)
        )
        adaptive_min_attempts = max(1, adaptive_min_attempts)  # At least 1

        postmortem_failing_nodes = get_failing_nodes_for_decomposition(
            min_attempts=adaptive_min_attempts,
            max_win_rate=DECOMP_MAX_WIN_RATE
        )
        logger.debug(
            "[solver] get_failing_nodes_for_decomposition(min_attempts=%d, accuracy=%.1f%%) returned %d nodes: %s",
            adaptive_min_attempts, accuracy * 100, len(postmortem_failing_nodes), postmortem_failing_nodes
        )
        for node_id in postmortem_failing_nodes:
            if node_id not in candidates:
                sig = self.step_db.get_signature(node_id)
                if sig:
                    # Flag for decomposition - this sets operational_failures so umbrella_learner finds it
                    # Per CLAUDE.md: "only signatures with operational_failures > 0 are decomposition candidates"
                    self.step_db.flag_for_split(node_id, reason="postmortem_analysis")
                    candidates.append(node_id)
                    logger.info(
                        "[solver] Signature '%s' (id=%d) flagged for decomposition (post-mortem analysis): "
                        "failing across step types",
                        sig.step_type, node_id
                    )

        # Also add nodes flagged by interference detection (destructive interference)
        # These already have operational_failures > 0 from record_interference_outcome
        for node_id in self._postmortem_flagged_nodes:
            if node_id not in candidates:
                sig = self.step_db.get_signature(node_id)
                if sig:
                    candidates.append(node_id)
                    logger.info(
                        "[solver] Signature '%s' (id=%d) candidate for decomposition (destructive interference)",
                        sig.step_type, node_id
                    )
        # Clear the flagged nodes list after processing
        self._postmortem_flagged_nodes = []

        return candidates

    async def maybe_run_dsl_regeneration(self, client) -> dict:
        """Run DSL regeneration if batch threshold was reached.

        Per beads mycelium-flbq: When post-mortem accumulates enough high-conf-wrong
        nodes, regenerate their DSLs using the LLM.

        Args:
            client: LLM client for DSL generation

        Returns:
            Dict with regeneration statistics, or empty dict if not ready
        """
        if not self._pending_dsl_regen:
            return {}

        from mycelium.data_layer import (
            trigger_dsl_regeneration_for_nodes,
            get_accumulated_failing_nodes,
            clear_accumulated_failing_nodes,
        )

        # Get accumulated failing nodes
        failing_nodes = get_accumulated_failing_nodes()
        if not failing_nodes:
            self._pending_dsl_regen = False
            return {}

        logger.info("[solver] Running DSL regeneration for %d accumulated nodes", len(failing_nodes))

        try:
            result = await trigger_dsl_regeneration_for_nodes(
                node_ids=failing_nodes,
                step_db=self.step_db,
                client=client,
            )
            # Clear accumulated nodes after processing
            clear_accumulated_failing_nodes()
            self._pending_dsl_regen = False

            if result.get("regenerated", 0) > 0:
                logger.info(
                    "[solver] DSL regeneration complete: %d regenerated, %d failed",
                    result.get("regenerated", 0), result.get("failed", 0)
                )
            return result
        except Exception as e:
            logger.error("[solver] DSL regeneration failed: %s", e)
            self._pending_dsl_regen = False
            return {"error": str(e)}

    async def maybe_run_reactive_exploration(self) -> dict:
        """Run reactive exploration if a problem failed.

        Per CLAUDE.md: "If we got the wrong answer that should trigger a larger MCTS
        rollout where we explore the tree more widely"

        This searches for alternative paths that would have produced the correct answer.
        When found, uses divergence analysis for precise blame assignment.

        Returns:
            Dict with exploration statistics, or empty dict if not needed
        """
        if not hasattr(self, '_pending_reactive_exploration') or not self._pending_reactive_exploration:
            return {}

        context = self._pending_reactive_exploration
        self._pending_reactive_exploration = None

        logger.info("[solver] Running reactive exploration for failed problem")

        try:
            result = await self._run_reactive_exploration(
                result=context["result"],
                ground_truth=context["ground_truth"],
                difficulty=context.get("difficulty"),
            )

            if result.get("winning_path_found"):
                logger.info(
                    "[solver] Reactive exploration found winning path: %d divergence points, %d blame assigned",
                    result.get("divergence_points", 0),
                    result.get("blame_assigned", 0),
                )
            else:
                logger.debug("[solver] Reactive exploration found no winning alternative")

            return result
        except Exception as e:
            logger.error("[solver] Reactive exploration failed: %s", e)
            return {"error": str(e)}

    async def _blocking_decompose_complex_steps(
        self,
        plan,
        problem: str,
        client,
        min_batch_size: int = 5,
        max_queue_age_sec: float = 15.0,
        poll_interval: float = 0.5,
        timeout: float = 15.0,
    ) -> tuple[list, dict]:
        """Check plan for complex steps, queue them, and block until decomposed.

        This implements blocking decomposition: complex steps are queued, we wait
        for batch decomposition to complete, then return decomposed atomic steps.

        Decomposition triggers when EITHER:
        - Queue size >= min_batch_size, OR
        - Oldest pending item is >= max_queue_age_sec old

        Args:
            plan: The execution plan with steps
            problem: Problem text for context
            client: LLM client for decomposition
            min_batch_size: Minimum queue size before triggering decomposition
            max_queue_age_sec: Max seconds oldest item can wait before triggering
            poll_interval: Seconds between polling for results
            timeout: Max seconds to wait for decomposition

        Returns:
            Tuple of (queue_ids for our complex steps, decomposed results dict)
        """
        import asyncio
        from mycelium.data_layer.mcts import (
            queue_for_decomposition,
            get_decomposition_queue_size,
            get_oldest_pending_age_seconds,
            get_decomposition_results,
            are_decompositions_ready,
        )
        from mycelium.embedding_cache import cached_embed

        # NOTE: Pre-execution complexity detection was removed.
        # Splitting now happens via divergence detection AFTER execution.
        # See step_signatures/divergence.py for the new natural splitting approach.
        complex_steps = []

        # Even if no complex steps in current plan, check if stale items need processing
        if not complex_steps:
            queue_size = get_decomposition_queue_size()
            oldest_age = get_oldest_pending_age_seconds()
            if queue_size > 0 and oldest_age >= max_queue_age_sec:
                logger.info(
                    "[solver] No complex steps in plan, but processing stale queue (age=%.1fs)",
                    oldest_age
                )
                await self.maybe_run_batch_decomposition(
                    client,
                    batch_size=queue_size,
                    min_queue_size=1,
                )
            return [], {}

        logger.info(
            "[solver] Found %d complex steps in plan, queueing for decomposition",
            len(complex_steps)
        )

        # 2. Queue complex steps with their dag_step_ids
        our_queue_ids = []
        for step, complexity_reason in complex_steps:
            embedding = cached_embed(step.task, self.embedder)
            dag_step_id = self._dag_step_ids.get(step.id)

            queue_id = queue_for_decomposition(
                step_text=step.task,
                complexity_reason=complexity_reason,
                embedding=embedding,
                dag_step_id=dag_step_id,
                problem_context=problem[:500],
            )
            our_queue_ids.append(queue_id)
            logger.debug(
                "[solver] Queued complex step %s (queue_id=%d): %s",
                step.id, queue_id, step.task[:50]
            )

        # 3. Check if threshold met (batch size OR age), trigger decomposition if so
        queue_size = get_decomposition_queue_size()
        oldest_age = get_oldest_pending_age_seconds()

        should_trigger = queue_size >= min_batch_size or oldest_age >= max_queue_age_sec

        if should_trigger and queue_size > 0:
            trigger_reason = (
                f"batch size ({queue_size} >= {min_batch_size})"
                if queue_size >= min_batch_size
                else f"age ({oldest_age:.1f}s >= {max_queue_age_sec}s)"
            )
            logger.info(
                "[solver] Decomposition triggered by %s, processing %d items",
                trigger_reason, queue_size
            )
            # Run decomposition for ALL pending items (not just ours)
            decomp_result = await self.maybe_run_batch_decomposition(
                client,
                batch_size=queue_size,  # Process all pending
                min_queue_size=1,  # Force run since we already checked threshold
            )
            logger.info(
                "[solver] Batch decomposition complete: %d processed",
                decomp_result.get("processed", 0)
            )

        # 4. Wait for our steps to be processed (may have been processed by another worker)
        start_time = asyncio.get_event_loop().time()
        while not are_decompositions_ready(our_queue_ids):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(
                    "[solver] Timeout waiting for decomposition after %.1fs",
                    elapsed
                )
                break
            await asyncio.sleep(poll_interval)

        # 5. Get decomposed results for our steps
        results = get_decomposition_results(our_queue_ids)
        logger.info(
            "[solver] Retrieved decomposition results for %d/%d steps",
            sum(1 for r in results.values() if r.get("processed")),
            len(our_queue_ids)
        )

        return our_queue_ids, results

    async def _expand_plan_with_decompositions(
        self,
        plan,
        queue_ids: list[int],
        decomposition_results: dict[int, dict],
    ):
        """Expand plan by replacing complex steps with their decomposed atomic sub-steps.

        Args:
            plan: The execution plan to modify
            queue_ids: Queue IDs for our complex steps
            decomposition_results: Results from get_decomposition_results()

        Returns:
            Modified plan with complex steps expanded
        """
        from mycelium.planner import Step

        if not queue_ids or not decomposition_results:
            return plan

        # Build mapping of step.task -> decomposed atomic steps
        # We need to match by task text since queue doesn't store step.id directly
        expanded_count = 0

        for step in plan.steps:
            # Find if this step was queued (by checking if it matches a queued step's decomposition)
            for queue_id in queue_ids:
                result = decomposition_results.get(queue_id)
                if not result or not result.get("processed"):
                    continue

                decomposed_steps = result.get("decomposition_steps", [])
                if not decomposed_steps:
                    continue

                # If this step has decomposed sub-steps, create a sub-plan
                if len(decomposed_steps) > 1:
                    # Create sub-steps from decomposed atomic operations
                    sub_steps = []
                    for i, atomic_task in enumerate(decomposed_steps):
                        sub_step = Step(
                            id=f"{step.id}_sub_{i}",
                            task=atomic_task,
                            is_atomic=True,
                            depends_on=[f"{step.id}_sub_{j}" for j in range(i)] if i > 0 else step.depends_on,
                        )
                        sub_steps.append(sub_step)

                    # Attach sub-plan to the original step
                    step.sub_plan = type(plan)(steps=sub_steps)
                    step.is_atomic = False  # Now a composite step
                    expanded_count += 1
                    logger.debug(
                        "[solver] Expanded step %s into %d sub-steps",
                        step.id, len(decomposed_steps)
                    )
                    break  # Found match, move to next step

        if expanded_count > 0:
            logger.info("[solver] Expanded %d complex steps with decompositions", expanded_count)

        return plan

    async def maybe_run_batch_decomposition(
        self,
        client,
        batch_size: int = 5,
        min_queue_size: int = 3,
    ) -> dict:
        """Process queued complex steps in batch.

        Per beads mycelium-mm08: Instead of decomposing immediately (1 LLM call per step),
        batch decompose queued complex steps periodically.

        Args:
            client: LLM client for decomposition
            batch_size: Max steps to process in one batch
            min_queue_size: Minimum queue size before processing

        Returns:
            Dict with decomposition statistics
        """
        from mycelium.data_layer.mcts import (
            get_pending_decompositions,
            get_decomposition_queue_size,
            mark_decomposition_processed,
        )

        queue_size = get_decomposition_queue_size()
        if queue_size < min_queue_size:
            return {"skipped": True, "reason": f"queue_size={queue_size} < min={min_queue_size}"}

        pending = get_pending_decompositions(limit=batch_size)
        if not pending:
            return {"skipped": True, "reason": "no pending items"}

        logger.info("[solver] Running batch decomposition: %d items", len(pending))

        # Build batch prompt
        prompt_parts = [
            "Break each of the following complex steps into simple atomic operations.",
            "Each atomic operation should do ONE thing (add, subtract, multiply, divide, etc.).",
            "Return JSON array where each element has 'original_step' and 'atomic_steps' (array of strings).",
            "",
            "Complex steps to decompose:",
        ]

        for i, item in enumerate(pending):
            context = f" (Context: {item['problem_context'][:100]})" if item.get('problem_context') else ""
            prompt_parts.append(f"{i+1}. {item['step_text']}{context}")

        prompt_parts.append("")
        prompt_parts.append("Return ONLY valid JSON array, no other text.")

        prompt = "\n".join(prompt_parts)

        try:
            messages = [
                {"role": "system", "content": "You decompose complex math steps into atomic operations. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            response = await client.generate(messages, temperature=0.0)

            # Parse response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                logger.warning("[solver] Batch decomposition returned no JSON: %s", response[:200])
                return {"error": "no JSON in response", "processed": 0}

            decompositions = json.loads(json_match.group())

            # Process each decomposition
            processed = 0
            signatures_created = 0

            for i, decomp in enumerate(decompositions):
                if i >= len(pending):
                    break

                item = pending[i]
                atomic_steps = decomp.get("atomic_steps", [])

                if not atomic_steps:
                    logger.debug("[solver] No atomic steps for: %s", item['step_text'][:50])
                    continue

                # Create signatures for atomic steps
                created_ids = []
                for atomic_step in atomic_steps:
                    if not atomic_step or len(atomic_step.strip()) < 3:
                        continue

                    # Embed and create signature
                    from mycelium.embedding_cache import cached_embed
                    embedding = cached_embed(atomic_step)
                    if embedding is not None:
                        sig, is_new = self.step_db.find_or_create(
                            step_text=atomic_step,
                            embedding=embedding,
                            min_similarity=0.85,
                            parent_problem=item.get('problem_context', ''),
                        )
                        created_ids.append(sig.id)
                        if is_new:
                            signatures_created += 1

                # Mark as processed
                mark_decomposition_processed(
                    queue_id=item['id'],
                    result_signature_ids=created_ids,
                    decomposition_steps=atomic_steps,
                )
                processed += 1

            logger.info(
                "[solver] Batch decomposition complete: %d processed, %d signatures created",
                processed, signatures_created
            )

            return {
                "processed": processed,
                "signatures_created": signatures_created,
                "queue_remaining": queue_size - processed,
            }

        except Exception as e:
            logger.error("[solver] Batch decomposition failed: %s", e)
            return {"error": str(e), "processed": 0}

    def _record_thread_outcomes(
        self,
        result: SolverResult,
        correct: bool,
        ground_truth: str = None,
    ) -> None:
        """Apply thread-based credit/blame and update MCTS thread state.

        Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"

        Args:
            result: The SolverResult with the final answer
            correct: Whether the final answer was correct
            ground_truth: The correct answer (for comparison)
        """
        if not self._problem_threads:
            return

        # Identify winning thread (the one whose answer matches the result)
        winning_thread_id = None
        for thread_id in self._problem_threads:
            thread = self._active_threads.get(thread_id)
            if thread and self._answers_match(thread.final_answer, result.answer):
                winning_thread_id = thread_id
                thread.is_winner = True
                break

        # If no thread matched, use root thread as winner
        if winning_thread_id is None and self._root_thread_id:
            winning_thread_id = self._root_thread_id
            root = self._active_threads.get(self._root_thread_id)
            if root:
                root.is_winner = True

        try:
            # Process all threads
            for thread_id in self._problem_threads:
                thread = self._active_threads.get(thread_id)
                if not thread:
                    continue

                is_winner = thread_id == winning_thread_id
                # Thread is correct if its answer matches ground truth
                is_correct = None
                if ground_truth:
                    is_correct = self._answers_match(thread.final_answer, ground_truth)
                    # Debug logging for cross-DAG divergence investigation
                    logger.debug(
                        "[solver] Thread %s: final_answer=%r, ground_truth=%r, is_correct=%s",
                        thread_id[:8], thread.final_answer, ground_truth, is_correct
                    )

                # Update mcts_threads table with final_answer and success
                thread_success = is_correct if is_correct is not None else None
                complete_thread(
                    thread_id=thread.thread_id,
                    final_answer=thread.final_answer or "",
                    success=thread_success,
                )
                # Note: Credit/blame to signatures happens via post-mortem analysis
                # (run_postmortem), not here. UCB1 stats are updated there.

            logger.info(
                "[solver] Recorded %d thread outcomes (winner=%s, threads_correct=%d)",
                len(self._problem_threads),
                winning_thread_id[:8] if winning_thread_id else "None",
                sum(
                    1 for t in self._problem_threads
                    if self._active_threads.get(t) and
                    self._answers_match(
                        self._active_threads[t].final_answer, ground_truth
                    )
                ) if ground_truth else 0,
            )

        except Exception as e:
            logger.warning("[solver] Failed to record thread outcomes: %s", e)

        finally:
            # Always clear thread state for next problem (even on exception)
            self._active_threads.clear()
            self._problem_threads.clear()
            self._root_thread_id = None

    async def _decompose_complex_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str],
        hint: str,
        log_tag: str,
        signature_type: str,
        difficulty: float = None,
        thread_id: str = None,
        decomp_depth: int = 0,
    ) -> Optional[StepResult]:
        """Decompose a complex step into sub-steps.

        Uses the planner to break down steps that can't be handled
        by a single leaf signature.

        Args:
            step: The step that needs decomposition
            problem: Original problem text
            context: step_id → result from previous steps
            step_descriptions: step_id → task description
            hint: Context hint explaining why decomposition is needed
            log_tag: Tag for logging
            signature_type: Type to use in StepResult
            difficulty: Problem difficulty for adaptive decomposition
            thread_id: Thread ID for multi-path credit tracking
            decomp_depth: Current inline decomposition depth (to prevent infinite loops)

        Returns:
            StepResult if decomposition and execution succeeded, None otherwise
        """
        import time
        from mycelium.config import INLINE_DECOMP_MAX_DEPTH

        start_time = time.time()

        # Check depth limit to prevent infinite decomposition loops
        if decomp_depth >= INLINE_DECOMP_MAX_DEPTH:
            logger.warning(
                "[%s] Max inline decomposition depth (%d) reached for '%s'",
                log_tag, INLINE_DECOMP_MAX_DEPTH, step.task[:40]
            )
            return None

        try:
            # Decompose the step using the planner
            sub_plan = await self.planner.decompose(
                problem=step.task,
                context=f"Original problem: {problem}\n{hint}",
            )

            if not sub_plan or not sub_plan.steps or len(sub_plan.steps) <= 1:
                logger.warning(
                    "[%s] Planner couldn't decompose '%s' further",
                    log_tag, step.task[:40]
                )
                return None

            logger.info(
                "[%s] Decomposed '%s' into %d sub-steps (depth=%d)",
                log_tag, step.task[:40], len(sub_plan.steps), decomp_depth
            )

            # Execute sub-steps recursively
            sub_context = context.copy()
            sub_results = []

            for sub_step in sub_plan.steps:
                sub_result = await self._execute_step(
                    sub_step,
                    problem,
                    sub_context,
                    step_descriptions,
                    depth=1,
                    difficulty=difficulty,
                    thread_id=thread_id,
                    decomp_depth=decomp_depth + 1,  # Track inline decomposition depth
                )
                sub_results.append(sub_result)
                sub_context[sub_step.id] = sub_result.result

            # Use the final sub-step's result as the overall result
            final_result = sub_results[-1].result if sub_results else None

            elapsed_ms = (time.time() - start_time) * 1000

            return StepResult(
                step_id=step.id,
                task=step.task,
                result=final_result,
                signature_type=signature_type,
                was_injected=False,
                was_routed=True,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("[%s] Step decomposition failed: %s", log_tag, e)
            return None

    async def _try_maturity_decomposition(
        self,
        step: Step,
        signature: StepSignature,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str],
        difficulty: float = None,
    ) -> Optional[str]:
        """Try decomposing a step to reuse existing signatures (maturity-based).

        Per mycelium-jaq9: When maturity sigmoid suggests decomposition, try to
        break the step into sub-steps that match existing signatures.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Uses operation embeddings (graph-based) for routing, not text embeddings.

        Escape hatch: If too many sub-steps don't match existing signatures,
        this recognizes a genuinely novel operation and returns None.

        Args:
            step: The step to decompose
            signature: The newly created signature (for logging)
            problem: Original problem text
            context: Results from previous steps
            step_descriptions: Task descriptions
            difficulty: Problem difficulty

        Returns:
            Combined result string if decomposition succeeds, None if escape hatch triggered
        """
        from mycelium.step_signatures.db import RoutingResult
        try:
            # Use planner to decompose the step
            # Build context string from problem and known values
            context_str = f"Original problem: {problem}"
            if context:
                context_str += f"\nKnown values: {context}"
            sub_plan = await self.planner.decompose(
                step.task,
                context=context_str,
            )

            if not sub_plan or not sub_plan.steps:
                logger.debug("[maturity] Decomposition returned empty plan - escape hatch")
                return None

            # Check if we have enough sub-steps
            if len(sub_plan.steps) < MATURITY_ESCAPE_MIN_SUBSTEPS:
                logger.debug(
                    "[maturity] Only %d sub-steps (need %d) - escape hatch",
                    len(sub_plan.steps), MATURITY_ESCAPE_MIN_SUBSTEPS
                )
                return None

            # Route each sub-step through existing signatures
            # Track how many miss (don't match existing)
            adaptive_threshold = get_adaptive_match_threshold()
            miss_count = 0
            results = []

            for sub_step in sub_plan.steps:
                # Use operation embedding for graph-based routing
                # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
                sub_operation_embedding = None
                if sub_step.id in self._operation_embeddings:
                    sub_operation_embedding = np.array(self._operation_embeddings[sub_step.id])

                # Use route_with_confidence to check for existing match
                # Pass dag_step_type for step-node stats lookup
                sub_dag_step_type = getattr(sub_step, 'dsl_hint', None) or sub_step.task
                if sub_operation_embedding is None:
                    # No operation embedding - treat as miss (can't route without graph embedding)
                    routing_result = RoutingResult(signature=None, path=[], confidence=0.0)
                else:
                    routing_result = self.step_db.route_with_confidence(
                        sub_operation_embedding,
                        min_similarity=adaptive_threshold,
                        dag_step_type=sub_dag_step_type,
                    )

                if routing_result.signature is None or routing_result.best_similarity < adaptive_threshold:
                    # No match found - count as miss
                    miss_count += 1
                    logger.debug(
                        "[maturity] Sub-step '%s' has no match (sim=%.2f) - miss %d/%d",
                        sub_step.task[:30],
                        routing_result.best_similarity or 0,
                        miss_count, MATURITY_ESCAPE_MAX_MISSES + 1
                    )

                    if miss_count > MATURITY_ESCAPE_MAX_MISSES:
                        logger.info(
                            "[maturity] Too many misses (%d > %d) - escape hatch triggered",
                            miss_count, MATURITY_ESCAPE_MAX_MISSES
                        )
                        return None
                else:
                    # Match found - execute the sub-step via existing signature
                    matched_sig = routing_result.signature
                    logger.debug(
                        "[maturity] Sub-step '%s' matched sig '%s' (sim=%.2f)",
                        sub_step.task[:30], matched_sig.step_type, routing_result.best_similarity
                    )

                    # Execute via the matched signature
                    sub_result = await self._execute_step(
                        sub_step, problem, context, step_descriptions,
                        depth=signature.depth + 1 if signature.depth else 1,
                        difficulty=difficulty,
                        compute_budget=1.0,  # Single-path for sub-steps
                    )

                    if sub_result and sub_result.result:
                        results.append(sub_result.result)
                        # Update context with sub-step result
                        context[sub_step.id] = sub_result.result

            # All sub-steps processed successfully
            if results:
                combined_result = results[-1]  # Use final sub-step result
                logger.info(
                    "[maturity] Decomposition succeeded with %d sub-steps (%d misses)",
                    len(sub_plan.steps), miss_count
                )
                return combined_result
            else:
                logger.debug("[maturity] No results from decomposition - escape hatch")
                return None

        except Exception as e:
            logger.warning("[maturity] Decomposition error: %s - escape hatch", e)
            return None

    async def _auto_decompose_signature(
        self, signature, recursion_depth: int = 0, difficulty: float = None
    ) -> bool:
        """Auto-decompose a decompose-type signature into computable children.

        Called when we encounter a decompose-type signature that needs children.
        Creates children with actual DSLs and promotes parent to umbrella.

        During BIG BANG phase, recursively decomposes children to explode tree structure.
        Decomposition depth is now DIFFICULTY-AWARE:
        - Harder problems (MATH L5) → deeper decomposition (up to 10 levels)
        - Easier problems (GSM8K) → shallower decomposition (3-4 levels)

        Args:
            signature: The decompose-type signature to decompose
            recursion_depth: Current recursion depth (to prevent runaway)
            difficulty: Problem difficulty (0.0-1.0) for adaptive depth

        Returns:
            True if decomposition succeeded, False otherwise
        """
        from mycelium.step_signatures.umbrella_learner import UmbrellaLearner
        from mycelium.difficulty import get_recommended_depth

        # Difficulty-aware depth limit: harder problems need deeper decomposition
        # Defaults to 5 if difficulty not provided (backward compatible)
        max_depth = get_recommended_depth(difficulty) if difficulty is not None else 5
        if recursion_depth >= max_depth:
            logger.debug(
                "[solver] Decomposition depth limit reached: depth=%d, max=%d (difficulty=%.2f)",
                recursion_depth, max_depth, difficulty or 0.0
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
                                "[expansion] Recursive decompose: '%s' at depth %d (difficulty=%.2f)",
                                child_sig.step_type, child_depth, difficulty or 0.0
                            )
                            await self._auto_decompose_signature(
                                child_sig, recursion_depth + 1, difficulty=difficulty
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

        Per CLAUDE.md: "aggressive branching early, tapering off later"
        - Skip during cold start (first 20 problems)
        - After cold start, batch every UMBRELLA_LEARNING_INTERVAL problems
        The periodic tree review handles optimization.

        Args:
            candidates: Signature IDs that may need decomposition

        Returns:
            Dict with learning statistics (empty if no candidates)
        """
        from mycelium.config import UMBRELLA_LEARNING_INTERVAL

        if not candidates:
            return {"candidates": 0, "decomposed": 0, "children_created": 0}

        # Skip during cold start - let periodic tree review handle optimization
        if self.step_db.is_cold_start():
            logger.debug(
                "[solver] Skipping umbrella learning during cold start (%d candidates deferred)",
                len(candidates)
            )
            return {"candidates": len(candidates), "decomposed": 0, "children_created": 0, "skipped": "cold_start"}

        # After cold start, batch every N problems to reduce LLM calls
        problems_solved = self.step_db.get_total_problems_solved()
        if problems_solved % UMBRELLA_LEARNING_INTERVAL != 0:
            logger.debug(
                "[solver] Deferring umbrella learning until problem %d (%d candidates)",
                (problems_solved // UMBRELLA_LEARNING_INTERVAL + 1) * UMBRELLA_LEARNING_INTERVAL,
                len(candidates)
            )
            return {"candidates": len(candidates), "decomposed": 0, "children_created": 0, "skipped": "batched"}

        logger.info("[solver] Auto-triggering umbrella learning for %d candidates (problem %d)", len(candidates), problems_solved)
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
