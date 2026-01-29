"""StepSignatureDB V2: SQLite-backed database with Natural Language Interface.

Lazy NL approach:
- New signatures start with just step_type + description (= step text)
- clarifying_questions, param_descriptions, dsl_script are empty initially
- These get filled in later as the system learns
"""

import asyncio
import json
import logging
import math
import os
import random
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

import numpy as np

# Version for centroid matrix cache (increment to invalidate old caches)
_CENTROID_CACHE_VERSION = 1

# Canonical atomic operations for embedding-based complexity detection
# Per CLAUDE.md: "prefer embedding similarity over keyword matching"
# Thresholds now imported from config.py per CLAUDE.md "The Flow"
ATOMIC_OPERATIONS = ["ADD(a, b)", "SUB(a, b)", "MUL(a, b)", "DIV(a, b)"]


class SignatureStat(Enum):
    """Types of signature statistics that can be incremented.

    Per CLAUDE.md "New Favorite Pattern": Single enum for all stat types
    to consolidate the increment_signature_stat() API.
    """
    SUCCESS = "successes"
    FAILURE = "operational_failures"


def _parse_centroid_data(data) -> Optional[np.ndarray]:
    """Parse centroid data which may be JSON string or binary bytes.

    SQLite stores centroids as JSON strings, but legacy code expected binary.
    This helper handles both formats.
    """
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    return np.frombuffer(data, dtype=np.float32)

from mycelium.config import (
    EMBEDDING_DIM,
    PARENT_CREDIT_DECAY,
    PARENT_CREDIT_MAX_DEPTH,
    PARENT_CREDIT_MIN,
    AUTO_DEMOTE_ENABLED,
    AUTO_DEMOTE_MAX_SUCCESS_RATE,
    AUTO_DEMOTE_EXCLUDED_TYPES,
    AUTO_DEMOTE_RAMP_DIVISOR,
    AUTO_DEMOTE_MIN_USES_FLOOR,
    AUTO_DEMOTE_MIN_USES_CAP,
    CENTROID_PROPAGATION_MAX_DEPTH,
    CENTROID_MAX_DRIFT,
    CENTROID_DRIFT_DECAY,
    DB_MAX_RETRIES,
    DB_BASE_RETRY_DELAY,
    # Atomic operation detection (per CLAUDE.md "The Flow")
    ATOMIC_SIMILARITY_THRESHOLD,
    ATOMIC_GAP_THRESHOLD,
    # Similarity thresholds (per CLAUDE.md "The Flow": thresholds from config)
    MIN_MATCH_THRESHOLD,
    FORK_GAP_SCALING_FACTOR,
    ROUTING_MIN_SIMILARITY,
    ROUTING_MIN_SIMILARITY_PERMISSIVE,
    ROUTING_BEST_MATCH_MIN_SIMILARITY,
    PLACEMENT_MIN_SIMILARITY,
    HINT_ALTERNATIVES_MIN_SIMILARITY,
    NEW_CHILD_SIMILARITY_THRESHOLD,
    # Welford-adaptive thresholds (per CLAUDE.md "The Flow")
    ADAPTIVE_THRESHOLD_MIN_SAMPLES,
    ADAPTIVE_THRESHOLD_K,
    ADAPTIVE_THRESHOLD_MIN,
    ADAPTIVE_THRESHOLD_MAX,
)

# Import from focused modules (scoring and DSL templates)
from mycelium.step_signatures.scoring import (
    compute_routing_score,
    compute_ucb1_score,
    normalize_step_text,
    increment_total_problems,
)
from mycelium.step_signatures.dsl_templates import infer_dsl_for_signature
from mycelium.step_signatures.graph_extractor import extract_computation_graph

from mycelium.data_layer import get_db, configure_connection, create_connection_manager
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import (
    cosine_similarity,
    batch_cosine_similarity,
    pack_embedding,
    unpack_embedding,
    get_cached_centroid,
    invalidate_centroid_cache,
    compute_centroid_bucket,
    # Signature lookup caches
    get_cached_signature,
    cache_signature,
    get_cached_children,
    cache_children,
    invalidate_signature_cache,
    invalidate_children_cache,
    # Cache manager
    get_cache_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BIG BANG - Smooth Fork Probability Functions
# =============================================================================
# Per CLAUDE.md: "smooth and continuous learning process"
# Per CLAUDE.md: "aggressively branch out early, tapering off later"

def get_system_maturity(sig_count: int) -> float:
    """Compute system maturity as smooth 0→1 value.

    Uses exponential approach: 1 - exp(-sig_count / (target / 3))
    This gives ~95% maturity when sig_count = target.

    Args:
        sig_count: Current number of signatures in the database

    Returns:
        Maturity value between 0 (fresh) and 1 (mature)
    """
    from mycelium.config import BIG_BANG_TARGET_SIGNATURES, BIG_BANG_TAU_DIVISOR

    # Smooth exponential curve: rises quickly at first, asymptotes to 1
    # Per config: TAU_DIVISOR = 3.0 means system reaches 95% maturity at TARGET_SIGNATURES
    tau = BIG_BANG_TARGET_SIGNATURES / BIG_BANG_TAU_DIVISOR
    maturity = 1.0 - math.exp(-sig_count / tau)
    return maturity


def get_fork_center(maturity: float) -> float:
    """Compute the depth where forking is most likely (fork center).

    Starts at MIN_FORK_DEPTH (level 6) and drifts toward root (level 1)
    as the system matures. Root (level 0) never forks.

    Args:
        maturity: System maturity from get_system_maturity() (0-1)

    Returns:
        Float representing the current fork center depth
    """
    from mycelium.config import MIN_FORK_DEPTH, BIG_BANG_FORK_CENTER_DRIFT_RATE

    # Start at MIN_FORK_DEPTH, drift toward level 1 (not 0, root never forks)
    # At maturity=0: center = MIN_FORK_DEPTH (e.g., 6)
    # At maturity=1: center = 1 + (MIN_FORK_DEPTH - 1) * (1 - drift_rate)
    drift = maturity * BIG_BANG_FORK_CENTER_DRIFT_RATE
    center = MIN_FORK_DEPTH - drift * (MIN_FORK_DEPTH - 1)
    return max(1.0, center)  # Never below level 1


def _sigmoid(x: float, steepness: float = 1.0) -> float:
    """Standard sigmoid function centered at 0.

    Args:
        x: Input value
        steepness: How sharp the transition is (higher = sharper)

    Returns:
        Value between 0 and 1
    """
    return 1.0 / (1.0 + math.exp(-steepness * x))


def compute_fork_probability(
    depth: int,
    sig_count: int,
    best_similarity: float,
    fork_threshold: float,
    has_existing_forks_at_level: bool = False,
) -> float:
    """Compute probability of forking at a given depth.

    Uses smooth, continuous functions with sigmoid transitions.
    Per CLAUDE.md: "no hard thresholds", "smooth and continuous"

    Factors:
    1. Depth factor: Sigmoid around fork center (protected levels → allowed levels)
    2. Similarity gap: Larger gap below threshold → more likely to fork
    3. Maturity: System maturity modulates overall fork aggressiveness
    4. Hysteresis: Levels with existing forks get bonus (easier to fork again)

    Args:
        depth: Current depth in the tree (0 = root)
        sig_count: Current signature count (for maturity calculation)
        best_similarity: Similarity of best matching child
        fork_threshold: Current fork threshold
        has_existing_forks_at_level: Whether this level has existing fork branches

    Returns:
        Fork probability between MIN_FORK_PROB and MAX_FORK_PROB
    """
    from mycelium.config import (
        BIG_BANG_SIGMOID_STEEPNESS,
        BIG_BANG_HYSTERESIS_BONUS,
        BIG_BANG_MIN_FORK_PROB,
        BIG_BANG_MAX_FORK_PROB,
        FORK_GAP_SCALING_FACTOR,
        MIN_FORK_DEPTH,
    )

    # Root (depth 0) NEVER forks
    if depth == 0:
        return 0.0

    # HARD CUTOFF: Protected levels (below MIN_FORK_DEPTH) NEVER fork
    # This preserves headroom for future domains
    if depth < MIN_FORK_DEPTH:
        return 0.0

    maturity = get_system_maturity(sig_count)
    fork_center = get_fork_center(maturity)

    # 1. DEPTH FACTOR: Sigmoid around fork center
    # Depths below fork center → low probability (protected)
    # Depths at/above fork center → high probability (allowed)
    depth_distance = depth - fork_center
    depth_factor = _sigmoid(depth_distance, BIG_BANG_SIGMOID_STEEPNESS)

    # 2. SIMILARITY GAP: How far below threshold is the best match?
    # Larger gap → more reason to fork (problem is divergent)
    gap = fork_threshold - best_similarity
    gap_factor = max(0.0, min(1.0, gap * FORK_GAP_SCALING_FACTOR))  # Scale to 0-1 range

    # 3. MATURITY MODULATION: Less aggressive forking as system matures
    # Cold start (maturity=0): fork more aggressively
    # Mature (maturity=1): be more selective
    maturity_factor = 1.0 - (maturity * 0.5)  # Range: 1.0 → 0.5

    # 4. HYSTERESIS: Levels with existing forks get bonus
    hysteresis = BIG_BANG_HYSTERESIS_BONUS if has_existing_forks_at_level else 0.0

    # Combine factors: depth is primary, gap and maturity modulate
    base_prob = depth_factor * gap_factor * maturity_factor
    final_prob = base_prob + hysteresis * (1.0 - base_prob)  # Hysteresis as boost

    # Clamp to configured range
    return max(BIG_BANG_MIN_FORK_PROB, min(BIG_BANG_MAX_FORK_PROB, final_prob))


def should_fork_at_depth(
    depth: int,
    sig_count: int,
    best_similarity: float,
    fork_threshold: float,
    has_existing_forks_at_level: bool = False,
) -> bool:
    """Probabilistic decision: should we fork at this depth?

    Uses compute_fork_probability() and random sampling.

    Args:
        depth: Current depth in the tree
        sig_count: Current signature count
        best_similarity: Similarity of best matching child
        fork_threshold: Current fork threshold
        has_existing_forks_at_level: Whether this level has existing forks

    Returns:
        True if we should fork, False otherwise
    """
    prob = compute_fork_probability(
        depth=depth,
        sig_count=sig_count,
        best_similarity=best_similarity,
        fork_threshold=fork_threshold,
        has_existing_forks_at_level=has_existing_forks_at_level,
    )

    # Sample from probability
    should_fork = random.random() < prob

    # Log at INFO level when fork happens (important event), DEBUG otherwise
    if should_fork:
        logger.info(
            "[FORK_DECISION] depth=%d sig_count=%d best_sim=%.3f threshold=%.3f "
            "gap=%.3f hysteresis=%s fork_prob=%.3f → FORK",
            depth, sig_count, best_similarity, fork_threshold,
            fork_threshold - best_similarity, has_existing_forks_at_level, prob
        )
    else:
        logger.debug(
            "[FORK_DECISION] depth=%d sig_count=%d best_sim=%.3f threshold=%.3f "
            "gap=%.3f hysteresis=%s fork_prob=%.3f → NO_FORK",
            depth, sig_count, best_similarity, fork_threshold,
            fork_threshold - best_similarity, has_existing_forks_at_level, prob
        )

    return should_fork


# =============================================================================
# ROUTING RESULT (for MCTS confidence scoring)
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class RoutingResult:
    """Result of routing an embedding through the signature hierarchy.

    Includes confidence signals for MCTS multi-path exploration:
    - confidence: Overall routing confidence (0-1), product of per-level confidences
    - ucb1_gaps: Gap between top-2 UCB1 scores at each level (larger = more certain)
    - alternatives: Top-k alternatives at each level (for multi-path exploration)
    - best_similarity: Cosine similarity of the best match (for amplitude logging)

    Usage:
        result = db.route_through_hierarchy_v2(embedding)
        if result.confidence < 0.5:
            # Low confidence - consider exploring alternative paths
            for level_alts in result.alternatives:
                for alt_sig, alt_score in level_alts:
                    # Try alternative path...
    """
    signature: Optional[StepSignature] = None  # Best matching signature (leaf)
    path: list[StepSignature] = field(default_factory=list)  # Path from root to leaf
    confidence: float = 1.0  # Overall confidence (0-1)
    ucb1_gaps: list[float] = field(default_factory=list)  # Gap at each routing level
    alternatives: list[list[tuple[StepSignature, float]]] = field(default_factory=list)  # Top-k alts per level
    best_similarity: Optional[float] = None  # Cosine similarity of best match at final level

    @property
    def is_match(self) -> bool:
        """Whether a signature was matched."""
        return self.signature is not None

    @property
    def depth(self) -> int:
        """Routing depth (number of hops from root)."""
        return len(self.path)

    @property
    def min_gap(self) -> float:
        """Minimum UCB1 gap across all levels (weakest link)."""
        return min(self.ucb1_gaps) if self.ucb1_gaps else 0.0


# =============================================================================
# ADAPTIVE SIMILARITY THRESHOLDS (Welford-based)
# =============================================================================
# Per CLAUDE.md: Route by what operations DO (graph_embedding), not what they SOUND LIKE.
# Instead of magic thresholds, we learn what "same" and "similar" mean from data.
#
# Two distributions tracked with Welford's algorithm:
# 1. match_sim_* - similarities when we successfully reuse a signature (dedup threshold)
# 2. cluster_sim_* - similarities between siblings (cluster threshold)
#
# Thresholds computed as: mean - k * stddev (adaptive, no magic numbers)

@dataclass
class AdaptiveThresholds:
    """Adaptive similarity thresholds learned from data."""
    dedup_threshold: float      # Above this = same node (return existing)
    cluster_threshold: float    # Above this = same cluster (share parent)
    # Welford stats for dedup (match) similarities
    match_count: int = 0
    match_mean: float = 0.0
    match_m2: float = 0.0
    # Welford stats for cluster (sibling) similarities
    cluster_count: int = 0
    cluster_mean: float = 0.0
    cluster_m2: float = 0.0


class PlacementDecision(Enum):
    """Welford-based placement decision for new signatures.

    Per mycelium-br28: After cold start, use z-scores relative to parent's
    child similarity distribution to decide placement.

    Decision logic (z-score thresholds from config):
    - MERGE: z > WELFORD_MERGE_THRESHOLD (very similar to existing sibling)
    - SIBLING: z > WELFORD_SIBLING_THRESHOLD (normal range, add as peer)
    - CHILD: z > WELFORD_CHILD_THRESHOLD (somewhat different, create sub-cluster)
    - NEW_CLUSTER: z <= WELFORD_CHILD_THRESHOLD (very different, new cluster under root)
    """
    SIBLING = "sibling"       # Normal: add as sibling (child of same parent)
    CHILD = "child"           # Somewhat different: create sub-cluster
    MERGE = "merge"           # Very similar: merge into existing signature
    NEW_CLUSTER = "new_cluster"  # Very different: new cluster under root


def get_adaptive_thresholds(conn) -> AdaptiveThresholds:
    """Get adaptive thresholds from db_metadata, with cold-start defaults.

    Cold-start defaults are conservative:
    - dedup_threshold: 0.95 (very high sim = same node)
    - cluster_threshold: 0.80 (moderately high sim = same cluster)

    As data accumulates, thresholds adapt to learned distributions.

    Per CLAUDE.md New Favorite Pattern: Uses StateManager for db_metadata access.
    """
    from mycelium.config import (
        ADAPTIVE_THRESHOLD_K,  # Number of stddevs below mean
        COLD_START_DEDUP_THRESHOLD,
        COLD_START_CLUSTER_THRESHOLD,
        ADAPTIVE_MIN_SAMPLES,  # Min samples before using learned thresholds
    )
    from mycelium.data_layer.state_manager import get_state_manager, StateManager

    # Read Welford stats via StateManager
    sm = get_state_manager()
    match_stats = sm.get_welford_stats(StateManager.PREFIX_SIM_STATS_MATCH)
    cluster_stats = sm.get_welford_stats(StateManager.PREFIX_SIM_STATS_CLUSTER)

    # Compute adaptive thresholds if we have enough data
    if match_stats.count >= ADAPTIVE_MIN_SAMPLES:
        dedup_threshold = match_stats.mean - ADAPTIVE_THRESHOLD_K * match_stats.stddev
        dedup_threshold = max(0.5, min(0.99, dedup_threshold))  # Clamp to reasonable range
    else:
        dedup_threshold = COLD_START_DEDUP_THRESHOLD

    if cluster_stats.count >= ADAPTIVE_MIN_SAMPLES:
        cluster_threshold = cluster_stats.mean - ADAPTIVE_THRESHOLD_K * cluster_stats.stddev
        cluster_threshold = max(0.3, min(0.95, cluster_threshold))  # Clamp to reasonable range
    else:
        cluster_threshold = COLD_START_CLUSTER_THRESHOLD

    return AdaptiveThresholds(
        dedup_threshold=dedup_threshold,
        cluster_threshold=cluster_threshold,
        match_count=match_stats.count,
        match_mean=match_stats.mean,
        match_m2=match_stats.m2,
        cluster_count=cluster_stats.count,
        cluster_mean=cluster_stats.mean,
        cluster_m2=cluster_stats.m2,
    )


def get_global_exec_success_floor(conn, k: float = 2.0, min_samples: int = 10) -> float:
    """Compute adaptive failure rate floor from aggregated Welford exec stats.

    Per CLAUDE.md "System Independence": Use learned stats instead of magic numbers.

    Aggregates exec_n and exec_successes across all signatures to compute
    a global success rate distribution. The floor is: mean - k*std

    Args:
        conn: Database connection
        k: Number of stddevs below mean for floor (default 2.0)
        min_samples: Minimum total observations before using adaptive floor

    Returns:
        Adaptive success rate floor (below which = "too low")
        During cold start, returns conservative default of 0.3
    """
    # Aggregate exec stats across all signatures
    row = conn.execute(
        """
        SELECT
            SUM(exec_n) as total_n,
            SUM(exec_successes) as total_successes
        FROM welford_stats
        WHERE exec_n > 0
        """
    ).fetchone()

    if not row or row["total_n"] is None or row["total_n"] < min_samples:
        # Cold start: use conservative default
        return 0.3

    total_n = row["total_n"]
    total_successes = row["total_successes"] or 0

    # Global success rate
    global_mean = total_successes / total_n if total_n > 0 else 0.5

    # Compute variance using aggregate method:
    # For binomial (success/fail), variance = p * (1-p) / n per signature
    # Aggregate variance is harder, so we'll use per-signature success rates
    rates_row = conn.execute(
        """
        SELECT
            AVG(CAST(exec_successes AS FLOAT) / exec_n) as mean_rate,
            COUNT(*) as n_sigs
        FROM welford_stats
        WHERE exec_n >= 3
        """
    ).fetchone()

    if not rates_row or rates_row["n_sigs"] < 3:
        # Not enough signatures with data, use simple floor
        floor = max(0.1, global_mean - 0.2)  # 20% below mean, min 0.1
        return floor

    # Compute variance of per-signature success rates
    var_row = conn.execute(
        """
        SELECT
            AVG((CAST(exec_successes AS FLOAT) / exec_n - ?) * (CAST(exec_successes AS FLOAT) / exec_n - ?)) as var
        FROM welford_stats
        WHERE exec_n >= 3
        """,
        (rates_row["mean_rate"], rates_row["mean_rate"])
    ).fetchone()

    if not var_row or var_row["var"] is None:
        floor = max(0.1, global_mean - 0.2)
        return floor

    std = math.sqrt(var_row["var"]) if var_row["var"] > 0 else 0.1

    # Floor = mean - k * std, clamped to [0.1, 0.5]
    floor = global_mean - k * std
    floor = max(0.1, min(0.5, floor))

    logger.debug(
        "[exec_floor] Global exec floor: %.3f (mean=%.3f, std=%.3f, n=%d)",
        floor, global_mean, std, total_n
    )

    return floor


def update_similarity_stats(conn, stat_type: str, similarity: float) -> None:
    """Update Welford stats for similarity tracking using SAME connection.

    IMPORTANT: Uses passed conn to avoid nested transactions that cause deadlock.
    The StateManager version opens a new connection which waits on busy_timeout.

    Per CLAUDE.md The Flow: Database Statistics -> Welford -> Tree Structure.

    Args:
        conn: Database connection - MUST use this to avoid deadlock
        stat_type: 'match' (for dedup) or 'cluster' (for clustering)
        similarity: The observed similarity value
    """
    from datetime import datetime, timezone

    prefix = f'sim_stats_{stat_type}'
    now = datetime.now(timezone.utc).isoformat()

    # Read current stats from same connection (no deadlock)
    cursor = conn.execute(
        "SELECT key, value FROM db_metadata WHERE key LIKE ?",
        (f'{prefix}_%',)
    )
    stats = {}
    for row in cursor:
        try:
            parsed = json.loads(row["value"] if isinstance(row, dict) else row[1])
            stats[row["key"] if isinstance(row, dict) else row[0]] = parsed.get("value", parsed) if isinstance(parsed, dict) else parsed
        except (json.JSONDecodeError, TypeError):
            pass

    count = int(stats.get(f'{prefix}_count', 0))
    mean = float(stats.get(f'{prefix}_mean', 0.0))
    m2 = float(stats.get(f'{prefix}_m2', 0.0))

    # Welford update
    count += 1
    delta = similarity - mean
    mean += delta / count
    delta2 = similarity - mean
    m2 += delta * delta2

    # Write back using same connection
    for key, val in [(f'{prefix}_count', count), (f'{prefix}_mean', mean), (f'{prefix}_m2', m2)]:
        json_val = json.dumps({'value': val})
        conn.execute(
            """INSERT INTO db_metadata (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (key, json_val, now, json_val, now)
        )


def find_global_best_match(
    conn,
    graph_embedding: np.ndarray,
    exclude_umbrellas: bool = False,  # Include umbrellas to find canonical operation type
) -> tuple[Optional["StepSignature"], float]:
    """Find the globally best matching signature by graph_embedding similarity.

    This searches ALL signatures (not just current routing branch) to find
    the most similar one. Used for dedup check before creating new signatures.

    Includes umbrellas because they represent canonical operation types.
    If matched against an umbrella, caller should look for/create leaf under it.

    Args:
        conn: Database connection
        graph_embedding: The query graph embedding
        exclude_umbrellas: If False, includes umbrellas in search (default False)

    Returns:
        (best_signature, similarity) or (None, 0.0) if no match
    """
    # Get all signatures with graph_embeddings (including umbrellas by default)
    umbrella_filter = "AND is_semantic_umbrella = 0" if exclude_umbrellas else ""
    cursor = conn.execute(f"""
        SELECT id, step_type, graph_embedding, is_semantic_umbrella, uses, successes, depth
        FROM step_signatures
        WHERE graph_embedding IS NOT NULL {umbrella_filter}
    """)

    best_sig = None
    best_sim = 0.0

    for row in cursor:
        sig_id, step_type, graph_emb_json, is_umbrella, uses, successes, depth = row
        sig_graph_emb = np.array(json.loads(graph_emb_json))

        sim = cosine_similarity(graph_embedding, sig_graph_emb)
        if sim > best_sim:
            best_sim = sim
            best_sig = StepSignature(
                id=sig_id,
                step_type=step_type,
                graph_embedding=json.loads(graph_emb_json),
                is_semantic_umbrella=bool(is_umbrella),
                uses=uses,
                successes=successes,
                depth=depth,
            )

    return best_sig, best_sim


def find_best_child_match(
    conn,
    umbrella_id: int,
    graph_embedding: np.ndarray,
) -> tuple[Optional["StepSignature"], float]:
    """Find the best matching LEAF child of an umbrella by graph_embedding similarity.

    When we match an umbrella above dedup threshold, we should check if there's
    already a leaf under it with matching graph_embedding before creating a new one.

    Args:
        conn: Database connection
        umbrella_id: The umbrella's signature ID
        graph_embedding: The query graph embedding

    Returns:
        (best_leaf, similarity) or (None, 0.0) if no matching leaf found
    """
    # Get all leaf children of this umbrella with graph_embeddings
    cursor = conn.execute("""
        SELECT s.id, s.step_type, s.graph_embedding, s.uses, s.successes, s.depth
        FROM step_signatures s
        JOIN signature_relationships r ON s.id = r.child_id
        WHERE r.parent_id = ?
          AND s.is_semantic_umbrella = 0
          AND s.graph_embedding IS NOT NULL
    """, (umbrella_id,))

    best_sig = None
    best_sim = 0.0

    for row in cursor:
        sig_id, step_type, graph_emb_json, uses, successes, depth = row
        sig_graph_emb = np.array(json.loads(graph_emb_json))

        sim = cosine_similarity(graph_embedding, sig_graph_emb)
        if sim > best_sim:
            best_sim = sim
            best_sig = StepSignature(
                id=sig_id,
                step_type=step_type,
                graph_embedding=json.loads(graph_emb_json),
                is_semantic_umbrella=False,
                uses=uses,
                successes=successes,
                depth=depth,
            )

    return best_sig, best_sim


class StepSignatureDB:
    """SQLite-backed database for step-level signatures.

    V2: Simplified schema with Natural Language Interface.
    """

    def __init__(self, db_path: str = None, embedder=None):
        """Initialize the database.

        Per CLAUDE.md "New Favorite Pattern": All database connections go through the data layer.

        Args:
            db_path: Optional path to SQLite database. If provided, creates
                     a ConnectionManager for that path instead of using the global singleton.
            embedder: Optional Embedder instance. If None, lazily fetches singleton on first use.
        """
        if db_path:
            # Per mycelium-7eqw: Use data layer's create_connection_manager instead of direct sqlite3.connect
            # This consolidates all DB connection logic through the data layer
            self._db = create_connection_manager(db_path)
            self._db_path = db_path
        else:
            self._db = get_db()
            # Use default DB path from config when using global singleton
            from mycelium.config import DB_PATH
            self._db_path = DB_PATH

        # Lazy-loaded centroid matrix for fast batch similarity
        self._centroid_matrix: Optional[np.ndarray] = None
        self._centroid_sig_ids: Optional[list[int]] = None
        self._centroid_rows: Optional[list] = None  # Cache rows for result building

        # Cached root signature (never changes after creation)
        self._cached_root: Optional[StepSignature] = None

        # Cached atomic operation embeddings for complexity detection
        # Per CLAUDE.md: "prefer embedding similarity over keyword matching"
        self._atomic_embeddings: Optional[list[np.ndarray]] = None

        # Lazy-loaded embedder (per CLAUDE.md: consolidate method calls)
        self._embedder = embedder

        self._init_schema()

        # Register with CacheManager for coordinated invalidation
        get_cache_manager().register_db(self)

    @property
    def embedder(self):
        """Get the embedder instance, lazily fetching singleton if needed."""
        if self._embedder is None:
            from mycelium.embedder import Embedder
            self._embedder = Embedder.get_instance()
        return self._embedder

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    def close(self):
        """Close the database connection."""
        if self._db:
            self._db.close()

    @property
    def _centroid_cache_path(self) -> str:
        """Get the path for the centroid matrix cache file."""
        return f"{self._db_path}.centroid_cache.npz"

    def _load_centroid_matrix_from_cache(self) -> bool:
        """Try to load centroid matrix from disk cache.

        Returns True if cache was loaded successfully, False otherwise.
        Cache is invalidated if signature count doesn't match.
        """
        cache_path = self._centroid_cache_path
        if not os.path.exists(cache_path):
            return False

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                version = int(data.get("version", 0))
                if version != _CENTROID_CACHE_VERSION:
                    logger.debug("[db] Centroid cache version mismatch, rebuilding")
                    return False

                cached_count = int(data["sig_count"])

                # Quick staleness check: compare non-archived signature count
                with self._connection() as conn:
                    row = conn.execute("SELECT COUNT(*) FROM step_signatures WHERE centroid IS NOT NULL AND is_archived = 0").fetchone()
                    current_count = row[0] if row else 0

                if cached_count != current_count:
                    logger.debug("[db] Centroid cache stale (%d vs %d sigs), rebuilding", cached_count, current_count)
                    return False

                # Load the cached data
                self._centroid_matrix = data["matrix"]
                self._centroid_sig_ids = data["sig_ids"].tolist()
                # Rows not cached - will be fetched lazily in find_similar
                self._centroid_rows = None

                logger.debug("[db] Loaded centroid matrix from cache: %d signatures", len(self._centroid_sig_ids))
                return True

        except Exception as e:
            logger.warning("[db] Failed to load centroid cache: %s", e)
            return False

    def _save_centroid_matrix_to_cache(self):
        """Save the centroid matrix to disk cache."""
        if self._centroid_matrix is None or len(self._centroid_sig_ids) == 0:
            return

        try:
            np.savez(
                self._centroid_cache_path,
                matrix=self._centroid_matrix,
                sig_ids=np.array(self._centroid_sig_ids, dtype=np.int64),
                sig_count=len(self._centroid_sig_ids),
                version=_CENTROID_CACHE_VERSION,
            )
            logger.debug("[db] Saved centroid matrix to cache: %d signatures", len(self._centroid_sig_ids))
        except Exception as e:
            logger.warning("[db] Failed to save centroid cache: %s", e)

    def _delete_centroid_cache(self):
        """Delete the centroid matrix cache file."""
        cache_path = self._centroid_cache_path
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.debug("[db] Deleted centroid cache file")
            except OSError as e:
                logger.warning("[db] Failed to delete centroid cache: %s", e)

    @contextmanager
    def _connection(self):
        """Get a database connection.

        Per CLAUDE.md "New Favorite Pattern": Single connection pathway through data layer.
        """
        with self._db.connection() as conn:
            yield conn

    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            init_db(conn)

    def get_signature_count(self) -> int:
        """Get total number of signatures in the database.

        Returns:
            Count of signatures (including archived ones)
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM step_signatures"
            ).fetchone()
            return row[0] if row else 0

    # =========================================================================
    # Root Management (Single Entry Point)
    # =========================================================================

    def get_root(self) -> Optional[StepSignature]:
        """Get the root signature (single entry point for all routing).

        The root is created automatically when the first signature is added.
        All problems route through the root first. Result is cached since
        root never changes after creation.

        Returns:
            The root signature, or None if database is empty
        """
        # Return cached root if available
        if self._cached_root is not None:
            return self._cached_root

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM step_signatures WHERE is_root = 1 LIMIT 1"
            ).fetchone()
            if row:
                self._cached_root = self._row_to_signature(row)
                return self._cached_root
            return None

    def has_root(self) -> bool:
        """Check if a root signature exists."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM step_signatures WHERE is_root = 1 LIMIT 1"
            ).fetchone()
            return row is not None

    def _create_scaffold_branch(
        self,
        conn,
        parent_id: int,
        embedding: np.ndarray,
        depth: int,
    ) -> Optional[StepSignature]:
        """Create a new scaffold branch (fork) for a divergent problem type.

        Called during routing when a problem doesn't match existing paths well.
        Creates a new placeholder LEAF that can be promoted to umbrella when
        children are added.

        Per CLAUDE.md System Independence: Don't create umbrellas without children.
        The signature will be promoted to umbrella when children are actually added.

        Args:
            conn: Database connection (within transaction)
            parent_id: ID of parent umbrella to branch from
            embedding: Embedding of the divergent problem
            depth: Depth of the new branch

        Returns:
            The new branch signature, or None on failure
        """
        branch_id = f"branch_L{depth}_{uuid.uuid4().hex[:8]}"

        try:
            return self._create_signature_atomic(
                conn=conn,
                step_text=f"Dynamic branch at level {depth}",
                embedding=embedding,
                parent_id=parent_id,
                is_umbrella=False,  # Per CLAUDE.md: create as leaf, promote when children added
                signature_id_override=branch_id,
                depth_override=depth,
                skip_example=True,
            )
        except sqlite3.IntegrityError as e:
            logger.warning("[db] Failed to create scaffold branch: %s", e)
            return None

    def compute_graph_centroid_from_children(
        self,
        conn,
        parent_id: int,
    ) -> Optional[np.ndarray]:
        """Compute graph_centroid as average of children's graph_embeddings.

        For routers: graph_embedding = average of children's graph_embeddings
        This enables graph-space routing through the entire hierarchy.

        Args:
            conn: Database connection
            parent_id: ID of the parent signature

        Returns:
            New graph_centroid as numpy array, or None if no children have embeddings
        """
        cursor = conn.execute(
            """SELECT graph_embedding
               FROM step_signatures s
               JOIN signature_relationships r ON s.id = r.child_id
               WHERE r.parent_id = ?
                 AND graph_embedding IS NOT NULL
                 AND graph_embedding != ''""",
            (parent_id,),
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        embeddings = []
        for row in rows:
            try:
                emb = np.array(json.loads(row["graph_embedding"]))
                embeddings.append(emb)
            except (json.JSONDecodeError, ValueError):
                continue

        if not embeddings:
            return None

        # Average of children's graph embeddings
        graph_centroid = np.mean(embeddings, axis=0)
        return graph_centroid

    def propagate_graph_centroid_to_parents(
        self,
        conn,
        signature_id: int,
        include_self: bool = False,
    ):
        """Single entry point for graph_centroid updates.

        Recomputes graph_centroids up the tree when a signature's graph_embedding changes.
        This is the ONLY function that should update router centroids.

        Args:
            conn: Database connection
            signature_id: ID of the signature whose graph_embedding changed
            include_self: If True, also recompute this signature's centroid first
                          (used when promoting to umbrella)
        """
        max_depth = CENTROID_PROPAGATION_MAX_DEPTH

        # Optionally recompute this node's own centroid first (for promote_to_umbrella)
        if include_self:
            graph_centroid = self.compute_graph_centroid_from_children(conn, signature_id)
            if graph_centroid is not None:
                conn.execute(
                    "UPDATE step_signatures SET graph_embedding = ? WHERE id = ?",
                    (json.dumps(graph_centroid.tolist()), signature_id),
                )
                invalidate_signature_cache(signature_id)
                logger.debug(
                    "[db] Computed graph_centroid for sig %d (include_self)",
                    signature_id
                )

        # Fetch all ancestors
        cursor = conn.execute(
            """
            WITH RECURSIVE ancestors AS (
                SELECT parent_id, 1 as depth
                FROM signature_relationships
                WHERE child_id = ?

                UNION ALL

                SELECT r.parent_id, a.depth + 1
                FROM signature_relationships r
                JOIN ancestors a ON r.child_id = a.parent_id
                WHERE a.depth < ?
            )
            SELECT DISTINCT parent_id, MIN(depth) as depth
            FROM ancestors
            GROUP BY parent_id
            ORDER BY depth
            """,
            (signature_id, max_depth),
        )
        ancestors = cursor.fetchall()

        if not ancestors:
            return

        # Update each ancestor's graph_centroid
        for parent_id, _ in ancestors:
            graph_centroid = self.compute_graph_centroid_from_children(conn, parent_id)
            if graph_centroid is not None:
                conn.execute(
                    "UPDATE step_signatures SET graph_embedding = ? WHERE id = ?",
                    (json.dumps(graph_centroid.tolist()), parent_id),
                )
                invalidate_signature_cache(parent_id)
                logger.debug(
                    "[db] Propagated graph_centroid to parent %d",
                    parent_id
                )

    # =========================================================================
    # Welford-Adaptive Similarity Threshold (per CLAUDE.md "The Flow")
    # =========================================================================

    def _get_adaptive_similarity_threshold(
        self,
        context: str = "match",
        fallback: float = None,
    ) -> float:
        """Get Welford-adaptive similarity threshold.

        Per CLAUDE.md "The Flow": Database Statistics -> Welford -> Tree Structure.
        Instead of static 0.85, adapt based on observed similarity distribution.

        Args:
            context: Welford stats prefix - 'match' (routing) or 'cluster' (clustering)
            fallback: Fallback if insufficient data (default: MIN_MATCH_THRESHOLD)

        Returns:
            Adaptive threshold clamped to [ADAPTIVE_THRESHOLD_MIN, ADAPTIVE_THRESHOLD_MAX]
        """
        from mycelium.data_layer.state_manager import get_state_manager

        if fallback is None:
            fallback = MIN_MATCH_THRESHOLD

        # Get Welford stats for this context
        stats = get_state_manager().get_welford_stats(f"sim_stats_{context}")

        # Need sufficient samples for reliable estimate
        if stats.count < ADAPTIVE_THRESHOLD_MIN_SAMPLES:
            logger.debug(
                "[db] Adaptive threshold: insufficient samples (%d < %d), using fallback %.3f",
                stats.count, ADAPTIVE_THRESHOLD_MIN_SAMPLES, fallback
            )
            return fallback

        # Adaptive: mean - k * std (captures ~93% of good matches at k=1.5)
        # This ensures we don't reject matches that are within normal variance
        adaptive = stats.mean - ADAPTIVE_THRESHOLD_K * stats.stddev

        # Clamp to reasonable range
        clamped = max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, adaptive))

        logger.debug(
            "[db] Adaptive threshold: mean=%.3f, std=%.3f, raw=%.3f, clamped=%.3f (n=%d)",
            stats.mean, stats.stddev, adaptive, clamped, stats.count
        )

        return clamped

    # =========================================================================
    # Consolidated Routing Core
    # =========================================================================

    def _route_core(
        self,
        operation_embedding: np.ndarray,
        min_similarity: Optional[float] = None,
        max_depth: int = None,
        track_alternatives: bool = False,
        top_k: int = 3,
        use_ucb1: bool = True,
        epsilon_exploration: bool = False,
        dag_step_type: Optional[str] = None,
        step_position: Optional[int] = None,
    ) -> RoutingResult:
        """Core DFS routing through signature hierarchy using graph embeddings.

        SINGLE PATHWAY for DFS routing. All variations use this function.

        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.

        Args:
            min_similarity: Minimum similarity threshold. None = use adaptive (recommended)
            step_position: Optional step position (1, 2, 3...) for position-aware
                routing. When provided, uses plan_step_stats to penalize nodes
                that historically fail at this position.
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

        # Resolve adaptive threshold if not specified
        if min_similarity is None:
            min_similarity = self._get_adaptive_similarity_threshold("match")

        if max_depth is None:
            max_depth = UMBRELLA_MAX_DEPTH
        max_depth = max(1, min(int(max_depth), 100))

        root = self.get_root()
        if root is None:
            return RoutingResult(signature=None, path=[], confidence=0.0)

        path = [root]
        current = root
        depth = 0
        ucb1_gaps = []
        alternatives = []
        confidence_factors = []
        best_similarity = None

        while depth < max_depth:
            if not current.is_semantic_umbrella:
                break

            children = self.get_children(current.id, for_routing=True)
            if not children:
                break

            step_stats_map = {}
            if dag_step_type and use_ucb1:
                from mycelium.data_layer.mcts import get_dag_step_node_stats_batch
                child_ids = [c.id for c, _ in children if c.id is not None]
                step_stats_map = get_dag_step_node_stats_batch(dag_step_type, child_ids)

            # Position-aware stats: look up historical success at this step position
            # Per CLAUDE.md: "Failures Are Valuable Data Points" - use position stats
            position_stats_map = {}
            if step_position is not None and use_ucb1:
                from mycelium.config import POSITION_STATS_ENABLED, POSITION_STATS_MIN_OBS
                if POSITION_STATS_ENABLED:
                    child_ids = [c.id for c, _ in children if c.id is not None]
                    for child_id in child_ids:
                        pos_stats = self.get_node_success_by_position(child_id)
                        if step_position in pos_stats:
                            stats = pos_stats[step_position]
                            if stats["n"] >= POSITION_STATS_MIN_OBS:
                                position_stats_map[child_id] = stats

            parent_uses = current.uses or 1
            scored_children = []

            for child_sig, _condition in children:
                graph_emb = child_sig.graph_embedding
                if graph_emb is None:
                    continue
                if not isinstance(graph_emb, np.ndarray):
                    graph_emb = np.array(graph_emb)

                sim = cosine_similarity(operation_embedding, graph_emb)
                threshold = min_similarity * 0.7 if track_alternatives else min_similarity

                if sim >= threshold:
                    if use_ucb1:
                        score = compute_ucb1_score(
                            cosine_sim=sim,
                            uses=child_sig.uses,
                            successes=child_sig.successes,
                            parent_uses=parent_uses,
                            last_used_at=child_sig.last_used_at,
                            step_node_stats=step_stats_map.get(child_sig.id),
                        )
                        # Apply position-aware penalty if we have stats for this position
                        if child_sig.id in position_stats_map:
                            pos_stats = position_stats_map[child_sig.id]
                            pos_success = pos_stats["mean_success"]
                            # Penalty: multiply score by position success rate
                            # Low success at this position = lower score
                            from mycelium.config import POSITION_STATS_WEIGHT
                            position_factor = POSITION_STATS_WEIGHT * pos_success + (1 - POSITION_STATS_WEIGHT)
                            score *= position_factor
                            logger.debug(
                                "[routing] Position penalty applied: node=%d pos=%d "
                                "pos_success=%.2f factor=%.2f",
                                child_sig.id, step_position, pos_success, position_factor
                            )
                    else:
                        score = compute_routing_score(
                            sim, child_sig.uses, child_sig.successes, child_sig.last_used_at
                        )
                    scored_children.append((child_sig, score, sim))

            if not scored_children:
                break

            scored_children.sort(key=lambda x: x[1], reverse=True)

            if epsilon_exploration and len(scored_children) > 1:
                from mycelium.config import EXPLORATION_EPSILON
                import random
                if EXPLORATION_EPSILON > 0 and random.random() < EXPLORATION_EPSILON:
                    idx = random.randint(0, len(scored_children) - 1)
                    scored_children[0], scored_children[idx] = scored_children[idx], scored_children[0]

            if track_alternatives:
                alternatives.append([(s, sc) for s, sc, _ in scored_children[:top_k]])

            best_score = scored_children[0][1]
            if len(scored_children) > 1:
                gap = best_score - scored_children[1][1]
                level_confidence = min(1.0, gap / 0.3)
            else:
                gap = 1.0
                level_confidence = 1.0

            ucb1_gaps.append(gap)
            confidence_factors.append(level_confidence)

            best_child, _, best_sim = scored_children[0]
            best_similarity = best_sim
            if best_sim < min_similarity:
                break

            # Record routing similarity for Welford stats (per periodic tree review plan)
            # This tracks the distribution of similarities at each routing decision
            self.update_welford_route(current.id, best_sim)

            path.append(best_child)
            current = best_child
            depth += 1

        overall_confidence = min(confidence_factors) if confidence_factors else (1.0 if current else 0.0)
        final_signature = current if not current.is_semantic_umbrella else None

        return RoutingResult(
            signature=final_signature,
            path=path,
            confidence=overall_confidence,
            ucb1_gaps=ucb1_gaps,
            alternatives=alternatives if track_alternatives else [],
            best_similarity=best_similarity,
        )

    def route_through_hierarchy(
        self,
        operation_embedding: np.ndarray,
        min_similarity: Optional[float] = None,
        max_depth: int = None,
    ) -> tuple[Optional[StepSignature], list[StepSignature]]:
        """Route an operation embedding through the signature hierarchy.

        Thin wrapper around _route_core() for simple routing without alternatives.

        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.

        Args:
            operation_embedding: The operation embedding to route
            min_similarity: Minimum similarity threshold. None = use adaptive (recommended)
            max_depth: Maximum depth to traverse

        Returns:
            Tuple of (best_leaf_signature, path_taken)
        """
        result = self._route_core(
            operation_embedding,
            min_similarity=min_similarity,
            max_depth=max_depth,
            track_alternatives=False,
            use_ucb1=False,  # Use simple routing score
        )
        # Return leaf from path if result.signature is None but path has leaf
        if result.signature is None and result.path:
            last = result.path[-1]
            if not last.is_semantic_umbrella:
                return last, result.path
        return result.signature, result.path

    def route_with_confidence(
        self,
        operation_embedding: np.ndarray,
        min_similarity: Optional[float] = None,
        max_depth: int = None,
        top_k: int = 3,
        dag_step_type: Optional[str] = None,
        step_position: Optional[int] = None,
    ) -> RoutingResult:
        """Route with confidence scoring for MCTS multi-path exploration.

        Thin wrapper around _route_core() with UCB1 scoring, alternatives
        tracking, and epsilon-greedy exploration enabled.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.

        Confidence interpretation:
        - High confidence (>0.8): Clear winner, single path likely sufficient
        - Medium confidence (0.5-0.8): Consider exploring 1-2 alternatives
        - Low confidence (<0.5): High uncertainty, explore multiple paths

        Args:
            operation_embedding: The operation embedding to route
            min_similarity: Minimum similarity threshold. None = use adaptive (recommended)
            max_depth: Maximum depth to traverse
            top_k: Number of top alternatives to track at each level
            dag_step_type: Optional step type for step-node stats lookup
            step_position: Optional step position (1, 2, 3...) for position-aware routing

        Returns:
            RoutingResult with signature, path, confidence, and alternatives
        """
        return self._route_core(
            operation_embedding,
            min_similarity=min_similarity,
            max_depth=max_depth,
            track_alternatives=True,
            top_k=top_k,
            use_ucb1=True,
            epsilon_exploration=True,
            dag_step_type=dag_step_type,
            step_position=step_position,
        )

    # =========================================================================
    # Core: Find or Create
    # =========================================================================
    #
    # CONSOLIDATED SIGNATURE CREATION PATTERN (per CLAUDE.md "New Favorite Pattern")
    #
    # All signature creation flows through these layers:
    #
    #   PUBLIC APIS:
    #   ├── find_or_create()      → Routes first, creates if no match
    #   ├── find_or_create_async() → Async version of above
    #   └── create_signature()    → Skips routing, direct creation with parent_id
    #
    #   INTERNAL LAYERS:
    #   ├── propose_signature()   → SINGLE ENTRY for placement decisions
    #   │                           (cold start, Welford-based, dedup/merge)
    #   └── _create_signature_atomic() → SINGLE ENTRY for INSERT
    #
    # Use cases:
    # - find_or_create: Normal routing - find existing match OR create new
    # - create_signature: Explicit creation - knows parent, skips routing
    #
    # =========================================================================

    def find_or_create(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: Optional[float] = None,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
        exclude_ids: set = None,
    ) -> tuple[StepSignature, bool]:
        """Find a matching signature or create a new one.

        Lazy NL: New signatures have empty clarifying_questions and param_descriptions.
        These get filled in later as the system learns.

        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            min_similarity: Minimum cosine similarity for matching. None = use adaptive (recommended)
            parent_problem: The parent problem this step came from
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication
            origin_depth: Decomposition depth at which this step was created
            extracted_values: Dict of semantic param names -> values from planner
            parent_id: Explicit parent ID for new signatures (overrides routing)
            exclude_ids: Signature IDs to exclude from matching (prevent circular routing)

        Returns:
            Tuple of (signature, is_new) where is_new=True if newly created

        Warning:
            This method uses blocking time.sleep() for retries. Use find_or_create_async()
            in async contexts to avoid blocking the event loop.
        """
        # Resolve adaptive threshold if not specified
        if min_similarity is None:
            min_similarity = self._get_adaptive_similarity_threshold("match")

        # Warn if called from async context - use find_or_create_async instead
        try:
            asyncio.get_running_loop()
            logger.warning(
                "[db] find_or_create() called from async context - "
                "use find_or_create_async() to avoid blocking the event loop"
            )
        except RuntimeError:
            pass  # No running loop, sync usage is fine

        for attempt in range(DB_MAX_RETRIES):
            try:
                return self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, origin_depth,
                    extracted_values=extracted_values, dsl_hint=dsl_hint, parent_id=parent_id,
                    exclude_ids=exclude_ids
                )
            except sqlite3.OperationalError as e:
                if attempt < DB_MAX_RETRIES - 1:
                    # Exponential backoff with jitter to avoid thundering herd
                    delay = DB_BASE_RETRY_DELAY * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.5)
                    time.sleep(delay + jitter)
                    logger.debug(
                        "[db] Retry %d/%d after OperationalError: %s (delay=%.3fs)",
                        attempt + 1, DB_MAX_RETRIES, str(e)[:50], delay + jitter
                    )
                    continue
                raise

    async def find_or_create_async(
        self,
        step_text: str,
        embedding: Optional[np.ndarray] = None,
        min_similarity: Optional[float] = None,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
        embedder=None,  # Optional sync embedder for graph embedding
        exclude_ids: set = None,  # Signature IDs to exclude from matching (prevent circular routing)
    ) -> tuple[StepSignature, bool]:
        """Async version of find_or_create with non-blocking retry sleep.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.
        Routing uses graph_embedding exclusively. Text embedding is optional/legacy.

        Use this from async contexts to avoid blocking the event loop during
        database contention retries.

        Args:
            step_text: The step description text
            embedding: Optional text embedding (legacy, not used for routing)
            min_similarity: Minimum cosine similarity for matching. None = use adaptive (recommended)
            parent_problem: The parent problem this step came from
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for graph routing
            origin_depth: Decomposition depth at which this step was created
            extracted_values: Dict of semantic param names -> values from planner
            parent_id: Explicit parent ID for new signatures (overrides routing)
            embedder: Optional sync embedder for computing graph_embedding on new signatures
            exclude_ids: Signature IDs to exclude from matching (prevent circular routing during decomposition)

        Returns:
            Tuple of (signature, is_new) where is_new=True if newly created
        """
        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync

        # Resolve adaptive threshold if not specified
        if min_similarity is None:
            min_similarity = self._get_adaptive_similarity_threshold("match")

        sig = None
        is_new = False

        for attempt in range(DB_MAX_RETRIES):
            try:
                sig, is_new = self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, origin_depth,
                    extracted_values=extracted_values, dsl_hint=dsl_hint, parent_id=parent_id,
                    exclude_ids=exclude_ids
                )
                break
            except sqlite3.OperationalError as e:
                if attempt < DB_MAX_RETRIES - 1:
                    # Exponential backoff with jitter to avoid thundering herd
                    delay = DB_BASE_RETRY_DELAY * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.5)
                    await asyncio.sleep(delay + jitter)
                    logger.debug(
                        "[db] Retry %d/%d after OperationalError: %s (delay=%.3fs)",
                        attempt + 1, DB_MAX_RETRIES, str(e)[:50], delay + jitter
                    )
                    continue
                raise

        # Cold start: Embed computation graph for new signatures
        # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
        if is_new and sig and sig.computation_graph and embedder is not None:
            try:
                graph_emb = embed_computation_graph_sync(embedder, sig.computation_graph)
                if graph_emb:
                    self.update_graph_embedding(sig.id, graph_emb)
                    logger.debug(
                        "[db] Cold start: embedded graph for new sig %d: %s",
                        sig.id, sig.computation_graph[:30]
                    )
            except Exception as e:
                # Don't fail signature creation if embedding fails
                logger.warning("[db] Failed to embed graph for new sig %d: %s", sig.id, e)

        return sig, is_new

    def create_signature(
        self,
        step_text: str,
        embedding: np.ndarray,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
        embedder=None,
    ) -> StepSignature:
        """Create a new signature via consolidated propose_signature pathway.

        Per CLAUDE.md New Favorite Pattern: Routes through propose_signature()
        which handles cold start, Welford-based placement, and dedup.

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            parent_problem: The parent problem this step came from
            origin_depth: Decomposition depth for this signature
            extracted_values: Dict of semantic param names -> values from planner
            dsl_hint: Explicit operation hint from planner (+, -, *, /)
            parent_id: ID of parent signature. If None, defaults to root.
            embedder: Optional sync embedder for computing graph_embedding

        Returns:
            The created or matched StepSignature
        """
        # Per CLAUDE.md "New Favorite Pattern": Use consolidated helper
        step_graph_emb = self._get_graph_embedding_from_hint(dsl_hint)

        # Use consolidated propose_signature pathway
        # Handles: cold start, Welford decisions, dedup (MERGE), placement
        sig_id, placement = self.propose_signature(
            step_text=step_text,
            embedding=embedding,
            graph_embedding=step_graph_emb,
            proposed_parent_id=parent_id,
            dsl_hint=dsl_hint,
            extracted_values=extracted_values,
            origin_depth=origin_depth,
            problem_context=parent_problem,
        )

        sig = self.get_signature(sig_id)
        logger.info(
            "[db] create_signature via propose_signature (placement=%s): step='%s' type='%s'",
            placement, step_text[:40], sig.step_type
        )
        return sig

    def _find_or_create_atomic(
        self,
        step_text: str,
        embedding: Optional[np.ndarray],
        min_similarity: float,
        parent_problem: str,
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
        exclude_ids: set = None,
    ) -> tuple[StepSignature, bool]:
        """Internal atomic find-or-create with hierarchical routing.

        Routes through the signature hierarchy using graph_embedding (operational similarity).
        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.

        Flow:
        1. If DB is empty → create root signature
        2. Route from root → best matching child via graph_embedding → recurse until leaf
        3. If leaf matches above threshold → return existing
        4. If no match → create new child under where routing stopped (or explicit parent_id)

        Args:
            step_text: The step description text
            embedding: Optional text embedding (legacy, not used for routing - graph_embedding used instead)
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for graph routing
            parent_id: Explicit parent ID for new signatures (overrides routing)
            exclude_ids: Signature IDs to exclude from matching (prevent circular routing)
        """
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Check if DB is empty (need to create root)
                root_row = conn.execute(
                    "SELECT id FROM step_signatures WHERE is_root = 1 LIMIT 1"
                ).fetchone()

                if root_row is None:
                    # Empty DB - create root signature via propose_signature
                    # Per CLAUDE.md "New Favorite Pattern": ALL creation through propose_signature
                    step_graph_emb = self._get_graph_embedding_from_hint(dsl_hint)
                    sig_id, placement = self.propose_signature(
                        step_text=step_text,
                        embedding=embedding,
                        graph_embedding=step_graph_emb,
                        proposed_parent_id=None,  # No parent for root
                        dsl_hint=dsl_hint,
                        extracted_values=extracted_values,
                        origin_depth=origin_depth,
                        problem_context=parent_problem,
                        conn=conn,
                    )
                    sig = self.get_signature(sig_id)
                    conn.commit()
                    logger.info(
                        "[db] Created ROOT signature via propose_signature: step='%s' type='%s'",
                        step_text[:40], sig.step_type
                    )
                    return sig, True

                # Route through hierarchy to find best match
                # Pass dsl_hint for graph-based routing (per CLAUDE.md: route by what operations DO)
                # Pass exclude_ids to prevent circular routing (e.g., child matching back to parent during decomposition)
                best_match, parent_for_new, best_sim = self._route_hierarchical(
                    conn, min_similarity, dsl_hint=dsl_hint, exclude_ids=exclude_ids
                )

                # NOTE: Cold start logic moved to propose_signature() per CLAUDE.md New Favorite Pattern
                # propose_signature is now the SINGLE ENTRY POINT for all signature creation

                # Accept match from hierarchical routing
                # Simplified: no match_score, just use routing result
                # Global dedup check happens later if no routing match
                # Use Welford-guided thresholds (per mycelium-808f)
                similarity_ok = best_sim >= min_similarity

                if best_match is not None and similarity_ok:
                    # Check if matched signature's step_type is compatible with dsl_hint
                    # This prevents matching "Calculate total distance" (sum) to a product signature
                    if dsl_hint and not self._is_step_type_compatible(best_match.step_type, dsl_hint):
                        logger.debug(
                            "[db] Step type mismatch: sig='%s' has type '%s' but dsl_hint='%s' - creating new",
                            best_match.step_type, best_match.step_type, dsl_hint
                        )
                        # Treat as no match - fall through to create new signature
                        best_match = None

                if best_match is not None and similarity_ok:
                    # LEAF REJECTION: Check if leaf should reject this step
                    # Reasons to reject:
                    # 1. Low similarity (operational mismatch)
                    # 2. Multi-part step (needs decomposition)
                    # Per CLAUDE.md: leaves use graph_embedding (operational), not centroid (semantic)
                    if not best_match.is_semantic_umbrella:
                        # Per CLAUDE.md "New Favorite Pattern": use consolidated check_rejection
                        from mycelium.step_signatures.rejection_utils import check_rejection
                        from mycelium.data_layer.mcts import reject_dag_step

                        # Use graph_embedding similarity if available (operational identity)
                        # Otherwise fall back to text similarity
                        rejection_sim = best_sim  # Default to text similarity
                        has_graph = best_match.graph_embedding is not None

                        if has_graph and dsl_hint:
                            # Per CLAUDE.md "New Favorite Pattern": Use consolidated helper
                            try:
                                step_graph_emb = self._get_graph_embedding_from_hint(dsl_hint)
                                if step_graph_emb is not None:
                                    leaf_graph_emb = np.array(best_match.graph_embedding)
                                    rejection_sim = cosine_similarity(step_graph_emb, leaf_graph_emb)
                                    logger.debug(
                                        "[routing] Leaf '%s' graph_sim=%.3f text_sim=%.3f",
                                        best_match.step_type, rejection_sim, best_sim
                                    )
                            except Exception as e:
                                logger.debug("[db] Graph embedding comparison failed: %s", e)

                        # Check 1: Low similarity rejection using adaptive Welford threshold
                        # Per CLAUDE.md "New Favorite Pattern": consolidated to check_rejection
                        from mycelium.config import COLD_START_SIGNATURE_THRESHOLD
                        sig_count = self.count_signatures()
                        is_cold_start = sig_count < COLD_START_SIGNATURE_THRESHOLD

                        # Check rejection using unified utility (uses config for k, min_samples, default_threshold)
                        rejection_result = check_rejection(
                            signature=best_match,
                            similarity=rejection_sim,
                            is_cold_start=is_cold_start,
                            step_text=step_text,
                            problem_context=parent_problem,
                            conn=conn,
                        )

                        if rejection_result.rejected:
                            logger.info(
                                "[db] Leaf '%s' REJECTED step (sim=%.3f < adaptive=%.3f, n=%d, mean=%.3f): '%s'",
                                best_match.step_type, rejection_sim, rejection_result.threshold,
                                best_match.success_sim_count, best_match.success_sim_mean,
                                step_text[:40]
                            )
                            # Step queued for decomposition - don't create new signature
                            conn.commit()
                            return None, False

                        # Check 2: Multi-part step rejection (needs decomposition)
                        # Leaves only handle atomic operations - detect via embedding similarity
                        # Per CLAUDE.md: "prefer embedding similarity over keyword matching"
                        if best_match is not None and dsl_hint:
                            # First check: does dsl_hint map to a known atomic operation?
                            known_op = self._dsl_hint_to_graph(dsl_hint)
                            if known_op is not None:
                                # Known atomic operation (+, -, *, /, add, subtract, etc.) - accept
                                pass
                            else:
                                # Unknown hint - use embedding similarity to detect if atomic
                                from mycelium.embedding_cache import cached_embed
                                step_emb_for_atomic_check = cached_embed(dsl_hint)

                                if step_emb_for_atomic_check is not None:
                                    is_atomic, max_atomic_sim, gap, best_atomic_op = self._is_step_atomic(
                                        np.array(step_emb_for_atomic_check)
                                    )
                                    if not is_atomic and not is_cold_start:
                                        # Small gap = matches multiple ops (multi-part like "add then multiply")
                                        # Low sim = unknown complex operation
                                        # Skip during cold start - let vocabulary build first
                                        reason = "multi_part" if gap < ATOMIC_GAP_THRESHOLD else "unknown_complex"

                                        # Per CLAUDE.md "New Favorite Pattern": Use consolidated reject_dag_step()
                                        decision = reject_dag_step(
                                            signature_id=best_match.id,
                                            similarity=max_atomic_sim,  # Use atomic similarity for tracking
                                            step_text=step_text,
                                            problem_context=parent_problem,
                                            reason=reason,
                                            conn=conn,  # Pass connection to avoid lock contention
                                        )

                                        logger.info(
                                            "[db] Leaf '%s' REJECTED %s step (sim=%.3f, gap=%.3f, best=%s, hint='%s', rejections=%d): '%s'",
                                            best_match.step_type, reason, max_atomic_sim, gap,
                                            best_atomic_op, dsl_hint, decision.rejection_count, step_text[:50]
                                        )
                                        # reject_dag_step already queued for decomposition
                                        # Return None - step is queued, don't create signature
                                        conn.commit()
                                        return None, False
                                    elif not is_atomic and is_cold_start:
                                        logger.debug(
                                            "[db] Cold start: skipping multi-part rejection (sig_count=%d)",
                                            sig_count
                                        )

                if best_match is not None and similarity_ok:
                    # Log routing decision with similarity for tuning
                    is_leaf = not best_match.is_semantic_umbrella
                    if is_leaf and rejection_sim != best_sim:
                        # Show both similarities when graph_embedding was used
                        logger.info(
                            "[routing] Leaf '%s' ACCEPTED (text=%.3f graph=%.3f): '%s'",
                            best_match.step_type, best_sim, rejection_sim, step_text[:40]
                        )
                    else:
                        logger.info(
                            "[routing] %s '%s' ACCEPTED (sim=%.3f): '%s'",
                            "Leaf" if is_leaf else "Router",
                            best_match.step_type, best_sim, step_text[:40]
                        )

                    # Found a match - update centroid using shared helper
                    new_count = self._update_centroid_atomic(
                        conn, best_match.id, embedding, update_last_used=True
                    )

                    # Update match similarity stats (Welford) for adaptive thresholds
                    # Use graph_embedding similarity if available for more accurate tracking
                    if dsl_hint and best_match.graph_embedding is not None:
                        # Per CLAUDE.md "New Favorite Pattern": Use consolidated helper
                        step_graph_emb = self._get_graph_embedding_from_hint(dsl_hint)
                        if step_graph_emb is not None:
                            sig_graph_emb = np.array(best_match.graph_embedding)
                            graph_sim = cosine_similarity(step_graph_emb, sig_graph_emb)
                            update_similarity_stats(conn, 'match', graph_sim)

                    conn.commit()
                    logger.debug(
                        "[db] Matched signature (hierarchical): step='%s' sig='%s' sim=%.3f count=%d",
                        step_text[:40], best_match.step_type, best_sim, new_count or 0
                    )
                    return best_match, False

                # No match found in routing - apply global dedup check
                # This prevents creating duplicate signatures with near-identical graph_embeddings

                # Compute graph_embedding for this step
                # Per CLAUDE.md "New Favorite Pattern": Use consolidated helper
                step_graph_emb = self._get_graph_embedding_from_hint(dsl_hint)

                # Global dedup: search ALL signatures for high-similarity match
                if step_graph_emb is not None:
                    global_match, global_sim = find_global_best_match(conn, step_graph_emb)
                    thresholds = get_adaptive_thresholds(conn)

                    if global_match is not None and global_sim >= thresholds.dedup_threshold:
                        if not global_match.is_semantic_umbrella:
                            # DEDUP: Found identical LEAF signature, return it
                            logger.info(
                                "[db] DEDUP: Found identical leaf (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                global_sim, thresholds.dedup_threshold, step_text[:40], global_match.id, global_match.step_type
                            )
                            update_similarity_stats(conn, 'match', global_sim)
                            self._update_centroid_atomic(conn, global_match.id, embedding, update_last_used=True)
                            return self._finalize_routing_result(
                                conn, global_match, False, parent_for_new, global_match.id
                            )
                        else:
                            # Matched an UMBRELLA - search its children for matching leaf first
                            child_match, child_sim = find_best_child_match(
                                conn, global_match.id, step_graph_emb
                            )
                            if child_match is not None and child_sim >= thresholds.dedup_threshold:
                                # Found matching leaf under umbrella - DEDUP
                                logger.info(
                                    "[db] DEDUP: Found matching leaf under umbrella (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                    child_sim, thresholds.dedup_threshold, step_text[:40], child_match.id, child_match.step_type
                                )
                                update_similarity_stats(conn, 'match', child_sim)
                                self._update_centroid_atomic(conn, child_match.id, embedding, update_last_used=True)
                                return self._finalize_routing_result(
                                    conn, child_match, False, parent_for_new, global_match.id
                                )
                            else:
                                # No matching leaf found - create new leaf
                                # Per CLAUDE.md New Favorite Pattern: use propose_signature as single entry point
                                update_similarity_stats(conn, 'cluster', global_sim)

                                sig_id, placement = self.propose_signature(
                                    step_text=step_text,
                                    embedding=embedding,
                                    graph_embedding=step_graph_emb,
                                    proposed_parent_id=global_match.id,
                                    best_match_id=global_match.id,
                                    best_match_sim=global_sim,
                                    dsl_hint=dsl_hint,
                                    extracted_values=extracted_values,
                                    origin_depth=origin_depth,
                                    problem_context=parent_problem,
                                    conn=conn,
                                )
                                sig = self.get_signature(sig_id)
                                logger.info(
                                    "[db] Created signature via propose_signature (placement=%s): step='%s'",
                                    placement, step_text[:40]
                                )
                                return self._finalize_routing_result(
                                    conn, sig, True, parent_for_new, sig.id
                                )

                    elif global_match is not None and global_sim >= thresholds.cluster_threshold:
                        # CLUSTER: Similar signature exists - determine parent
                        if global_match.is_semantic_umbrella:
                            cluster_parent_id = global_match.id
                        else:
                            cursor = conn.execute(
                                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                                (global_match.id,)
                            )
                            row = cursor.fetchone()
                            cluster_parent_id = row[0] if row else None

                        update_similarity_stats(conn, 'cluster', global_sim)

                        # Per CLAUDE.md New Favorite Pattern: use propose_signature as single entry point
                        # propose_signature handles cold start + Welford-based placement decisions
                        sig_id, placement = self.propose_signature(
                            step_text=step_text,
                            embedding=embedding,
                            graph_embedding=step_graph_emb,
                            proposed_parent_id=cluster_parent_id,
                            best_match_id=global_match.id,
                            best_match_sim=global_sim,
                            dsl_hint=dsl_hint,
                            extracted_values=extracted_values,
                            origin_depth=origin_depth,
                            problem_context=parent_problem,
                            conn=conn,
                        )
                        sig = self.get_signature(sig_id)
                        logger.info(
                            "[db] Created signature via propose_signature (placement=%s): step='%s'",
                            placement, step_text[:40]
                        )
                        return self._finalize_routing_result(
                            conn, sig, True, parent_for_new, sig.id
                        )

                # No global match above thresholds - create new signature
                # Per CLAUDE.md New Favorite Pattern: use propose_signature as single entry point
                actual_parent_id = parent_id if parent_id is not None else (parent_for_new.id if parent_for_new else None)

                sig_id, placement = self.propose_signature(
                    step_text=step_text,
                    embedding=embedding,
                    graph_embedding=step_graph_emb,
                    proposed_parent_id=actual_parent_id,
                    best_match_id=None,
                    best_match_sim=None,
                    dsl_hint=dsl_hint,
                    extracted_values=extracted_values,
                    origin_depth=origin_depth,
                    problem_context=parent_problem,
                    conn=conn,
                )
                sig = self.get_signature(sig_id)
                logger.info(
                    "[db] Created NEW signature via propose_signature (placement=%s): step='%s'",
                    placement, step_text[:40]
                )
                return self._finalize_routing_result(
                    conn, sig, True, parent_for_new, sig.id
                )

            except Exception:
                conn.rollback()
                raise

    def _route_hierarchical(
        self,
        conn,
        min_similarity: float,
        dsl_hint: str = None,
        exclude_ids: set = None,
    ) -> tuple[Optional[StepSignature], Optional[StepSignature], float]:
        """Route through hierarchy using graph_embedding (operational similarity).

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        - Routers: use graph_centroid (avg of descendants' graph_embeddings)
        - Leaves: use graph_embedding (fixed operational identity)

        Graph-only routing: No text centroid fallback. If no graph_embedding
        is available, similarity is 0.0 (cold start triggers new signature creation).

        Uses UCB1 scoring to balance exploitation (high-similarity, high-success)
        with exploration (under-visited signatures that might be better).

        Args:
            min_similarity: Minimum similarity threshold
            dsl_hint: Operation hint from planner (+, -, *, /) for graph routing
            exclude_ids: Signature IDs to exclude from matching (prevent circular routing)

        Returns:
            (best_match, parent_for_new, best_similarity)
            - best_match: Leaf signature if found above threshold
            - parent_for_new: Umbrella where routing stopped (for creating new child)
            - best_similarity: Similarity of best_match
        """
        # Normalize exclude_ids
        exclude_ids = exclude_ids or set()
        from mycelium.config import (
            UMBRELLA_MAX_DEPTH, SCAFFOLD_ENABLED, MIN_SIGNATURE_DEPTH,
            SCAFFOLD_FORK_THRESHOLD, SCAFFOLD_FORK_THRESHOLD_COLD_START,
            SCAFFOLD_FORK_RAMP_SIGNATURES, MIN_FORK_DEPTH
        )

        # Validate max depth to prevent unbounded recursion
        max_depth = max(1, min(int(UMBRELLA_MAX_DEPTH or 10), 100))  # Hard cap at 100

        # Compute cold-start aware fork threshold
        # During cold start, use HIGHER threshold (more forking / big bang)
        # As system matures, lower to standard threshold (consolidation)
        sig_count = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()[0]
        if sig_count >= SCAFFOLD_FORK_RAMP_SIGNATURES:
            fork_threshold = SCAFFOLD_FORK_THRESHOLD  # Mature
        else:
            # Linear ramp from cold_start to mature
            progress = sig_count / SCAFFOLD_FORK_RAMP_SIGNATURES
            fork_threshold = SCAFFOLD_FORK_THRESHOLD_COLD_START - (
                progress * (SCAFFOLD_FORK_THRESHOLD_COLD_START - SCAFFOLD_FORK_THRESHOLD)
            )
        logger.debug("[db] Fork threshold: %.3f (sigs=%d, cold_start=%.2f, mature=%.2f)",
                     fork_threshold, sig_count, SCAFFOLD_FORK_THRESHOLD_COLD_START, SCAFFOLD_FORK_THRESHOLD)

        # Per CLAUDE.md "New Favorite Pattern": Use consolidated helper
        step_graph_embedding = self._get_graph_embedding_from_hint(dsl_hint)
        if step_graph_embedding is not None:
            logger.debug("[db] Computed step_graph_embedding from dsl_hint=%s", dsl_hint)

        # Start at root
        root_row = conn.execute(
            "SELECT * FROM step_signatures WHERE is_root = 1 LIMIT 1"
        ).fetchone()

        if root_row is None:
            return None, None, 0.0

        current = self._row_to_signature(root_row)
        parent_for_new = current  # Track where to create new child
        depth = 0

        while depth < max_depth:
            # Check similarity to current node using graph_embedding (operational)
            # No text/centroid fallback - graph-only routing per CLAUDE.md

            # Get graph_embedding for current node (leaves have fixed, routers have centroid of children)
            current_graph_emb = current.graph_embedding
            if current_graph_emb is not None and not isinstance(current_graph_emb, np.ndarray):
                current_graph_emb = np.array(current_graph_emb)

            # Determine which embedding to compare against (graph-only routing)
            if step_graph_embedding is not None and current_graph_emb is not None:
                # Graph_embedding routing (operational similarity)
                sim = cosine_similarity(step_graph_embedding, current_graph_emb)
                used_graph = True
            else:
                # No graph_embedding available - cold start, low similarity triggers inline decomposition
                sim = 0.0
                used_graph = False
                if current_graph_emb is None:
                    logger.debug("[db] Cold start: node %d has no graph_embedding", current.id)

            # If current is a leaf, return it (always return leaf regardless of threshold)
            # UNLESS it's in exclude_ids (e.g., parent being decomposed - prevent circular matching)
            if not current.is_semantic_umbrella:
                if current.id in exclude_ids:
                    logger.debug("[db] Leaf %d excluded from matching (in exclude_ids)", current.id)
                    return None, parent_for_new, 0.0
                if used_graph:
                    logger.debug("[db] Leaf %d routed by graph_embedding (sim=%.3f)", current.id, sim)
                return current, parent_for_new, sim

            # Get children of current umbrella (exclude archived)
            cursor = conn.execute(
                """SELECT s.* FROM signature_relationships r
                   JOIN step_signatures s ON r.child_id = s.id
                   WHERE r.parent_id = ?
                     AND s.is_archived = 0
                   ORDER BY r.routing_order ASC""",
                (current.id,)
            )
            children = [self._row_to_signature(row) for row in cursor.fetchall()]

            if not children:
                # Umbrella with no children - no leaf found, caller should create one
                # Return None for best_match (not the umbrella!), keep current as parent_for_new
                # Bug fix: was returning umbrella as best_match, should return None
                return None, current, sim

            # MCTS UCB1 Selection: balance exploitation vs exploration
            # parent_uses = current node's uses (N in UCB1 formula)
            parent_uses = current.uses or 1

            best_child = None
            best_child_sim = 0.0
            best_child_score = 0.0

            # Separate children with graph_embeddings from those without
            # Per CLAUDE.md: route by what operations DO (graph_embedding), not what they SOUND LIKE (centroid)
            children_with_embeddings = []
            null_embedding_children = []

            for child in children:
                # Skip excluded signatures (prevent circular routing during decomposition)
                if child.id in exclude_ids:
                    logger.debug("[db] Skipping child %d (in exclude_ids) during routing", child.id)
                    continue

                # Prefer graph_embedding for routing
                child_graph_emb = child.graph_embedding
                if child_graph_emb is not None and not isinstance(child_graph_emb, np.ndarray):
                    child_graph_emb = np.array(child_graph_emb)

                if step_graph_embedding is not None and child_graph_emb is not None:
                    # Route by graph_embedding (operational similarity)
                    child_sim = cosine_similarity(step_graph_embedding, child_graph_emb)
                    children_with_embeddings.append((child, child_sim, True))  # True = used graph
                else:
                    # No graph_embedding - treat as cold start placeholder
                    null_embedding_children.append(child)

            # Try children with embeddings (standard UCB1 selection)
            # Use Welford-guided thresholds (per mycelium-808f)
            for child, child_sim, used_graph in children_with_embeddings:
                if child_sim >= min_similarity:
                    score = compute_ucb1_score(
                        child_sim,
                        child.uses,
                        child.successes,
                        parent_uses,
                        child.last_used_at
                    )
                    if score > best_child_score:
                        best_child = child
                        best_child_sim = child_sim
                        best_child_score = score
                        if used_graph:
                            logger.debug("[db] Child %d selected via graph_embedding (sim=%.3f, score=%.3f)",
                                       child.id, child_sim, score)

            # SCAFFOLD: If no match found but we have null-embedding placeholders,
            # and we haven't reached MIN_SIGNATURE_DEPTH yet, route through one
            if best_child is None and null_embedding_children and SCAFFOLD_ENABLED:
                if depth < MIN_SIGNATURE_DEPTH - 1:
                    # Pick least-used placeholder (exploration) or random if all equal
                    null_embedding_children.sort(key=lambda c: c.uses or 0)
                    placeholder = null_embedding_children[0]

                    # Initialize placeholder's centroid with this embedding
                    # Also set graph_embedding if available
                    logger.info(
                        "[db] Initializing scaffold placeholder: id=%d depth=%d",
                        placeholder.id, depth + 1
                    )
                    self._update_centroid_atomic(conn, placeholder.id, embedding, update_last_used=False)
                    if step_graph_embedding is not None:
                        self.update_graph_embedding(placeholder.id, step_graph_embedding.tolist())

                    # Update our local object to reflect the change
                    placeholder.centroid = embedding
                    placeholder.graph_embedding = step_graph_embedding
                    best_child = placeholder
                    best_child_sim = 1.0  # Perfect match since we just set it

            if best_child is None:
                # No child matches above threshold
                # SCAFFOLD: If we haven't reached MIN_SIGNATURE_DEPTH, keep routing
                if SCAFFOLD_ENABLED and depth < MIN_SIGNATURE_DEPTH - 1:
                    # Find best below-threshold child
                    best_below = None
                    best_below_sim = 0.0
                    best_below_score = -float('inf')

                    for child, child_sim, _used_graph in children_with_embeddings:
                        score = compute_ucb1_score(
                            child_sim, child.uses, child.successes,
                            parent_uses, child.last_used_at
                        )
                        if score > best_below_score:
                            best_below = child
                            best_below_sim = child_sim
                            best_below_score = score

                    # DYNAMIC FORKING (BIG BANG): Use smooth probability function
                    # Per CLAUDE.md: "smooth and continuous", "no hard thresholds"
                    # Factors: depth, maturity, similarity gap, hysteresis
                    # Note: depth + 1 because we're creating a child at the next level

                    # Check for hysteresis: does this level have existing forks?
                    # (more than 1 child with embedding at this level = already forked)
                    has_existing_forks = len(children_with_embeddings) > 1

                    # Use smooth probabilistic forking decision
                    fork_decision = should_fork_at_depth(
                        depth=depth + 1,  # We're creating at next level
                        sig_count=sig_count,
                        best_similarity=best_below_sim if best_below else 0.0,
                        fork_threshold=fork_threshold,
                        has_existing_forks_at_level=has_existing_forks,
                    )

                    if fork_decision:
                        # Create new branch (fork) for this divergent problem type
                        new_branch = self._create_scaffold_branch(
                            conn, current.id, embedding, depth + 1
                        )
                        if new_branch:
                            # Structured fork event log for analysis
                            logger.info(
                                "[FORK_CREATED] depth=%d new_sig_id=%d parent_id=%d "
                                "best_sim=%.3f threshold=%.3f gap=%.3f "
                                "existing_children=%d sig_count=%d",
                                depth + 1, new_branch.id, current.id,
                                best_below_sim, fork_threshold,
                                fork_threshold - best_below_sim,
                                len(children_with_embeddings), sig_count
                            )
                            parent_for_new = current
                            current = new_branch
                            depth += 1
                            continue

                    # If no children with embeddings, use a placeholder
                    if best_below is None and null_embedding_children:
                        null_embedding_children.sort(key=lambda c: c.uses or 0)
                        placeholder = null_embedding_children[0]
                        # Initialize centroid and graph_embedding
                        self._update_centroid_atomic(conn, placeholder.id, embedding, update_last_used=False)
                        placeholder.centroid = embedding
                        if step_graph_embedding is not None:
                            self.update_graph_embedding(placeholder.id, step_graph_embedding.tolist())
                            placeholder.graph_embedding = step_graph_embedding
                        best_below = placeholder
                        best_below_sim = 1.0
                        logger.info(
                            "[db] Initializing scaffold placeholder: id=%d depth=%d",
                            placeholder.id, depth + 1
                        )

                    if best_below:
                        # Continue routing deeper through existing path
                        parent_for_new = current
                        current = best_below
                        depth += 1
                        continue

                # Either scaffold disabled or we've reached MIN_SIGNATURE_DEPTH
                # Return current as where we'd add new child
                parent_for_new = current
                # Return best child below threshold as "best effort" (or None)
                best_below = None
                best_below_sim = 0.0
                best_below_score = 0.0
                for child, child_sim, _used_graph in children_with_embeddings:
                    score = compute_ucb1_score(
                        child_sim, child.uses, child.successes,
                        parent_uses, child.last_used_at
                    )
                    if score > best_below_score:
                        best_below = child
                        best_below_sim = child_sim
                        best_below_score = score
                return best_below, parent_for_new, best_below_sim

            # Move to best child (selected by UCB1)
            parent_for_new = current  # Current umbrella is parent if we create here
            current = best_child
            depth += 1

        # Hit max depth
        # Capture centroid once to avoid TOCTOU race condition
        max_depth_centroid = current.centroid
        sim = cosine_similarity(embedding, max_depth_centroid) if max_depth_centroid is not None else 0.0
        return current, parent_for_new, sim

    def _create_signature_atomic(
        self,
        conn,
        step_text: str,
        embedding: Optional[np.ndarray],
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        parent_id: int = None,
        dsl_hint: str = None,
        graph_embedding: Optional[np.ndarray] = None,
        is_umbrella: bool = False,
        signature_id_override: str = None,
        depth_override: int = None,
        skip_example: bool = False,
        skip_parent_relationship: bool = False,
    ) -> StepSignature:
        """Create a new signature within an existing transaction.

        This is the SINGLE pathway for creating signatures in the database.
        All signature creation flows through this function.

        For execution signatures (leaves): Auto-assigns DSL based on step_type,
        description, extracted_values, and dsl_hint.

        For router signatures (umbrellas): Skips DSL generation, sets dsl_type="router".

        Hierarchical routing:
        - First signature becomes THE root (is_root=1, is_semantic_umbrella=1)
        - Subsequent signatures become children of specified parent (or root if not specified)

        Args:
            step_text: The step description text
            embedding: Optional text embedding (legacy, for centroid initialization)
            parent_id: ID of parent signature. If None, defaults to root.
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication.
            graph_embedding: Pre-computed graph embedding for dedup. If not provided, computed from computation_graph.
            is_umbrella: If True, create a router signature (no DSL, dsl_type="router").
            signature_id_override: Custom signature_id (e.g., "branch_L2_abc123"). If None, generates UUID.
            depth_override: Explicit depth. If None, computed from parent.
            skip_example: If True, don't insert into step_examples table.
            skip_parent_relationship: If True, don't add as child of parent. Useful for upward restructuring.
        """
        sig_id = signature_id_override or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Check if this will be the root (first signature in DB)
        root_row = conn.execute(
            "SELECT id FROM step_signatures WHERE is_root = 1 LIMIT 1"
        ).fetchone()
        is_first_signature = root_row is None

        # Determine actual parent: use specified parent_id, or fall back to root
        if parent_id is not None:
            actual_parent_id = parent_id
        elif root_row is not None:
            actual_parent_id = root_row[0]
        else:
            actual_parent_id = None  # Creating the root

        step_type = self._infer_step_type(step_text, dsl_hint=dsl_hint)
        # Text centroid is legacy - graph_embedding is used for routing
        centroid_packed = pack_embedding(embedding) if embedding is not None else None
        centroid_bucket = compute_centroid_bucket(embedding) if embedding is not None else None

        # Umbrella signatures route, they don't execute - no DSL
        if is_umbrella:
            dsl_script = None
            dsl_type = "router"
        else:
            # Auto-assign DSL based on step_type, description, planner's extracted_values, and dsl_hint
            # dsl_hint enables bidirectional LLM-signature communication
            dsl_script, dsl_type = infer_dsl_for_signature(
                step_type, step_text, extracted_values=extracted_values, dsl_hint=dsl_hint,
                embedder=self.embedder
            )

        # Auto-generate NL interface from extracted_values if we created a math DSL
        # The param names ARE the semantic descriptions - use them!
        # Skip for umbrellas (routers don't need NL interface)
        clarifying_questions = []
        param_descriptions = {}
        if not is_umbrella and extracted_values and dsl_type == "math":
            for param_name, value in extracted_values.items():
                # Convert param_name to readable question/description
                readable = param_name.replace("_", " ")
                clarifying_questions.append(f"What is the {readable}?")
                param_descriptions[param_name] = f"The {readable} value"
            logger.debug(
                "[db] Auto-generated NL interface from extracted_values: %d questions",
                len(clarifying_questions)
            )

        # Initialize embedding_sum = embedding, embedding_count = 1
        embedding_sum_packed = centroid_packed  # Same as centroid initially

        # Serialize NL interface
        clarifying_json = json.dumps(clarifying_questions)
        params_json = json.dumps(param_descriptions)

        # Extract computation graph from DSL (per CLAUDE.md: route by what operations DO)
        computation_graph = extract_computation_graph(dsl_script) if dsl_script else None
        if computation_graph:
            logger.debug("[db] Extracted computation graph: %s", computation_graph)

        # Compute graph_embedding if not provided but we have computation_graph
        # This is the SINGLE pathway for graph_embedding assignment
        graph_emb_to_store = graph_embedding
        if graph_emb_to_store is None and computation_graph:
            try:
                from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                emb_list = embed_computation_graph_sync(self.embedder, computation_graph)
                if emb_list:
                    graph_emb_to_store = np.array(emb_list)
                    logger.debug("[db] Computed graph_embedding from computation_graph")
            except Exception as e:
                logger.debug("[db] Failed to compute graph_embedding: %s", e)

        # Pack graph_embedding for storage
        graph_emb_packed = json.dumps(graph_emb_to_store.tolist()) if graph_emb_to_store is not None else None

        # Set flags based on whether this is the root
        is_root_flag = 1 if is_first_signature else 0
        # Root is always an umbrella (routes to children)
        # For non-root, use the is_umbrella parameter
        is_umbrella_flag = 1 if (is_first_signature or is_umbrella) else 0

        # Calculate depth based on parent (or use override if provided)
        if depth_override is not None:
            actual_depth = depth_override
        elif is_first_signature:
            actual_depth = 0  # Root is depth 0
        elif actual_parent_id is not None:
            # Get parent's depth
            parent_row = conn.execute(
                "SELECT depth FROM step_signatures WHERE id = ?",
                (actual_parent_id,)
            ).fetchone()
            parent_depth = parent_row["depth"] if parent_row and parent_row["depth"] else 0
            actual_depth = parent_depth + 1
        else:
            actual_depth = 1  # Fallback

        try:
            cursor = conn.execute(
                """INSERT INTO step_signatures
                   (signature_id, centroid, centroid_bucket, embedding_sum, embedding_count, step_type, description,
                    dsl_script, dsl_type, clarifying_questions, param_descriptions, depth,
                    is_root, is_semantic_umbrella, computation_graph, graph_embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, centroid_bucket, embedding_sum_packed, 1, step_type, step_text,
                 dsl_script, dsl_type, clarifying_json, params_json, actual_depth,
                 is_root_flag, is_umbrella_flag, computation_graph, graph_emb_packed, now),
            )
            row_id = cursor.lastrowid

            # Defensive check: ensure we got a valid row ID
            if not row_id:
                # Fallback: query by signature_id which is unique
                row = conn.execute(
                    "SELECT id FROM step_signatures WHERE signature_id = ?",
                    (sig_id,)
                ).fetchone()
                if row:
                    row_id = row["id"]
                else:
                    raise RuntimeError(f"Failed to get row ID after INSERT for signature_id={sig_id}")

            # If not root, add as child of the appropriate parent
            # skip_parent_relationship is used for upward restructuring where we add child manually
            if not is_first_signature and actual_parent_id is not None and not skip_parent_relationship:
                # Prevent self-references (circular dependency bug)
                if actual_parent_id == row_id:
                    logger.warning(
                        "[db] Rejecting self-reference in signature creation: parent_id=%d == child_id=%d",
                        actual_parent_id, row_id
                    )
                else:
                    # Add parent-child relationship
                    conn.execute(
                        """INSERT OR IGNORE INTO signature_relationships
                           (parent_id, child_id, condition, routing_order, created_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (actual_parent_id, row_id, step_type, 0, now),
                    )
                    # Use single pathway for umbrella promotion
                    self._promote_to_umbrella_internal(conn, actual_parent_id)
                    # Invalidate parent's children cache since we added a new child
                    invalidate_children_cache(actual_parent_id)
                    logger.debug(
                        "[db] Added as child: parent_id=%d (depth=%d) -> child_id=%d (depth=%d) (condition='%s')",
                        actual_parent_id, actual_depth - 1, row_id, actual_depth, step_type
                    )
        except sqlite3.IntegrityError as e:
            # Centroid bucket collision - find existing signature by bucket hash
            if "centroid_bucket" in str(e):
                row = conn.execute(
                    "SELECT * FROM step_signatures WHERE centroid_bucket = ?",
                    (centroid_bucket,)
                ).fetchone()
                if row:
                    logger.debug(
                        "[db] Centroid bucket collision, reusing existing signature: id=%d type='%s'",
                        row["id"], row["step_type"]
                    )
                    return self._row_to_signature(row)
            # Unknown integrity error - re-raise
            raise

        # Also add as first example (skip for umbrella/router signatures)
        if not skip_example and not is_umbrella:
            conn.execute(
                """INSERT INTO step_examples
                   (signature_id, step_text, embedding, parent_problem, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (row_id, step_text, pack_embedding(embedding), parent_problem, now),
            )

        if is_first_signature:
            logger.info("[db] Created ROOT signature: type=%s (first in DB)", step_type)
        elif is_umbrella:
            logger.info("[db] Created UMBRELLA signature: id=%d, type=%s, depth=%d", row_id, step_type, actual_depth)
        else:
            logger.debug("[db] Auto-assigned DSL type=%s for step_type=%s", dsl_type, step_type)

        # Invalidate centroid matrix cache since we added a new signature
        self.invalidate_centroid_matrix()

        return StepSignature(
            id=row_id,
            signature_id=sig_id,
            centroid=embedding,
            step_type=step_type,
            description=step_text,
            clarifying_questions=clarifying_questions,
            param_descriptions=param_descriptions,
            dsl_script=dsl_script,
            dsl_type=dsl_type,
            computation_graph=computation_graph,
            graph_embedding=graph_emb_to_store.tolist() if graph_emb_to_store is not None else None,
            examples=[],
            uses=0,
            successes=0,
            depth=actual_depth,
            is_root=is_first_signature,
            is_semantic_umbrella=bool(is_umbrella_flag),
            created_at=now,
        )

    # =========================================================================
    # Lookup
    # =========================================================================

    def get_signature(self, signature_id: int) -> Optional[StepSignature]:
        """Get a signature by ID.

        Uses LRU cache with TTL to skip DB for hot signatures.
        """
        # Check cache first
        cached = get_cached_signature(signature_id)
        if cached is not None:
            return cached

        # Cache miss - fetch from DB
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM step_signatures WHERE id = ?",
                (signature_id,)
            ).fetchone()
            if row:
                sig = self._row_to_signature(row)
                cache_signature(signature_id, sig)
                return sig
            return None

    def _ensure_centroid_matrix(self):
        """Build or refresh the cached centroid matrix for fast similarity search.

        Tries to load from disk cache first for fast startup.
        Falls back to building from DB if cache is missing or stale.
        """
        if self._centroid_matrix is not None:
            return  # Already loaded

        # Try loading from disk cache first (fast path)
        if self._load_centroid_matrix_from_cache():
            return  # Loaded successfully, rows will be fetched lazily

        # Slow path: build from DB
        # Select only columns needed for matrix building + from_row_fast()
        # Skips: centroid_bucket, embedding_sum, clarifying_questions, examples, last_rewrite_at
        # IMPORTANT: Exclude archived signatures from routing
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT id, signature_id, centroid, embedding_count, step_type,
                       description, param_descriptions, dsl_script, dsl_type,
                       uses, successes, is_semantic_umbrella, is_root, depth,
                       created_at, last_used_at
                FROM step_signatures
                WHERE is_archived = 0
            """)
            rows = cursor.fetchall()

        valid_rows = []
        centroids = []
        sig_ids = []
        for row in rows:
            if not row["centroid"]:
                continue
            centroid = get_cached_centroid(row["id"], row["centroid"])
            if centroid is not None:
                valid_rows.append(row)
                centroids.append(centroid)
                sig_ids.append(row["id"])

        if centroids:
            matrix = np.array(centroids, dtype=np.float32)
            # Pre-normalize for faster similarity (just dot product needed)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._centroid_matrix = matrix / norms  # Normalized!
            self._centroid_sig_ids = sig_ids
            self._centroid_rows = valid_rows
            logger.debug("[db] Built pre-normalized centroid matrix: %d signatures", len(sig_ids))
            # Save to disk cache for next startup
            self._save_centroid_matrix_to_cache()
        else:
            self._centroid_matrix = np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)
            self._centroid_sig_ids = []
            self._centroid_rows = []

    def invalidate_centroid_matrix(self):
        """Invalidate cached centroid matrix (call when signatures change)."""
        self._centroid_matrix = None
        self._centroid_sig_ids = None
        self._centroid_rows = None
        # Also delete disk cache so it rebuilds on next load
        self._delete_centroid_cache()

    def invalidate_root_cache(self):
        """Invalidate cached root signature (call when DB is cleared)."""
        self._cached_root = None

    # =========================================================================
    # Consolidated Cache Invalidation Helpers
    # =========================================================================
    # These delegate to CacheManager for coordinated invalidation.
    # See issue mycelium-wrvq for consolidation details.

    def _invalidate_on_embedding_change(self, signature_id: int):
        """Invalidate when centroid or graph_embedding changes.

        Delegates to CacheManager.on_embedding_change().
        """
        get_cache_manager().on_embedding_change(signature_id)

    def _invalidate_on_relationship_change(self, parent_id: int, child_id: int):
        """Invalidate when parent-child relationship is added or removed.

        Delegates to CacheManager.on_relationship_change().
        """
        get_cache_manager().on_relationship_change(parent_id, child_id)

    def _invalidate_on_dsl_change(self, signature_id: int):
        """Invalidate when DSL script or signature metadata changes.

        Delegates to CacheManager.on_dsl_change().
        """
        get_cache_manager().on_dsl_change(signature_id)

    def find_similar(
        self,
        embedding: np.ndarray,
        threshold: float = NEW_CHILD_SIMILARITY_THRESHOLD,
        limit: int = 10,
    ) -> list[tuple[StepSignature, float]]:
        """Find signatures similar to the given embedding.

        Args:
            embedding: Query embedding vector
            threshold: Minimum cosine similarity
            limit: Maximum number of results

        Returns:
            List of (signature, similarity) tuples, sorted by similarity descending
        """
        self._ensure_centroid_matrix()

        if len(self._centroid_matrix) == 0:
            return []

        # Batch cosine similarity (~50x faster than loop, pre-normalized matrix)
        scores = batch_cosine_similarity(embedding, self._centroid_matrix, matrix_normalized=True)

        # Get indices above threshold, sorted by score (descending)
        above_threshold = np.where(scores >= threshold)[0]
        if len(above_threshold) == 0:
            return []

        # Sort by score and take top limit (avoid converting all matches to Signature objects)
        sorted_indices = above_threshold[np.argsort(scores[above_threshold])[::-1]][:limit]

        # Only convert the top results to StepSignature objects (major speedup!)
        # Use fast parsing - skip JSON fields we don't need for similarity results
        results = []

        if self._centroid_rows is not None:
            # Fast path: rows cached in memory
            for i in sorted_indices:
                sig = self._row_to_signature_fast(self._centroid_rows[i])
                results.append((sig, float(scores[i])))
        else:
            # Lazy path: rows not cached (loaded from disk), fetch by ID
            # Only fetch the rows we need (typically 10 or fewer)
            ids_to_fetch = [self._centroid_sig_ids[i] for i in sorted_indices]
            if ids_to_fetch:
                placeholders = ",".join("?" * len(ids_to_fetch))
                with self._connection() as conn:
                    cursor = conn.execute(
                        f"""SELECT id, signature_id, centroid, embedding_count, step_type,
                                   description, param_descriptions, dsl_script, dsl_type,
                                   uses, successes, is_semantic_umbrella, is_root, depth,
                                   created_at, last_used_at
                            FROM step_signatures WHERE id IN ({placeholders})""",
                        ids_to_fetch,
                    )
                    rows_by_id = {row["id"]: row for row in cursor.fetchall()}

                for i in sorted_indices:
                    sig_id = self._centroid_sig_ids[i]
                    if sig_id in rows_by_id:
                        sig = self._row_to_signature_fast(rows_by_id[sig_id])
                        results.append((sig, float(scores[i])))

        return results

    def count_signatures(self) -> int:
        """Get total number of signatures."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
            return row[0] if row else 0

    def get_structure_stats(self) -> dict:
        """Get structural statistics about the signature database.

        Returns:
            Dict with:
            - total: Total signature count
            - by_type: Count by dsl_type (router, math, decompose, etc.)
            - by_role: Count by role (router vs leaf)
            - depth_histogram: Count of signatures at each depth
            - umbrella_count: Number of umbrella signatures
            - orphan_umbrellas: Umbrellas with no children
            - avg_depth: Average depth of all signatures
            - max_depth: Maximum depth in the tree
            - success_rate: Overall success rate
        """
        with self._connection() as conn:
            stats = {}

            # Total count
            row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
            stats["total"] = row[0] if row else 0

            if stats["total"] == 0:
                return {
                    "total": 0,
                    "by_type": {},
                    "by_role": {"router": 0, "leaf": 0},
                    "depth_histogram": {},
                    "umbrella_count": 0,
                    "orphan_umbrellas": 0,
                    "avg_depth": 0.0,
                    "max_depth": 0,
                    "success_rate": 0.0,
                }

            # Count by dsl_type
            rows = conn.execute(
                "SELECT dsl_type, COUNT(*) FROM step_signatures GROUP BY dsl_type"
            ).fetchall()
            stats["by_type"] = {row[0] or "unknown": row[1] for row in rows}

            # Count by role: router (is_semantic_umbrella=1 OR dsl_type='router') vs leaf
            router_count = conn.execute(
                """SELECT COUNT(*) FROM step_signatures
                   WHERE is_semantic_umbrella = 1 OR dsl_type = 'router'"""
            ).fetchone()[0]
            stats["by_role"] = {
                "router": router_count,
                "leaf": stats["total"] - router_count,
            }

            # Depth histogram
            rows = conn.execute(
                """SELECT COALESCE(depth, 0) as d, COUNT(*)
                   FROM step_signatures
                   GROUP BY d
                   ORDER BY d"""
            ).fetchall()
            stats["depth_histogram"] = {row[0]: row[1] for row in rows}

            # Umbrella stats
            umbrella_count = conn.execute(
                "SELECT COUNT(*) FROM step_signatures WHERE is_semantic_umbrella = 1"
            ).fetchone()[0]
            stats["umbrella_count"] = umbrella_count

            # Orphan umbrellas (umbrellas with no children)
            try:
                orphan_count = conn.execute(
                    """SELECT COUNT(*) FROM step_signatures s
                       WHERE s.is_semantic_umbrella = 1
                       AND NOT EXISTS (
                           SELECT 1 FROM signature_children c WHERE c.parent_id = s.id
                       )"""
                ).fetchone()[0]
                stats["orphan_umbrellas"] = orphan_count
            except Exception:
                # Table may not exist in older DBs
                stats["orphan_umbrellas"] = 0

            # Depth stats
            depth_row = conn.execute(
                """SELECT AVG(COALESCE(depth, 0)), MAX(COALESCE(depth, 0))
                   FROM step_signatures"""
            ).fetchone()
            stats["avg_depth"] = round(depth_row[0] or 0.0, 2)
            stats["max_depth"] = depth_row[1] or 0

            # Overall success rate
            totals = conn.execute(
                "SELECT SUM(uses), SUM(successes) FROM step_signatures"
            ).fetchone()
            total_uses = totals[0] or 0
            total_successes = totals[1] or 0
            stats["success_rate"] = round(
                total_successes / total_uses if total_uses > 0 else 0.0, 3
            )

            # Success rate by type
            rows = conn.execute(
                """SELECT dsl_type, SUM(uses), SUM(successes)
                   FROM step_signatures
                   GROUP BY dsl_type"""
            ).fetchall()
            stats["success_by_type"] = {}
            for row in rows:
                dsl_type = row[0] or "unknown"
                uses = row[1] or 0
                successes = row[2] or 0
                if uses > 0:
                    stats["success_by_type"][dsl_type] = round(successes / uses, 3)

            return stats

    def get_total_signature_uses(self) -> int:
        """Get total uses across all signatures."""
        with self._connection() as conn:
            row = conn.execute("SELECT SUM(uses) FROM step_signatures").fetchone()
            return row[0] if row and row[0] else 0

    def reset_signature_stats(self, signature_id: int) -> None:
        """Reset uses and successes for a signature after DSL rewrite."""
        with self._connection() as conn:
            self._update_signature_fields(
                conn, signature_id,
                log_reason="reset_stats",
                uses=0, successes=0,
            )

    def mark_signature_rewritten(self, signature_id: int) -> None:
        """Mark a signature as recently rewritten (for cooldown tracking)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            # Store in last_used_at for now (could add dedicated column later)
            self._update_signature_fields(
                conn, signature_id,
                log_reason="rewritten",
                last_used_at=now,
            )

    # =========================================================================
    # Centroid Management (Running Average Embeddings)
    # =========================================================================

    def _update_centroid_atomic(
        self,
        conn,
        signature_id: int,
        new_embedding: np.ndarray,
        update_last_used: bool = False,
    ) -> Optional[int]:
        """NO-OP: Text centroid updates removed - routing uses graph_embedding only.

        Only updates last_used_at if requested. Text centroid updates are skipped.
        See propagate_graph_centroid_to_parents() for graph-based routing.

        Args:
            conn: Database connection
            signature_id: ID of the signature
            new_embedding: Unused (kept for API compatibility)
            update_last_used: If True, update last_used_at timestamp

        Returns:
            1 (for API compatibility with callers expecting a count)
        """
        if update_last_used:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE step_signatures SET last_used_at = ? WHERE id = ?",
                (now, signature_id),
            )
            invalidate_signature_cache(signature_id)
        return 1

    def record_interference_outcome(
        self,
        signature_id: int,
        interference_type: str,
        thread_count: int,
        success_count: int,
    ):
        """Record an interference pattern outcome for a signature.

        Per CLAUDE.md: When multiple threads visit the same (dag_step_id, node_id):
        - Constructive interference (all succeed): Reinforce the node
        - Destructive interference (mixed results): Signal to split the cluster

        This updates signature statistics based on interference patterns:
        - Constructive: Boost successes (the node is operationally correct)
        - Destructive: Increment operational_failures (cluster is too generic)

        Note: Interference outcomes do NOT propagate to parents - they are
        local signals about this specific signature's cluster quality.

        Args:
            signature_id: ID of the signature
            interference_type: 'constructive' or 'destructive'
            thread_count: How many threads visited this combination
            success_count: How many threads succeeded
        """
        with self._connection() as conn:
            if interference_type == "constructive":
                # Constructive interference: all threads succeeded
                # This is strong evidence the signature is operationally correct
                # Boost success count (scaled by thread_count for multi-thread signal)
                self._increment_signature_stat(
                    conn, signature_id, "successes",
                    amount=thread_count, propagate_to_parents=False
                )
                logger.debug(
                    "[db] Constructive interference: sig %d boosted by %d (all %d threads succeeded)",
                    signature_id, thread_count, thread_count,
                )

            elif interference_type == "destructive":
                # Destructive interference: mixed results (some succeeded, some failed)
                # This signals the cluster is too generic and may need splitting
                # Record as operational failures to trigger decomposition consideration
                failure_count = thread_count - success_count
                self._increment_signature_stat(
                    conn, signature_id, "operational_failures",
                    amount=failure_count, propagate_to_parents=False
                )
                logger.debug(
                    "[db] Destructive interference: sig %d recorded %d failures "
                    "(%d/%d threads failed)",
                    signature_id, failure_count, failure_count, thread_count,
                )

    def _increment_signature_stat(
        self,
        conn,
        signature_id: int,
        stat_column: str,
        amount: float = 1.0,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """Single pathway for all signature stat increments with parent propagation.

        This is the ONLY function that should increment successes/operational_failures
        with parent credit propagation. All public methods are thin wrappers.

        Pattern follows propagate_graph_centroid_to_parents() - single entry point
        for updates that need to propagate up the tree.

        Args:
            conn: Database connection (caller manages transaction)
            signature_id: ID of the signature to update
            stat_column: Column to increment ('successes' or 'operational_failures')
            amount: Amount to increment by (default 1.0)
            propagate_to_parents: If True, propagate credit up to parent routers with decay
            _depth: Internal recursion depth tracker
        """
        from mycelium.config import PARENT_CREDIT_DECAY, PARENT_CREDIT_MAX_DEPTH, PARENT_CREDIT_MIN

        # Validate stat_column to prevent SQL injection
        if stat_column not in ("successes", "operational_failures"):
            raise ValueError(f"Invalid stat_column: {stat_column}")

        # Update this signature
        conn.execute(
            f"""UPDATE step_signatures
               SET {stat_column} = COALESCE({stat_column}, 0) + ?
               WHERE id = ?""",
            (amount, signature_id)
        )

        # Propagate to parent with decay
        if propagate_to_parents and _depth < PARENT_CREDIT_MAX_DEPTH:
            parent_row = conn.execute(
                "SELECT parent_id FROM signature_relationships WHERE child_id = ? LIMIT 1",
                (signature_id,)
            ).fetchone()

            if parent_row and parent_row[0]:
                decayed_amount = amount * PARENT_CREDIT_DECAY
                if decayed_amount >= PARENT_CREDIT_MIN:
                    self._increment_signature_stat(
                        conn,
                        parent_row[0],
                        stat_column,
                        amount=decayed_amount,
                        propagate_to_parents=True,
                        _depth=_depth + 1,
                    )

    def increment_signature_stat(
        self,
        signature_id: int,
        stat_type: SignatureStat,
        *,
        amount: float = 1.0,
        propagate_to_parents: bool = True,
    ):
        """Single entry point for all signature stat updates.

        Per CLAUDE.md "New Favorite Pattern": Consolidated pathway for all stat increments.
        Per CLAUDE.md "Credit Propagation": Automatically propagates to parents with decay.

        Args:
            signature_id: ID of the signature to update
            stat_type: SignatureStat.SUCCESS or SignatureStat.FAILURE
            amount: Amount to increment by (default 1.0, use fractional for partial credit)
            propagate_to_parents: If True, propagate credit up to parent routers with decay
        """
        with self._connection() as conn:
            self._increment_signature_stat(
                conn, signature_id, stat_type.value,
                amount=amount, propagate_to_parents=propagate_to_parents, _depth=0
            )

    # -------------------------------------------------------------------------
    # DEPRECATED: Old wrapper methods - use increment_signature_stat() instead
    # -------------------------------------------------------------------------

    def increment_signature_successes(
        self,
        signature_id: int,
        count: int = 1,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """DEPRECATED: Use increment_signature_stat(id, SignatureStat.SUCCESS) instead."""
        import warnings
        warnings.warn(
            "increment_signature_successes() is deprecated. "
            "Use increment_signature_stat(id, SignatureStat.SUCCESS, amount=count) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.increment_signature_stat(
            signature_id, SignatureStat.SUCCESS,
            amount=count, propagate_to_parents=propagate_to_parents
        )

    def increment_signature_failures(
        self,
        signature_id: int,
        count: int = 1,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """DEPRECATED: Use increment_signature_stat(id, SignatureStat.FAILURE) instead."""
        import warnings
        warnings.warn(
            "increment_signature_failures() is deprecated. "
            "Use increment_signature_stat(id, SignatureStat.FAILURE, amount=count) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.increment_signature_stat(
            signature_id, SignatureStat.FAILURE,
            amount=count, propagate_to_parents=propagate_to_parents
        )

    def increment_signature_partial_success(
        self,
        signature_id: int,
        weight: float = 0.5,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """DEPRECATED: Use increment_signature_stat(id, SignatureStat.SUCCESS, amount=weight) instead."""
        import warnings
        warnings.warn(
            "increment_signature_partial_success() is deprecated. "
            "Use increment_signature_stat(id, SignatureStat.SUCCESS, amount=weight) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.increment_signature_stat(
            signature_id, SignatureStat.SUCCESS,
            amount=weight, propagate_to_parents=propagate_to_parents
        )

    def merge_signatures(
        self,
        survivor_id: int,
        absorbed_id: int,
    ) -> bool:
        """Merge two signatures into one (constructive interference optimization).

        Per CLAUDE.md: When centroids are close AND both succeed consistently,
        merge into single node.

        The survivor keeps its identity but absorbs:
        - Centroid: weighted average by embedding_count
        - Stats: summed uses, successes
        - The absorbed signature is archived

        Args:
            survivor_id: ID of signature to keep
            absorbed_id: ID of signature to absorb and archive

        Returns:
            True if merge succeeded, False otherwise
        """
        with self._connection() as conn:
            # Get both signatures
            cursor = conn.execute(
                """SELECT id, centroid, embedding_sum, embedding_count, uses, successes,
                          operational_failures, step_type, description
                   FROM step_signatures WHERE id IN (?, ?)""",
                (survivor_id, absorbed_id)
            )
            rows = {row[0]: row for row in cursor.fetchall()}

            if survivor_id not in rows or absorbed_id not in rows:
                logger.warning(
                    "[db] Cannot merge: one or both signatures not found (survivor=%d, absorbed=%d)",
                    survivor_id, absorbed_id
                )
                return False

            survivor = rows[survivor_id]
            absorbed = rows[absorbed_id]

            # Parse centroids (handles both JSON string and binary formats)
            survivor_centroid = _parse_centroid_data(survivor[1])
            absorbed_centroid = _parse_centroid_data(absorbed[1])

            # Parse embedding sums
            survivor_sum = _parse_centroid_data(survivor[2])
            absorbed_sum = _parse_centroid_data(absorbed[2])

            survivor_count = survivor[3] or 1
            absorbed_count = absorbed[3] or 1

            # Compute merged centroid (weighted average by embedding_count)
            if survivor_centroid is not None and absorbed_centroid is not None:
                total_count = survivor_count + absorbed_count
                merged_centroid = (
                    survivor_centroid * survivor_count + absorbed_centroid * absorbed_count
                ) / total_count

                # Compute merged embedding sum
                if survivor_sum is not None and absorbed_sum is not None:
                    merged_sum = survivor_sum + absorbed_sum
                else:
                    merged_sum = merged_centroid * total_count  # Reconstruct from centroid

                # Update survivor with merged values
                conn.execute(
                    """UPDATE step_signatures SET
                       centroid = ?,
                       embedding_sum = ?,
                       embedding_count = ?,
                       uses = uses + ?,
                       successes = successes + ?,
                       operational_failures = COALESCE(operational_failures, 0) + ?
                       WHERE id = ?""",
                    (
                        merged_centroid.astype(np.float32).tobytes(),
                        merged_sum.astype(np.float32).tobytes(),
                        total_count,
                        absorbed[4] or 0,  # uses
                        absorbed[5] or 0,  # successes
                        absorbed[6] or 0,  # operational_failures
                        survivor_id,
                    )
                )

            # Archive the absorbed signature
            self._update_signature_fields(
                conn, absorbed_id,
                log_reason="merged_into_survivor",
                is_archived=1,
            )

            # Repoint any children of absorbed to survivor
            conn.execute(
                "UPDATE signature_relationships SET parent_id = ? WHERE parent_id = ?",
                (survivor_id, absorbed_id)
            )

            logger.info(
                "[db] Merged signature %d ('%s') into %d ('%s'): "
                "combined count=%d, absorbed archived",
                absorbed_id, absorbed[8][:30] if absorbed[8] else "?",
                survivor_id, survivor[8][:30] if survivor[8] else "?",
                survivor_count + absorbed_count,
            )

            return True

    def find_merge_candidates(
        self,
        min_success_rate: float = 0.7,
        min_uses: int = 5,
        min_similarity: Optional[float] = None,
        limit: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Find pairs of signatures that are candidates for merging.

        Candidates are signatures that:
        - Both have high success rates (operationally correct)
        - Have similar centroids (semantically similar)
        - Are not already archived

        Per CLAUDE.md "The Flow": min_similarity defaults to Welford-adaptive threshold.

        Args:
            min_success_rate: Minimum success rate for both signatures
            min_uses: Minimum uses for both signatures (need data to trust)
            min_similarity: Minimum cosine similarity. None = use adaptive (recommended)
            limit: Maximum number of pairs to return

        Returns:
            List of (sig1_id, sig2_id, similarity) tuples, ordered by similarity desc
        """
        # Resolve adaptive threshold if not specified
        if min_similarity is None:
            min_similarity = self._get_adaptive_similarity_threshold("cluster")
        with self._connection() as conn:
            # Get candidate signatures (high success rate, enough uses)
            cursor = conn.execute(
                """SELECT id, centroid, uses, successes
                   FROM step_signatures
                   WHERE is_archived = 0
                     AND uses >= ?
                     AND centroid IS NOT NULL
                     AND (CAST(successes AS REAL) / uses) >= ?
                   ORDER BY uses DESC
                   LIMIT 100""",  # Limit to top 100 for performance
                (min_uses, min_success_rate)
            )

            candidates = []
            for row in cursor.fetchall():
                sig_id, centroid_data, uses, successes = row
                if centroid_data:
                    # Centroid is stored as JSON string, not binary
                    if isinstance(centroid_data, str):
                        centroid = np.array(json.loads(centroid_data), dtype=np.float32)
                    else:
                        centroid = _parse_centroid_data(centroid_data)
                    candidates.append((sig_id, centroid, uses, successes))

            if len(candidates) < 2:
                return []

            # Vectorized similarity computation (much faster than O(n²) loop)
            # Stack centroids into matrix and compute all pairwise similarities at once
            ids = [c[0] for c in candidates]
            centroids = np.vstack([c[1] for c in candidates])  # (n, dim)

            # Normalize for cosine similarity
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1e-9)  # Avoid division by zero
            normalized = centroids / norms

            # All pairwise cosine similarities in one matrix multiply
            # similarity_matrix[i,j] = cosine_similarity(centroid_i, centroid_j)
            similarity_matrix = normalized @ normalized.T  # (n, n)

            # Vectorized upper triangle extraction (avoid O(n²) loop)
            n = len(ids)
            i_idx, j_idx = np.triu_indices(n, k=1)  # Upper triangle indices, k=1 excludes diagonal
            sims = similarity_matrix[i_idx, j_idx]

            # Filter by threshold
            mask = sims >= min_similarity
            if not np.any(mask):
                return []

            # Build results from filtered indices
            filtered_i = i_idx[mask]
            filtered_j = j_idx[mask]
            filtered_sims = sims[mask]

            # Sort by similarity descending
            sort_idx = np.argsort(-filtered_sims)[:limit]

            merge_candidates = [
                (ids[filtered_i[k]], ids[filtered_j[k]], float(filtered_sims[k]))
                for k in sort_idx
            ]

            return merge_candidates

    def find_similar_successful_steps(
        self,
        embedding,
        exclude_signature_id: int = None,
        min_similarity: float = ROUTING_BEST_MATCH_MIN_SIMILARITY,
        limit: int = 5,
        lookback_days: int = None,
    ) -> list[dict]:
        """Find successful steps from OTHER signatures that are similar to this embedding.

        Used by diagnostic post-mortem to detect routing misses:
        "Similar steps succeeded with different signatures"

        Args:
            embedding: Query embedding (numpy array or list)
            exclude_signature_id: Exclude results from this signature (the one that failed)
            min_similarity: Minimum cosine similarity threshold
            limit: Maximum results to return
            lookback_days: Only consider examples from last N days (None = no limit)

        Returns:
            List of dicts with: signature_id, similarity, success_rate, step_desc
        """
        if embedding is None:
            return []

        # Convert to numpy if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Build date filter if lookback specified
        date_filter = ""
        params = [exclude_signature_id, exclude_signature_id]
        if lookback_days is not None and lookback_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
            date_filter = "AND e.created_at >= ?"
            params.append(cutoff)

        # Query step_examples for successful steps with embeddings
        with self._connection() as conn:
            # Join step_examples with step_signatures to get success rate
            # Filter for successful examples from non-excluded signatures
            cursor = conn.execute(
                f"""SELECT e.signature_id, e.step_text, e.embedding,
                          s.uses, s.successes
                   FROM step_examples e
                   JOIN step_signatures s ON e.signature_id = s.id
                   WHERE e.success = 1
                     AND e.embedding IS NOT NULL
                     AND s.is_archived = 0
                     AND (? IS NULL OR e.signature_id != ?)
                     {date_filter}
                   LIMIT 500""",  # Limit scan for performance
                params
            )

            candidates = []
            for row in cursor.fetchall():
                sig_id, step_text, emb_data, uses, successes = row
                if emb_data:
                    try:
                        if isinstance(emb_data, str):
                            step_emb = np.array(json.loads(emb_data), dtype=np.float32)
                        else:
                            step_emb = np.frombuffer(emb_data, dtype=np.float32)

                        # Compute cosine similarity
                        norm_a = np.linalg.norm(embedding)
                        norm_b = np.linalg.norm(step_emb)
                        if norm_a > 0 and norm_b > 0:
                            similarity = float(np.dot(embedding, step_emb) / (norm_a * norm_b))
                            if similarity >= min_similarity:
                                success_rate = (successes / uses) if uses and uses > 0 else 0.0
                                candidates.append({
                                    "signature_id": sig_id,
                                    "similarity": similarity,
                                    "success_rate": success_rate,
                                    "step_desc": step_text[:100] if step_text else "",
                                })
                    except (json.JSONDecodeError, ValueError):
                        continue

            # Sort by similarity descending and return top results
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            return candidates[:limit]

    def has_mcts_alternatives(self, signature_id: int, conn=None) -> tuple[bool, Optional[int]]:
        """Check if alternative leaves exist for a signature's operations.

        This is the core MCTS check used by all leaf decomposition decisions.
        If alternatives exist with high similarity, the issue is likely routing,
        not the leaf itself.

        Args:
            signature_id: ID of the leaf signature to check
            conn: Optional existing DB connection

        Returns:
            Tuple of (has_alternatives, best_alternative_id)
            - has_alternatives: True if viable alternatives exist
            - best_alternative_id: ID of best alternative (if any)
        """
        use_conn = conn if conn else self._connection().__enter__()
        try:
            # Get the signature's centroid
            cursor = use_conn.execute(
                "SELECT centroid, is_semantic_umbrella FROM step_signatures WHERE id = ?",
                (signature_id,),
            )
            row = cursor.fetchone()

            if not row or not row[0] or row[1]:  # No centroid or is umbrella
                return False, None

            centroid = _parse_centroid_data(row[0])

            # Check if alternative leaves could handle this operation type
            # Use permissive min_similarity - each candidate has its own adaptive threshold
            alternatives = self.match_step_to_leaves_mcts(
                operation_embedding=centroid,
                dag_step_type=None,
                top_k=3,
                min_similarity=ROUTING_MIN_SIMILARITY_PERMISSIVE,  # Permissive - adaptive threshold applied per-candidate
            )

            # Filter out self
            other_alternatives = [
                (sig, ucb1, sim) for sig, ucb1, sim in alternatives
                if sig.id != signature_id
            ]

            if other_alternatives:
                best_alt = other_alternatives[0]
                return True, best_alt[0].id

            return False, None
        finally:
            if conn is None:
                use_conn.__exit__(None, None, None)

    def flag_for_split(
        self,
        signature_id: int,
        reason: str = "destructive_interference",
        skip_mcts_check: bool = False,
    ) -> bool:
        """Flag a signature for potential decomposition/split.

        MCTS-aware: Before flagging, checks if alternative leaves could handle
        the operations this leaf handles. If alternatives exist, it's likely
        a routing issue, not a leaf issue - skip decomposition.

        This marks a signature as needing attention due to mixed interference
        results. The actual decomposition is triggered by umbrella_learner.

        This increments operational_failures which gates decomposition in umbrella_learner.
        Per CLAUDE.md: only signatures with operational_failures > 0 are decomposition candidates.

        Args:
            signature_id: ID of signature to flag
            reason: Why it's being flagged
            skip_mcts_check: If True, bypass MCTS alternative check (force flag)

        Returns:
            True if flagged successfully, False if skipped (alternatives exist)
        """
        # MCTS alternative check: is this a routing issue or a leaf issue?
        if not skip_mcts_check:
            has_alts, alt_id = self.has_mcts_alternatives(signature_id)
            if has_alts:
                logger.info(
                    "[db] Skipping flag_for_split for sig %d (%s): "
                    "alternative sig %d exists (routing issue)",
                    signature_id, reason, alt_id
                )
                return False

        # No alternatives or check skipped - proceed with flagging
        with self._connection() as conn:
            self._update_signature_fields(
                conn, signature_id,
                log_reason=f"flag_for_split:{reason}",
                operational_failures_increment=True,
            )

            logger.info(
                "[db] Flagged signature %d for split (reason: %s, no alternatives)",
                signature_id, reason
            )

            return True

    def archive_signature(self, signature_id: int, reason: str = "retirement") -> bool:
        """Archive a signature (soft delete).

        Archived signatures are excluded from routing but kept for analysis.
        This is a soft delete - the signature data remains in the database.

        Args:
            signature_id: ID of signature to archive
            reason: Why it's being archived

        Returns:
            True if archived successfully
        """
        with self._connection() as conn:
            # Per CLAUDE.md "System Independence": Invalidate parent's Welford stats
            # when tree structure changes (child archived)
            parent_row = conn.execute(
                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                (signature_id,)
            ).fetchone()
            if parent_row:
                self.reset_welford_child_stats(parent_row["parent_id"], conn=conn)

            self._update_signature_fields(
                conn, signature_id,
                log_reason=f"archive:{reason}",
                is_archived=1,
            )

            logger.info(
                "[db] Archived signature %d (reason: %s)",
                signature_id, reason
            )

            return True

    def archive_signature_with_reparent(
        self,
        signature_id: int,
        parent_id: int,
        child_ids: list[int],
        reason: str = "retirement",
    ) -> bool:
        """DEPRECATED: Archive a signature and reparent its children atomically.

        Per CLAUDE.md System Independence (mycelium-zlza): Immediate reparenting
        bypasses Welford-guided decisions in periodic review. Use archive_signature()
        instead - orphan children will be adopted by run_periodic_tree_review().

        This function is kept for backwards compatibility but logs a warning.

        Args:
            signature_id: ID of signature to archive
            parent_id: ID of parent to reparent children to
            child_ids: List of child signature IDs to reparent
            reason: Why it's being archived

        Returns:
            True if archived successfully
        """
        import warnings
        warnings.warn(
            "archive_signature_with_reparent is deprecated per CLAUDE.md System Independence. "
            "Use archive_signature() instead - periodic review will adopt orphan children.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "[db] DEPRECATED: archive_signature_with_reparent called for sig %d. "
            "Use archive_signature() - periodic review handles orphan adoption.",
            signature_id
        )
        with self._connection() as conn:
            # Per CLAUDE.md "System Independence": Invalidate Welford stats
            # when tree structure changes (reparenting + archive)

            # Reset child stats for the signature being archived (its stats are now stale)
            self.reset_welford_child_stats(signature_id, conn=conn)

            # Reset child stats for the new parent (it's getting new children)
            self.reset_welford_child_stats(parent_id, conn=conn)

            # Re-parent all children to the grandparent
            for child_id in child_ids:
                conn.execute(
                    """INSERT OR REPLACE INTO signature_hierarchy
                       (parent_id, child_id, condition, routing_order)
                       VALUES (?, ?, 'reparented', 0)""",
                    (parent_id, child_id)
                )
                # Remove old parent relationship
                conn.execute(
                    """DELETE FROM signature_hierarchy
                       WHERE parent_id = ? AND child_id = ?""",
                    (signature_id, child_id)
                )

            # Archive the signature
            self._update_signature_fields(
                conn, signature_id,
                log_reason=f"archive_with_reparent:{reason}",
                is_archived=1,
            )

            logger.info(
                "[db] Archived signature %d with %d children reparented to %d (reason: %s)",
                signature_id, len(child_ids), parent_id, reason
            )

            return True

    def unarchive_signature(self, signature_id: int, conn=None) -> bool:
        """Restore an archived signature (un-soft-delete).

        Args:
            signature_id: ID of signature to restore
            conn: Optional connection for transaction support

        Returns:
            True if restored successfully
        """
        def _do_unarchive(c):
            self._update_signature_fields(
                c, signature_id,
                log_reason="unarchive",
                is_archived=0,
            )
            logger.info("[db] Unarchived signature %d", signature_id)
            return True

        if conn is not None:
            return _do_unarchive(conn)
        else:
            with self._connection() as c:
                return _do_unarchive(c)

    def demote_umbrella_to_leaf(
        self,
        signature_id: int,
        reason: str = "no_children",
        dsl_type: str = "decompose",
        conn=None,
    ) -> bool:
        """Demote an umbrella signature to a leaf signature.

        Per CLAUDE.md "New Favorite Pattern": SINGLE PATHWAY for all umbrella demotion.
        All code that needs to demote an umbrella should call this function.
        Ensures cache invalidation, logging, and consistency.

        Args:
            signature_id: ID of umbrella to demote
            reason: Why demotion is happening (for logging)
            dsl_type: DSL type for the demoted leaf ("decompose" or "math")
            conn: Optional connection for transaction support

        Returns:
            True if demotion succeeded
        """
        def _do_demote(c):
            result = self._update_signature_fields(
                c, signature_id,
                log_reason=f"demote_umbrella:{reason}",
                is_semantic_umbrella=0,
                dsl_type=dsl_type,
            )
            if result:
                logger.info(
                    "[db] Demoted umbrella %d to leaf (reason: %s, dsl_type: %s)",
                    signature_id, reason, dsl_type
                )
            return result

        if conn is not None:
            return _do_demote(conn)
        else:
            with self._connection() as c:
                return _do_demote(c)

    def mark_signature_atomic(
        self,
        signature_id: int,
        reason: str,
        conn=None,
    ) -> bool:
        """Mark a signature as atomic (non-decomposable).

        Per CLAUDE.md "New Favorite Pattern": Consolidated method for atomic marking.
        Prevents repeated failed decomposition attempts.

        Used by umbrella learner when decomposition has failed.

        Args:
            signature_id: ID of signature to mark atomic
            reason: Why it's atomic (e.g., "single_operation", "decomposition_failed")
            conn: Optional connection for transaction support

        Returns:
            True if update succeeded
        """
        def _do_mark(c):
            result = self._update_signature_fields(
                c, signature_id,
                log_reason=f"mark_atomic:{reason}",
                is_atomic=1,
                atomic_reason=reason,
            )
            if result:
                logger.info("[db] Marked signature %d as atomic (reason: %s)", signature_id, reason)
            return result

        if conn is not None:
            return _do_mark(conn)
        else:
            with self._connection() as c:
                return _do_mark(c)

    def increment_rejection_count(self, signature_id: int, conn=None) -> int:
        """Increment rejection count and return new value.

        Args:
            signature_id: ID of signature that rejected a step
            conn: Optional connection for transaction support

        Returns:
            New rejection_count value, or 0 on error
        """
        def _do_increment(c):
            self._update_signature_fields(
                c, signature_id,
                log_reason="rejection",
                rejection_count_increment=True,
            )
            # Get updated count
            cursor = c.execute(
                "SELECT rejection_count FROM step_signatures WHERE id = ?",
                (signature_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

        try:
            if conn is not None:
                return _do_increment(conn)
            else:
                with self._connection() as c:
                    return _do_increment(c)
        except Exception as e:
            logger.warning("[db] Error incrementing rejection count for %d: %s", signature_id, e)
            return 0

    def update_signature_depth(self, signature_id: int, depth: int, conn=None) -> bool:
        """Update signature depth.

        Args:
            signature_id: ID of signature to update
            depth: New depth value
            conn: Optional connection for transaction support

        Returns:
            True if updated successfully
        """
        def _do_update(c):
            return self._update_signature_fields(
                c, signature_id,
                log_reason="depth_update",
                depth=depth,
            )

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                return _do_update(c)

    # =========================================================================
    # Usage Recording
    # =========================================================================

    def record_usage(
        self,
        signature_id: int,
        step_text: str,
        step_completed: bool,
        was_injected: bool = False,
        params_extracted: dict = None,
        difficulty: float = None,
    ) -> int:
        """Record usage of a signature (step-level, not problem-level).

        Note: step_completed tracks whether the step returned a result, NOT
        whether the final problem answer was correct. Use update_problem_outcome()
        after grading to track actual problem correctness (which updates
        step_signatures.successes).

        Args:
            signature_id: ID of the signature used
            step_text: The step text that was executed
            step_completed: Whether the step returned a result (NOT problem correctness)
            was_injected: Whether DSL was injected
            params_extracted: Parameters that were extracted (for learning)
            difficulty: Problem difficulty (0.0-1.0) for difficulty tracking

        Returns:
            New uses count (for triggering DSL regeneration on mod 10)
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            # Insert usage log (step_completed = step returned result, not problem correct)
            conn.execute(
                """INSERT INTO step_usage_log
                   (signature_id, step_text, step_completed, was_injected, params_extracted, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    signature_id,
                    step_text,
                    1 if step_completed else 0,
                    1 if was_injected else 0,
                    json.dumps(params_extracted) if params_extracted else None,
                    now,
                ),
            )

            # Only increment uses count here, NOT successes
            # Successes and last_used_at are updated by update_problem_outcome() after grading
            # (last_used_at only updates on success to prevent stale signatures staying fresh)
            self._update_signature_fields(
                conn, signature_id,
                uses_increment=True,
            )

            # Update difficulty_stats if difficulty provided
            if difficulty is not None:
                self._update_difficulty_stats(conn, signature_id, difficulty, success=step_completed)

            # Get current stats for auto-demotion check
            cursor = conn.execute(
                """SELECT uses, successes, dsl_type, is_semantic_umbrella
                   FROM step_signatures WHERE id = ?""",
                (signature_id,),
            )
            row = cursor.fetchone()
            if not row:
                return 0

            uses, successes, dsl_type, is_umbrella = row

            # Auto-demote failing DSLs to umbrellas
            # Graduated threshold: min_uses = FLOOR + (sig_count // DIVISOR), capped
            # Branch fast early (centroid averaging will stabilize good paths)
            # NOTE: Only demote if THIS step failed (step_completed=False)
            # OR if we have enough history showing poor success_rate
            # This prevents premature demotion before problem grading updates successes
            if AUTO_DEMOTE_ENABLED and not is_umbrella and dsl_type not in AUTO_DEMOTE_EXCLUDED_TYPES:
                sig_count_row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
                sig_count = sig_count_row[0] if sig_count_row else 0
                min_uses = min(
                    AUTO_DEMOTE_MIN_USES_FLOOR + sig_count // AUTO_DEMOTE_RAMP_DIVISOR,
                    AUTO_DEMOTE_MIN_USES_CAP
                )

                if uses >= min_uses:
                    success_rate = successes / uses if uses > 0 else 0
                    # Only demote based on historical success rate (problem correctness)
                    # NOT based on step_completed - that conflates DSL failure with multi-path loss
                    # Multi-path losers get step_completed=False but their DSL didn't actually fail
                    should_demote = success_rate < AUTO_DEMOTE_MAX_SUCCESS_RATE
                    if should_demote:
                        # MCTS check: only demote if no alternatives exist
                        # If alternatives exist, it's a routing issue, not this leaf's fault
                        has_alts, alt_id = self.has_mcts_alternatives(signature_id, conn=conn)
                        if has_alts:
                            logger.info(
                                "[db] Skipping auto-demotion for sig %d: "
                                "alternative sig %d exists (routing issue, not leaf issue)",
                                signature_id, alt_id
                            )
                        else:
                            # No alternatives - check if has children before promoting
                            # An umbrella without children can't route and breaks dedup
                            child_count_row = conn.execute(
                                "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                                (signature_id,)
                            ).fetchone()
                            has_children = child_count_row and child_count_row[0] > 0

                            if has_children:
                                # Has children - safe to promote to umbrella/router
                                self._promote_to_umbrella_internal(conn, signature_id)
                                logger.info(
                                    "[db] Auto-demoted sig %d to umbrella/router (%.0f%% after %d uses, step_ok=%s, min=%d, no alternatives)",
                                    signature_id, success_rate * 100, uses, step_completed, min_uses
                                )
                            else:
                                # No children - keep as leaf, let dedup handle it
                                logger.info(
                                    "[db] Skipping auto-demotion for sig %d: no children (would break dedup)",
                                    signature_id
                                )

            return uses

    def _update_difficulty_stats(
        self,
        conn,
        signature_id: int,
        difficulty: float,
        success: bool,
    ):
        """Update difficulty_stats for a signature.

        Tracks usage and success by difficulty level for routing optimization.
        Uses bucketed difficulty (rounded to 0.1) as keys.

        Args:
            conn: Database connection
            signature_id: Signature ID
            difficulty: Problem difficulty (0.0-1.0)
            success: Whether the step succeeded
        """
        # Bucket difficulty to nearest 0.1
        diff_key = str(round(difficulty, 1))

        # Get current stats
        row = conn.execute(
            "SELECT difficulty_stats, max_difficulty_solved FROM step_signatures WHERE id = ?",
            (signature_id,),
        ).fetchone()

        if not row:
            return

        # Parse existing stats
        try:
            stats = json.loads(row["difficulty_stats"]) if row["difficulty_stats"] else {}
        except json.JSONDecodeError:
            stats = {}

        # Update stats for this difficulty bucket
        if diff_key not in stats:
            stats[diff_key] = {"uses": 0, "successes": 0}

        stats[diff_key]["uses"] += 1
        if success:
            stats[diff_key]["successes"] += 1

        # Update max_difficulty_solved if this is a new high
        max_diff = row["max_difficulty_solved"] or 0.0
        if success and difficulty > max_diff:
            max_diff = difficulty

        # Save updated stats
        self._update_signature_fields(
            conn, signature_id,
            difficulty_stats=json.dumps(stats),
            max_difficulty_solved=max_diff,
        )

    def record_failure(
        self,
        step_text: str,
        failure_type: str,
        error_message: str = None,
        signature_id: int = None,
        context: dict = None,
        increment_operational_failures: bool = False,
    ) -> int:
        """Record a step failure for pattern learning.

        Per CLAUDE.md: "Failures Are Valuable Data Points"
        - Record every failure—it feeds the refinement loop
        - Failed signatures get decomposed
        - Success/failure stats drive routing decisions

        This is the single entry point for recording failures. It always logs
        to step_failures table, and optionally increments the signature's
        operational_failures stat (for routing decisions).

        Args:
            step_text: The step that failed
            failure_type: Category of failure:
                - 'dsl_error': DSL execution failed
                - 'no_match': No signature matched
                - 'llm_error': LLM call failed
                - 'timeout': Operation timed out
                - 'validation': Result validation failed
                - 'routing': Umbrella routing failed
                - 'operational': Signature produced wrong answer vs ground truth
            error_message: The actual error text
            signature_id: ID of signature that failed (None if no match)
            context: Additional context dict (params, expected, problem, etc.)
                For operational failures, include 'produced_answer' and 'expected_answer'
            increment_operational_failures: If True and signature_id provided,
                also increment the signature's operational_failures stat.
                Use this when the signature is at fault (e.g., wrong answer).

        Returns:
            ID of the failure record
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO step_failures
                   (signature_id, step_text, failure_type, error_message, context, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    signature_id,
                    step_text,
                    failure_type,
                    error_message,
                    json.dumps(context) if context else None,
                    now,
                ),
            )
            failure_id = cursor.lastrowid

            # Optionally increment operational_failures stat on the signature
            # This affects routing decisions (signatures with many failures may be decomposed)
            # Note: Does NOT propagate to parents - operational failures are local signal
            if increment_operational_failures and signature_id is not None:
                self._increment_signature_stat(
                    conn, signature_id, "operational_failures",
                    amount=1.0, propagate_to_parents=False
                )
                logger.debug(
                    "[db] Incremented operational_failures for sig %d",
                    signature_id
                )

            logger.debug(
                "[db] Recorded failure: type=%s sig=%s step='%s'",
                failure_type, signature_id, step_text[:50]
            )

            return failure_id

    def get_signature_examples(
        self,
        signature_id: int,
        limit: int = 10,
    ) -> list[dict]:
        """Get usage examples for a signature (for DSL generation).

        Uses step_examples table which has actual results (not just usage logs).

        Args:
            signature_id: ID of the signature
            limit: Maximum number of examples to return

        Returns:
            List of example dicts with step_text, result, success, expression, inputs
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT step_text, result, success, expression, inputs
                   FROM step_examples
                   WHERE signature_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (signature_id, limit),
            )
            examples = []
            for row in cursor.fetchall():
                examples.append({
                    'step_text': row[0],
                    'result': row[1] if row[1] else '',
                    'success': bool(row[2]),
                    'expression': row[3] if row[3] else '',
                    'inputs': row[4] if row[4] else '',
                })
            return examples

    def update_example_result(
        self,
        signature_id: int,
        step_text: str,
        result: str,
        success: bool,
        expression: str | None = None,
        inputs: str | None = None,
    ) -> bool:
        """Update an example with its execution result.

        Called after DSL execution to record the result for DSL regeneration.
        Finds the most recent example for this signature matching step_text.

        Args:
            signature_id: ID of the signature
            step_text: The step text to match
            result: The DSL execution result
            success: Whether execution succeeded
            expression: The DSL script that was executed (e.g., "a * b")
            inputs: JSON string of parameter values used (e.g., '{"a": 5, "b": 3}')

        Returns:
            True if an example was updated, False otherwise
        """
        with self._connection() as conn:
            # Update the most recent example for this signature
            # Match on first 100 chars of step_text to handle truncation
            cursor = conn.execute(
                """UPDATE step_examples
                   SET result = ?, success = ?, expression = ?, inputs = ?
                   WHERE id = (
                       SELECT id FROM step_examples
                       WHERE signature_id = ?
                         AND substr(step_text, 1, 100) = substr(?, 1, 100)
                       ORDER BY created_at DESC
                       LIMIT 1
                   )""",
                (result, 1 if success else 0, expression, inputs, signature_id, step_text),
            )
            updated = cursor.rowcount > 0
            if updated:
                logger.debug(
                    "[db] Updated example result for sig %d: success=%s expr=%s",
                    signature_id, success, expression[:30] if expression else None
                )
            return updated

    def update_problem_outcome(
        self,
        signature_ids: list[int],
        problem_correct: bool,
        decay_factor: float = None,
        difficulty: float = None,
    ):
        """Update signature success counts based on problem outcome.

        Call this after grading a problem to propagate correctness back to
        all signatures that were used. Also propagates credit up to parent
        umbrella signatures with decay.

        NOTE: This function intentionally does NOT use _increment_signature_stat
        because it has batch-specific semantics:
        - Bulk SQL update for multiple signatures (efficiency)
        - Updates last_used_at alongside successes
        - Parent propagation uses MAX dedup (vs SUM in single-sig recursive)
        - Parent updates filter by is_semantic_umbrella = 1

        DIFFICULTY-WEIGHTED CREDIT: Harder problems provide more valuable signal.
        - difficulty=0.0 (trivial) → 1.0x credit
        - difficulty=0.5 (GSM8K) → 3.0x credit
        - difficulty=1.0 (competition) → 5.0x credit

        This ensures signatures that solve hard problems gain more confidence
        than those solving easy problems (per CLAUDE.md universal tree design).

        Args:
            signature_ids: IDs of signatures used in the solved problem
            problem_correct: Whether the final answer was correct
            decay_factor: Credit decay per level (default from config)
            difficulty: Problem difficulty for weighted credit (0.0-1.0)
        """
        # Increment global problem counter (for traffic-based decay)
        increment_total_problems()

        if decay_factor is None:
            decay_factor = PARENT_CREDIT_DECAY
        if not signature_ids:
            return

        # Calculate difficulty-weighted credit multiplier
        # Harder problems → more credit (1.0x to 5.0x)
        from mycelium.difficulty import get_credit_multiplier
        credit_multiplier = get_credit_multiplier(difficulty) if difficulty is not None else 1.0
        base_credit = credit_multiplier  # Base credit for direct signatures

        now = datetime.now().isoformat()

        with self._connection() as conn:
            if problem_correct:
                # Increment success count with difficulty weighting
                # (last_used_at only updates on success to properly penalize stale failures)
                # Use weighted credit: harder problems count more
                placeholders = ",".join("?" * len(signature_ids))
                conn.execute(
                    f"""UPDATE step_signatures
                       SET successes = successes + ?, last_used_at = ?
                       WHERE id IN ({placeholders})""",
                    [base_credit, now] + signature_ids,
                )
                logger.debug(
                    "[db] Problem correct: credited %d signatures (%.1fx multiplier, diff=%.2f)",
                    len(signature_ids), credit_multiplier, difficulty or 0.0
                )

                # Propagate credit to parent umbrellas with decay
                # Track credits per parent to avoid double-counting
                parent_credits: dict[int, float] = {}

                for sig_id in signature_ids:
                    self._collect_parent_credits(
                        conn, sig_id, decay_factor, 1, parent_credits
                    )

                # Apply accumulated credits with difficulty weighting
                for parent_id, credit in parent_credits.items():
                    weighted_credit = credit * credit_multiplier
                    if weighted_credit >= PARENT_CREDIT_MIN:
                        conn.execute(
                            """UPDATE step_signatures
                               SET successes = successes + ?
                               WHERE id = ? AND is_semantic_umbrella = 1""",
                            (weighted_credit, parent_id),
                        )

                if parent_credits:
                    logger.debug(
                        "[db] Propagated weighted credit to %d parent umbrellas (%.1fx multiplier)",
                        len(parent_credits), credit_multiplier
                    )
            else:
                # Problem failed - signatures don't get success credit
                # This is how we detect negative lift
                logger.debug(
                    "[db] Problem incorrect: %d signatures get no success credit",
                    len(signature_ids)
                )

    def record_weighted_failure(
        self,
        signature_id: int,
        step_text: str,
        weight: float,
        is_llm_blamed: bool = False,
    ) -> None:
        """Record a weighted failure for a signature.

        Per beads mycelium-b5tq: LLM failure diagnosis applies weighted blame.
        LLM-blamed steps get higher weight, others get lower weight.

        This updates the signature's Welford failure stats with the weight,
        allowing the system to learn which (dag_step, leaf_node) pairs are
        more likely to be the actual culprit.

        Args:
            signature_id: The signature that was used
            step_text: The step description (for logging)
            weight: Blame weight (0.0 to 1.0, higher = more blame)
            is_llm_blamed: True if LLM identified this as the likely culprit
        """
        with self._connection() as conn:
            # Update weighted failure count in signature
            # We use a weighted failure counter that accumulates blame
            # Higher weight = more contribution to failure stats
            conn.execute(
                """UPDATE step_signatures
                   SET weighted_failures = COALESCE(weighted_failures, 0) + ?,
                       llm_blamed_count = COALESCE(llm_blamed_count, 0) + ?
                   WHERE id = ?""",
                (weight, 1 if is_llm_blamed else 0, signature_id)
            )

            # Also update Welford stats for this signature's failure rate
            # This feeds into the adaptive threshold calculation
            from mycelium.data_layer.state_manager import get_state_manager
            prefix = f"sig_{signature_id}_failure"
            try:
                get_state_manager().update_welford_stats(prefix, weight)
            except Exception as e:
                logger.debug("[db] Failed to update Welford stats for sig %d: %s", signature_id, e)

            logger.debug(
                "[db] Recorded weighted failure: sig=%d step='%s' weight=%.2f blamed=%s",
                signature_id, step_text[:30], weight, is_llm_blamed
            )

    def _collect_parent_credits(
        self,
        conn,
        signature_id: int,
        decay_factor: float,
        current_depth: int,
        credits: dict[int, float],
        max_depth: int = None,
    ):
        """Collect credits for parent umbrellas using recursive CTE (single query).

        Replaces recursive function calls with one SQL query that fetches
        all ancestor umbrellas up to max_depth in a single round-trip.

        Args:
            conn: Database connection
            signature_id: Current signature ID
            decay_factor: Credit multiplier per level (e.g., 0.7^depth)
            current_depth: Starting depth (usually 1)
            credits: Dict accumulating {parent_id: total_credit}
            max_depth: Max depth to traverse (default from config)
        """
        if max_depth is None:
            max_depth = PARENT_CREDIT_MAX_DEPTH
        if current_depth > max_depth:
            return

        # Fetch all ancestor umbrellas in ONE query using recursive CTE
        cursor = conn.execute(
            """WITH RECURSIVE ancestors AS (
                -- Base case: direct parent
                SELECT r.parent_id, 1 as depth
                FROM signature_relationships r
                JOIN step_signatures s ON r.parent_id = s.id
                WHERE r.child_id = ? AND s.is_semantic_umbrella = 1

                UNION ALL

                -- Recursive case: grandparents and beyond
                SELECT r.parent_id, a.depth + 1
                FROM signature_relationships r
                JOIN step_signatures s ON r.parent_id = s.id
                JOIN ancestors a ON r.child_id = a.parent_id
                WHERE a.depth < ? AND s.is_semantic_umbrella = 1
            )
            SELECT parent_id, depth FROM ancestors
            ORDER BY depth""",
            (signature_id, max_depth),
        )

        # Compute credits for each ancestor based on depth
        for row in cursor:
            parent_id = row[0]
            depth = row[1] + current_depth - 1  # Adjust for starting depth
            credit = decay_factor ** depth
            # Take max credit if already seen (from multiple children)
            if parent_id not in credits or credit > credits[parent_id]:
                credits[parent_id] = credit

    # =========================================================================
    # NL Interface Updates (for later learning)
    # =========================================================================

    def update_nl_interface(
        self,
        signature_id: int,
        clarifying_questions: list[str] = None,
        param_descriptions: dict[str, str] = None,
        dsl_script: str = None,
        dsl_type: str = None,
        examples: list[dict] = None,
    ):
        """Update the Natural Language interface for a signature.

        Called when we learn better questions/DSLs for a signature.
        """
        updates = []
        params = []

        if clarifying_questions is not None:
            updates.append("clarifying_questions = ?")
            params.append(json.dumps(clarifying_questions))

        if param_descriptions is not None:
            updates.append("param_descriptions = ?")
            params.append(json.dumps(param_descriptions))

        if dsl_script is not None:
            updates.append("dsl_script = ?")
            params.append(dsl_script)

        if dsl_type is not None:
            # Only allow: decompose, math, router
            if dsl_type not in ("decompose", "math", "router"):
                raise ValueError(f"Invalid dsl_type '{dsl_type}'. Must be: decompose, math, or router")
            updates.append("dsl_type = ?")
            params.append(dsl_type)

        if examples is not None:
            updates.append("examples = ?")
            params.append(json.dumps(examples))

        if not updates:
            return

        params.append(signature_id)

        with self._connection() as conn:
            conn.execute(
                f"UPDATE step_signatures SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            self._invalidate_on_dsl_change(signature_id)

    def update_dsl_script(
        self,
        signature_id: int,
        dsl_script: str,
    ):
        """Update the DSL script for a signature and mark as rewritten.

        Called by DSL regeneration to update the script based on learned patterns.
        Also updates last_rewrite_at timestamp.

        Args:
            signature_id: ID of the signature to update
            dsl_script: The new DSL script
        """
        from datetime import datetime

        now = datetime.utcnow().isoformat()
        with self._connection() as conn:
            self._update_signature_fields(
                conn, signature_id,
                log_reason="dsl_update",
                dsl_script=dsl_script,
                last_rewrite_at=now,
            )
            self._invalidate_on_dsl_change(signature_id)
            logger.info(
                "[db] Updated DSL script for signature %d, marked as rewritten",
                signature_id
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _update_signature_fields(
        self,
        conn,
        signature_id: int,
        log_reason: str = None,
        **field_updates,
    ) -> bool:
        """Update signature fields atomically with proper cache invalidation.

        Consolidated helper for all signature field updates. Handles:
        - Building the UPDATE query dynamically
        - Cache invalidation based on which fields changed
        - Common side effects (e.g., propagation for graph fields)
        - Audit logging

        Args:
            conn: Database connection (caller's transaction context)
            signature_id: ID of signature to update
            log_reason: Optional reason for logging (e.g., "auto_demotion", "postmortem")
            **field_updates: Field names and values to update, e.g.:
                - is_semantic_umbrella=True
                - dsl_type="router"
                - dsl_script=None
                - operational_failures=1 (use _increment=True for increment)
                - last_used_at="2024-01-01T00:00:00"

        Returns:
            True if update succeeded (rowcount > 0), False otherwise

        Example:
            self._update_signature_fields(
                conn, sig_id,
                log_reason="auto_demotion",
                is_semantic_umbrella=True,
                dsl_type="router",
                dsl_script=None,
            )
        """
        if not field_updates:
            return False

        # Build UPDATE clause
        set_clauses = []
        params = []

        # Fields that need special handling
        increment_fields = {"operational_failures", "uses", "successes", "rejection_count"}

        for field, value in field_updates.items():
            # Check for increment syntax: field_increment=True means += 1
            if field.endswith("_increment") and value:
                base_field = field.replace("_increment", "")
                set_clauses.append(f"{base_field} = COALESCE({base_field}, 0) + 1")
            elif value is None:
                set_clauses.append(f"{field} = NULL")
            else:
                set_clauses.append(f"{field} = ?")
                params.append(value)

        params.append(signature_id)

        # Execute UPDATE
        cursor = conn.execute(
            f"UPDATE step_signatures SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )

        if cursor.rowcount == 0:
            return False

        # Determine which caches to invalidate based on fields changed
        fields_changed = set(field_updates.keys())

        # Always invalidate signature cache
        invalidate_signature_cache(signature_id)

        # Invalidate centroid cache if centroid-related fields changed
        centroid_fields = {"centroid", "graph_centroid", "graph_embedding"}
        if fields_changed & centroid_fields:
            invalidate_centroid_cache(signature_id)
            self.invalidate_centroid_matrix()

        # Invalidate children cache if parent relationship changed
        if "is_semantic_umbrella" in fields_changed:
            invalidate_children_cache(signature_id)

        # Invalidate parent's children cache if depth changed (tree structure changed)
        if "depth" in fields_changed:
            # Find and invalidate parent's children cache
            parent_row = conn.execute(
                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                (signature_id,)
            ).fetchone()
            if parent_row and parent_row[0]:
                invalidate_children_cache(parent_row[0])

        # Propagate graph centroid if graph_embedding changed
        if "graph_embedding" in fields_changed:
            self.propagate_graph_centroid_to_parents(conn, signature_id)

        # Log the update
        if log_reason:
            fields_str = ", ".join(f"{k}={v}" for k, v in field_updates.items())
            logger.debug(
                "[db] Updated sig %d (%s): %s",
                signature_id, log_reason, fields_str[:100]
            )

        return True

    def update_success_similarity(
        self,
        signature_id: int,
        similarity: float,
    ) -> bool:
        """Update success similarity stats using Welford's online algorithm.

        Per mycelium-i601: Track similarity scores of SUCCESSFUL matches to compute
        adaptive rejection threshold: threshold = mean - k * std

        Called by post-mortem when a thread succeeds. The similarity value is the
        cosine similarity that was used when matching the dag_step to this leaf.

        Uses Welford's algorithm for numerically stable online variance:
        - delta = x - mean
        - new_mean = mean + delta/n
        - delta2 = x - new_mean
        - new_m2 = m2 + delta * delta2

        Args:
            signature_id: ID of the leaf signature
            similarity: Cosine similarity score from successful match

        Returns:
            True if update succeeded
        """
        with self._connection() as conn:
            # Get current stats
            row = conn.execute(
                """SELECT success_sim_count, success_sim_mean, success_sim_m2
                   FROM step_signatures WHERE id = ?""",
                (signature_id,)
            ).fetchone()

            if not row:
                return False

            n = (row["success_sim_count"] or 0) + 1
            old_mean = row["success_sim_mean"] or 0.0
            old_m2 = row["success_sim_m2"] or 0.0

            # Welford's update
            delta = similarity - old_mean
            new_mean = old_mean + delta / n
            delta2 = similarity - new_mean
            new_m2 = old_m2 + delta * delta2

            # Update stats
            conn.execute(
                """UPDATE step_signatures
                   SET success_sim_count = ?,
                       success_sim_mean = ?,
                       success_sim_m2 = ?
                   WHERE id = ?""",
                (n, new_mean, new_m2, signature_id)
            )
            conn.commit()
            invalidate_signature_cache(signature_id)

            logger.debug(
                "[db] Updated success_sim for sig %d: n=%d, mean=%.4f, std=%.4f",
                signature_id, n, new_mean, (new_m2/n)**0.5 if n > 0 else 0
            )
            return True

    def update_success_similarity_batch(
        self,
        updates: list[tuple[int, float]],
    ) -> int:
        """Batch update success similarity stats for multiple signatures.

        More efficient than calling update_success_similarity() in a loop.

        Args:
            updates: List of (signature_id, similarity) tuples

        Returns:
            Number of signatures updated
        """
        if not updates:
            return 0

        updated = 0
        with self._connection() as conn:
            for signature_id, similarity in updates:
                # Get current stats
                row = conn.execute(
                    """SELECT success_sim_count, success_sim_mean, success_sim_m2
                       FROM step_signatures WHERE id = ?""",
                    (signature_id,)
                ).fetchone()

                if not row:
                    continue

                n = (row["success_sim_count"] or 0) + 1
                old_mean = row["success_sim_mean"] or 0.0
                old_m2 = row["success_sim_m2"] or 0.0

                # Welford's update
                delta = similarity - old_mean
                new_mean = old_mean + delta / n
                delta2 = similarity - new_mean
                new_m2 = old_m2 + delta * delta2

                # Update stats
                conn.execute(
                    """UPDATE step_signatures
                       SET success_sim_count = ?,
                           success_sim_mean = ?,
                           success_sim_m2 = ?
                       WHERE id = ?""",
                    (n, new_mean, new_m2, signature_id)
                )
                invalidate_signature_cache(signature_id)
                updated += 1

            conn.commit()

        logger.debug("[db] Batch updated success_sim for %d signatures", updated)
        return updated

    def _row_to_signature(self, row) -> StepSignature:
        """Convert a database row to a StepSignature object."""
        return StepSignature.from_row(dict(row))

    def _row_to_signature_fast(self, row) -> StepSignature:
        """Convert a database row to StepSignature (skip JSON parsing).

        ~3x faster. Use when you only need basic fields.
        """
        return StepSignature.from_row_fast(dict(row))

    def _row_to_signature_for_routing(self, row) -> StepSignature:
        """Convert a database row to StepSignature optimized for routing.

        ~4x faster than full parsing. Parses centroid (required for similarity)
        but skips all other JSON. Per CLAUDE.md: "Umbrella signature routing
        should not require an LLM call" - routing is purely embedding-based.
        """
        return StepSignature.from_row_for_routing(dict(row))

    def _is_step_type_compatible(self, step_type: str, dsl_hint: str) -> bool:
        """Check if a step_type is compatible with a dsl_hint.

        Used during routing to prevent matching a sum step to a product signature.

        Args:
            step_type: The signature's step_type (e.g., "compute_product")
            dsl_hint: The planner's operation hint (+, -, *, /)

        Returns:
            True if compatible, False if they conflict
        """
        hint = dsl_hint.strip().lower()

        # Map dsl_hint to expected step_type
        HINT_TO_TYPE = {
            "+": "compute_sum", "add": "compute_sum", "sum": "compute_sum",
            "-": "compute_difference", "subtract": "compute_difference", "difference": "compute_difference",
            "*": "compute_product", "multiply": "compute_product", "product": "compute_product",
            "/": "compute_quotient", "divide": "compute_quotient", "quotient": "compute_quotient",
        }

        expected_type = HINT_TO_TYPE.get(hint)
        if expected_type is None:
            # Unknown hint - allow any match
            return True

        # Check if step_type matches expected
        # Also allow matching to abstract/branch types (they route, don't execute)
        if step_type == expected_type:
            return True
        if step_type.startswith("abstract_") or step_type.startswith("branch_"):
            return True

        return False

    def _dsl_hint_to_graph(self, dsl_hint: str) -> str:
        """Convert a dsl_hint to a canonical computation graph string.

        Used for graph_embedding comparison during leaf rejection.
        Per CLAUDE.md: route by what operations DO, not what they SOUND LIKE.

        Args:
            dsl_hint: Operation hint from planner (+, -, *, /)

        Returns:
            Canonical graph string like "ADD(a, b)" or None if unknown
        """
        hint = dsl_hint.strip().lower()

        # Map dsl_hint to canonical graph representation
        # IMPORTANT: Must match the format used in signature computation_graph
        # Signatures use "ADD(param_0, param_1)" not "ADD(a, b)"
        HINT_TO_GRAPH = {
            "+": "ADD(param_0, param_1)",
            "add": "ADD(param_0, param_1)",
            "sum": "ADD(param_0, param_1)",
            "-": "SUB(param_0, param_1)",
            "subtract": "SUB(param_0, param_1)",
            "difference": "SUB(param_0, param_1)",
            "*": "MUL(param_0, param_1)",
            "multiply": "MUL(param_0, param_1)",
            "product": "MUL(param_0, param_1)",
            "/": "DIV(param_0, param_1)",
            "divide": "DIV(param_0, param_1)",
            "quotient": "DIV(param_0, param_1)",
        }

        return HINT_TO_GRAPH.get(hint)

    def _get_graph_embedding_from_hint(self, dsl_hint: Optional[str]) -> Optional[np.ndarray]:
        """Convert DSL hint to graph embedding.

        Per CLAUDE.md "New Favorite Pattern": Consolidates repeated pattern of
        _dsl_hint_to_graph() + embed_computation_graph_sync().

        Args:
            dsl_hint: DSL operation hint like "+", "multiply", etc.

        Returns:
            Graph embedding as numpy array, or None if conversion fails
        """
        if not dsl_hint:
            return None

        graph = self._dsl_hint_to_graph(dsl_hint)
        if not graph:
            return None

        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
        emb_list = embed_computation_graph_sync(self.embedder, graph)
        if emb_list:
            return np.array(emb_list)
        return None

    def _get_atomic_embeddings(self) -> list[np.ndarray]:
        """Get cached embeddings for atomic operations.

        Lazily computes and caches embeddings for ADD, SUB, MUL, DIV.
        Per CLAUDE.md: "prefer embedding similarity over keyword matching"

        Returns:
            List of numpy arrays, one embedding per atomic operation
        """
        if self._atomic_embeddings is None:
            from mycelium.embedding_cache import cached_embed
            self._atomic_embeddings = []
            for op in ATOMIC_OPERATIONS:
                emb = cached_embed(op)
                if emb is not None:
                    self._atomic_embeddings.append(np.array(emb))
            logger.debug("[db] Cached %d atomic operation embeddings", len(self._atomic_embeddings))
        return self._atomic_embeddings

    def _is_step_atomic(self, step_embedding: np.ndarray) -> tuple[bool, float, float, str]:
        """Check if a step embedding matches an atomic operation.

        Compares the step's embedding against all atomic operations (ADD, SUB, MUL, DIV).
        Uses two signals:
        1. Max similarity - if too low, step is unknown/complex
        2. Gap between best and 2nd best - if too small, step matches multiple ops (multi-part)

        Per CLAUDE.md: "prefer embedding similarity over keyword matching"

        Args:
            step_embedding: Embedding of the step (from dsl_hint or step_text)

        Returns:
            Tuple of (is_atomic, max_similarity, gap, best_match_op)
            - is_atomic: True if step clearly matches one atomic operation
            - max_similarity: Similarity to best matching atomic op
            - gap: Difference between best and 2nd best match (small gap = multi-part)
            - best_match_op: The atomic operation that matched best
        """
        atomic_embeddings = self._get_atomic_embeddings()
        if not atomic_embeddings or len(atomic_embeddings) < 2:
            # Not enough atomic embeddings available, assume atomic
            return True, 1.0, 1.0, "unknown"

        # Compute similarity to all atomic operations
        similarities = []
        for i, atomic_emb in enumerate(atomic_embeddings):
            sim = cosine_similarity(step_embedding, atomic_emb)
            similarities.append((sim, ATOMIC_OPERATIONS[i]))

        # Sort by similarity descending
        similarities.sort(key=lambda x: -x[0])
        best_sim, best_op = similarities[0]
        second_sim, _ = similarities[1]
        gap = best_sim - second_sim

        # Step is atomic if:
        # 1. Max similarity is above threshold (it matches some atomic op)
        # 2. Gap is above threshold (it clearly matches ONE op, not multiple)
        is_atomic = best_sim >= ATOMIC_SIMILARITY_THRESHOLD and gap >= ATOMIC_GAP_THRESHOLD

        return is_atomic, best_sim, gap, best_op

    def _infer_step_type(self, step_text: str, dsl_hint: str = None) -> str:
        """Infer a step type from step text.

        Priority:
        1. If dsl_hint provided, derive step_type from it (authoritative)
        2. Fall back to keyword matching

        The dsl_hint from the planner is the LLM's analysis of the actual
        operation needed, so it's more reliable than keyword matching.
        """
        # Priority 1: Use dsl_hint from planner (LLM's operation analysis)
        if dsl_hint:
            hint = dsl_hint.strip().lower()
            HINT_TO_TYPE = {
                "+": "compute_sum", "add": "compute_sum", "sum": "compute_sum",
                "-": "compute_difference", "subtract": "compute_difference", "difference": "compute_difference",
                "*": "compute_product", "multiply": "compute_product", "product": "compute_product",
                "/": "compute_quotient", "divide": "compute_quotient", "quotient": "compute_quotient",
            }
            if hint in HINT_TO_TYPE:
                return HINT_TO_TYPE[hint]

        # Priority 2: Keyword matching fallback
        text_lower = step_text.lower()

        # Keyword patterns
        patterns = [
            (r"factorial|(\d+)!", "compute_factorial"),
            (r"gcd|greatest common divisor", "compute_gcd"),
            (r"lcm|least common multiple", "compute_lcm"),
            (r"power|exponent|\^|raised to", "compute_power"),
            (r"sqrt|square root", "compute_sqrt"),
            (r"modulo|remainder|mod\s", "compute_modulo"),
            (r"sum|add|total|plus", "compute_sum"),
            (r"product|multiply|times", "compute_product"),
            (r"difference|subtract|minus", "compute_difference"),
            (r"quotient|divide|ratio", "compute_quotient"),
            (r"average|mean", "compute_average"),
            (r"probability", "compute_probability"),
            (r"permutation", "count_permutations"),
            (r"combination|choose|C\(", "count_combinations"),
            (r"solve.*equation|find.*root", "solve_equation"),
            (r"factor", "factor_expression"),
            (r"simplify", "simplify_expression"),
            (r"evaluate|compute|calculate", "evaluate_expression"),
            (r"area", "compute_area"),
            (r"perimeter", "compute_perimeter"),
            (r"distance", "compute_distance"),
            (r"angle", "compute_angle"),
        ]

        for pattern, step_type in patterns:
            if re.search(pattern, text_lower):
                return step_type

        return "general_step"

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def get_all_signatures(self) -> list[StepSignature]:
        """Get all signatures in the database (fast variant).

        Uses selective columns and skips expensive JSON parsing.
        For full signatures with all fields, use get_signature() by ID.
        """
        # Select only columns needed for from_row_fast()
        # Skips: centroid_bucket, embedding_sum, clarifying_questions, examples, is_archived, last_rewrite_at
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT id, signature_id, centroid, embedding_count, step_type,
                       description, param_descriptions, dsl_script, dsl_type,
                       uses, successes, operational_failures, is_semantic_umbrella,
                       is_root, depth, created_at, last_used_at
                FROM step_signatures
            """)
            return [self._row_to_signature_fast(row) for row in cursor.fetchall()]

    def get_signature_hints(
        self,
        limit: int = 20,
        problem_embedding: np.ndarray = None,
        min_similarity: float = HINT_ALTERNATIVES_MIN_SIMILARITY,
    ) -> list:
        """Get hierarchical signature hints for the decomposer.

        Returns a multi-level hierarchy:
        1. Level-1 clusters (root's children) - top-level operation categories
        2. Level-2 children per cluster - specific operation types
        3. Level-3 grandchildren (if HINT_MAX_DEPTH >= 2) - specialized variants

        This gives the planner visibility into deeper specialized operations,
        not just top-level clusters. Controlled by config:
        - HINT_MAX_DEPTH: How deep to traverse (1=level-1 only, 2+=grandchildren)
        - HINT_MAX_GRANDCHILDREN: Max grandchildren per level-2 umbrella

        PERF: Uses selective columns, batched queries, minimal JSON parsing (~3x faster).
        Only parses param_descriptions and clarifying_questions, skips examples/centroid.

        Args:
            limit: Maximum number of top-level hints to return
            problem_embedding: Optional embedding to filter by semantic similarity
            min_similarity: Minimum cosine similarity to include hint (default 0.3)

        Returns:
            List of SignatureHint objects (some with nested children for clusters)
        """
        from mycelium.planner import SignatureHint
        from mycelium.config import (
            HINT_MAX_CHILDREN_PER_CLUSTER,
            HINT_MAX_DEPTH,
            HINT_MAX_GRANDCHILDREN,
        )

        # Columns needed for hints (avoid SELECT * and full JSON parsing)
        HINT_COLUMNS = """s.id, s.step_type, s.description, s.param_descriptions,
                         s.clarifying_questions, s.dsl_script, s.centroid,
                         s.is_semantic_umbrella, s.successes, s.uses"""

        hints = []
        seen_ids = set()

        with self._connection() as conn:
            # First, get level-1 clusters (umbrellas that are children of root)
            root = self.get_root()
            if root is not None:
                # Get root's children (level-1 clusters) - selective columns
                cursor = conn.execute(
                    f"""SELECT {HINT_COLUMNS} FROM signature_relationships r
                       JOIN step_signatures s ON r.child_id = s.id
                       WHERE r.parent_id = ?
                       ORDER BY s.successes DESC, s.uses DESC
                       LIMIT ?""",
                    (root.id, limit)
                )
                level1_rows = cursor.fetchall()

                # Filter by embedding similarity if provided
                level1_rows = self._filter_rows_by_similarity(
                    level1_rows, problem_embedding, min_similarity
                )

                # Collect umbrella IDs for batched child query
                umbrella_ids = [
                    row["id"] for row in level1_rows
                    if row["is_semantic_umbrella"]
                ]

                # Batch fetch all children for all umbrellas (avoids N+1 queries)
                children_by_parent = {}
                level2_umbrella_ids = []
                if umbrella_ids:
                    placeholders = ",".join("?" * len(umbrella_ids))
                    child_cursor = conn.execute(
                        f"""SELECT r.parent_id, {HINT_COLUMNS}
                           FROM signature_relationships r
                           JOIN step_signatures s ON r.child_id = s.id
                           WHERE r.parent_id IN ({placeholders})
                           ORDER BY r.parent_id, s.successes DESC""",
                        umbrella_ids
                    )
                    for child_row in child_cursor.fetchall():
                        parent_id = child_row["parent_id"]
                        if parent_id not in children_by_parent:
                            children_by_parent[parent_id] = []
                        # Limit children per cluster
                        if len(children_by_parent[parent_id]) < HINT_MAX_CHILDREN_PER_CLUSTER:
                            children_by_parent[parent_id].append(child_row)
                            seen_ids.add(child_row["id"])
                            # Track level-2 umbrellas for deeper fetch
                            if child_row["is_semantic_umbrella"]:
                                level2_umbrella_ids.append(child_row["id"])

                # Fetch grandchildren (level-3) if depth allows
                grandchildren_by_parent = {}
                if HINT_MAX_DEPTH >= 2 and level2_umbrella_ids:
                    placeholders = ",".join("?" * len(level2_umbrella_ids))
                    gc_cursor = conn.execute(
                        f"""SELECT r.parent_id, {HINT_COLUMNS}
                           FROM signature_relationships r
                           JOIN step_signatures s ON r.child_id = s.id
                           WHERE r.parent_id IN ({placeholders})
                           ORDER BY r.parent_id, s.successes DESC""",
                        level2_umbrella_ids
                    )
                    gc_rows = gc_cursor.fetchall()

                    # Filter grandchildren by similarity if embedding provided
                    if problem_embedding is not None:
                        gc_rows = self._filter_rows_by_similarity(
                            gc_rows, problem_embedding, min_similarity
                        )

                    for gc_row in gc_rows:
                        parent_id = gc_row["parent_id"]
                        if parent_id not in grandchildren_by_parent:
                            grandchildren_by_parent[parent_id] = []
                        if len(grandchildren_by_parent[parent_id]) < HINT_MAX_GRANDCHILDREN:
                            grandchildren_by_parent[parent_id].append(gc_row)
                            seen_ids.add(gc_row["id"])

                # Build cluster hints
                for row in level1_rows:
                    sig_id = row["id"]
                    seen_ids.add(sig_id)
                    is_umbrella = bool(row["is_semantic_umbrella"])

                    # Get pre-fetched children for this cluster
                    child_hints = []
                    if is_umbrella and sig_id in children_by_parent:
                        for child_row in children_by_parent[sig_id]:
                            child_hint = self._row_to_hint(child_row)
                            # Attach grandchildren if this child is an umbrella
                            child_id = child_row["id"]
                            if child_row["is_semantic_umbrella"] and child_id in grandchildren_by_parent:
                                gc_hints = [self._row_to_hint(gc) for gc in grandchildren_by_parent[child_id]]
                                child_hint.is_cluster = len(gc_hints) > 0
                                child_hint.children = gc_hints
                            child_hints.append(child_hint)

                    hint = self._row_to_hint(row)
                    hint.is_cluster = is_umbrella and len(child_hints) > 0
                    hint.children = child_hints
                    hints.append(hint)

            # Fill remaining slots with high-quality leaf signatures not already included
            remaining = limit - len(hints)
            if remaining > 0:
                placeholders = ",".join("?" * len(seen_ids)) if seen_ids else "0"
                cursor = conn.execute(
                    f"""SELECT {HINT_COLUMNS} FROM step_signatures s
                       WHERE s.id NOT IN ({placeholders})
                       AND (s.clarifying_questions IS NOT NULL AND s.clarifying_questions != '[]'
                            OR s.param_descriptions IS NOT NULL AND s.param_descriptions != '{{}}')
                       AND s.is_semantic_umbrella = 0
                       ORDER BY s.successes DESC, s.uses DESC
                       LIMIT ?""",
                    list(seen_ids) + [remaining * 2]
                )
                leaf_rows = cursor.fetchall()

                # Filter by embedding if provided
                leaf_rows = self._filter_rows_by_similarity(
                    leaf_rows, problem_embedding, min_similarity, limit=remaining
                )

                for row in leaf_rows:
                    hints.append(self._row_to_hint(row))

        # Count nested clusters (grandchildren with children)
        nested_count = sum(
            1 for h in hints if h.is_cluster
            for c in h.children if c.is_cluster
        )
        logger.debug(
            "[db] Retrieved %d hierarchical hints (%d clusters, %d nested)",
            len(hints), sum(1 for h in hints if h.is_cluster), nested_count
        )
        return hints

    def _filter_rows_by_similarity(
        self,
        rows: list,
        problem_embedding: np.ndarray,
        min_similarity: float,
        limit: int = None,
    ) -> list:
        """Filter rows by centroid similarity to problem embedding.

        Args:
            rows: Database rows with 'centroid' column
            problem_embedding: Embedding to compare against (None = no filtering)
            min_similarity: Minimum cosine similarity threshold
            limit: Optional max rows to return after filtering

        Returns:
            Filtered (and optionally limited) list of rows
        """
        if problem_embedding is None:
            return rows[:limit] if limit else rows

        scored = []
        for row in rows:
            centroid_packed = row["centroid"]
            if centroid_packed:
                centroid = unpack_embedding(centroid_packed)
                if centroid is not None:
                    sim = cosine_similarity(problem_embedding, centroid)
                    if sim >= min_similarity:
                        scored.append((row, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [row for row, _ in scored]
        return filtered[:limit] if limit else filtered

    def _row_to_hint(self, row) -> "SignatureHint":
        """Convert a lightweight row to SignatureHint (minimal JSON parsing).

        Only parses param_descriptions and clarifying_questions.
        ~3x faster than full _row_to_signature().
        """
        from mycelium.planner import SignatureHint

        # Parse only what we need
        param_descriptions = {}
        if row["param_descriptions"]:
            try:
                param_descriptions = json.loads(row["param_descriptions"])
            except (json.JSONDecodeError, TypeError):
                pass

        clarifying_questions = []
        if row["clarifying_questions"]:
            try:
                clarifying_questions = json.loads(row["clarifying_questions"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract param names from DSL script
        param_names = []
        dsl_script = row["dsl_script"]
        if dsl_script and dsl_script.startswith('{'):
            try:
                dsl_data = json.loads(dsl_script)
                param_names = dsl_data.get('params', [])
            except (json.JSONDecodeError, TypeError):
                pass

        return SignatureHint(
            step_type=row["step_type"],
            description=row["description"] or "",
            param_names=param_names,
            param_descriptions=param_descriptions,
            clarifying_questions=clarifying_questions,
            is_cluster=False,
            children=[],
        )

    def _extract_param_names(self, sig) -> list[str]:
        """Extract parameter names from a signature's DSL spec."""
        if not sig.dsl_script:
            return []
        try:
            dsl_data = json.loads(sig.dsl_script) if sig.dsl_script.startswith('{') else {}
            return dsl_data.get('params', [])
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug("[db] Failed to parse DSL params for sig %d: %s", sig.id, e)
            return []

    # =========================================================================
    # Umbrella Routing (DAG of DAGs)
    # =========================================================================

    def get_children(
        self, parent_id: int, for_routing: bool = False, skip_cache: bool = False
    ) -> list[tuple[StepSignature, str]]:
        """Get child signatures for an umbrella parent.

        Uses LRU cache with TTL to skip DB for hot umbrella nodes during routing.

        Args:
            parent_id: ID of the parent signature
            for_routing: If True, use fast parsing (centroid only, skip JSON).
                        Per CLAUDE.md: "Umbrella routing should not require LLM call"
            skip_cache: If True, bypass cache and query DB directly.
                       Use for critical checks like "umbrella has no children"
                       in multiprocess environments where cache may be stale.

        Returns:
            List of (child_signature, condition) tuples, ordered by routing_order
        """
        # Check cache first (unless explicitly skipped)
        if not skip_cache:
            cached = get_cached_children(parent_id, for_routing)
            if cached is not None:
                return cached

        # Cache miss - fetch from DB (exclude archived signatures)
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT s.*, r.condition
                   FROM signature_relationships r
                   JOIN step_signatures s ON r.child_id = s.id
                   WHERE r.parent_id = ?
                     AND s.is_archived = 0
                   ORDER BY r.routing_order ASC""",
                (parent_id,)
            )
            results = []
            row_converter = (
                self._row_to_signature_for_routing if for_routing
                else self._row_to_signature
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                condition = row_dict.pop("condition")
                sig = row_converter(row_dict)
                results.append((sig, condition))

            # Cache the result
            cache_children(parent_id, results, for_routing)
            return results

    def get_parent(self, child_id: int) -> Optional[StepSignature]:
        """Get the parent signature for a child (tree structure - single parent).

        Args:
            child_id: ID of the child signature

        Returns:
            Parent signature or None if no parent (root node)
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT s.*
                   FROM signature_relationships r
                   JOIN step_signatures s ON r.parent_id = s.id
                   WHERE r.child_id = ?""",
                (child_id,)
            )
            row = cursor.fetchone()
            return self._row_to_signature(dict(row)) if row else None

    def _find_best_sibling(
        self,
        new_embedding: np.ndarray,
        parent_id: int,
        conn=None,
    ) -> tuple[Optional["StepSignature"], float]:
        """Find the best matching sibling (child of same parent) by graph_embedding similarity.

        Per mycelium-br28: Helper for decide_signature_placement().
        Compares new_embedding against all children of parent_id that have graph_embeddings.

        Args:
            new_embedding: The new signature's graph_embedding
            parent_id: The parent's signature ID (siblings are children of this parent)
            conn: Optional database connection

        Returns:
            (best_sibling, similarity) tuple, or (None, 0.0) if no siblings with embeddings
        """
        def _do_find(c):
            # Get all children of parent with graph_embeddings
            cursor = c.execute(
                """SELECT s.id, s.step_type, s.graph_embedding, s.is_semantic_umbrella,
                          s.uses, s.successes, s.depth
                   FROM step_signatures s
                   JOIN signature_relationships r ON s.id = r.child_id
                   WHERE r.parent_id = ?
                     AND s.graph_embedding IS NOT NULL
                     AND s.is_archived = 0""",
                (parent_id,)
            )

            best_sig = None
            best_sim = 0.0

            for row in cursor:
                sig_id, step_type, graph_emb_json, is_umbrella, uses, successes, depth = row
                sig_graph_emb = np.array(json.loads(graph_emb_json))

                sim = cosine_similarity(new_embedding, sig_graph_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_sig = StepSignature(
                        id=sig_id,
                        step_type=step_type,
                        graph_embedding=json.loads(graph_emb_json),
                        is_semantic_umbrella=bool(is_umbrella),
                        uses=uses,
                        successes=successes,
                        depth=depth,
                    )

            return best_sig, best_sim

        if conn is not None:
            return _do_find(conn)
        else:
            with self._connection() as c:
                return _do_find(c)

    def get_all_leaves(self, min_uses: int = 0) -> list[StepSignature]:
        """Get all leaf signatures (non-umbrellas) with graph embeddings.

        Used for MCTS-style leaf matching where we want to find the best
        leaf for a dag_step regardless of tree routing path.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Filters on graph_embedding (not text centroid).

        Args:
            min_uses: Minimum use count to include (filters cold signatures)

        Returns:
            List of leaf signatures with graph_embeddings
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT *
                   FROM step_signatures
                   WHERE is_semantic_umbrella = 0
                     AND is_archived = 0
                     AND graph_embedding IS NOT NULL
                     AND uses >= ?
                   ORDER BY uses DESC""",
                (min_uses,)
            )
            return [
                self._row_to_signature_for_routing(dict(row))
                for row in cursor.fetchall()
            ]

    def match_step_to_leaves_mcts(
        self,
        operation_embedding: np.ndarray,
        dag_step_type: str = None,
        top_k: int = 3,
        min_similarity: float = ROUTING_MIN_SIMILARITY_PERMISSIVE,
        use_adaptive_threshold: bool = None,  # None = use config default
    ) -> list[tuple[StepSignature, float, float]]:
        """MCTS-style matching: find top-k leaf candidates for a dag_step using graph embeddings.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        Uses graph_embedding (not text centroid) for matching.

        Instead of routing through tree hierarchy, directly scores all leaves
        using UCB1 to balance:
        - Exploitation: similarity + success rate
        - Exploration: bonus for under-visited leaves

        Per mycelium-i601: Uses adaptive rejection thresholds per leaf.
        Each leaf computes its own threshold based on historical success similarities:
        threshold = mean - 1.5 * std. Cold-start leaves use min_similarity as fallback.

        Per CLAUDE.md: Returns top-k candidates so caller can:
        1. Try best match first
        2. On rejection, try alternatives (sideways)
        3. Only decompose if ALL reject (depth)

        Args:
            operation_embedding: Operation embedding to match (from step.operation)
            dag_step_type: Optional step type for step-node stats lookup
            top_k: Number of candidates to return (default 3)
            min_similarity: Fallback threshold for cold-start leaves
            use_adaptive_threshold: If True, use per-leaf adaptive thresholds

        Returns:
            List of (leaf, ucb1_score, similarity) tuples, sorted by UCB1 desc
        """
        from mycelium.data_layer.mcts import get_dag_step_node_stats_batch
        from mycelium.config import ADAPTIVE_REJECTION_ENABLED

        # Resolve use_adaptive_threshold: None means use config default
        if use_adaptive_threshold is None:
            use_adaptive_threshold = ADAPTIVE_REJECTION_ENABLED

        leaves = self.get_all_leaves(min_uses=0)
        if not leaves:
            return []

        # Get step-node stats for all leaves if dag_step_type provided
        step_stats_map = {}
        if dag_step_type:
            leaf_ids = [leaf.id for leaf in leaves if leaf.id is not None]
            step_stats_map = get_dag_step_node_stats_batch(dag_step_type, leaf_ids)

        # Estimate total visits for exploration bonus
        total_visits = sum(leaf.uses or 1 for leaf in leaves)

        candidates = []
        rejections = 0
        for leaf in leaves:
            # Use graph_embedding for matching (operational similarity)
            graph_emb = leaf.graph_embedding
            if graph_emb is None:
                continue
            if not isinstance(graph_emb, np.ndarray):
                graph_emb = np.array(graph_emb)

            sim = cosine_similarity(operation_embedding, graph_emb)

            # Per mycelium-i601: Use adaptive threshold based on leaf's historical successes
            # Cold-start leaves (few samples) fall back to min_similarity
            if use_adaptive_threshold:
                from mycelium.config import (
                    ADAPTIVE_REJECTION_K,
                    ADAPTIVE_REJECTION_MIN_SAMPLES,
                )
                threshold = leaf.get_adaptive_rejection_threshold(
                    k=ADAPTIVE_REJECTION_K,
                    min_samples=ADAPTIVE_REJECTION_MIN_SAMPLES,
                    default_threshold=min_similarity,
                )
            else:
                threshold = min_similarity

            if sim < threshold:
                rejections += 1
                continue

            # Get step-node stats for this leaf
            leaf_step_stats = step_stats_map.get(leaf.id)

            ucb1 = compute_ucb1_score(
                cosine_sim=sim,
                uses=leaf.uses or 0,
                successes=leaf.successes or 0,
                parent_uses=total_visits,
                last_used_at=leaf.last_used_at,
                step_node_stats=leaf_step_stats,
            )
            candidates.append((leaf, ucb1, sim))

        # Sort by UCB1 score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            logger.debug(
                "[mcts_match] Top-%d leaves for '%s': %s (rejected %d via adaptive threshold)",
                top_k,
                dag_step_type or "unknown",
                [(c[0].step_type, f"ucb1={c[1]:.3f}", f"sim={c[2]:.3f}") for c in candidates[:top_k]],
                rejections,
            )
        elif rejections > 0:
            logger.debug(
                "[mcts_match] All %d leaves rejected for '%s' via adaptive threshold",
                rejections,
                dag_step_type or "unknown",
            )

        return candidates[:top_k]

    def get_decomposition_hints(
        self,
        step_description: str,
        operation_embedding: np.ndarray,
        similarity_threshold: Optional[float] = None,
        step_position: Optional[int] = None,
    ) -> dict:
        """Get hints for decomposing a dag_step using existing vocabulary.

        Per CLAUDE.md "Negotiation between Tree and Planner":
        Bias towards decomposing dag_steps (cheap, per-problem) over
        decomposing leaf_nodes (permanent tree change).

        Per CLAUDE.md "Cluster Boundaries" & "System Independence":
        Uses adaptive z-score thresholds from Welford stats, not hard-coded values.
        similarity_threshold: If None, uses adaptive cluster_threshold from global match stats.
        Rejection based on: similarity + historical (node, position) performance.

        When a dag_step has a poor match, this method suggests how to
        break it down into operations that already exist in the tree.

        Args:
            step_description: The step that needs decomposition
            operation_embedding: The step's operation embedding
            similarity_threshold: Below this, step is considered "poor match"
            step_position: Position in plan (1, 2, 3...) for position-aware stats

        Returns:
            Dict with:
            - 'needs_decomposition': bool
            - 'best_match': (step_type, similarity, node_id) or None
            - 'vocabulary': list of available operations
            - 'suggested_decomposition': list of operation names that might compose this step
            - 'rejection_reason': why decomposition is needed (if applicable)
        """
        # Defensive check: validate embedding dimension
        from mycelium.config import EMBEDDING_DIM
        if operation_embedding is None or len(operation_embedding.shape) == 0:
            logger.warning("[decomp_hints] Invalid embedding: None or scalar")
            return {
                'needs_decomposition': True,
                'best_match': None,
                'vocabulary': [],
                'suggested_decomposition': [],
                'rejection_reason': 'invalid_embedding',
            }
        if operation_embedding.shape[0] != EMBEDDING_DIM:
            logger.warning(
                "[decomp_hints] Embedding dimension mismatch: got %s, expected %d",
                operation_embedding.shape, EMBEDDING_DIM
            )
            return {
                'needs_decomposition': True,
                'best_match': None,
                'vocabulary': [],
                'suggested_decomposition': [],
                'rejection_reason': f'embedding_dim_mismatch_{operation_embedding.shape[0]}',
            }

        # Per CLAUDE.md "System Independence": Compute adaptive thresholds from Welford stats
        # instead of hard-coded magic numbers
        with self._connection() as conn:
            adaptive = get_adaptive_thresholds(conn)
            exec_success_floor = get_global_exec_success_floor(conn)

        # Use adaptive cluster_threshold if no explicit threshold provided
        if similarity_threshold is None:
            similarity_threshold = adaptive.cluster_threshold
            logger.debug(
                "[decomp_hints] Using adaptive thresholds: similarity=%.3f, exec_floor=%.3f (match_count=%d)",
                similarity_threshold, exec_success_floor, adaptive.match_count
            )

        leaves = self.get_all_leaves(min_uses=0)

        # Find best match
        best_match = None
        best_sim = 0.0
        best_node_id = None

        # Collect vocabulary and similarities
        vocabulary = []
        for leaf in leaves:
            graph_emb = leaf.graph_embedding
            if graph_emb is None:
                continue
            if not isinstance(graph_emb, np.ndarray):
                graph_emb = np.array(graph_emb)

            sim = cosine_similarity(operation_embedding, graph_emb)
            vocabulary.append((leaf.step_type, sim, leaf.uses or 0, leaf.id))

            if sim > best_sim:
                best_sim = sim
                best_match = (leaf.step_type, sim, leaf.id)
                best_node_id = leaf.id

        # Sort vocabulary by similarity (descending)
        vocabulary.sort(key=lambda x: x[1], reverse=True)

        # === ADAPTIVE Z-SCORE REJECTION LOGIC ===
        needs_decomposition = False
        rejection_reason = None

        # 1. Similarity-based rejection (baseline)
        if best_sim < similarity_threshold:
            needs_decomposition = True
            rejection_reason = f"low_similarity ({best_sim:.3f} < {similarity_threshold})"

        # 2. Welford-based rejection (adaptive thresholds)
        elif best_node_id is not None and step_position is not None:
            # Get stats for this (node, position) pair
            position_stats = self.get_node_position_stats(best_node_id, step_position)

            if position_stats and position_stats["n"] >= 5:
                # Enough observations for reliable z-score

                # Method A: Self-comparison (node vs its own history)
                node_overall = self.get_node_stats_all_positions(best_node_id)
                if node_overall["n"] >= 5 and node_overall["std"] > 0.01:
                    # Z-score: how does this position compare to node's overall performance?
                    z_self = (position_stats["mean_success"] - node_overall["mean_success"]) / node_overall["std"]
                    if z_self < -2.0:
                        # This node performs 2+ std worse at this position than usual
                        needs_decomposition = True
                        rejection_reason = f"position_underperform (z={z_self:.2f}, pos_success={position_stats['mean_success']:.2f})"

                # Method B: Cluster comparison (node vs siblings)
                if not needs_decomposition:
                    cluster_stats = self.get_cluster_stats(best_node_id)
                    if cluster_stats["sibling_count"] >= 3 and cluster_stats["cluster_std"] > 0.01:
                        # Z-score: how does this node compare to siblings?
                        z_cluster = (position_stats["mean_success"] - cluster_stats["cluster_mean"]) / cluster_stats["cluster_std"]
                        if z_cluster < -2.0:
                            # This node performs 2+ std worse than sibling average
                            needs_decomposition = True
                            rejection_reason = f"cluster_underperform (z={z_cluster:.2f}, cluster_mean={cluster_stats['cluster_mean']:.2f})"

                # Method C: Coefficient of Variation check (adaptive instability)
                # Per CLAUDE.md "Cluster Boundaries": Adaptive thresholds, not hard-coded
                # CV = std / mean; CV > 1.0 means std larger than mean = very unstable
                if not needs_decomposition and position_stats["mean_success"] > 0.1:
                    cv = position_stats["std"] / position_stats["mean_success"]
                    if cv > 1.0:
                        # Results are all over the place - node is unreliable at this position
                        needs_decomposition = True
                        rejection_reason = f"high_cv (cv={cv:.2f}, std={position_stats['std']:.2f}, mean={position_stats['mean_success']:.2f})"

                # Method D: Adaptive failure rate floor (per CLAUDE.md "System Independence")
                # Uses global exec success distribution: floor = mean - 2*std
                if not needs_decomposition and position_stats["mean_success"] < exec_success_floor:
                    needs_decomposition = True
                    rejection_reason = f"high_failure_rate ({position_stats['mean_success']:.2f} < floor={exec_success_floor:.2f})"

        # Suggest decomposition based on existing vocabulary
        suggested_decomposition = []
        if needs_decomposition:
            # Top operations by similarity might be components
            for step_type, sim, uses, node_id in vocabulary[:5]:
                if sim > 0.4:  # Minimum relevance
                    suggested_decomposition.append(step_type)

        result = {
            'needs_decomposition': needs_decomposition,
            'best_match': best_match,
            'vocabulary': [(v[0], v[2]) for v in vocabulary[:10]],  # (step_type, uses)
            'suggested_decomposition': suggested_decomposition,
            'similarity_threshold': similarity_threshold,
            'rejection_reason': rejection_reason,
        }

        if needs_decomposition:
            logger.info(
                "[db] Decomposition hints for '%s': reason=%s, best_match=%s (sim=%.3f)",
                step_description[:40],
                rejection_reason,
                best_match[0] if best_match else None,
                best_sim,
            )

        return result

    def add_child(
        self,
        parent_id: int,
        child_id: int,
        condition: str,
        routing_order: int = 0,
    ) -> bool:
        """Add a parent-child relationship (tree structure - single parent per child).

        Args:
            parent_id: ID of the parent signature
            child_id: ID of the child signature
            condition: Routing condition (e.g., "counting outcomes")
            routing_order: Priority (lower = higher priority)

        Returns:
            True if relationship was created, False if child already has a parent
        """
        # Prevent self-references
        if parent_id == child_id:
            logger.warning("[db] Rejecting self-reference: parent_id=%d == child_id=%d", parent_id, child_id)
            return False

        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            # Check if child already has a parent (tree structure - single parent)
            existing = conn.execute(
                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                (child_id,)
            ).fetchone()

            if existing:
                if existing["parent_id"] == parent_id:
                    logger.debug("[db] Relationship already exists: parent=%d → child=%d", parent_id, child_id)
                else:
                    logger.warning(
                        "[db] Child %d already has parent %d, rejecting new parent %d (tree structure)",
                        child_id, existing["parent_id"], parent_id
                    )
                return False

            # Cycle prevention: check if parent is already a descendant of child
            cycle_check = conn.execute(
                """
                WITH RECURSIVE ancestors AS (
                    SELECT parent_id FROM signature_relationships WHERE child_id = ?
                    UNION ALL
                    SELECT sr.parent_id
                    FROM signature_relationships sr
                    JOIN ancestors a ON sr.child_id = a.parent_id
                )
                SELECT 1 FROM ancestors WHERE parent_id = ? LIMIT 1
                """,
                (parent_id, child_id),
            ).fetchone()

            if cycle_check:
                logger.warning(
                    "[db] Rejecting cycle-creating relationship: %d -> %d (child is ancestor of parent)",
                    parent_id, child_id
                )
                return False

            # Get parent's depth to set child's depth
            parent_row = conn.execute(
                "SELECT depth FROM step_signatures WHERE id = ?",
                (parent_id,)
            ).fetchone()
            parent_depth = parent_row["depth"] if parent_row and parent_row["depth"] else 0

            conn.execute(
                """INSERT INTO signature_relationships
                   (parent_id, child_id, condition, routing_order, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (parent_id, child_id, condition, routing_order, now),
            )
            # Use single pathway for umbrella promotion (handles centroid propagation)
            self._promote_to_umbrella_internal(conn, parent_id)

            # Set child's depth = parent_depth + 1
            child_depth = parent_depth + 1
            self._update_signature_fields(
                conn, child_id,
                log_reason="set_child_depth",
                depth=child_depth,
            )
            # Invalidate caches (relationship change: new child added)
            self._invalidate_on_relationship_change(parent_id, child_id)

            logger.info(
                "[db] Added child: parent=%d (depth=%d) → child=%d (depth=%d) (condition='%s')",
                parent_id, parent_depth, child_id, child_depth, condition[:30]
            )
            return True

    def _promote_to_umbrella_internal(self, conn, signature_id: int, skip_children_check: bool = False) -> bool:
        """Internal: Mark signature as umbrella within existing transaction.

        This is the SINGLE PATHWAY for umbrella promotion. All code that needs
        to mark a signature as umbrella should call this function.

        Per CLAUDE.md System Independence: Don't create umbrellas without children.
        This function validates children exist before promoting (unless skip_children_check=True
        for cases where children are being added in the same transaction).

        Per mycelium-5cn0: No umbrella promotions during cold start. The first N problems
        create a flat structure under root to collect Welford stats before restructuring.

        Args:
            conn: Database connection (existing transaction)
            signature_id: ID of the signature to promote
            skip_children_check: If True, skip validation (caller guarantees children exist/will exist)

        Returns:
            True if updated, False if signature not found, no children, or cold start active
        """
        # Cold start check: no umbrella promotions during cold start (per mycelium-5cn0)
        if self.is_cold_start(conn=conn):
            logger.debug(
                "[db] Refusing to promote sig %d to umbrella: cold start active (flat structure)",
                signature_id
            )
            return False

        # Validate children exist (unless caller explicitly skips check)
        if not skip_children_check:
            child_count = conn.execute(
                "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                (signature_id,)
            ).fetchone()[0]
            if child_count == 0:
                logger.warning(
                    "[db] Refusing to promote sig %d to umbrella: no children (would create orphan)",
                    signature_id
                )
                return False

        result = self._update_signature_fields(
            conn, signature_id,
            log_reason="promote_to_umbrella",
            is_semantic_umbrella=1,
            dsl_type="router",
            dsl_script=None,
        )
        if result:
            # Use consolidated centroid update pathway:
            # include_self=True computes this node's centroid, then propagates to ancestors
            self.propagate_graph_centroid_to_parents(conn, signature_id, include_self=True)
            logger.debug(
                "[db] Promoted signature %d to umbrella with graph_centroid",
                signature_id
            )
            return True
        return False

    def promote_to_umbrella(self, signature_id: int) -> bool:
        """Mark a signature as a semantic umbrella (pure router, no DSL).

        Umbrellas are routers that dispatch to child signatures based on
        semantic similarity. They don't execute DSLs directly - their job
        is to route problems to the right specialized child.

        Per Option B (tlax): Routers use graph_centroid (avg of children's
        graph_embeddings) for graph-space routing.

        Args:
            signature_id: ID of the signature to promote

        Returns:
            True if updated, False if signature not found
        """
        with self._connection() as conn:
            result = self._promote_to_umbrella_internal(conn, signature_id)
            if result:
                logger.info(
                    "[db] Promoted signature %d to umbrella with graph_centroid",
                    signature_id
                )
            return result

    def find_deeper_signature(
        self,
        embedding: np.ndarray,
        min_depth: int,
        min_similarity: float = PLACEMENT_MIN_SIMILARITY,
        exclude_ids: set[int] = None,
    ) -> Optional[StepSignature]:
        """Find existing signature at deeper depth for repointing.

        When decomposing a parent into children, prefer repointing to existing
        deeper signatures over creating new ones. This reduces fragmentation
        and reuses learned knowledge.

        Args:
            embedding: Query embedding to match against
            min_depth: Minimum depth required (parent_depth + 1)
            min_similarity: Minimum cosine similarity threshold
            exclude_ids: Signature IDs to exclude (e.g., parent itself)

        Returns:
            Best matching deeper signature, or None if no suitable match
        """
        from mycelium.step_signatures.utils import cosine_similarity, unpack_embedding

        exclude_ids = exclude_ids or set()

        with self._connection() as conn:
            # Get signatures at required depth or deeper (not umbrellas - we want leaf executors)
            cursor = conn.execute(
                """SELECT * FROM step_signatures
                   WHERE depth >= ?
                   AND is_semantic_umbrella = 0""",
                (min_depth,)
            )
            rows = cursor.fetchall()

        best_match = None
        best_score = 0.0

        for row in rows:
            sig_id = row["id"]
            if sig_id in exclude_ids:
                continue

            if not row["centroid"]:
                continue
            # Use cached centroid to avoid repeated JSON parsing
            centroid = get_cached_centroid(sig_id, row["centroid"])
            if centroid is None:
                continue

            sim = cosine_similarity(embedding, centroid)
            if sim < min_similarity:
                continue

            # Use routing score (similarity + success rate - staleness)
            uses = row["uses"] or 0
            successes = row["successes"] or 0
            last_used_at = row["last_used_at"]
            score = compute_routing_score(sim, uses, successes, last_used_at)

            if score > best_score:
                best_match = self._row_to_signature(row)
                best_score = score

        if best_match:
            logger.info(
                "[db] Found deeper signature for repoint: id=%d depth=%d sim=%.3f",
                best_match.id, best_match.depth, best_score
            )

        return best_match

    def create_upward_umbrella(
        self,
        child_signature: StepSignature,
        problem_embedding: np.ndarray,
        difficulty: float,
        description: str,
    ) -> Optional[StepSignature]:
        """Create a new umbrella above an existing signature for upward restructuring.

        This is called when detect_upward_restructuring identifies that a new
        problem represents a higher abstraction level than an existing signature.

        This is a thin wrapper around _create_signature_atomic with is_umbrella=True
        and skip_parent_relationship=True, then manually adds the child relationship.

        The new umbrella:
        1. Becomes the parent of the existing signature
        2. Has a centroid averaged from the existing sig and new problem
        3. Has a higher depth threshold for routing

        Args:
            child_signature: The existing signature to place under the new umbrella
            problem_embedding: Embedding of the harder problem
            difficulty: Difficulty of the harder problem
            description: Description for the new umbrella

        Returns:
            The new umbrella signature, or None on failure
        """
        # Create averaged centroid from child + new problem
        if child_signature.centroid is not None:
            new_centroid = (child_signature.centroid + problem_embedding) / 2
        else:
            new_centroid = problem_embedding

        sig_id = f"umbrella_{uuid.uuid4().hex[:8]}"
        target_depth = max(0, child_signature.depth - 1)

        with self._connection() as conn:
            now = datetime.now(timezone.utc).isoformat()

            try:
                # Per CLAUDE.md System Independence: Create as LEAF first, then promote
                # to umbrella AFTER child relationship exists (prevents orphan umbrellas)
                new_sig = self._create_signature_atomic(
                    conn=conn,
                    step_text=description,
                    embedding=new_centroid,
                    is_umbrella=False,  # Start as leaf, promote after child added
                    signature_id_override=sig_id,
                    depth_override=target_depth,
                    skip_example=True,
                    skip_parent_relationship=True,  # We'll add child manually
                )

                # Create parent-child relationship (new umbrella is PARENT of child_signature)
                conn.execute(
                    """INSERT INTO signature_relationships (parent_id, child_id, condition, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (new_sig.id, child_signature.id, f"difficulty <= {child_signature.max_difficulty_solved}", now),
                )

                # NOW promote to umbrella (child exists, safe from orphan state)
                self._promote_to_umbrella_internal(conn, new_sig.id)

                # Update max_difficulty_solved
                conn.execute(
                    "UPDATE step_signatures SET max_difficulty_solved = ? WHERE id = ?",
                    (difficulty, new_sig.id),
                )

                # Invalidate caches
                self._invalidate_on_relationship_change(new_sig.id, child_signature.id)

                logger.info(
                    "[db] Created upward umbrella: id=%d, child=%d, depth=%d, max_diff=%.2f",
                    new_sig.id, child_signature.id, target_depth, difficulty
                )

                # Refresh signature object (is_semantic_umbrella changed)
                new_sig = new_sig._replace(is_semantic_umbrella=True, dsl_type="router")
                return new_sig

            except sqlite3.IntegrityError as e:
                logger.warning("[db] Failed to create upward umbrella: %s", e)
                return None

    def remove_child(self, parent_id: int, child_id: int) -> bool:
        """Remove a parent-child relationship.

        Per CLAUDE.md "New Favorite Pattern": Uses demote_umbrella_to_leaf() if
        parent becomes orphaned after child removal.

        Args:
            parent_id: ID of the parent signature
            child_id: ID of the child signature

        Returns:
            True if relationship was removed
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM signature_relationships WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            )
            if cursor.rowcount > 0:
                # Check if parent still has children
                remaining_row = conn.execute(
                    "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                    (parent_id,),
                ).fetchone()
                remaining = remaining_row[0] if remaining_row else 0
                if remaining == 0:
                    # Demote from umbrella using consolidated pathway
                    self.demote_umbrella_to_leaf(
                        parent_id,
                        reason="last_child_removed",
                        dsl_type="decompose",  # Keep decompose since it was an umbrella
                        conn=conn,
                    )
                # Invalidate caches (relationship change: child removed)
                self._invalidate_on_relationship_change(parent_id, child_id)
                logger.info("[db] Removed child relationship: parent=%d → child=%d", parent_id, child_id)
                return True
            return False

    def demote_orphan_umbrellas(self) -> int:
        """Demote umbrellas with no children back to leaves.

        Per CLAUDE.md "New Favorite Pattern": Uses demote_umbrella_to_leaf() as
        the single pathway for all demotion operations.

        Per CLAUDE.md: An umbrella is a router that routes to children.
        If an umbrella has no children, it's a broken state - demote it back to leaf.

        Returns:
            Number of umbrellas demoted
        """
        with self._connection() as conn:
            # Find orphan umbrellas (umbrellas with no children, not archived)
            cursor = conn.execute("""
                SELECT s.id, s.step_type
                FROM step_signatures s
                WHERE s.is_semantic_umbrella = 1
                  AND s.is_archived = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM signature_relationships r WHERE r.parent_id = s.id
                  )
            """)
            orphans = cursor.fetchall()

            if not orphans:
                logger.info("[db] No orphan umbrellas found")
                return 0

            # Demote each orphan using consolidated pathway
            demoted = 0
            for sig_id, step_type in orphans:
                if self.demote_umbrella_to_leaf(
                    sig_id,
                    reason=f"orphan_umbrella:{step_type}",
                    dsl_type="math",  # Orphans default to math for execution
                    conn=conn,
                ):
                    demoted += 1

            logger.info("[db] Demoted %d orphan umbrellas back to leaves", demoted)
            return demoted

    def _demote_if_orphan(self, conn, signature_id: int) -> bool:
        """Demote a specific signature to leaf if it's an orphan umbrella.

        Per CLAUDE.md "New Favorite Pattern": Uses demote_umbrella_to_leaf() as
        the single pathway for all demotion operations.

        Called when global dedup redirects to a different parent, leaving
        a scaffold branch (created during routing) without children.

        Args:
            conn: Database connection
            signature_id: ID of signature to check and potentially demote

        Returns:
            True if signature was demoted, False otherwise
        """
        if signature_id is None:
            return False

        # Check if it's an orphan umbrella (umbrella with no children)
        cursor = conn.execute("""
            SELECT s.is_semantic_umbrella, s.step_type,
                   EXISTS(SELECT 1 FROM signature_relationships r WHERE r.parent_id = s.id) as has_children
            FROM step_signatures s
            WHERE s.id = ?
        """, (signature_id,))
        row = cursor.fetchone()

        if not row:
            return False

        is_umbrella = row[0]
        step_type = row[1]
        has_children = row[2]

        # Demote if it's an umbrella with no children using consolidated pathway
        if is_umbrella and not has_children:
            return self.demote_umbrella_to_leaf(
                signature_id,
                reason=f"abandoned_scaffold:{step_type}",
                dsl_type="math",  # Abandoned scaffolds default to math
                conn=conn,
            )

        return False

    def _finalize_routing_result(
        self,
        conn,
        result_sig: "StepSignature",
        was_created: bool,
        parent_for_new: Optional["StepSignature"],
        actual_parent_id: Optional[int],
    ) -> tuple["StepSignature", bool]:
        """Finalize routing: demote orphan scaffolds if unused, then commit.

        Per CLAUDE.md "New Favorite Pattern": consolidate method calls for features
        to simplify codebase and reduce bugs. This is the SINGLE exit point for
        routing results that need orphan cleanup.

        Args:
            conn: Database connection
            result_sig: The signature being returned
            was_created: Whether result_sig was newly created
            parent_for_new: The parent from routing (may be orphan scaffold)
            actual_parent_id: The parent actually used (may differ from parent_for_new)

        Returns:
            Tuple of (result_sig, was_created)
        """
        # Demote orphan scaffold if routing created one but we used a different parent
        if parent_for_new is not None:
            used_routing_parent = (actual_parent_id is not None and
                                   actual_parent_id == parent_for_new.id)
            if not used_routing_parent:
                self._demote_if_orphan(conn, parent_for_new.id)

        conn.commit()
        return result_sig, was_created

    def clear_all_data(self, force: bool = False) -> dict:
        """Clear all signature data for a fresh start.

        Args:
            force: If True, bypass DB_PROTECTED check

        Returns:
            Dict with counts of deleted rows

        Raises:
            RuntimeError: If DB_PROTECTED=True and force=False
        """
        from mycelium.config import DB_PROTECTED

        if DB_PROTECTED and not force:
            raise RuntimeError(
                "Database is protected (DB_PROTECTED=True). "
                "Set MYCELIUM_DB_PROTECTED=false or pass force=True to clear. "
                "This protection exists because the DB contains valuable learned data."
            )

        with self._connection() as conn:
            # Get counts before deletion (defensive None checks for race conditions)
            sig_row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
            sig_count = sig_row[0] if sig_row else 0
            ex_row = conn.execute("SELECT COUNT(*) FROM step_examples").fetchone()
            ex_count = ex_row[0] if ex_row else 0
            log_row = conn.execute("SELECT COUNT(*) FROM step_usage_log").fetchone()
            log_count = log_row[0] if log_row else 0
            if self._table_exists(conn, "signature_relationships"):
                rel_row = conn.execute("SELECT COUNT(*) FROM signature_relationships").fetchone()
                rel_count = rel_row[0] if rel_row else 0
            else:
                rel_count = 0

            # Delete in order (relationships first due to FK constraints)
            if self._table_exists(conn, "signature_relationships"):
                conn.execute("DELETE FROM signature_relationships")
            conn.execute("DELETE FROM step_usage_log")
            conn.execute("DELETE FROM step_examples")
            conn.execute("DELETE FROM step_signatures")

            # Invalidate all caches
            self.invalidate_centroid_matrix()
            self.invalidate_root_cache()

            logger.warning(
                "[db] Cleared all data: signatures=%d examples=%d usage_log=%d relationships=%d",
                sig_count, ex_count, log_count, rel_count
            )
            return {
                "signatures_deleted": sig_count,
                "examples_deleted": ex_count,
                "usage_log_deleted": log_count,
                "relationships_deleted": rel_count,
            }

    def _table_exists(self, conn, table_name: str) -> bool:
        """Check if a table exists in the database."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None

    # =========================================================================
    # Graph Embedding Methods
    # Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE
    # =========================================================================

    def update_graph_embedding(
        self, signature_id: int, graph_embedding: list[float]
    ) -> bool:
        """Update a signature's graph_embedding.

        Called after async embedding computation to store the result.

        Args:
            signature_id: ID of the signature
            graph_embedding: Embedding vector (will be JSON-serialized)

        Returns:
            True if updated, False if signature not found
        """
        embedding_json = json.dumps(graph_embedding)

        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE step_signatures SET graph_embedding = ? WHERE id = ?",
                (embedding_json, signature_id),
            )
            if cursor.rowcount > 0:
                self._invalidate_on_dsl_change(signature_id)
                logger.debug("[db] Updated graph_embedding for sig %d", signature_id)
                # Propagate centroid update to parent routers
                self.propagate_graph_centroid_to_parents(conn, signature_id)
                return True
            return False

    def get_signatures_needing_graph_embedding(
        self, limit: int = 100
    ) -> list[tuple[int, str]]:
        """Get signatures that have computation_graph but no graph_embedding.

        Used for batch population of graph embeddings.

        Args:
            limit: Maximum number of signatures to return

        Returns:
            List of (signature_id, computation_graph) tuples
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT id, computation_graph
                   FROM step_signatures
                   WHERE computation_graph IS NOT NULL
                     AND computation_graph != ''
                     AND (graph_embedding IS NULL OR graph_embedding = '')
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [(row["id"], row["computation_graph"]) for row in rows]

    def route_by_graph_embedding(
        self,
        operation_embedding: np.ndarray,
        min_similarity: float = PLACEMENT_MIN_SIMILARITY,
        top_k: int = 5,
    ) -> list[tuple[StepSignature, float]]:
        """Route by comparing operation embedding to graph embeddings.

        Per Option B (tlax): ALL routing happens in graph space.
        - Routers have graph_centroid (avg of children's graph_embeddings)
        - Leaves have graph_embedding (their computation graph embedded)
        - Routes by what operations DO, not what they SOUND LIKE

        Args:
            operation_embedding: Embedding of the extracted operation
            min_similarity: Minimum cosine similarity threshold
            top_k: Maximum number of matches to return

        Returns:
            List of (signature, similarity) tuples, sorted by similarity descending
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

        return self._route_by_graph_hierarchical(
            operation_embedding, min_similarity, top_k, UMBRELLA_MAX_DEPTH
        )

    def _route_by_graph_hierarchical(
        self,
        operation_embedding: np.ndarray,
        min_similarity: float,
        top_k: int,
        max_depth: int,
    ) -> list[tuple[StepSignature, float]]:
        """Hierarchical graph routing through routers and leaves.

        Traverses the tree using graph_embeddings at each level:
        - Routers: compare against graph_centroid (avg of children)
        - Leaves: compare against graph_embedding (computation graph)
        """
        root = self.get_root()
        if root is None:
            return []

        # BFS through tree, collecting leaf matches
        matches = []
        queue = [(root, 1.0)]  # (signature, accumulated_sim)
        visited = set()

        while queue and len(matches) < top_k * 2:  # Collect extra, then trim
            current, parent_sim = queue.pop(0)

            if current.id in visited:
                continue
            visited.add(current.id)

            # Get graph_embedding for this node
            graph_emb = current.graph_embedding
            if graph_emb is None:
                # Try to load it if not cached
                with self._connection() as conn:
                    row = conn.execute(
                        "SELECT graph_embedding FROM step_signatures WHERE id = ?",
                        (current.id,)
                    ).fetchone()
                    if row and row["graph_embedding"]:
                        try:
                            graph_emb = np.array(json.loads(row["graph_embedding"]))
                        except (json.JSONDecodeError, ValueError):
                            pass

            if graph_emb is None:
                # No graph_embedding - skip or use children directly
                if current.is_semantic_umbrella:
                    children = self.get_children(current.id, for_routing=True)
                    for child_sig, _condition in children:
                        queue.append((child_sig, parent_sim))
                continue

            # Compute similarity
            sim = cosine_similarity(operation_embedding, graph_emb)

            if current.is_semantic_umbrella:
                # Per CLAUDE.md "System Independence": Welford-guided adaptive threshold
                # Routers with high route_mean are selective - use higher threshold
                router_threshold = min_similarity * 0.8  # Default
                with self._connection() as conn:
                    stats = self.get_welford_stats(current.id, conn=conn)
                    if stats and stats.get("route_n", 0) >= 5:
                        route_mean = stats.get("route_mean", 0.0)
                        route_std = self._welford_std_from_stats(stats, "route")
                        # Adaptive threshold: route_mean - 1.5*std (accept if within 1.5 std)
                        adaptive_threshold = max(min_similarity * 0.7, route_mean - 1.5 * route_std)
                        router_threshold = min(0.95, adaptive_threshold)

                if sim >= router_threshold:
                    # Update Welford route stats (per CLAUDE.md "System Independence")
                    self.update_welford_route(current.id, sim)

                    children = self.get_children(current.id, for_routing=True)
                    for child_sig, _condition in children:
                        queue.append((child_sig, sim))
            else:
                # Leaf: if matches, add to results
                if sim >= min_similarity:
                    matches.append((current, sim))

        # Sort by similarity descending and return top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    # =========================================================================
    # WELFORD STATS (per mycelium-bjrf)
    # =========================================================================
    # Consolidated stats for tree restructuring decisions.
    # Per CLAUDE.md "New Favorite Pattern": single entry points for stats updates.

    def _ensure_welford_stats(self, signature_id: int, conn) -> None:
        """Ensure a welford_stats row exists for this signature."""
        conn.execute(
            """
            INSERT OR IGNORE INTO welford_stats (signature_id, created_at, updated_at)
            VALUES (?, datetime('now'), datetime('now'))
            """,
            (signature_id,)
        )

    def update_welford_route(
        self,
        signature_id: int,
        similarity: float,
        conn=None,
    ) -> bool:
        """Update routing similarity stats using Welford's algorithm.

        Called every time a step is routed to this signature.
        Tracks: how consistent are the similarities when routing here?

        Per mycelium-bjrf: used by restructuring to detect:
        - High variance = routing is inconsistent = maybe split
        - Low variance = routing is stable = good cluster

        Args:
            signature_id: The signature being routed to
            similarity: The cosine similarity used in routing decision

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            self._ensure_welford_stats(signature_id, c)

            row = c.execute(
                "SELECT route_n, route_mean, route_m2 FROM welford_stats WHERE signature_id = ?",
                (signature_id,)
            ).fetchone()

            if not row:
                return False

            n = row["route_n"] + 1
            old_mean = row["route_mean"]
            old_m2 = row["route_m2"]

            # Welford's update
            delta = similarity - old_mean
            new_mean = old_mean + delta / n
            delta2 = similarity - new_mean
            new_m2 = old_m2 + delta * delta2

            c.execute(
                """
                UPDATE welford_stats
                SET route_n = ?, route_mean = ?, route_m2 = ?, updated_at = datetime('now')
                WHERE signature_id = ?
                """,
                (n, new_mean, new_m2, signature_id)
            )
            return True

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                result = _do_update(c)
                c.commit()
                return result

    def update_welford_child(
        self,
        signature_id: int,
        similarity: float,
        conn=None,
    ) -> bool:
        """Update child cluster similarity stats using Welford's algorithm.

        Called when measuring similarity between children of an umbrella.
        Tracks: how tight is this umbrella's cluster?

        Per mycelium-bjrf: used by restructuring to detect:
        - High variance = children are dissimilar = maybe over-clustered
        - Low variance = children are similar = good umbrella

        Args:
            signature_id: The umbrella signature
            similarity: Similarity between two of its children

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            self._ensure_welford_stats(signature_id, c)

            row = c.execute(
                "SELECT child_n, child_mean, child_m2 FROM welford_stats WHERE signature_id = ?",
                (signature_id,)
            ).fetchone()

            if not row:
                return False

            n = row["child_n"] + 1
            old_mean = row["child_mean"]
            old_m2 = row["child_m2"]

            # Welford's update
            delta = similarity - old_mean
            new_mean = old_mean + delta / n
            delta2 = similarity - new_mean
            new_m2 = old_m2 + delta * delta2

            c.execute(
                """
                UPDATE welford_stats
                SET child_n = ?, child_mean = ?, child_m2 = ?, updated_at = datetime('now')
                WHERE signature_id = ?
                """,
                (n, new_mean, new_m2, signature_id)
            )
            return True

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                result = _do_update(c)
                c.commit()
                return result

    def update_welford_exec(
        self,
        signature_id: int,
        success: bool,
        conn=None,
    ) -> bool:
        """Update execution success stats.

        Called after every step execution. Simple success rate tracking.

        Per mycelium-bjrf: used by restructuring to detect:
        - Low success rate = signature needs refinement
        - High success rate = signature is reliable

        Args:
            signature_id: The signature that was executed
            success: Whether execution succeeded

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            self._ensure_welford_stats(signature_id, c)

            c.execute(
                """
                UPDATE welford_stats
                SET exec_n = exec_n + 1,
                    exec_successes = exec_successes + ?,
                    updated_at = datetime('now')
                WHERE signature_id = ?
                """,
                (1 if success else 0, signature_id)
            )
            return True

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                result = _do_update(c)
                c.commit()
                return result

    def update_welford_decomp(
        self,
        signature_id: int,
        success: bool,
        conn=None,
    ) -> bool:
        """Update decomposition attempt stats.

        Called after every decomposition attempt. Tracks whether decomposing
        this signature into sub-steps tends to succeed.

        Used to determine if future decomposition attempts are worthwhile:
        - Low success rate = signature is effectively atomic (don't decompose)
        - High success rate = decomposition helps

        Args:
            signature_id: The signature that was decomposed
            success: Whether decomposition led to successful execution

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            self._ensure_welford_stats(signature_id, c)

            c.execute(
                """
                UPDATE welford_stats
                SET decomp_attempts = decomp_attempts + 1,
                    decomp_successes = decomp_successes + ?,
                    updated_at = datetime('now')
                WHERE signature_id = ?
                """,
                (1 if success else 0, signature_id)
            )
            return True

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                result = _do_update(c)
                c.commit()
                return result

    def reset_welford_child_stats(
        self,
        signature_id: int,
        conn=None,
    ) -> bool:
        """Reset Welford child stats for a signature.

        Per CLAUDE.md "System Independence": Invalidate stale stats when tree structure changes.
        Called when a signature is archived, reparented, or its children change.
        Stats will be rebuilt by the next tree review backfill.

        Args:
            signature_id: The signature whose child stats should be reset

        Returns:
            True if reset succeeded
        """
        def _do_reset(c):
            c.execute(
                """
                UPDATE welford_stats
                SET child_n = 0, child_mean = 0.0, child_m2 = 0.0, updated_at = datetime('now')
                WHERE signature_id = ?
                """,
                (signature_id,)
            )
            logger.debug("[welford] Reset child stats for signature %d", signature_id)
            return True

        if conn is not None:
            return _do_reset(conn)
        else:
            with self._connection() as c:
                result = _do_reset(c)
                c.commit()
                return result

    def get_decomp_success_rate(
        self,
        signature_id: int,
        min_attempts: int = 3,
        conn=None,
    ) -> tuple[float, int]:
        """Get decomposition success rate for a signature.

        Used to decide if decomposition should be attempted.

        Args:
            signature_id: The signature to check
            min_attempts: Minimum attempts required for meaningful rate

        Returns:
            Tuple of (success_rate, attempt_count)
            Returns (1.0, 0) if insufficient data (allows initial attempts)
        """
        def _do_get(c):
            row = c.execute(
                """
                SELECT decomp_attempts, decomp_successes
                FROM welford_stats
                WHERE signature_id = ?
                """,
                (signature_id,)
            ).fetchone()

            if not row or row["decomp_attempts"] < min_attempts:
                # Insufficient data - allow decomposition attempts
                return (1.0, row["decomp_attempts"] if row else 0)

            attempts = row["decomp_attempts"]
            successes = row["decomp_successes"]
            rate = successes / attempts if attempts > 0 else 0.0
            return (rate, attempts)

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_welford_stats(
        self,
        signature_id: int,
        conn=None,
    ) -> Optional[dict]:
        """Get Welford stats for a signature.

        Returns:
            Dict with route_*, child_*, exec_* stats, or None if not found
        """
        def _do_get(c):
            row = c.execute(
                """
                SELECT signature_id, route_n, route_mean, route_m2,
                       child_n, child_mean, child_m2,
                       exec_n, exec_successes,
                       decomp_attempts, decomp_successes,
                       created_at, updated_at
                FROM welford_stats
                WHERE signature_id = ?
                """,
                (signature_id,)
            ).fetchone()

            if not row:
                return None

            return dict(row)

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_welford_variance(
        self,
        signature_id: int,
        stat_type: str = "route",
        conn=None,
    ) -> float:
        """Get variance from Welford stats.

        Args:
            signature_id: The signature
            stat_type: "route" or "child"

        Returns:
            Sample variance (M2 / (N-1)), or 0.0 if insufficient data
        """
        stats = self.get_welford_stats(signature_id, conn=conn)
        if not stats:
            return 0.0

        n = stats.get(f"{stat_type}_n", 0)
        m2 = stats.get(f"{stat_type}_m2", 0.0)

        if n < 2:
            return 0.0

        return m2 / (n - 1)

    def get_welford_std(
        self,
        signature_id: int,
        stat_type: str = "route",
        conn=None,
    ) -> float:
        """Get standard deviation from Welford stats.

        Args:
            signature_id: The signature
            stat_type: "route" or "child"

        Returns:
            Sample standard deviation, or 0.0 if insufficient data
        """
        import math
        variance = self.get_welford_variance(signature_id, stat_type, conn=conn)
        return math.sqrt(variance) if variance > 0 else 0.0

    # =========================================================================
    # UCB1 GAP STATS: Welford-guided exploration thresholds (per mycelium-02nn)
    # =========================================================================
    # Per CLAUDE.md "System Independence": Replace _force_exploration with
    # Welford-guided adaptive thresholds based on historical gap outcomes.

    def update_ucb1_gap_stats(
        self,
        gap: float,
        success: bool,
        conn=None,
    ) -> bool:
        """Update Welford stats for UCB1 gap outcomes.

        Records UCB1 gap values that led to successful or failed routing
        decisions. Used to compute adaptive gap threshold.

        Per mycelium-02nn: "Track gap values that led to correct vs incorrect
        routing. Use Welford to compute adaptive gap threshold."

        Args:
            gap: The UCB1 gap value from routing decision
            success: Whether the routing decision led to correct answer

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            # Get current stats (singleton row)
            row = c.execute("""
                SELECT success_n, success_mean, success_m2,
                       failure_n, failure_mean, failure_m2,
                       total_n, total_mean, total_m2
                FROM ucb1_gap_stats WHERE id = 1
            """).fetchone()

            if not row:
                # Initialize if missing (shouldn't happen after migration)
                c.execute("INSERT OR IGNORE INTO ucb1_gap_stats (id) VALUES (1)")
                row = c.execute("""
                    SELECT success_n, success_mean, success_m2,
                           failure_n, failure_mean, failure_m2,
                           total_n, total_mean, total_m2
                    FROM ucb1_gap_stats WHERE id = 1
                """).fetchone()

            # Update success or failure stats
            if success:
                n = row["success_n"] + 1
                old_mean = row["success_mean"]
                old_m2 = row["success_m2"]
                delta = gap - old_mean
                new_mean = old_mean + delta / n
                delta2 = gap - new_mean
                new_m2 = old_m2 + delta * delta2

                c.execute("""
                    UPDATE ucb1_gap_stats
                    SET success_n = ?, success_mean = ?, success_m2 = ?,
                        updated_at = datetime('now')
                    WHERE id = 1
                """, (n, new_mean, new_m2))
            else:
                n = row["failure_n"] + 1
                old_mean = row["failure_mean"]
                old_m2 = row["failure_m2"]
                delta = gap - old_mean
                new_mean = old_mean + delta / n
                delta2 = gap - new_mean
                new_m2 = old_m2 + delta * delta2

                c.execute("""
                    UPDATE ucb1_gap_stats
                    SET failure_n = ?, failure_mean = ?, failure_m2 = ?,
                        updated_at = datetime('now')
                    WHERE id = 1
                """, (n, new_mean, new_m2))

            # Also update total stats
            total_n = row["total_n"] + 1
            total_old_mean = row["total_mean"]
            total_old_m2 = row["total_m2"]
            total_delta = gap - total_old_mean
            total_new_mean = total_old_mean + total_delta / total_n
            total_delta2 = gap - total_new_mean
            total_new_m2 = total_old_m2 + total_delta * total_delta2

            c.execute("""
                UPDATE ucb1_gap_stats
                SET total_n = ?, total_mean = ?, total_m2 = ?
                WHERE id = 1
            """, (total_n, total_new_mean, total_new_m2))

            c.connection.commit()
            return True

        if conn:
            return _do_update(conn)
        with self._connection() as c:
            return _do_update(c)

    def get_ucb1_gap_stats(self, conn=None) -> Optional[dict]:
        """Get UCB1 gap statistics for adaptive threshold calculation.

        Returns:
            Dictionary with gap stats, or None if no data
        """
        def _do_get(c):
            row = c.execute("""
                SELECT success_n, success_mean, success_m2,
                       failure_n, failure_mean, failure_m2,
                       total_n, total_mean, total_m2,
                       created_at, updated_at
                FROM ucb1_gap_stats WHERE id = 1
            """).fetchone()
            if row:
                return dict(row)
            return None

        if conn:
            return _do_get(conn)
        with self._connection() as c:
            return _do_get(c)

    def get_adaptive_gap_threshold(self, conn=None) -> float:
        """Compute adaptive UCB1 gap threshold from Welford stats.

        Per mycelium-02nn: threshold = success_mean - k * success_std
        This uses gaps from successful routings to set the branching threshold.

        Falls back to static UCB1_GAP_BRANCH_THRESHOLD during cold start.

        Returns:
            Adaptive gap threshold
        """
        import math
        from mycelium.config import (
            ADAPTIVE_GAP_ENABLED,
            ADAPTIVE_GAP_K,
            ADAPTIVE_GAP_MIN_SAMPLES,
            ADAPTIVE_GAP_MIN_THRESHOLD,
            ADAPTIVE_GAP_MAX_THRESHOLD,
            UCB1_GAP_BRANCH_THRESHOLD,
        )

        if not ADAPTIVE_GAP_ENABLED:
            return UCB1_GAP_BRANCH_THRESHOLD

        stats = self.get_ucb1_gap_stats(conn=conn)
        if not stats or stats.get("success_n", 0) < ADAPTIVE_GAP_MIN_SAMPLES:
            # Cold start: use static threshold
            return UCB1_GAP_BRANCH_THRESHOLD

        # Compute threshold from successful routings
        success_n = stats["success_n"]
        success_mean = stats["success_mean"]
        success_m2 = stats["success_m2"]

        # Compute sample variance and std
        if success_n > 1:
            variance = success_m2 / (success_n - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
        else:
            std = 0.0

        # Adaptive threshold: mean - k * std
        # Lower threshold = more branching (for uncertain decisions)
        threshold = success_mean - ADAPTIVE_GAP_K * std

        # Clamp to bounds
        threshold = max(ADAPTIVE_GAP_MIN_THRESHOLD, min(ADAPTIVE_GAP_MAX_THRESHOLD, threshold))

        logger.debug(
            "[db] Adaptive gap threshold: %.3f (success_mean=%.3f, std=%.3f, n=%d)",
            threshold, success_mean, std, success_n
        )

        return threshold

    # =========================================================================
    # REACTIVE EXPLORATION STATS: Welford-adaptive multipliers (per mycelium-02nn)
    # =========================================================================
    # Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure
    # Tracks reactive exploration outcomes to adapt gap/budget multipliers.

    def update_reactive_exploration_stats(
        self,
        found_winner: bool,
        gap_mult_used: float,
        budget_mult_used: float,
        conn=None,
    ) -> bool:
        """Update Welford stats for reactive exploration outcomes.

        Records whether reactive exploration found a winning path and what
        multipliers were used. Used to adapt future multipliers.

        Per CLAUDE.md "The Flow": This is the DB Statistics part of the flow.

        Args:
            found_winner: Whether reactive exploration found a winning path
            gap_mult_used: The gap multiplier that was used
            budget_mult_used: The budget multiplier that was used

        Returns:
            True if update succeeded
        """
        def _do_update(c):
            # Get current stats (singleton row)
            row = c.execute("""
                SELECT n, success_mean, success_m2,
                       gap_mult_n, gap_mult_mean, gap_mult_m2,
                       budget_mult_n, budget_mult_mean, budget_mult_m2
                FROM reactive_exploration_stats WHERE id = 1
            """).fetchone()

            if not row:
                # Initialize if missing
                c.execute("INSERT OR IGNORE INTO reactive_exploration_stats (id) VALUES (1)")
                row = c.execute("""
                    SELECT n, success_mean, success_m2,
                           gap_mult_n, gap_mult_mean, gap_mult_m2,
                           budget_mult_n, budget_mult_mean, budget_mult_m2
                    FROM reactive_exploration_stats WHERE id = 1
                """).fetchone()

            # Update success rate stats (Welford)
            success_val = 1.0 if found_winner else 0.0
            n = row["n"] + 1
            old_mean = row["success_mean"]
            old_m2 = row["success_m2"]
            delta = success_val - old_mean
            new_mean = old_mean + delta / n
            delta2 = success_val - new_mean
            new_m2 = old_m2 + delta * delta2

            c.execute("""
                UPDATE reactive_exploration_stats
                SET n = ?, success_mean = ?, success_m2 = ?,
                    updated_at = datetime('now')
                WHERE id = 1
            """, (n, new_mean, new_m2))

            # If found winner, record the multipliers that worked (Welford)
            if found_winner:
                # Update gap_mult stats
                gap_n = row["gap_mult_n"] + 1
                gap_old_mean = row["gap_mult_mean"]
                gap_old_m2 = row["gap_mult_m2"]
                gap_delta = gap_mult_used - gap_old_mean
                gap_new_mean = gap_old_mean + gap_delta / gap_n
                gap_delta2 = gap_mult_used - gap_new_mean
                gap_new_m2 = gap_old_m2 + gap_delta * gap_delta2

                c.execute("""
                    UPDATE reactive_exploration_stats
                    SET gap_mult_n = ?, gap_mult_mean = ?, gap_mult_m2 = ?
                    WHERE id = 1
                """, (gap_n, gap_new_mean, gap_new_m2))

                # Update budget_mult stats
                budget_n = row["budget_mult_n"] + 1
                budget_old_mean = row["budget_mult_mean"]
                budget_old_m2 = row["budget_mult_m2"]
                budget_delta = budget_mult_used - budget_old_mean
                budget_new_mean = budget_old_mean + budget_delta / budget_n
                budget_delta2 = budget_mult_used - budget_new_mean
                budget_new_m2 = budget_old_m2 + budget_delta * budget_delta2

                c.execute("""
                    UPDATE reactive_exploration_stats
                    SET budget_mult_n = ?, budget_mult_mean = ?, budget_mult_m2 = ?
                    WHERE id = 1
                """, (budget_n, budget_new_mean, budget_new_m2))

            c.connection.commit()
            logger.debug(
                "[db] Reactive exploration stats updated: found_winner=%s, n=%d, success_rate=%.2f",
                found_winner, n, new_mean
            )
            return True

        if conn:
            return _do_update(conn)
        with self._connection() as c:
            return _do_update(c)

    def get_reactive_exploration_stats(self, conn=None) -> Optional[dict]:
        """Get reactive exploration statistics.

        Returns:
            Dictionary with reactive exploration stats, or None if no data
        """
        def _do_get(c):
            row = c.execute("""
                SELECT n, success_mean, success_m2,
                       gap_mult_n, gap_mult_mean, gap_mult_m2,
                       budget_mult_n, budget_mult_mean, budget_mult_m2,
                       created_at, updated_at
                FROM reactive_exploration_stats WHERE id = 1
            """).fetchone()
            if row:
                return dict(row)
            return None

        if conn:
            return _do_get(conn)
        with self._connection() as c:
            return _do_get(c)

    def get_adaptive_reactive_multipliers(self, conn=None) -> tuple[float, float]:
        """Compute adaptive reactive exploration multipliers from Welford stats.

        Per CLAUDE.md "The Flow": This is the Welford → Tree Structure part.
        Uses success rate to adjust multipliers:
        - Low success rate → need more exploration → increase multipliers
        - High success rate → current settings work → use learned means

        Returns:
            Tuple of (gap_mult, budget_mult)
        """
        import math
        from mycelium.config import (
            ADAPTIVE_REACTIVE_ENABLED,
            ADAPTIVE_REACTIVE_MIN_SAMPLES,
            REACTIVE_EXPLORATION_GAP_MULT,
            REACTIVE_EXPLORATION_BUDGET_MULT,
            REACTIVE_EXPLORATION_GAP_MULT_MIN,
            REACTIVE_EXPLORATION_GAP_MULT_MAX,
            REACTIVE_EXPLORATION_BUDGET_MULT_MIN,
            REACTIVE_EXPLORATION_BUDGET_MULT_MAX,
            REACTIVE_EXPLORATION_ADJUST_K,
        )

        # Cold start: use default multipliers
        if not ADAPTIVE_REACTIVE_ENABLED:
            return (REACTIVE_EXPLORATION_GAP_MULT, REACTIVE_EXPLORATION_BUDGET_MULT)

        stats = self.get_reactive_exploration_stats(conn=conn)
        if not stats or stats.get("n", 0) < ADAPTIVE_REACTIVE_MIN_SAMPLES:
            return (REACTIVE_EXPLORATION_GAP_MULT, REACTIVE_EXPLORATION_BUDGET_MULT)

        # Get success rate and variance
        n = stats["n"]
        success_mean = stats["success_mean"]
        success_m2 = stats["success_m2"]

        if n > 1:
            variance = success_m2 / (n - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
        else:
            std = 0.0

        # Adaptive logic:
        # - If success rate is HIGH (exploration is working): use learned multipliers
        # - If success rate is LOW (exploration isn't helping): increase multipliers
        # - Use variance to determine confidence in adjustment

        # Compute adjustment factor based on success rate
        # success_mean=1.0 → factor=0 (no extra boost needed)
        # success_mean=0.0 → factor=1 (max boost needed)
        # Adjust by k*std for conservative bounds
        failure_rate = 1.0 - success_mean
        adjustment = failure_rate + REACTIVE_EXPLORATION_ADJUST_K * std

        # Interpolate between MIN and MAX based on adjustment
        # adjustment=0 → use MIN (or learned mean)
        # adjustment=1+ → use MAX
        adjustment = min(1.0, max(0.0, adjustment))

        # For gap_mult: if we have successful data, blend learned mean with adjustment
        if stats.get("gap_mult_n", 0) >= 3:
            learned_gap = stats["gap_mult_mean"]
            gap_mult = learned_gap + adjustment * (REACTIVE_EXPLORATION_GAP_MULT_MAX - learned_gap)
        else:
            # Interpolate between default and max
            gap_mult = REACTIVE_EXPLORATION_GAP_MULT + adjustment * (
                REACTIVE_EXPLORATION_GAP_MULT_MAX - REACTIVE_EXPLORATION_GAP_MULT
            )

        # For budget_mult: same logic
        if stats.get("budget_mult_n", 0) >= 3:
            learned_budget = stats["budget_mult_mean"]
            budget_mult = learned_budget + adjustment * (REACTIVE_EXPLORATION_BUDGET_MULT_MAX - learned_budget)
        else:
            budget_mult = REACTIVE_EXPLORATION_BUDGET_MULT + adjustment * (
                REACTIVE_EXPLORATION_BUDGET_MULT_MAX - REACTIVE_EXPLORATION_BUDGET_MULT
            )

        # Clamp to bounds
        gap_mult = max(REACTIVE_EXPLORATION_GAP_MULT_MIN, min(REACTIVE_EXPLORATION_GAP_MULT_MAX, gap_mult))
        budget_mult = max(REACTIVE_EXPLORATION_BUDGET_MULT_MIN, min(REACTIVE_EXPLORATION_BUDGET_MULT_MAX, budget_mult))

        logger.debug(
            "[db] Adaptive reactive multipliers: gap=%.2f, budget=%.2f "
            "(success_rate=%.2f, std=%.2f, n=%d)",
            gap_mult, budget_mult, success_mean, std, n
        )

        return (gap_mult, budget_mult)

    # =========================================================================
    # EMBEDDING DRIFT: Semantic Attractor Updates (per mycelium-ieq4)
    # =========================================================================
    # Per CLAUDE.md: "High-traffic signatures become semantic attractors"
    # Accumulate successful dag_step embeddings for batch drift updates.

    def accumulate_embedding_drift(
        self,
        signature_id: int,
        success_embedding: list[float],
        conn=None,
    ) -> bool:
        """Accumulate a successful dag_step embedding for later drift update.

        Called on successful (leaf_node, dag_step) matches. Embeddings are
        accumulated and averaged during periodic batch drift updates.

        Args:
            signature_id: The leaf node signature ID
            success_embedding: The dag_step's graph embedding that succeeded
            conn: Optional connection

        Returns:
            True if accumulated successfully
        """
        def _do_accumulate(c):
            # Check if we have existing accumulator for this signature
            row = c.execute("""
                SELECT embedding_sum, success_count
                FROM pending_embedding_drifts
                WHERE signature_id = ?
            """, (signature_id,)).fetchone()

            if row:
                # Add to existing sum
                import json
                existing_sum = json.loads(row["embedding_sum"])
                new_sum = [a + b for a, b in zip(existing_sum, success_embedding)]
                new_count = row["success_count"] + 1

                c.execute("""
                    UPDATE pending_embedding_drifts
                    SET embedding_sum = ?, success_count = ?, updated_at = datetime('now')
                    WHERE signature_id = ?
                """, (json.dumps(new_sum), new_count, signature_id))
            else:
                # Create new accumulator
                import json
                c.execute("""
                    INSERT INTO pending_embedding_drifts (signature_id, embedding_sum, success_count)
                    VALUES (?, ?, 1)
                """, (signature_id, json.dumps(success_embedding)))

            c.connection.commit()
            return True

        try:
            if conn:
                return _do_accumulate(conn)
            with self._connection() as c:
                return _do_accumulate(c)
        except Exception as e:
            logger.warning("[db] Failed to accumulate embedding drift: %s", e)
            return False

    def get_pending_embedding_drifts(self, min_successes: int = 1, conn=None) -> list[dict]:
        """Get all pending embedding drifts ready for batch update.

        Args:
            min_successes: Minimum success count to include
            conn: Optional connection

        Returns:
            List of dicts with signature_id, avg_embedding, success_count
        """
        def _do_get(c):
            rows = c.execute("""
                SELECT signature_id, embedding_sum, success_count
                FROM pending_embedding_drifts
                WHERE success_count >= ?
            """, (min_successes,)).fetchall()

            results = []
            import json
            for row in rows:
                embedding_sum = json.loads(row["embedding_sum"])
                count = row["success_count"]
                # Compute average embedding
                avg_embedding = [x / count for x in embedding_sum]
                results.append({
                    "signature_id": row["signature_id"],
                    "avg_embedding": avg_embedding,
                    "success_count": count,
                })
            return results

        if conn:
            return _do_get(conn)
        with self._connection() as c:
            return _do_get(c)

    def clear_pending_embedding_drifts(self, signature_ids: list[int] = None, conn=None) -> int:
        """Clear pending embedding drifts after batch update.

        Args:
            signature_ids: Specific IDs to clear, or None for all
            conn: Optional connection

        Returns:
            Number of rows cleared
        """
        def _do_clear(c):
            if signature_ids:
                placeholders = ",".join("?" * len(signature_ids))
                result = c.execute(f"""
                    DELETE FROM pending_embedding_drifts
                    WHERE signature_id IN ({placeholders})
                """, signature_ids)
            else:
                result = c.execute("DELETE FROM pending_embedding_drifts")
            c.connection.commit()
            return result.rowcount

        if conn:
            return _do_clear(conn)
        with self._connection() as c:
            return _do_clear(c)

    def apply_embedding_drift_batch(self, conn=None) -> dict:
        """Apply Welford-adaptive embedding drift to leaf nodes.

        Vectorized batch update using NumPy for performance.
        Called during periodic tree review (every ~50 problems).

        Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure

        Returns:
            Dict with update stats: nodes_updated, avg_drift_magnitude
        """
        from mycelium.config import (
            EMBEDDING_DRIFT_ENABLED,
            EMBEDDING_DRIFT_MIN_SUCCESSES,
            EMBEDDING_DRIFT_VARIANCE_K,
        )

        if not EMBEDDING_DRIFT_ENABLED:
            return {"nodes_updated": 0, "skipped": "drift_disabled"}

        def _do_apply(c):
            import numpy as np

            # Get pending drifts with sufficient successes
            pending = self.get_pending_embedding_drifts(
                min_successes=EMBEDDING_DRIFT_MIN_SUCCESSES, conn=c
            )

            if not pending:
                return {"nodes_updated": 0, "skipped": "no_pending_drifts"}

            # Get current embeddings and Welford stats for these signatures
            sig_ids = [p["signature_id"] for p in pending]
            placeholders = ",".join("?" * len(sig_ids))

            rows = c.execute(f"""
                SELECT s.id, s.graph_embedding,
                       COALESCE(w.embedding_n, 0) as emb_n,
                       COALESCE(w.embedding_mean, 0) as emb_mean,
                       COALESCE(w.embedding_m2, 0) as emb_m2
                FROM step_signatures s
                LEFT JOIN welford_stats w ON s.id = w.signature_id
                WHERE s.id IN ({placeholders})
                  AND s.graph_embedding IS NOT NULL
            """, sig_ids).fetchall()

            if not rows:
                return {"nodes_updated": 0, "skipped": "no_valid_signatures"}

            # Build lookup for pending data
            pending_lookup = {p["signature_id"]: p for p in pending}

            # Vectorized computation
            updates = []
            drift_magnitudes = []

            for row in rows:
                sig_id = row["id"]
                if sig_id not in pending_lookup:
                    continue

                # Parse current embedding
                import json
                current_emb = np.array(json.loads(row["graph_embedding"]), dtype=np.float32)
                success_emb = np.array(pending_lookup[sig_id]["avg_embedding"], dtype=np.float32)

                # Compute Welford-adaptive alpha
                # α = 1 - (k / (k + variance))
                # High variance → lower α → faster drift
                emb_n = row["emb_n"]
                emb_m2 = row["emb_m2"]
                if emb_n > 1:
                    variance = emb_m2 / (emb_n - 1)
                else:
                    variance = 1.0  # High variance for new nodes = fast drift

                k = EMBEDDING_DRIFT_VARIANCE_K
                alpha = 1.0 - (k / (k + variance))
                alpha = max(0.5, min(0.99, alpha))  # Clamp to reasonable range

                # EMA update: new = α * old + (1-α) * success
                new_emb = alpha * current_emb + (1 - alpha) * success_emb

                # Track drift magnitude
                drift_mag = float(np.linalg.norm(new_emb - current_emb))
                drift_magnitudes.append(drift_mag)

                updates.append({
                    "id": sig_id,
                    "new_embedding": json.dumps(new_emb.tolist()),
                    "alpha": alpha,
                })

            # Apply updates
            for upd in updates:
                c.execute("""
                    UPDATE step_signatures
                    SET graph_embedding = ?, updated_at = datetime('now')
                    WHERE id = ?
                """, (upd["new_embedding"], upd["id"]))

            # Clear processed drifts
            processed_ids = [u["id"] for u in updates]
            self.clear_pending_embedding_drifts(processed_ids, conn=c)

            c.connection.commit()

            avg_drift = sum(drift_magnitudes) / len(drift_magnitudes) if drift_magnitudes else 0.0

            logger.info(
                "[db] Embedding drift applied: %d nodes updated, avg_drift=%.4f",
                len(updates), avg_drift
            )

            return {
                "nodes_updated": len(updates),
                "avg_drift_magnitude": avg_drift,
                "updates": [{"id": u["id"], "alpha": u["alpha"]} for u in updates],
            }

        if conn:
            return _do_apply(conn)
        with self._connection() as c:
            return _do_apply(c)

    def recompute_router_centroids(self, conn=None) -> int:
        """Recompute router centroids as average of children graph embeddings.

        Called after embedding drift updates to propagate changes up the tree.

        Returns:
            Number of routers updated
        """
        def _do_recompute(c):
            import numpy as np
            import json

            # Get all router nodes (non-leaf with children)
            routers = c.execute("""
                SELECT DISTINCT p.id
                FROM step_signatures p
                JOIN step_signatures ch ON ch.parent_id = p.id
                WHERE p.is_archived = 0
            """).fetchall()

            updated = 0
            for row in routers:
                router_id = row["id"]

                # Get children graph embeddings
                children = c.execute("""
                    SELECT graph_embedding
                    FROM step_signatures
                    WHERE parent_id = ? AND graph_embedding IS NOT NULL AND is_archived = 0
                """, (router_id,)).fetchall()

                if not children:
                    continue

                # Compute centroid (mean of children)
                embeddings = [np.array(json.loads(ch["graph_embedding"]), dtype=np.float32)
                              for ch in children]
                centroid = np.mean(embeddings, axis=0)

                # Update router
                c.execute("""
                    UPDATE step_signatures
                    SET graph_embedding = ?, updated_at = datetime('now')
                    WHERE id = ?
                """, (json.dumps(centroid.tolist()), router_id))
                updated += 1

            c.connection.commit()
            logger.info("[db] Recomputed %d router centroids", updated)
            return updated

        if conn:
            return _do_recompute(conn)
        with self._connection() as c:
            return _do_recompute(c)

    # =========================================================================
    # PLAN_STEP_STATS: Statistical blame accumulation for (plan, position, node)
    # =========================================================================
    # Per CLAUDE.md: "Failures Are Valuable Data Points" - accumulate blame statistically

    def update_plan_step_stats(
        self,
        plan_signature: str,
        step_position: int,
        node_id: int,
        success: bool,
        conn=None,
    ) -> bool:
        """Update Welford stats for (plan, position, node) success rate.

        Called after grading a problem. Records whether each step in the plan
        succeeded based on the overall problem outcome.

        Per CLAUDE.md: "Failures Are Valuable Data Points" - accumulate blame
        without reactive exploration. This enables:
        - Identifying which step positions consistently fail
        - Detecting which nodes are problematic at certain positions
        - Order-aware tracking (same node at step 1 vs step 5 may differ)

        Args:
            plan_signature: Hash of plan structure (from dag_plan_stats)
            step_position: Position in plan (1, 2, 3...)
            node_id: Which signature handled this step
            success: Whether the problem was solved correctly

        Returns:
            True if update succeeded
        """
        success_val = 1.0 if success else 0.0

        def _do_update(c):
            # Check if row exists
            row = c.execute(
                """
                SELECT n, mean_success, m2
                FROM plan_step_stats
                WHERE plan_signature = ? AND step_position = ? AND node_id = ?
                """,
                (plan_signature, step_position, node_id)
            ).fetchone()

            if row:
                # Update existing row with Welford's algorithm
                n = row["n"] + 1
                old_mean = row["mean_success"]
                old_m2 = row["m2"]

                delta = success_val - old_mean
                new_mean = old_mean + delta / n
                delta2 = success_val - new_mean
                new_m2 = old_m2 + delta * delta2

                c.execute(
                    """
                    UPDATE plan_step_stats
                    SET n = ?, mean_success = ?, m2 = ?, last_updated_at = datetime('now')
                    WHERE plan_signature = ? AND step_position = ? AND node_id = ?
                    """,
                    (n, new_mean, new_m2, plan_signature, step_position, node_id)
                )
            else:
                # Insert new row (first observation: n=1, mean=success_val, m2=0)
                c.execute(
                    """
                    INSERT INTO plan_step_stats
                        (plan_signature, step_position, node_id, n, mean_success, m2)
                    VALUES (?, ?, ?, 1, ?, 0.0)
                    """,
                    (plan_signature, step_position, node_id, success_val)
                )
            return True

        if conn is not None:
            return _do_update(conn)
        else:
            with self._connection() as c:
                result = _do_update(c)
                c.commit()
                return result

    def get_plan_step_stats(
        self,
        plan_signature: str,
        step_position: int,
        node_id: int,
        conn=None,
    ) -> Optional[dict]:
        """Get Welford stats for a specific (plan, position, node) combo.

        Args:
            plan_signature: Hash of plan structure
            step_position: Position in plan
            node_id: Which signature

        Returns:
            Dict with n, mean_success, m2, variance, std, or None if not found
        """
        def _do_get(c):
            row = c.execute(
                """
                SELECT plan_signature, step_position, node_id, n, mean_success, m2,
                       first_seen_at, last_updated_at
                FROM plan_step_stats
                WHERE plan_signature = ? AND step_position = ? AND node_id = ?
                """,
                (plan_signature, step_position, node_id)
            ).fetchone()

            if not row:
                return None

            result = dict(row)
            n = result["n"]
            m2 = result["m2"]

            # Compute variance and std
            if n >= 2:
                result["variance"] = m2 / (n - 1)
                result["std"] = (result["variance"]) ** 0.5
            else:
                result["variance"] = 0.0
                result["std"] = 0.0

            return result

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_plan_step_stats_for_plan(
        self,
        plan_signature: str,
        conn=None,
    ) -> list[dict]:
        """Get all step stats for a plan, ordered by position.

        Useful for understanding which positions are problematic.

        Args:
            plan_signature: Hash of plan structure

        Returns:
            List of stats dicts, ordered by step_position
        """
        def _do_get(c):
            rows = c.execute(
                """
                SELECT pss.plan_signature, pss.step_position, pss.node_id,
                       pss.n, pss.mean_success, pss.m2,
                       pss.first_seen_at, pss.last_updated_at,
                       ss.step_type
                FROM plan_step_stats pss
                LEFT JOIN step_signatures ss ON pss.node_id = ss.id
                WHERE pss.plan_signature = ?
                ORDER BY pss.step_position
                """,
                (plan_signature,)
            ).fetchall()

            results = []
            for row in rows:
                result = dict(row)
                n = result["n"]
                m2 = result["m2"]
                if n >= 2:
                    result["variance"] = m2 / (n - 1)
                    result["std"] = (result["variance"]) ** 0.5
                else:
                    result["variance"] = 0.0
                    result["std"] = 0.0
                results.append(result)

            return results

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_low_performing_plan_steps(
        self,
        min_observations: int = 5,
        max_success_rate: float = 0.5,
        conn=None,
    ) -> list[dict]:
        """Find (plan, position, node) combinations with low success rates.

        Used to identify problematic patterns that need attention.
        Per CLAUDE.md: "Failures Are Valuable Data Points"

        Args:
            min_observations: Minimum n to be considered significant
            max_success_rate: Maximum mean_success to be flagged as low

        Returns:
            List of stats dicts for problematic combinations
        """
        def _do_get(c):
            rows = c.execute(
                """
                SELECT pss.plan_signature, pss.step_position, pss.node_id,
                       pss.n, pss.mean_success, pss.m2,
                       pss.first_seen_at, pss.last_updated_at,
                       ss.step_type, ss.description
                FROM plan_step_stats pss
                LEFT JOIN step_signatures ss ON pss.node_id = ss.id
                WHERE pss.n >= ? AND pss.mean_success <= ?
                ORDER BY pss.mean_success ASC, pss.n DESC
                """,
                (min_observations, max_success_rate)
            ).fetchall()

            results = []
            for row in rows:
                result = dict(row)
                n = result["n"]
                m2 = result["m2"]
                if n >= 2:
                    result["variance"] = m2 / (n - 1)
                    result["std"] = (result["variance"]) ** 0.5
                else:
                    result["variance"] = 0.0
                    result["std"] = 0.0
                results.append(result)

            return results

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_node_success_by_position(
        self,
        node_id: int,
        conn=None,
    ) -> dict[int, dict]:
        """Get success rates for a node grouped by step position.

        Useful for detecting if a node performs differently at different positions.
        E.g., a "compute_sum" node might work well at step 1 but fail at step 5.

        Args:
            node_id: The signature to analyze

        Returns:
            Dict mapping step_position -> stats dict
        """
        def _do_get(c):
            rows = c.execute(
                """
                SELECT step_position, n, mean_success, m2
                FROM plan_step_stats
                WHERE node_id = ?
                ORDER BY step_position
                """,
                (node_id,)
            ).fetchall()

            results = {}
            for row in rows:
                pos = row["step_position"]
                n = row["n"]
                m2 = row["m2"]

                results[pos] = {
                    "n": n,
                    "mean_success": row["mean_success"],
                    "m2": m2,
                    "variance": m2 / (n - 1) if n >= 2 else 0.0,
                    "std": (m2 / (n - 1)) ** 0.5 if n >= 2 else 0.0,
                }

            return results

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def should_avoid_node_at_position(
        self,
        node_id: int,
        step_position: int,
        plan_signature: Optional[str] = None,
        min_observations: int = 5,
        max_success_rate: float = 0.3,
        conn=None,
    ) -> tuple[bool, Optional[dict]]:
        """Check if a node should be avoided at a specific position.

        Used during routing to warn about historically problematic combinations.
        Per CLAUDE.md: "Failures Are Valuable Data Points"

        Args:
            node_id: The signature to check
            step_position: Position in the plan
            plan_signature: Optional - check specific plan, else check across all plans
            min_observations: Minimum n to be considered significant
            max_success_rate: Below this = should avoid

        Returns:
            Tuple of (should_avoid, stats_dict)
        """
        def _do_check(c):
            if plan_signature:
                # Check specific (plan, position, node) combo
                stats = self.get_plan_step_stats(
                    plan_signature, step_position, node_id, conn=c
                )
                if stats and stats["n"] >= min_observations:
                    if stats["mean_success"] <= max_success_rate:
                        return (True, stats)
                return (False, stats)
            else:
                # Check node at this position across ALL plans
                row = c.execute(
                    """
                    SELECT SUM(n) as total_n,
                           SUM(n * mean_success) / SUM(n) as weighted_mean
                    FROM plan_step_stats
                    WHERE node_id = ? AND step_position = ? AND n > 0
                    """,
                    (node_id, step_position)
                ).fetchone()

                if row and row["total_n"] and row["total_n"] >= min_observations:
                    weighted_mean = row["weighted_mean"]
                    if weighted_mean <= max_success_rate:
                        return (True, {
                            "n": row["total_n"],
                            "mean_success": weighted_mean,
                            "scope": "all_plans"
                        })
                    return (False, {
                        "n": row["total_n"],
                        "mean_success": weighted_mean,
                        "scope": "all_plans"
                    })
                return (False, None)

        if conn is not None:
            return _do_check(conn)
        else:
            with self._connection() as c:
                return _do_check(c)

    def get_best_nodes_for_position(
        self,
        step_position: int,
        min_observations: int = 3,
        limit: int = 5,
        conn=None,
    ) -> list[dict]:
        """Get the best performing nodes at a specific position.

        Useful for suggesting alternatives when a node is flagged as problematic.

        Args:
            step_position: Position in the plan
            min_observations: Minimum n to be considered
            limit: Max nodes to return

        Returns:
            List of dicts with node_id, mean_success, n, sorted by success desc
        """
        def _do_get(c):
            rows = c.execute(
                """
                SELECT node_id,
                       SUM(n) as total_n,
                       SUM(n * mean_success) / SUM(n) as weighted_mean
                FROM plan_step_stats
                WHERE step_position = ? AND n > 0
                GROUP BY node_id
                HAVING total_n >= ?
                ORDER BY weighted_mean DESC
                LIMIT ?
                """,
                (step_position, min_observations, limit)
            ).fetchall()

            return [
                {
                    "node_id": row["node_id"],
                    "n": row["total_n"],
                    "mean_success": row["weighted_mean"],
                }
                for row in rows
            ]

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_plan_step_stats_summary(self, conn=None) -> dict:
        """Get summary of plan_step_stats table.

        Useful for debugging and monitoring.

        Returns:
            Dict with total entries, avg success rate, problematic combos count
        """
        def _do_get(c):
            row = c.execute(
                """
                SELECT
                    COUNT(*) as total_entries,
                    SUM(n) as total_observations,
                    AVG(mean_success) as avg_success_rate,
                    COUNT(CASE WHEN n >= 5 AND mean_success < 0.5 THEN 1 END) as low_performers
                FROM plan_step_stats
                """
            ).fetchone()

            if row:
                return {
                    "total_entries": row["total_entries"] or 0,
                    "total_observations": row["total_observations"] or 0,
                    "avg_success_rate": row["avg_success_rate"] or 0.0,
                    "low_performers": row["low_performers"] or 0,
                }
            return {
                "total_entries": 0,
                "total_observations": 0,
                "avg_success_rate": 0.0,
                "low_performers": 0,
            }

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_node_position_stats(
        self,
        node_id: int,
        step_position: int,
        conn=None,
    ) -> Optional[dict]:
        """Get aggregated Welford stats for (node_id, step_position) across ALL plans.

        Per CLAUDE.md "Cluster Boundaries": Welford statistics guide tree structure.
        This aggregates stats for a node at a specific position regardless of plan.

        Args:
            node_id: Which signature
            step_position: Position in plan (1, 2, 3...)

        Returns:
            Dict with aggregated n, mean_success, variance, std, or None if no data
        """
        def _do_get(c):
            # Aggregate across all plans for this (node, position) pair
            row = c.execute(
                """
                SELECT
                    SUM(n) as total_n,
                    SUM(n * mean_success) as weighted_success_sum,
                    COUNT(*) as plan_count
                FROM plan_step_stats
                WHERE node_id = ? AND step_position = ?
                """,
                (node_id, step_position)
            ).fetchone()

            if not row or not row["total_n"] or row["total_n"] == 0:
                return None

            total_n = row["total_n"]
            weighted_mean = row["weighted_success_sum"] / total_n if total_n > 0 else 0.5

            # Get variance by computing weighted M2 sum
            rows = c.execute(
                """
                SELECT n, mean_success, m2
                FROM plan_step_stats
                WHERE node_id = ? AND step_position = ? AND n > 0
                """,
                (node_id, step_position)
            ).fetchall()

            # Combine Welford stats using parallel algorithm
            combined_m2 = 0.0
            for r in rows:
                n_i = r["n"]
                mean_i = r["mean_success"]
                m2_i = r["m2"]
                # Delta between this batch mean and combined mean
                delta = mean_i - weighted_mean
                combined_m2 += m2_i + delta * delta * n_i

            variance = combined_m2 / (total_n - 1) if total_n >= 2 else 0.0
            std = variance ** 0.5

            return {
                "node_id": node_id,
                "step_position": step_position,
                "n": total_n,
                "mean_success": weighted_mean,
                "variance": variance,
                "std": std,
                "plan_count": row["plan_count"],
            }

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_node_stats_all_positions(
        self,
        node_id: int,
        conn=None,
    ) -> dict:
        """Get this node's aggregated stats across all positions.

        Returns the node's overall mean_success and std for self-comparison.

        Args:
            node_id: Which signature

        Returns:
            Dict with overall n, mean_success, variance, std
        """
        def _do_get(c):
            row = c.execute(
                """
                SELECT
                    SUM(n) as total_n,
                    SUM(n * mean_success) as weighted_success_sum
                FROM plan_step_stats
                WHERE node_id = ?
                """,
                (node_id,)
            ).fetchone()

            if not row or not row["total_n"] or row["total_n"] == 0:
                return {"n": 0, "mean_success": 0.5, "variance": 0.0, "std": 0.0}

            total_n = row["total_n"]
            weighted_mean = row["weighted_success_sum"] / total_n if total_n > 0 else 0.5

            # Get variance
            rows = c.execute(
                """
                SELECT n, mean_success, m2
                FROM plan_step_stats
                WHERE node_id = ? AND n > 0
                """,
                (node_id,)
            ).fetchall()

            combined_m2 = 0.0
            for r in rows:
                delta = r["mean_success"] - weighted_mean
                combined_m2 += r["m2"] + delta * delta * r["n"]

            variance = combined_m2 / (total_n - 1) if total_n >= 2 else 0.0
            std = variance ** 0.5

            return {
                "node_id": node_id,
                "n": total_n,
                "mean_success": weighted_mean,
                "variance": variance,
                "std": std,
            }

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def get_cluster_stats(
        self,
        node_id: int,
        conn=None,
    ) -> dict:
        """Get sibling node stats for cluster-relative z-score comparison.

        Per CLAUDE.md "Cluster Boundaries": Compare node against its peers.

        Args:
            node_id: Which signature

        Returns:
            Dict with cluster_mean, cluster_std, sibling_count
        """
        def _do_get(c):
            # Get parent of this node
            parent_row = c.execute(
                """
                SELECT parent_id FROM signature_relationships WHERE child_id = ?
                """,
                (node_id,)
            ).fetchone()

            if not parent_row:
                return {"cluster_mean": 0.5, "cluster_std": 0.25, "sibling_count": 0}

            parent_id = parent_row["parent_id"]

            # Get all sibling leaf nodes under same parent
            siblings = c.execute(
                """
                SELECT sr.child_id as node_id
                FROM signature_relationships sr
                JOIN step_signatures ss ON sr.child_id = ss.id
                WHERE sr.parent_id = ? AND ss.is_semantic_umbrella = 0
                """,
                (parent_id,)
            ).fetchall()

            if not siblings:
                return {"cluster_mean": 0.5, "cluster_std": 0.25, "sibling_count": 0}

            # Get mean_success for each sibling
            sibling_means = []
            for sib in siblings:
                sib_stats = c.execute(
                    """
                    SELECT SUM(n) as total_n, SUM(n * mean_success) as weighted_sum
                    FROM plan_step_stats WHERE node_id = ?
                    """,
                    (sib["node_id"],)
                ).fetchone()

                if sib_stats and sib_stats["total_n"] and sib_stats["total_n"] > 0:
                    sib_mean = sib_stats["weighted_sum"] / sib_stats["total_n"]
                    sibling_means.append(sib_mean)

            if not sibling_means:
                return {"cluster_mean": 0.5, "cluster_std": 0.25, "sibling_count": len(siblings)}

            cluster_mean = sum(sibling_means) / len(sibling_means)
            if len(sibling_means) >= 2:
                variance = sum((m - cluster_mean) ** 2 for m in sibling_means) / (len(sibling_means) - 1)
                cluster_std = variance ** 0.5
            else:
                cluster_std = 0.25  # Default std when not enough siblings

            return {
                "cluster_mean": cluster_mean,
                "cluster_std": max(cluster_std, 0.05),  # Floor to avoid division by zero
                "sibling_count": len(siblings),
            }

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def decide_signature_placement(
        self,
        new_embedding: np.ndarray,
        parent_id: int,
        conn=None,
    ) -> tuple["PlacementDecision", Optional["StepSignature"], float]:
        """Decide placement for new signature using Welford-based z-scores.

        Per mycelium-br28: After cold start, use z-scores relative to parent's
        child_* Welford stats to determine if new signature is:
        - SIBLING: normal similarity (within 2 sigma of mean)
        - CHILD: somewhat different (2-3 sigma below mean)
        - MERGE: very similar (>3 sigma above mean)
        - NEW_CLUSTER: very different (>3 sigma below mean)

        The decision uses the parent's child similarity distribution to determine
        what is "normal" for this cluster. Z-score thresholds are defined in config.

        Args:
            new_embedding: The new signature's graph_embedding
            parent_id: The prospective parent's signature ID

        Returns:
            (decision, best_sibling, similarity) tuple where:
            - decision: PlacementDecision enum value
            - best_sibling: The most similar existing sibling (for MERGE), or None
            - similarity: Cosine similarity to best_sibling
        """
        from mycelium.config import (
            WELFORD_MERGE_THRESHOLD,      # 3.0 - z-score above which to merge
            WELFORD_SIBLING_THRESHOLD,    # -2.0 - z-score above which to add as sibling
            WELFORD_CHILD_THRESHOLD,      # -3.0 - z-score above which to add as child
        )

        def _do_decide(c):
            # 1. Get parent's Welford stats for child similarities
            stats = self.get_welford_stats(parent_id, conn=c)

            # 2. Find best matching sibling (needed for both Welford update and decision)
            best_sibling, best_sim = self._find_best_sibling(new_embedding, parent_id, conn=c)

            # 3. Update parent's child similarity distribution (per CLAUDE.md "System Independence")
            # CRITICAL: Update Welford stats BEFORE early return to bootstrap stats during cold start
            # This tracks inter-child similarity distribution for adaptive threshold decisions
            if best_sibling is not None:
                self.update_welford_child(parent_id, best_sim, conn=c)
                logger.debug(
                    "[placement] Updated Welford child stats for parent %d: sim=%.3f",
                    parent_id, best_sim
                )

            # 4. Handle insufficient data (cold start for this parent)
            # During cold start, default to SIBLING placement but stats are now being collected
            if stats is None or stats.get("child_n", 0) < 2:
                logger.debug(
                    "[placement] parent_id=%d has insufficient Welford data (n=%d), defaulting to SIBLING",
                    parent_id, stats.get("child_n", 0) if stats else 0
                )
                return PlacementDecision.SIBLING, best_sibling, best_sim if best_sibling else 0.0

            # If no siblings with embeddings, default to SIBLING
            if best_sibling is None:
                logger.debug(
                    "[placement] parent_id=%d has no siblings with embeddings, defaulting to SIBLING",
                    parent_id
                )
                return PlacementDecision.SIBLING, None, 0.0

            # 4. Compute z-score relative to parent's child similarity distribution
            child_mean = stats.get("child_mean", 0.0)
            child_m2 = stats.get("child_m2", 0.0)
            child_n = stats.get("child_n", 0)

            # Sample standard deviation
            std_sim = math.sqrt(child_m2 / max(1, child_n - 1)) if child_n > 1 else 0.0
            std_sim = max(0.01, std_sim)  # Avoid division by zero

            z_score = (best_sim - child_mean) / std_sim

            logger.debug(
                "[placement] parent_id=%d best_sibling=%d sim=%.3f "
                "child_mean=%.3f child_std=%.3f z_score=%.2f",
                parent_id, best_sibling.id, best_sim, child_mean, std_sim, z_score
            )

            # 5. Decision based on z-score thresholds
            if z_score > WELFORD_MERGE_THRESHOLD:
                # Very similar to existing sibling - consider merging
                logger.info(
                    "[placement] MERGE: z=%.2f > %.1f, merge with sibling %d (sim=%.3f)",
                    z_score, WELFORD_MERGE_THRESHOLD, best_sibling.id, best_sim
                )
                return PlacementDecision.MERGE, best_sibling, best_sim

            elif z_score > WELFORD_SIBLING_THRESHOLD:
                # Normal range - add as sibling (peer to existing children)
                logger.debug(
                    "[placement] SIBLING: z=%.2f in normal range [%.1f, %.1f]",
                    z_score, WELFORD_SIBLING_THRESHOLD, WELFORD_MERGE_THRESHOLD
                )
                return PlacementDecision.SIBLING, best_sibling, best_sim

            elif z_score > WELFORD_CHILD_THRESHOLD:
                # Somewhat different - create sub-cluster under best sibling
                logger.info(
                    "[placement] CHILD: z=%.2f in [%.1f, %.1f], create sub-cluster",
                    z_score, WELFORD_CHILD_THRESHOLD, WELFORD_SIBLING_THRESHOLD
                )
                return PlacementDecision.CHILD, best_sibling, best_sim

            else:
                # Very different - new cluster under root
                logger.info(
                    "[placement] NEW_CLUSTER: z=%.2f < %.1f, too different for this cluster",
                    z_score, WELFORD_CHILD_THRESHOLD
                )
                return PlacementDecision.NEW_CLUSTER, best_sibling, best_sim

        if conn is not None:
            return _do_decide(conn)
        else:
            with self._connection() as c:
                return _do_decide(c)

    def get_total_problems_solved(self, conn=None) -> int:
        """Get total number of problems solved (for cold start detection).

        Per mycelium-5cn0: Cold start = first 20 problems.
        During cold start, all leaves are flat under root collecting stats.

        Returns:
            Count of distinct successful problem_ids in mcts_dags
        """
        def _do_get(c):
            row = c.execute(
                "SELECT COUNT(DISTINCT problem_id) FROM mcts_dags WHERE success = 1"
            ).fetchone()
            return row[0] if row else 0

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def is_cold_start(self, conn=None) -> bool:
        """Check if we're in cold start mode.

        Per mycelium-5cn0: Cold start = first 20 problems.
        During cold start:
        - All new signatures auto-accepted as ROOT children
        - No umbrella promotions
        - No sibling vs child decisions
        - Just collect Welford stats

        Returns:
            True if total_problems_solved < COLD_START_THRESHOLD
        """
        from mycelium.config import COLD_START_PROBLEMS_THRESHOLD
        return self.get_total_problems_solved(conn=conn) < COLD_START_PROBLEMS_THRESHOLD

    # =========================================================================
    # PROPOSED SIGNATURES STAGING (per mycelium-xv09)
    # =========================================================================
    # Per CLAUDE.md "New Favorite Pattern": Single entry point for proposing signatures.
    # During cold start: auto-accept as root children.
    # After cold start: stage for Welford-based decision.

    # -------------------------------------------------------------------------
    # Private helpers for proposed_signatures table (consolidation pattern)
    # -------------------------------------------------------------------------

    def _fetch_pending_proposal(self, proposal_id: int, conn) -> Optional["ProposedSignature"]:
        """Fetch a single pending proposal by ID. Consolidated SQL helper."""
        from mycelium.step_signatures.models import ProposedSignature

        row = conn.execute(
            "SELECT * FROM proposed_signatures WHERE id = ? AND status = 'pending'",
            (proposal_id,)
        ).fetchone()

        if not row:
            return None
        return ProposedSignature.from_row(dict(row))

    def _update_proposal_status(
        self, proposal_id: int, status: str, reason: str, conn
    ) -> bool:
        """Update proposal status. Consolidated SQL helper."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """UPDATE proposed_signatures
               SET status = ?, decision_reason = ?, decided_at = ?
               WHERE id = ? AND status = 'pending'""",
            (status, reason, now, proposal_id),
        )
        return cursor.rowcount > 0

    def _fetch_pending_proposals(self, limit: int, conn) -> list["ProposedSignature"]:
        """Fetch pending proposals ordered by created_at. Consolidated SQL helper."""
        from mycelium.step_signatures.models import ProposedSignature

        cursor = conn.execute(
            """SELECT * FROM proposed_signatures
               WHERE status = 'pending'
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,)
        )
        return [ProposedSignature.from_row(dict(row)) for row in cursor]

    # -------------------------------------------------------------------------
    # Public API for proposals
    # -------------------------------------------------------------------------

    def stage_proposal(
        self,
        step_text: str,
        embedding: Optional[np.ndarray] = None,
        graph_embedding: Optional[np.ndarray] = None,
        computation_graph: Optional[str] = None,
        proposed_parent_id: Optional[int] = None,
        best_match_id: Optional[int] = None,
        best_match_sim: Optional[float] = None,
        dsl_hint: Optional[str] = None,
        extracted_values: Optional[dict] = None,
        origin_depth: int = 0,
        problem_context: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        conn=None,
    ) -> int:
        """Stage a signature proposal for later review by periodic tree review.

        Per CLAUDE.md "Negotiation between Tree and Planner":
        When refinement fails (can't decompose further), stage the proposal
        for review. Periodic tree review will use Welford stats to decide
        whether to accept (child or sibling) or reject.

        Args:
            step_text: The step description text
            embedding: Text embedding (for centroid)
            graph_embedding: Computation graph embedding (for routing)
            computation_graph: Structural graph representation
            proposed_parent_id: Suggested parent from routing
            best_match_id: Most similar existing signature
            best_match_sim: Similarity to best match
            dsl_hint: Operation hint from planner
            extracted_values: Extracted parameter values
            origin_depth: Depth where proposal originated
            problem_context: Original problem text (for context)
            rejection_reason: Why this step was rejected during negotiation

        Returns:
            The proposal ID
        """
        import json

        def _do_stage(c):
            # Serialize embeddings as blobs
            embedding_blob = embedding.tobytes() if embedding is not None else None
            graph_embedding_blob = graph_embedding.tobytes() if graph_embedding is not None else None
            extracted_values_json = json.dumps(extracted_values) if extracted_values else None

            # Include rejection reason in problem_context for review
            context = problem_context or ""
            if rejection_reason:
                context = f"[REJECTION: {rejection_reason}] {context}"

            cursor = c.execute(
                """INSERT INTO proposed_signatures
                   (step_text, embedding, graph_embedding, computation_graph,
                    proposed_parent_id, best_match_id, best_match_sim,
                    dsl_hint, extracted_values, status, origin_depth, problem_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    step_text,
                    embedding_blob,
                    graph_embedding_blob,
                    computation_graph,
                    proposed_parent_id,
                    best_match_id,
                    best_match_sim,
                    dsl_hint,
                    extracted_values_json,
                    origin_depth,
                    context,
                ),
            )
            proposal_id = cursor.lastrowid

            logger.info(
                "[proposals] Staged proposal id=%d: step='%s' best_match=%s (sim=%.3f) reason='%s'",
                proposal_id,
                step_text[:40],
                best_match_id,
                best_match_sim or 0.0,
                rejection_reason or "none",
            )
            return proposal_id

        if conn is not None:
            return _do_stage(conn)
        else:
            with self._connection() as c:
                result = _do_stage(c)
                c.commit()
                return result

    def propose_signature(
        self,
        step_text: str,
        embedding: Optional[np.ndarray],
        graph_embedding: Optional[np.ndarray] = None,
        computation_graph: Optional[str] = None,
        proposed_parent_id: Optional[int] = None,
        best_match_id: Optional[int] = None,
        best_match_sim: Optional[float] = None,
        dsl_hint: Optional[str] = None,
        extracted_values: Optional[dict] = None,
        origin_depth: int = 0,
        problem_context: Optional[str] = None,
        conn=None,
    ) -> tuple[int, bool]:
        """SINGLE ENTRY POINT for creating new signatures.

        Per mycelium-xv09 + CLAUDE.md New Favorite Pattern: Consolidates all
        signature creation logic into one entry point.

        - Cold start: Creates signature under root (flat structure)
        - Post cold start: Uses Welford-based placement decision

        Per CLAUDE.md System Independence: Automated placement decisions,
        no manual tree intervention required.

        Args:
            step_text: The step description text
            embedding: Text embedding (for centroid)
            graph_embedding: Computation graph embedding (for routing)
            computation_graph: Structural graph representation
            proposed_parent_id: Suggested parent from routing
            best_match_id: Most similar existing signature
            best_match_sim: Similarity to best match
            dsl_hint: Operation hint from planner
            extracted_values: Extracted parameter values
            origin_depth: Depth where proposal originated
            problem_context: Original problem text

        Returns:
            Tuple of (signature_id, placement_info):
            - signature_id: The created (or merged) signature ID
            - placement_info: String describing placement decision
        """
        def _do_propose(c):
            root = self.get_root()
            root_id = root.id if root else None

            # Check if we're in cold start mode
            if self.is_cold_start(conn=c):
                # Cold start: auto-accept as root child (flat structure)
                parent_id = root_id

                sig = self._create_signature_atomic(
                    c,
                    step_text=step_text,
                    embedding=embedding,
                    parent_id=parent_id,
                    dsl_hint=dsl_hint,
                    graph_embedding=graph_embedding,
                    extracted_values=extracted_values,
                    origin_depth=origin_depth,
                )
                logger.info(
                    "[proposals] Cold start: created signature id=%d type='%s' under root",
                    sig.id, sig.step_type
                )
                return (sig.id, "cold_start_root")

            # Post cold start: use Welford-based placement decision
            # Per mycelium-br28: decide_signature_placement uses z-scores
            actual_parent_id = proposed_parent_id if proposed_parent_id is not None else root_id

            if graph_embedding is not None and actual_parent_id is not None:
                decision, best_sibling, sim = self.decide_signature_placement(
                    graph_embedding, actual_parent_id, conn=c
                )

                if decision == PlacementDecision.MERGE and best_sibling is not None:
                    # Very similar to existing - merge (dedup)
                    logger.info(
                        "[proposals] MERGE: dedup to existing sig %d (sim=%.3f)",
                        best_sibling.id, sim
                    )
                    # Update centroid of existing signature
                    if embedding is not None:
                        self._update_centroid_atomic(c, best_sibling.id, embedding, update_last_used=True)
                    return (best_sibling.id, "merge_dedup")

                elif decision == PlacementDecision.CHILD and best_sibling is not None:
                    # Create as child of best_sibling (sub-cluster)
                    # First promote best_sibling to umbrella if needed
                    if not best_sibling.is_semantic_umbrella:
                        self._promote_to_umbrella_internal(c, best_sibling.id, skip_children_check=True)

                    sig = self._create_signature_atomic(
                        c,
                        step_text=step_text,
                        embedding=embedding,
                        parent_id=best_sibling.id,
                        dsl_hint=dsl_hint,
                        graph_embedding=graph_embedding,
                        extracted_values=extracted_values,
                        origin_depth=origin_depth,
                    )
                    logger.info(
                        "[proposals] CHILD: created sig %d under umbrella %d (z-score indicated sub-cluster)",
                        sig.id, best_sibling.id
                    )
                    return (sig.id, "child_subcluster")

                elif decision == PlacementDecision.NEW_CLUSTER:
                    # Very different from all existing - create under root
                    sig = self._create_signature_atomic(
                        c,
                        step_text=step_text,
                        embedding=embedding,
                        parent_id=root_id,
                        dsl_hint=dsl_hint,
                        graph_embedding=graph_embedding,
                        extracted_values=extracted_values,
                        origin_depth=origin_depth,
                    )
                    logger.info(
                        "[proposals] NEW_CLUSTER: created sig %d under root (very different from siblings)",
                        sig.id
                    )
                    return (sig.id, "new_cluster_root")

                # Default: SIBLING - create under proposed parent
                sig = self._create_signature_atomic(
                    c,
                    step_text=step_text,
                    embedding=embedding,
                    parent_id=actual_parent_id,
                    dsl_hint=dsl_hint,
                    graph_embedding=graph_embedding,
                    extracted_values=extracted_values,
                    origin_depth=origin_depth,
                )
                logger.info(
                    "[proposals] SIBLING: created sig %d under parent %d (normal similarity)",
                    sig.id, actual_parent_id
                )
                return (sig.id, "sibling_normal")

            # Fallback: no graph_embedding or no parent - create under root
            sig = self._create_signature_atomic(
                c,
                step_text=step_text,
                embedding=embedding,
                parent_id=root_id,
                dsl_hint=dsl_hint,
                graph_embedding=graph_embedding,
                extracted_values=extracted_values,
                origin_depth=origin_depth,
            )
            logger.info(
                "[proposals] FALLBACK: created sig %d under root (no graph_embedding)",
                sig.id
            )
            return (sig.id, "fallback_root")

        if conn is not None:
            return _do_propose(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_propose(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def accept_proposal(
        self,
        proposal_id: int,
        parent_id: Optional[int] = None,
        reason: str = "welford_accepted",
        conn=None,
    ) -> Optional[int]:
        """Accept a staged proposal and create signature.

        Args:
            proposal_id: ID of the proposal to accept
            parent_id: Override parent (defaults to proposal's proposed_parent_id)
            reason: Why the proposal was accepted

        Returns:
            Created signature ID, or None if proposal not found
        """
        def _do_accept(c):
            # Fetch using consolidated helper
            proposal = self._fetch_pending_proposal(proposal_id, c)
            if not proposal:
                logger.warning("[proposals] Proposal id=%d not found or not pending", proposal_id)
                return None

            # Determine parent
            actual_parent_id = parent_id if parent_id is not None else proposal.proposed_parent_id
            if actual_parent_id is None:
                root = self.get_root()
                actual_parent_id = root.id if root else None

            # Create the signature
            sig = self._create_signature_atomic(
                c,
                step_text=proposal.step_text,
                embedding=proposal.embedding,
                parent_id=actual_parent_id,
                dsl_hint=proposal.dsl_hint,
                graph_embedding=proposal.graph_embedding,
                extracted_values=proposal.extracted_values,
                origin_depth=proposal.origin_depth,
            )

            # Update using consolidated helper
            self._update_proposal_status(proposal_id, "accepted", reason, c)

            logger.info(
                "[proposals] Accepted proposal id=%d -> signature id=%d, reason='%s'",
                proposal_id, sig.id, reason
            )
            return sig.id

        if conn is not None:
            return _do_accept(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_accept(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def reject_proposal(
        self,
        proposal_id: int,
        reason: str,
        conn=None,
    ) -> bool:
        """Reject a staged proposal.

        Args:
            proposal_id: ID of the proposal to reject
            reason: Why the proposal was rejected

        Returns:
            True if rejection succeeded, False if proposal not found
        """
        def _do_reject(c):
            # Use consolidated helper
            success = self._update_proposal_status(proposal_id, "rejected", reason, c)
            if success:
                logger.info("[proposals] Rejected proposal id=%d, reason='%s'", proposal_id, reason)
            else:
                logger.warning("[proposals] Proposal id=%d not found or not pending", proposal_id)
            return success

        if conn is not None:
            return _do_reject(conn)
        else:
            with self._connection() as c:
                result = _do_reject(c)
                c.commit()
                return result

    def merge_proposal(
        self,
        proposal_id: int,
        merge_into_sig_id: int,
        reason: str = "welford_merged",
        conn=None,
    ) -> bool:
        """Merge a proposal into an existing signature.

        Updates the target signature's centroid with the proposal's embedding
        using running average (embedding_sum / embedding_count).

        Args:
            proposal_id: ID of the proposal to merge
            merge_into_sig_id: ID of signature to merge into
            reason: Why the proposal was merged

        Returns:
            True if merge succeeded, False if proposal or signature not found
        """
        def _do_merge(c):
            # Fetch using consolidated helper
            proposal = self._fetch_pending_proposal(proposal_id, c)
            if not proposal:
                logger.warning("[proposals] Proposal id=%d not found or not pending", proposal_id)
                return False

            if proposal.embedding is None:
                logger.warning("[proposals] Proposal id=%d has no embedding, cannot merge", proposal_id)
                return False

            # Fetch target signature
            sig_row = c.execute(
                "SELECT id, embedding_sum, embedding_count FROM step_signatures WHERE id = ?",
                (merge_into_sig_id,)
            ).fetchone()

            if not sig_row:
                logger.warning("[proposals] Target signature id=%d not found", merge_into_sig_id)
                return False

            # Update centroid using running average
            old_sum = unpack_embedding(sig_row["embedding_sum"])
            old_count = sig_row["embedding_count"] or 1

            if old_sum is not None:
                new_sum = old_sum + proposal.embedding
            else:
                new_sum = proposal.embedding
            new_count = old_count + 1
            new_centroid = new_sum / new_count

            # Pack for storage
            centroid_packed = pack_embedding(new_centroid)
            sum_packed = pack_embedding(new_sum)
            centroid_bucket = compute_centroid_bucket(new_centroid)

            c.execute(
                """UPDATE step_signatures
                   SET centroid = ?, centroid_bucket = ?, embedding_sum = ?, embedding_count = ?
                   WHERE id = ?""",
                (centroid_packed, centroid_bucket, sum_packed, new_count, merge_into_sig_id),
            )

            # Update using consolidated helper (with extended reason)
            merge_reason = f"{reason}:into_sig_{merge_into_sig_id}"
            self._update_proposal_status(proposal_id, "merged", merge_reason, c)

            # Invalidate caches
            invalidate_centroid_cache(merge_into_sig_id)
            invalidate_signature_cache(merge_into_sig_id)
            self.invalidate_centroid_matrix()

            logger.info(
                "[proposals] Merged proposal id=%d into signature id=%d (count=%d)",
                proposal_id, merge_into_sig_id, new_count
            )
            return True

        if conn is not None:
            return _do_merge(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_merge(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def get_pending_proposals(
        self,
        limit: int = 100,
        conn=None,
    ) -> list:
        """Get all pending proposals for review.

        Args:
            limit: Maximum number of proposals to return

        Returns:
            List of ProposedSignature objects
        """
        # Use consolidated helper
        if conn is not None:
            return self._fetch_pending_proposals(limit, conn)
        else:
            with self._connection() as c:
                return self._fetch_pending_proposals(limit, c)

    def get_proposal_stats(self, conn=None) -> dict:
        """Get summary statistics about proposals.

        Returns:
            Dict with counts by status and other stats
        """
        def _do_get(c):
            # Count by status
            cursor = c.execute(
                """SELECT status, COUNT(*) as count
                   FROM proposed_signatures
                   GROUP BY status"""
            )
            status_counts = {row["status"]: row["count"] for row in cursor}

            # Get total and recent counts
            total = sum(status_counts.values())
            pending = status_counts.get("pending", 0)
            accepted = status_counts.get("accepted", 0)
            rejected = status_counts.get("rejected", 0)
            merged = status_counts.get("merged", 0)

            # Recent activity (last 24 hours)
            cursor = c.execute(
                """SELECT COUNT(*) FROM proposed_signatures
                   WHERE created_at > datetime('now', '-1 day')"""
            )
            recent = cursor.fetchone()[0]

            return {
                "total": total,
                "pending": pending,
                "accepted": accepted,
                "rejected": rejected,
                "merged": merged,
                "recent_24h": recent,
                "acceptance_rate": accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0,
            }

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    # =========================================================================
    # AUTO-RESTRUCTURE (mycelium-heh3)
    # =========================================================================
    # Per CLAUDE.md System Independence: Fully automated tree restructuring.
    # Per CLAUDE.md New Favorite Pattern: Single entry point (maybe_restructure).

    def _get_last_restructure_count(self, conn) -> int:
        """Get last restructure problem count from db_metadata.

        Per CLAUDE.md New Favorite Pattern: Uses StateManager for db_metadata access.
        """
        from mycelium.data_layer.state_manager import get_state_manager
        return get_state_manager().get_last_restructure_count()

    def _set_last_restructure_count(self, conn, count: int) -> None:
        """Set last restructure problem count in db_metadata.

        Per CLAUDE.md New Favorite Pattern: Uses StateManager for db_metadata access.
        """
        from mycelium.data_layer.state_manager import get_state_manager
        get_state_manager().set_last_restructure_count(count)

    def maybe_restructure(self, problem_count: int, conn=None) -> dict:
        """SINGLE ENTRY POINT: Check if restructure should run and execute if needed.

        Per mycelium-heh3: Auto-restructure process runs every N problems.
        Per CLAUDE.md System Independence: Fully automated, no manual intervention.

        Uses last_restructure_count tracking instead of mod check for robustness.
        This ensures we never miss a restructure window if counts skip over intervals.

        Args:
            problem_count: Current total problems solved
            conn: Optional database connection

        Returns:
            Dict with restructure results:
            - ran: bool (whether restructure ran)
            - reason: str (why it did/didn't run)
            - clusters_created: int (umbrellas created)
            - orphans_cleaned: int (orphan umbrellas removed)
        """
        from mycelium.config import COLD_START_PROBLEMS_THRESHOLD, RESTRUCTURE_INTERVAL

        def _do_maybe(c):
            # Skip during cold start - collecting stats
            if problem_count < COLD_START_PROBLEMS_THRESHOLD:
                return {
                    "ran": False,
                    "reason": f"cold_start (problem {problem_count} < {COLD_START_PROBLEMS_THRESHOLD})",
                    "clusters_created": 0,
                    "orphans_cleaned": 0,
                }

            # Check if we've passed the interval since last restructure
            last_count = self._get_last_restructure_count(c)
            next_trigger = max(last_count + RESTRUCTURE_INTERVAL, COLD_START_PROBLEMS_THRESHOLD)

            if problem_count < next_trigger:
                return {
                    "ran": False,
                    "reason": f"not_due (problem {problem_count} < {next_trigger}, last={last_count})",
                    "clusters_created": 0,
                    "orphans_cleaned": 0,
                }

            # Run comprehensive tree review
            logger.info("[review] Running periodic tree review at problem %d (last=%d)", problem_count, last_count)
            result = self.run_periodic_tree_review(conn=c)

            # Update last restructure count
            self._set_last_restructure_count(c, problem_count)

            return result

        if conn is not None:
            return _do_maybe(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_maybe(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _run_restructure_pass(self, conn=None) -> dict:
        """Internal: Execute a full restructure pass.

        Steps:
        1. Get all root children with graph embeddings
        2. Compute pairwise similarities
        3. Detect clusters using Welford-guided thresholds
        4. Create umbrellas for clusters with >1 member
        5. Cleanup orphan umbrellas

        Returns:
            Dict with restructure results
        """
        def _do_pass(c):
            # 1. Get root children with embeddings
            root = self.get_root()
            if root is None:
                logger.warning("[restructure] No root found, skipping")
                return {"ran": True, "reason": "no_root", "clusters_created": 0, "orphans_cleaned": 0}

            children = self._get_children_with_embeddings(root.id, conn=c)
            if len(children) < 2:
                logger.info("[restructure] Not enough children (%d) for clustering", len(children))
                return {"ran": True, "reason": f"too_few_children ({len(children)})", "clusters_created": 0, "orphans_cleaned": 0}

            logger.info("[restructure] Analyzing %d root children for clustering", len(children))

            # 2. Compute pairwise similarities
            sim_matrix = self._compute_pairwise_similarities(children)

            # 3. Detect clusters using Welford-guided threshold
            # Get root's Welford stats for adaptive threshold
            stats = self.get_welford_stats(root.id, conn=c)
            if stats and stats.get("child_n", 0) > 5:
                # Use 2 * std as cluster threshold (similar items)
                child_std = self.get_welford_std(root.id, "child", conn=c)
                cluster_threshold = max(0.85, 1.0 - 2 * child_std) if child_std else 0.90
            else:
                # Cold start fallback
                cluster_threshold = 0.90

            logger.debug("[restructure] Cluster threshold: %.3f", cluster_threshold)

            clusters = self._detect_clusters(children, sim_matrix, cluster_threshold)
            logger.info("[restructure] Detected %d clusters", len(clusters))

            # 4. Create umbrellas for clusters with >1 member
            clusters_created = 0
            for cluster in clusters:
                if len(cluster) > 1:
                    success = self._create_umbrella_for_cluster(cluster, root.id, conn=c)
                    if success:
                        clusters_created += 1

            # 5. Cleanup orphan umbrellas
            orphans_cleaned = self._cleanup_orphan_umbrellas(conn=c)

            logger.info(
                "[restructure] Complete: %d clusters created, %d orphans cleaned",
                clusters_created, orphans_cleaned
            )

            return {
                "ran": True,
                "reason": "success",
                "clusters_created": clusters_created,
                "orphans_cleaned": orphans_cleaned,
            }

        if conn is not None:
            return _do_pass(conn)
        else:
            with self._connection() as c:
                return _do_pass(c)

    def _get_children_with_embeddings(self, parent_id: int, conn=None) -> list[tuple[int, str, np.ndarray]]:
        """Get all children of a signature that have graph embeddings.

        Returns:
            List of (signature_id, step_type, graph_embedding) tuples
        """
        def _do_get(c):
            cursor = c.execute(
                """SELECT s.id, s.step_type, s.graph_embedding
                   FROM step_signatures s
                   JOIN signature_relationships r ON s.id = r.child_id
                   WHERE r.parent_id = ? AND s.graph_embedding IS NOT NULL""",
                (parent_id,)
            )
            results = []
            for row in cursor:
                sig_id, step_type, emb_blob = row
                if emb_blob:
                    emb = unpack_embedding(emb_blob)
                    if emb is not None:
                        results.append((sig_id, step_type, emb))
            return results

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def _compute_pairwise_similarities(self, children: list[tuple[int, str, np.ndarray]]) -> np.ndarray:
        """Compute pairwise cosine similarities between children.

        Args:
            children: List of (sig_id, step_type, embedding) tuples

        Returns:
            NxN similarity matrix
        """
        n = len(children)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i][j] = 1.0
                else:
                    sim = cosine_similarity(children[i][2], children[j][2])
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

        return sim_matrix

    def _detect_clusters(
        self,
        children: list[tuple[int, str, np.ndarray]],
        sim_matrix: np.ndarray,
        threshold: float
    ) -> list[list[tuple[int, str, np.ndarray]]]:
        """Detect clusters of similar signatures using single-linkage clustering.

        Uses union-find for efficient cluster detection.

        Args:
            children: List of (sig_id, step_type, embedding) tuples
            sim_matrix: Pairwise similarity matrix
            threshold: Minimum similarity for clustering

        Returns:
            List of clusters, each cluster is a list of (sig_id, step_type, embedding)
        """
        n = len(children)
        if n == 0:
            return []

        # Union-find for clustering
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union similar items
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    union(i, j)

        # Group by cluster
        clusters_map = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(children[i])

        return list(clusters_map.values())

    def _create_umbrella_for_cluster(
        self,
        cluster: list[tuple[int, str, np.ndarray]],
        current_parent_id: int,
        conn=None
    ) -> bool:
        """Create an umbrella signature and move cluster members under it.

        Args:
            cluster: List of (sig_id, step_type, embedding) tuples to cluster
            current_parent_id: Current parent (will be umbrella's parent)
            conn: Database connection

        Returns:
            True if umbrella created successfully
        """
        def _do_create(c):
            if len(cluster) < 2:
                return False

            # Generate umbrella name from cluster step types
            step_types = [item[1] for item in cluster]
            common_prefix = self._find_common_prefix(step_types)
            umbrella_name = f"cluster:{common_prefix}" if common_prefix else f"cluster:{len(cluster)}_ops"

            # Compute centroid embedding for umbrella
            embeddings = [item[2] for item in cluster]
            centroid = np.mean(embeddings, axis=0)

            # Create umbrella signature
            umbrella = self._create_signature_atomic(
                c,
                step_text=umbrella_name,
                embedding=centroid,  # Use centroid as text embedding
                parent_problem="",
                origin_depth=0,
                parent_id=current_parent_id,
                graph_embedding=centroid,  # graph_centroid = centroid of children
            )

            # Promote to umbrella (skip children check - we're adding them next)
            self._promote_to_umbrella_internal(c, umbrella.id, skip_children_check=True)

            # Reparent cluster members under umbrella
            now = datetime.now(timezone.utc).isoformat()
            for sig_id, step_type, _ in cluster:
                # Remove old parent relationship
                c.execute(
                    "DELETE FROM signature_relationships WHERE child_id = ?",
                    (sig_id,)
                )
                # Add new parent relationship (condition = step_type)
                c.execute(
                    """INSERT INTO signature_relationships
                       (parent_id, child_id, condition, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (umbrella.id, sig_id, step_type or "clustered", now)
                )

            # Propagate centroid up
            self.propagate_graph_centroid_to_parents(c, umbrella.id, include_self=True)

            logger.info(
                "[restructure] Created umbrella '%s' (id=%d) with %d children: %s",
                umbrella_name, umbrella.id, len(cluster),
                [item[1][:20] for item in cluster]
            )
            return True

        if conn is not None:
            return _do_create(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_create(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _find_common_prefix(self, strings: list[str]) -> str:
        """Find common prefix of a list of strings."""
        if not strings:
            return ""
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        # Clean up trailing underscores/spaces
        return prefix.rstrip("_- ")

    def _cleanup_orphan_umbrellas(self, conn=None) -> int:
        """Archive umbrella signatures that have no children.

        Per CLAUDE.md System Independence: Uses soft-delete (is_archived=1) to
        preserve learning history. Permanent deletion loses valuable data.

        Per CLAUDE.md: Don't create umbrellas without children.
        This cleans up any orphaned umbrellas from previous operations.

        Returns:
            Number of orphan umbrellas archived
        """
        def _do_cleanup(c):
            # Find non-archived umbrellas with no children
            cursor = c.execute(
                """SELECT s.id, s.step_type FROM step_signatures s
                   WHERE s.is_semantic_umbrella = 1
                   AND s.is_root = 0
                   AND (s.is_archived = 0 OR s.is_archived IS NULL)
                   AND NOT EXISTS (
                       SELECT 1 FROM signature_relationships r WHERE r.parent_id = s.id
                   )"""
            )
            orphans = cursor.fetchall()

            if not orphans:
                return 0

            orphan_ids = [row[0] for row in orphans]
            logger.info(
                "[restructure] Found %d orphan umbrellas: %s",
                len(orphans), [(row[0], row[1][:30]) for row in orphans]
            )

            # Archive orphans (soft-delete per CLAUDE.md System Independence)
            for orphan_id in orphan_ids:
                # Remove from relationships (as child) - clean up graph structure
                c.execute(
                    "DELETE FROM signature_relationships WHERE child_id = ?",
                    (orphan_id,)
                )
                # Soft-delete: archive instead of permanent delete
                c.execute(
                    "UPDATE step_signatures SET is_archived = 1 WHERE id = ?",
                    (orphan_id,)
                )

            logger.info("[restructure] Archived %d orphan umbrellas", len(orphan_ids))
            return len(orphan_ids)

        if conn is not None:
            return _do_cleanup(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_cleanup(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _adopt_orphan_children(self, conn=None) -> int:
        """Adopt orphan children (non-root nodes without parents) under root.

        Per CLAUDE.md System Independence: When a parent is archived, its children
        may become orphans. Instead of immediate reparenting (which bypasses
        Welford-guided decisions), we defer to periodic review to adopt orphans.

        Orphan children are attached to root, where subsequent periodic reviews
        can use Welford stats to place them in appropriate clusters.

        Returns:
            Number of orphan children adopted
        """
        def _do_adopt(c):
            # Get root ID
            root = self.get_root()
            if not root:
                logger.warning("[review] No root found, cannot adopt orphans")
                return 0

            # Find non-root, non-archived signatures that have no parent
            cursor = c.execute(
                """SELECT s.id, s.step_type FROM step_signatures s
                   WHERE s.is_root = 0
                   AND (s.is_archived = 0 OR s.is_archived IS NULL)
                   AND NOT EXISTS (
                       SELECT 1 FROM signature_relationships r WHERE r.child_id = s.id
                   )"""
            )
            orphans = cursor.fetchall()

            if not orphans:
                return 0

            orphan_ids = [row[0] for row in orphans]
            logger.info(
                "[review] Found %d orphan children (no parent): %s",
                len(orphans), [(row[0], row[1][:30]) for row in orphans]
            )

            # Adopt orphans under root
            for orphan_id in orphan_ids:
                c.execute(
                    """INSERT OR IGNORE INTO signature_relationships
                       (parent_id, child_id, condition, routing_order)
                       VALUES (?, ?, 'adopted_orphan', 0)""",
                    (root.id, orphan_id)
                )

            logger.info("[review] Adopted %d orphan children under root", len(orphan_ids))
            return len(orphan_ids)

        if conn is not None:
            return _do_adopt(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_adopt(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _process_pending_proposals(self, conn=None) -> dict:
        """Process staged proposals using Welford stats to decide placement.

        Per CLAUDE.md "Negotiation between Tree and Planner":
        Proposals are staged when refinement fails. This method reviews them
        using Welford stats to decide:
        - ACCEPT as sibling: Normal similarity, reasonable performance expected
        - ACCEPT as child: Sub-cluster under best match
        - REJECT: Pattern already well-covered or consistently fails

        Uses adaptive z-scores just like negotiation rejection.

        Returns:
            Dict with accepted and rejected counts
        """
        from mycelium.step_signatures.models import ProposedSignature

        def _do_process(c):
            stats = {"accepted": 0, "rejected": 0, "deferred": 0}

            # Get pending proposals (oldest first)
            proposals = self.get_pending_proposals(limit=20, conn=c)
            if not proposals:
                return stats

            logger.info("[proposals] Processing %d pending proposals", len(proposals))

            for proposal in proposals:
                # Get best match info
                best_match_id = proposal.best_match_id
                best_match_sim = proposal.best_match_sim or 0.0

                # Decision criteria using Welford stats
                decision = self._decide_proposal_fate(proposal, conn=c)

                if decision["action"] == "accept":
                    # Accept and create signature
                    parent_id = decision.get("parent_id")
                    reason = decision.get("reason", "welford_accepted")

                    sig_id = self.accept_proposal(
                        proposal.id,
                        parent_id=parent_id,
                        reason=reason,
                        conn=c,
                    )
                    if sig_id:
                        stats["accepted"] += 1
                        logger.info(
                            "[proposals] Accepted proposal %d -> sig %d (parent=%s, reason=%s)",
                            proposal.id, sig_id, parent_id, reason
                        )

                elif decision["action"] == "reject":
                    # Reject proposal
                    reason = decision.get("reason", "welford_rejected")
                    self.reject_proposal(proposal.id, reason=reason, conn=c)
                    stats["rejected"] += 1
                    logger.info(
                        "[proposals] Rejected proposal %d: %s",
                        proposal.id, reason
                    )

                else:
                    # Defer - not enough data yet
                    stats["deferred"] += 1

            return stats

        if conn is not None:
            return _do_process(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_process(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _decide_proposal_fate(self, proposal: "ProposedSignature", conn=None) -> dict:
        """Decide whether to accept/reject a proposal using Welford stats.

        Per CLAUDE.md "Cluster Boundaries": Welford statistics guide
        accept/reject decisions. Uses adaptive z-scores.

        Returns:
            Dict with 'action' (accept/reject/defer) and 'reason'
        """
        def _do_decide(c):
            best_match_id = proposal.best_match_id
            best_match_sim = proposal.best_match_sim or 0.0

            # If no best match, accept as root child
            if best_match_id is None:
                root = self.get_root()
                return {
                    "action": "accept",
                    "parent_id": root.id if root else None,
                    "reason": "no_existing_match"
                }

            # Get best match's overall performance
            match_stats = self.get_node_stats_all_positions(best_match_id, conn=c)

            # Get cluster stats for comparison
            cluster_stats = self.get_cluster_stats(best_match_id, conn=c)

            # Decision logic using adaptive thresholds:

            # 1. Very high similarity -> merge/dedup (reject proposal, use existing)
            if best_match_sim >= 0.97:
                return {
                    "action": "reject",
                    "reason": f"high_similarity_dedup (sim={best_match_sim:.3f})"
                }

            # 2. Check if best match performs well (z-score relative to cluster)
            if match_stats["n"] >= 5 and cluster_stats["sibling_count"] >= 2:
                if cluster_stats["cluster_std"] > 0.01:
                    z_score = (match_stats["mean_success"] - cluster_stats["cluster_mean"]) / cluster_stats["cluster_std"]

                    # Best match is significantly underperforming -> accept proposal as sibling
                    if z_score < -1.5:
                        # Get parent of best match
                        best_match_parent = self.get_parent(best_match_id)
                        parent_id = best_match_parent.id if best_match_parent else None

                        return {
                            "action": "accept",
                            "parent_id": parent_id,
                            "reason": f"sibling_alternative (best_match z={z_score:.2f} underperforms)"
                        }

                    # Best match performs well and similar -> reject (covered)
                    if z_score > -0.5 and best_match_sim >= 0.85:
                        return {
                            "action": "reject",
                            "reason": f"well_covered (sim={best_match_sim:.3f}, match z={z_score:.2f})"
                        }

            # 3. Moderate similarity -> accept as child (sub-cluster)
            if best_match_sim >= 0.75:
                return {
                    "action": "accept",
                    "parent_id": best_match_id,  # Child of best match
                    "reason": f"child_subcluster (sim={best_match_sim:.3f})"
                }

            # 4. Low similarity -> accept as sibling under same parent
            if best_match_sim >= 0.5:
                best_match_parent = self.get_parent(best_match_id)
                parent_id = best_match_parent.id if best_match_parent else None
                return {
                    "action": "accept",
                    "parent_id": parent_id,
                    "reason": f"sibling_new_pattern (sim={best_match_sim:.3f})"
                }

            # 5. Very low similarity -> accept under root (new cluster)
            root = self.get_root()
            return {
                "action": "accept",
                "parent_id": root.id if root else None,
                "reason": f"new_cluster (sim={best_match_sim:.3f})"
            }

        if conn is not None:
            return _do_decide(conn)
        else:
            with self._connection() as c:
                return _do_decide(c)

    # =========================================================================
    # PERIODIC TREE REVIEW (per periodic tree review plan)
    # =========================================================================
    # Comprehensive tree review that runs every RESTRUCTURE_INTERVAL problems
    # Uses Welford stats to guide: deduplication, outlier relocation, sub-clustering

    def _backfill_missing_embeddings(self, conn) -> int:
        """Backfill graph_embedding for signatures missing them.

        Per CLAUDE.md "System Independence": Automated maintenance, no manual intervention.
        Ensures all signatures have embeddings before tree review can subcluster them.

        For math signatures: Extract computation_graph from DSL, embed it
        For decompose signatures: Embed the description (no computation graph)

        Returns:
            Number of signatures updated
        """
        from mycelium.step_signatures.graph_extractor import (
            extract_computation_graph,
            embed_computation_graph_sync,
        )
        from mycelium.embedder import get_embedding

        updated = 0

        # Find signatures missing graph_embedding
        rows = conn.execute(
            """SELECT id, dsl_script, description, dsl_type
               FROM step_signatures
               WHERE (graph_embedding IS NULL OR graph_embedding = '')
                 AND is_archived = 0
               LIMIT 100"""
        ).fetchall()

        for row in rows:
            sig_id = row["id"]
            dsl_script = row["dsl_script"]
            description = row["description"]
            dsl_type = row["dsl_type"]

            try:
                graph_embedding = None
                computation_graph = None

                # For math signatures: extract and embed computation graph
                if dsl_type == "math" and dsl_script:
                    computation_graph = extract_computation_graph(dsl_script)
                    if computation_graph:
                        graph_embedding = embed_computation_graph_sync(computation_graph)

                # For decompose signatures (or if math extraction failed): embed description
                if graph_embedding is None and description:
                    emb = get_embedding(description)
                    if emb is not None:
                        graph_embedding = emb

                if graph_embedding is None:
                    continue

                # Update signature
                updates = ["graph_embedding = ?"]
                values = [json.dumps(graph_embedding.tolist() if hasattr(graph_embedding, 'tolist') else list(graph_embedding))]

                if computation_graph:
                    updates.append("computation_graph = ?")
                    values.append(computation_graph)

                values.append(sig_id)
                conn.execute(
                    f"UPDATE step_signatures SET {', '.join(updates)} WHERE id = ?",
                    tuple(values)
                )
                updated += 1

            except Exception as e:
                logger.warning("[backfill] Failed to backfill sig %d: %s", sig_id, e)
                continue

        if updated > 0:
            logger.info("[backfill] Updated %d signatures with embeddings", updated)

        return updated

    def _backfill_welford_child_stats(self, conn) -> int:
        """Backfill Welford child stats for umbrellas missing them.

        Per CLAUDE.md "System Independence": Automated maintenance.
        Computes pairwise similarities among children and updates Welford stats.
        This bootstraps stats for existing umbrellas that were created before
        Welford tracking was added.

        Returns:
            Number of umbrellas updated
        """
        # Find umbrellas with children but no child stats
        rows = conn.execute(
            """SELECT u.id, COUNT(sr.child_id) as num_children
               FROM step_signatures u
               JOIN signature_relationships sr ON sr.parent_id = u.id
               LEFT JOIN welford_stats ws ON ws.signature_id = u.id
               WHERE u.is_semantic_umbrella = 1
                 AND u.is_archived = 0
                 AND (ws.child_n IS NULL OR ws.child_n = 0)
               GROUP BY u.id
               HAVING COUNT(sr.child_id) >= 2
               LIMIT 20"""
        ).fetchall()

        updated = 0
        for row in rows:
            umbrella_id = row[0]
            try:
                # Get children with embeddings
                children = self._get_children_with_embeddings(umbrella_id, conn=conn)
                if len(children) < 2:
                    continue

                # Compute pairwise similarities and update Welford stats
                sim_matrix = self._compute_pairwise_similarities(children)
                n = len(children)

                for i in range(n):
                    for j in range(i + 1, n):
                        self.update_welford_child(umbrella_id, sim_matrix[i][j], conn=conn)

                updated += 1
                logger.debug(
                    "[backfill] Updated Welford child stats for umbrella %d (%d pairs)",
                    umbrella_id, n * (n - 1) // 2
                )

            except Exception as e:
                logger.warning("[backfill] Failed to backfill Welford for umbrella %d: %s", umbrella_id, e)
                continue

        if updated > 0:
            logger.info("[backfill] Updated Welford child stats for %d umbrellas", updated)

        return updated

    def run_periodic_tree_review(self, conn=None) -> dict:
        """Comprehensive tree review using Welford stats.

        Per CLAUDE.md: System independence - automated optimization.
        Reviews all umbrella nodes (not just root) for:
        - Deduplication (very similar signatures → merge)
        - Outlier detection (poor fits → relocate)
        - Sub-clustering (high variance → split)

        Returns:
            Dict with review stats: merges, moves, clusters_created, orphans_adopted, orphans_cleaned
        """
        from mycelium import config

        def _do_review(c):
            stats = {"ran": True, "merges": 0, "moves": 0, "clusters_created": 0,
                     "orphans_adopted": 0, "orphans_cleaned": 0,
                     "embeddings_backfilled": 0, "welford_backfilled": 0}

            # 0a. BACKFILL: Ensure all signatures have embeddings (system independence)
            stats["embeddings_backfilled"] = self._backfill_missing_embeddings(c)

            # 0b. BACKFILL: Ensure all umbrellas have Welford child stats (system independence)
            stats["welford_backfilled"] = self._backfill_welford_child_stats(c)

            # 1. Get all umbrella nodes (routers) for review
            umbrellas = self._get_all_umbrellas(conn=c)
            logger.info("[review] Starting periodic tree review: %d umbrellas to check", len(umbrellas))

            for umbrella in umbrellas:
                # 2. Get Welford stats for this cluster
                welford = self.get_welford_stats(umbrella.id, conn=c)

                # 3. Get total child count (for fan-out check) and children with embeddings
                total_children = c.execute(
                    "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                    (umbrella.id,)
                ).fetchone()[0]
                children = self._get_children_with_embeddings(umbrella.id, conn=c)

                # Log if many children lack embeddings (limits subclustering ability)
                if total_children > 0 and len(children) < total_children * 0.5:
                    logger.warning(
                        "[review] Umbrella %d has %d total children but only %d with embeddings",
                        umbrella.id, total_children, len(children)
                    )

                if len(children) < 2:
                    continue  # Nothing to review

                # 4. Compute pairwise similarities
                sim_matrix = self._compute_pairwise_similarities(children)

                # 5. DEDUPLICATION: Merge very similar signatures
                merge_threshold = self._compute_merge_threshold(welford)
                merges = self._merge_duplicates(children, sim_matrix, merge_threshold, conn=c)
                stats["merges"] += merges

                # Refresh children list after merges (some may have been archived)
                if merges > 0:
                    children = self._get_children_with_embeddings(umbrella.id, conn=c)
                    if len(children) < 2:
                        continue
                    sim_matrix = self._compute_pairwise_similarities(children)

                # 6. OUTLIER DETECTION: Find nodes that don't fit
                outlier_threshold = self._compute_outlier_threshold(welford)
                outliers = self._detect_outliers_in_cluster(children, sim_matrix, outlier_threshold)
                moves = self._relocate_outliers(outliers, umbrella.id, conn=c)
                stats["moves"] += moves

                # 7. SUB-CLUSTERING: Split if Welford stats indicate heterogeneity
                should_split, split_reason = self._should_subcluster(welford, children, total_children)
                if should_split:
                    logger.info(
                        "[review] Subclustering umbrella %d: %s",
                        umbrella.id, split_reason
                    )
                    # Refresh children after any moves
                    children = self._get_children_with_embeddings(umbrella.id, conn=c)
                    if len(children) >= config.RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT:
                        sim_matrix = self._compute_pairwise_similarities(children)
                        new_clusters = self._create_subclusters_for_umbrella(
                            umbrella.id, children, sim_matrix, welford, conn=c
                        )
                        stats["clusters_created"] += new_clusters

            # 8. Adopt orphan children (nodes without parents) under root
            # Per CLAUDE.md System Independence: deferred reparenting via periodic review
            stats["orphans_adopted"] = self._adopt_orphan_children(conn=c)

            # 9. Cleanup orphan umbrellas (empty routers)
            stats["orphans_cleaned"] = self._cleanup_orphan_umbrellas(conn=c)

            # 10. Process pending proposals (staged signatures from failed refinement)
            proposals_result = self._process_pending_proposals(conn=c)
            stats["proposals_accepted"] = proposals_result.get("accepted", 0)
            stats["proposals_rejected"] = proposals_result.get("rejected", 0)

            # 11. Apply embedding drift batch update (per mycelium-ieq4)
            # Per CLAUDE.md: "High-traffic signatures become semantic attractors"
            drift_result = self.apply_embedding_drift_batch(conn=c)
            stats["drift_nodes_updated"] = drift_result.get("nodes_updated", 0)
            stats["drift_avg_magnitude"] = drift_result.get("avg_drift_magnitude", 0.0)

            # 12. Recompute router centroids after drift updates
            if stats["drift_nodes_updated"] > 0:
                stats["routers_recomputed"] = self.recompute_router_centroids(conn=c)
            else:
                stats["routers_recomputed"] = 0

            logger.info(
                "[review] Tree review complete: %d merges, %d moves, %d clusters, %d adopted, "
                "%d orphan umbrellas, %d proposals accepted, %d rejected, %d drift updates (avg=%.4f), %d routers recomputed",
                stats["merges"], stats["moves"], stats["clusters_created"], stats["orphans_adopted"],
                stats["orphans_cleaned"], stats["proposals_accepted"], stats["proposals_rejected"],
                stats["drift_nodes_updated"], stats["drift_avg_magnitude"], stats["routers_recomputed"]
            )

            return stats

        if conn is not None:
            return _do_review(conn)
        else:
            with self._connection() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    result = _do_review(c)
                    c.commit()
                    return result
                except Exception:
                    c.rollback()
                    raise

    def _get_all_umbrellas(self, conn=None) -> list["StepSignature"]:
        """Get all umbrella (router) signatures for tree review.

        Returns:
            List of StepSignature objects that are semantic umbrellas
        """
        def _do_get(c):
            cursor = c.execute(
                """SELECT * FROM step_signatures
                   WHERE is_semantic_umbrella = 1 AND is_archived = 0"""
            )
            return [StepSignature.from_row(dict(row)) for row in cursor]

        if conn is not None:
            return _do_get(conn)
        else:
            with self._connection() as c:
                return _do_get(c)

    def _compute_merge_threshold(self, welford: Optional[dict]) -> float:
        """Compute merge threshold from cluster's Welford stats.

        Very high similarity relative to cluster's distribution = duplicate.
        Uses z-score: merge if sim > mean + 3*std

        Args:
            welford: Welford stats dict or None

        Returns:
            Similarity threshold for merging
        """
        from mycelium.config import RESTRUCTURE_MERGE_FLOOR

        if welford is None or welford.get("child_n", 0) < 3:
            return 0.98  # Default: very conservative during cold start

        child_mean = welford.get("child_mean", 0.85)
        child_std = self._welford_std_from_stats(welford, "child")

        # Merge threshold: 3 sigma above mean (very similar = duplicate)
        threshold = min(0.99, child_mean + 3 * child_std)
        return max(RESTRUCTURE_MERGE_FLOOR, threshold)  # Floor to avoid false merges

    def _compute_outlier_threshold(self, welford: Optional[dict]) -> float:
        """Compute outlier threshold from cluster's Welford stats.

        Very low similarity relative to cluster = outlier, should move.
        Uses z-score: outlier if avg_sim < mean - 2*std

        Args:
            welford: Welford stats dict or None

        Returns:
            Similarity threshold for outlier detection
        """
        if welford is None or welford.get("child_n", 0) < 3:
            return 0.5  # Default: lenient during cold start

        child_mean = welford.get("child_mean", 0.85)
        child_std = self._welford_std_from_stats(welford, "child")

        # Outlier threshold: 2 sigma below mean
        threshold = max(0.3, child_mean - 2 * child_std)
        return threshold

    def _welford_std_from_stats(self, welford: dict, stat_type: str) -> float:
        """Compute standard deviation from Welford stats dict.

        Args:
            welford: Stats dict with {stat_type}_n and {stat_type}_m2
            stat_type: "route", "child", etc.

        Returns:
            Sample standard deviation
        """
        import math
        n = welford.get(f"{stat_type}_n", 0)
        m2 = welford.get(f"{stat_type}_m2", 0.0)
        if n < 2:
            return 0.0
        variance = m2 / (n - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    def _should_subcluster(
        self, welford: Optional[dict], children: list, total_children: int = None
    ) -> tuple[bool, str]:
        """Decide if cluster should be split based on Welford stats.

        Per CLAUDE.md "System Independence": Use Welford-guided adaptive thresholds.

        Split conditions (checked in order):
        1. High fan-out: total_children > MAX_CHILDREN_PER_PARENT (too many for efficient routing)
        2. High CV: child_std/child_mean > CV_THRESHOLD (relative heterogeneity)
        3. High variance: child_std > VARIANCE_THRESHOLD (absolute heterogeneity)

        Args:
            welford: Welford stats for the umbrella
            children: List of children with embeddings (for similarity computation)
            total_children: Total children count from relationships (for fan-out check)

        Returns:
            Tuple of (should_split, reason_string)
        """
        from mycelium.config import (
            RESTRUCTURE_VARIANCE_THRESHOLD,
            RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT,
            MAX_CHILDREN_PER_PARENT,
            RESTRUCTURE_CV_THRESHOLD,
        )

        num_children_with_embeddings = len(children)
        # Use total_children for fan-out check if provided, else fall back to children with embeddings
        num_children_total = total_children if total_children is not None else num_children_with_embeddings

        # Condition 1: High fan-out forces split regardless of Welford stats
        # Per CLAUDE.md: System should automatically balance tree structure
        # Check total children (not just ones with embeddings) for fan-out
        if num_children_total > MAX_CHILDREN_PER_PARENT:
            if num_children_with_embeddings < RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT:
                return False, f"high_fanout_but_no_embeddings (total={num_children_total}, with_emb={num_children_with_embeddings})"
            return True, f"high_fanout ({num_children_total} > {MAX_CHILDREN_PER_PARENT})"

        # Minimum children with embeddings required to split meaningfully
        if num_children_with_embeddings < RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT:
            return False, f"too_few_children ({num_children_with_embeddings} < {RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT})"

        # Need Welford stats for remaining checks
        if welford is None:
            return False, "no_welford_stats"

        child_n = welford.get("child_n", 0)
        if child_n < 3:
            return False, f"insufficient_welford_data (n={child_n})"

        child_mean = welford.get("child_mean", 0.0)
        child_std = self._welford_std_from_stats(welford, "child")

        # Condition 2: High Coefficient of Variation (relative heterogeneity)
        # CV = std/mean; high CV means children vary significantly relative to mean similarity
        # Per CLAUDE.md: Prefer adaptive thresholds over hard-coded values
        if child_mean > 0.1:  # Avoid division issues with very low mean
            cv = child_std / child_mean
            if cv > RESTRUCTURE_CV_THRESHOLD:
                return True, f"high_cv (cv={cv:.3f} > {RESTRUCTURE_CV_THRESHOLD}, std={child_std:.3f}, mean={child_mean:.3f})"

        # Condition 3: High absolute variance (fallback)
        if child_std > RESTRUCTURE_VARIANCE_THRESHOLD:
            return True, f"high_variance (std={child_std:.3f} > {RESTRUCTURE_VARIANCE_THRESHOLD})"

        return False, f"cohesive_cluster (cv={child_std/child_mean:.3f}, std={child_std:.3f}, n={num_children_with_embeddings})"

    def _merge_duplicates(
        self,
        children: list[tuple[int, str, np.ndarray]],
        sim_matrix: np.ndarray,
        threshold: float,
        conn=None,
    ) -> int:
        """Merge signatures with similarity above threshold.

        Keeps the signature with more usage (higher exec_n).
        Archives the merged signature.

        Args:
            children: List of (sig_id, step_type, embedding) tuples
            sim_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for merging
            conn: Database connection

        Returns:
            Number of merges performed
        """
        merged_count = 0
        merged_ids = set()

        for i in range(len(children)):
            if children[i][0] in merged_ids:
                continue

            for j in range(i + 1, len(children)):
                if children[j][0] in merged_ids:
                    continue

                if sim_matrix[i][j] >= threshold:
                    # Merge: keep the one with more usage
                    keeper_id, merged_id = self._choose_keeper_by_usage(
                        children[i][0], children[j][0], conn
                    )
                    success = self._merge_signatures_internal(keeper_id, merged_id, conn)
                    if success:
                        merged_ids.add(merged_id)
                        merged_count += 1
                        logger.info(
                            "[review] Merged duplicate: %d -> %d (sim=%.3f)",
                            merged_id, keeper_id, sim_matrix[i][j]
                        )

        return merged_count

    def _choose_keeper_by_usage(self, sig_a_id: int, sig_b_id: int, conn) -> tuple[int, int]:
        """Choose which signature to keep based on usage stats.

        Returns:
            Tuple of (keeper_id, merged_id)
        """
        stats_a = self.get_welford_stats(sig_a_id, conn=conn)
        stats_b = self.get_welford_stats(sig_b_id, conn=conn)

        usage_a = (stats_a.get("exec_n", 0) if stats_a else 0)
        usage_b = (stats_b.get("exec_n", 0) if stats_b else 0)

        if usage_a >= usage_b:
            return (sig_a_id, sig_b_id)
        return (sig_b_id, sig_a_id)

    def _merge_signatures_internal(self, keeper_id: int, merged_id: int, conn) -> bool:
        """Merge two signatures, keeping one and archiving the other.

        - Updates keeper's centroid (average with merged)
        - Reparents any children of merged to keeper
        - Removes merged from tree
        - Archives merged signature

        Args:
            keeper_id: Signature to keep
            merged_id: Signature to archive
            conn: Database connection

        Returns:
            True if merge succeeded
        """
        try:
            keeper = self.get_signature(keeper_id)
            merged = self.get_signature(merged_id)
            if not keeper or not merged:
                return False

            # Update keeper's graph centroid (average with merged)
            # Per CLAUDE.md "New Favorite Pattern": Use consistent JSON format for graph_embedding
            if keeper.graph_embedding is not None and merged.graph_embedding is not None:
                keeper_emb = np.array(keeper.graph_embedding)
                merged_emb = np.array(merged.graph_embedding)
                new_centroid = (keeper_emb + merged_emb) / 2
                conn.execute(
                    "UPDATE step_signatures SET graph_embedding = ? WHERE id = ?",
                    (json.dumps(new_centroid.tolist()), keeper_id)
                )
                invalidate_signature_cache(keeper_id)

            # Reparent any children of merged to keeper
            conn.execute(
                "UPDATE signature_relationships SET parent_id = ? WHERE parent_id = ?",
                (keeper_id, merged_id)
            )

            # If merged had children, keeper becomes umbrella
            cursor = conn.execute(
                "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                (keeper_id,)
            )
            child_count = cursor.fetchone()[0]
            if child_count > 0 and not keeper.is_semantic_umbrella:
                self._promote_to_umbrella_internal(conn, keeper_id, skip_children_check=True)

            # Remove merged from its parent
            conn.execute(
                "DELETE FROM signature_relationships WHERE child_id = ?",
                (merged_id,)
            )

            # Archive merged signature (don't delete - preserve history)
            conn.execute(
                "UPDATE step_signatures SET is_archived = 1 WHERE id = ?",
                (merged_id,)
            )

            # Propagate centroid changes
            self.propagate_graph_centroid_to_parents(conn, keeper_id, include_self=True)

            return True
        except Exception as e:
            logger.warning("[review] Failed to merge %d -> %d: %s", merged_id, keeper_id, e)
            return False

    def _detect_outliers_in_cluster(
        self,
        children: list[tuple[int, str, np.ndarray]],
        sim_matrix: np.ndarray,
        threshold: float,
    ) -> list[tuple[int, float]]:
        """Find children whose avg similarity to siblings is below threshold.

        Args:
            children: List of (sig_id, step_type, embedding) tuples
            sim_matrix: Pairwise similarity matrix
            threshold: Outlier threshold

        Returns:
            List of (sig_id, avg_similarity) for outliers
        """
        outliers = []
        n = len(children)

        for i in range(n):
            # Compute average similarity to all siblings
            sims = [sim_matrix[i][j] for j in range(n) if i != j]
            avg_sim = sum(sims) / len(sims) if sims else 0

            if avg_sim < threshold:
                outliers.append((children[i][0], avg_sim))

        return outliers

    def _relocate_outliers(
        self,
        outliers: list[tuple[int, float]],
        current_parent_id: int,
        conn=None,
    ) -> int:
        """Move outliers to better-fitting clusters or root.

        Args:
            outliers: List of (sig_id, avg_similarity) tuples
            current_parent_id: ID of current parent umbrella
            conn: Database connection

        Returns:
            Number of signatures moved
        """
        from mycelium.config import RESTRUCTURE_OUTLIER_IMPROVEMENT

        moved = 0

        for sig_id, avg_sim in outliers:
            # Find best matching cluster for this signature
            best_parent_id, best_sim = self._find_best_cluster_for_signature(sig_id, current_parent_id, conn)

            if best_parent_id is not None and best_sim > avg_sim + RESTRUCTURE_OUTLIER_IMPROVEMENT:
                # Found better home - move it
                success = self._move_signature_to_parent(sig_id, best_parent_id, conn)
                if success:
                    moved += 1
                    logger.info(
                        "[review] Moved outlier %d to cluster %d (old_sim=%.3f, new_sim=%.3f)",
                        sig_id, best_parent_id, avg_sim, best_sim
                    )
            elif avg_sim < 0.5:
                # Very poor fit everywhere - move to root as new cluster seed
                root = self.get_root()
                if root and root.id != current_parent_id:
                    success = self._move_signature_to_parent(sig_id, root.id, conn)
                    if success:
                        moved += 1
                        logger.info("[review] Moved outlier %d to root (avg_sim=%.3f)", sig_id, avg_sim)

        return moved

    def _find_best_cluster_for_signature(
        self,
        sig_id: int,
        exclude_parent_id: int,
        conn=None,
    ) -> tuple[Optional[int], float]:
        """Find the best-fitting cluster for a signature.

        Compares signature's graph_embedding to all umbrella centroids.

        Args:
            sig_id: Signature to find home for
            exclude_parent_id: Don't consider this parent (current parent)
            conn: Database connection

        Returns:
            Tuple of (best_parent_id, similarity) or (None, 0)
        """
        sig = self.get_signature(sig_id)
        if sig is None or sig.graph_embedding is None:
            return (None, 0.0)

        sig_emb = np.array(sig.graph_embedding)

        # Get all umbrellas
        umbrellas = self._get_all_umbrellas(conn=conn)

        best_parent_id = None
        best_sim = 0.0

        for umbrella in umbrellas:
            if umbrella.id == exclude_parent_id:
                continue
            if umbrella.graph_embedding is None:
                continue

            umb_emb = np.array(umbrella.graph_embedding)
            sim = cosine_similarity(sig_emb, umb_emb)

            if sim > best_sim:
                best_sim = sim
                best_parent_id = umbrella.id

        return (best_parent_id, best_sim)

    def _move_signature_to_parent(self, sig_id: int, new_parent_id: int, conn) -> bool:
        """Move a signature from current parent to new parent.

        Args:
            sig_id: Signature to move
            new_parent_id: New parent ID
            conn: Database connection

        Returns:
            True if move succeeded
        """
        try:
            # Get current parent
            old_parent = self.get_parent(sig_id)
            old_parent_id = old_parent.id if old_parent else None

            # Remove from old parent
            conn.execute(
                "DELETE FROM signature_relationships WHERE child_id = ?",
                (sig_id,)
            )

            # Get signature for condition
            sig = self.get_signature(sig_id)
            condition = sig.step_type if sig else ""

            # Add to new parent
            conn.execute(
                "INSERT INTO signature_relationships (parent_id, child_id, condition) VALUES (?, ?, ?)",
                (new_parent_id, sig_id, condition)
            )

            # Update depth
            new_parent = self.get_signature(new_parent_id)
            new_depth = (new_parent.depth + 1) if new_parent and new_parent.depth else 1
            conn.execute(
                "UPDATE step_signatures SET depth = ? WHERE id = ?",
                (new_depth, sig_id)
            )

            # Propagate centroid changes to both old and new parents
            if old_parent_id:
                self.propagate_graph_centroid_to_parents(conn, old_parent_id, include_self=True)
            self.propagate_graph_centroid_to_parents(conn, new_parent_id, include_self=True)

            return True
        except Exception as e:
            logger.warning("[review] Failed to move sig %d to parent %d: %s", sig_id, new_parent_id, e)
            return False

    def _create_subclusters_for_umbrella(
        self,
        parent_id: int,
        children: list[tuple[int, str, np.ndarray]],
        sim_matrix: np.ndarray,
        welford: Optional[dict],
        conn=None,
    ) -> int:
        """Split heterogeneous umbrella into tighter sub-clusters.

        Uses existing union-find clustering with adaptive threshold.

        Args:
            parent_id: The umbrella to split
            children: List of (sig_id, step_type, embedding) tuples
            sim_matrix: Pairwise similarity matrix
            welford: Welford stats for adaptive threshold
            conn: Database connection

        Returns:
            Number of new clusters created
        """
        import math

        # Per CLAUDE.md "System Independence": Welford-guided adaptive thresholds
        # To SPLIT a heterogeneous cluster into tighter groups:
        # - Use threshold = mean + k*std when std is meaningful (k > 0)
        # - Fall back to percentile when std is too small (tight distribution)
        # - Goal: create meaningful splits without fragmenting into singletons

        n = len(children)
        if n < 2:
            return 0

        # Compute mean and std from actual similarity matrix (Welford-style)
        sims = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        if not sims:
            return 0

        # Online Welford computation for mean and std
        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims) if len(sims) > 1 else 0
        std_sim = math.sqrt(variance)
        cv = std_sim / mean_sim if mean_sim > 0 else 0  # Coefficient of variation

        # Choose threshold strategy based on distribution characteristics
        if cv > 0.1:
            # High CV: Welford-guided threshold works well
            cluster_threshold = min(0.95, mean_sim + 2.0 * std_sim)
            method = "welford"
        else:
            # Low CV (tight distribution): Use percentile-based threshold
            # Top 3% creates meaningful clusters without over-fragmenting
            sorted_sims = sorted(sims, reverse=True)
            percentile_idx = max(1, int(len(sorted_sims) * 0.03))  # Top 3%
            cluster_threshold = sorted_sims[percentile_idx]
            method = "percentile_3"

        # Ensure threshold is in reasonable range
        cluster_threshold = max(0.85, min(0.95, cluster_threshold))

        logger.info(
            "[subcluster] %s threshold %.3f (mean=%.3f, std=%.3f, cv=%.3f, n=%d)",
            method, cluster_threshold, mean_sim, std_sim, cv, len(sims)
        )

        # Detect sub-clusters using union-find
        clusters = self._detect_clusters(children, sim_matrix, cluster_threshold)

        # Only create umbrellas for clusters with 2+ members
        clusters_created = 0
        for cluster in clusters:
            if len(cluster) >= 2:
                success = self._create_umbrella_for_cluster(cluster, parent_id, conn=conn)
                if success:
                    clusters_created += 1

        return clusters_created


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_step_db: Optional[StepSignatureDB] = None


def get_step_db() -> StepSignatureDB:
    """Get the singleton StepSignatureDB instance.

    Per CLAUDE.md "New Favorite Pattern": Consolidate database access through
    a single data layer. Use this instead of creating new StepSignatureDB()
    instances throughout the codebase.

    Returns:
        StepSignatureDB: The singleton instance
    """
    global _step_db
    if _step_db is None:
        _step_db = StepSignatureDB()
        logger.debug("[db] Created singleton StepSignatureDB instance")
    return _step_db


def reset_step_db() -> None:
    """Reset the singleton StepSignatureDB instance.

    Primarily for testing - allows tests to get a fresh instance.
    Also clears all caches to ensure fresh state.
    """
    global _step_db
    _step_db = None
    # Clear signature and children caches to avoid stale data between tests
    from mycelium.step_signatures.utils import invalidate_signature_cache
    invalidate_signature_cache()
    logger.debug("[db] Reset singleton StepSignatureDB instance and caches")
