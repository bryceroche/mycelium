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
from typing import Optional

import numpy as np

# Version for centroid matrix cache (increment to invalidate old caches)
_CENTROID_CACHE_VERSION = 1

# Canonical atomic operations for embedding-based complexity detection
# Per CLAUDE.md: "prefer embedding similarity over keyword matching"
ATOMIC_OPERATIONS = ["ADD(a, b)", "SUB(a, b)", "MUL(a, b)", "DIV(a, b)"]
ATOMIC_SIMILARITY_THRESHOLD = 0.70  # Below this, step is unknown/complex
ATOMIC_GAP_THRESHOLD = 0.03  # Gap between best and 2nd best match; below this = multi-part


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

from mycelium.data_layer import get_db, configure_connection
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
    from mycelium.config import BIG_BANG_TARGET_SIGNATURES

    # Smooth exponential curve: rises quickly at first, asymptotes to 1
    tau = BIG_BANG_TARGET_SIGNATURES / 3.0  # ~3tau to reach 95%
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
    gap_factor = max(0.0, min(1.0, gap * 2.0))  # Scale to 0-1 range

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


def get_adaptive_thresholds(conn) -> AdaptiveThresholds:
    """Get adaptive thresholds from db_metadata, with cold-start defaults.

    Cold-start defaults are conservative:
    - dedup_threshold: 0.95 (very high sim = same node)
    - cluster_threshold: 0.80 (moderately high sim = same cluster)

    As data accumulates, thresholds adapt to learned distributions.
    """
    from mycelium.config import (
        ADAPTIVE_THRESHOLD_K,  # Number of stddevs below mean
        COLD_START_DEDUP_THRESHOLD,
        COLD_START_CLUSTER_THRESHOLD,
        ADAPTIVE_MIN_SAMPLES,  # Min samples before using learned thresholds
    )

    # Read Welford stats from db_metadata
    cursor = conn.execute(
        "SELECT key, value FROM db_metadata WHERE key LIKE 'sim_stats_%'"
    )
    stats = {row[0]: json.loads(row[1]) for row in cursor}

    match_count = stats.get('sim_stats_match_count', {}).get('value', 0)
    match_mean = stats.get('sim_stats_match_mean', {}).get('value', 0.0)
    match_m2 = stats.get('sim_stats_match_m2', {}).get('value', 0.0)

    cluster_count = stats.get('sim_stats_cluster_count', {}).get('value', 0)
    cluster_mean = stats.get('sim_stats_cluster_mean', {}).get('value', 0.0)
    cluster_m2 = stats.get('sim_stats_cluster_m2', {}).get('value', 0.0)

    # Compute adaptive thresholds if we have enough data
    if match_count >= ADAPTIVE_MIN_SAMPLES:
        match_stddev = math.sqrt(match_m2 / match_count) if match_count > 1 else 0.0
        dedup_threshold = match_mean - ADAPTIVE_THRESHOLD_K * match_stddev
        dedup_threshold = max(0.5, min(0.99, dedup_threshold))  # Clamp to reasonable range
    else:
        dedup_threshold = COLD_START_DEDUP_THRESHOLD

    if cluster_count >= ADAPTIVE_MIN_SAMPLES:
        cluster_stddev = math.sqrt(cluster_m2 / cluster_count) if cluster_count > 1 else 0.0
        cluster_threshold = cluster_mean - ADAPTIVE_THRESHOLD_K * cluster_stddev
        cluster_threshold = max(0.3, min(0.95, cluster_threshold))  # Clamp to reasonable range
    else:
        cluster_threshold = COLD_START_CLUSTER_THRESHOLD

    return AdaptiveThresholds(
        dedup_threshold=dedup_threshold,
        cluster_threshold=cluster_threshold,
        match_count=match_count,
        match_mean=match_mean,
        match_m2=match_m2,
        cluster_count=cluster_count,
        cluster_mean=cluster_mean,
        cluster_m2=cluster_m2,
    )


def update_similarity_stats(conn, stat_type: str, similarity: float) -> None:
    """Update Welford stats for similarity tracking.

    Args:
        conn: Database connection
        stat_type: 'match' (for dedup) or 'cluster' (for clustering)
        similarity: The observed similarity value
    """
    now = datetime.now(timezone.utc).isoformat()
    prefix = f'sim_stats_{stat_type}'

    # Read current stats
    cursor = conn.execute(
        "SELECT key, value FROM db_metadata WHERE key LIKE ?",
        (f'{prefix}_%',)
    )
    stats = {row[0]: json.loads(row[1]).get('value', 0) for row in cursor}

    count = stats.get(f'{prefix}_count', 0)
    mean = stats.get(f'{prefix}_mean', 0.0)
    m2 = stats.get(f'{prefix}_m2', 0.0)

    # Welford update
    count += 1
    delta = similarity - mean
    mean += delta / count
    delta2 = similarity - mean
    m2 += delta * delta2

    # Write back
    for key, value in [(f'{prefix}_count', count), (f'{prefix}_mean', mean), (f'{prefix}_m2', m2)]:
        conn.execute(
            """INSERT INTO db_metadata (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (key, json.dumps({'value': value}), now, json.dumps({'value': value}), now)
        )

    logger.debug("[db] Updated %s stats: count=%d, mean=%.3f, stddev=%.3f",
                 stat_type, count, mean, math.sqrt(m2/count) if count > 1 else 0.0)


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

    def __init__(self, db_path: str = None):
        """Initialize the database.

        Args:
            db_path: Optional path to SQLite database. If provided, creates
                     a direct connection instead of using the global singleton.
        """
        if db_path:
            self._direct_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
            configure_connection(self._direct_conn, enable_foreign_keys=False)
            self._db = None
            self._db_path = db_path
        else:
            self._db = get_db()
            self._direct_conn = None
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

        self._init_schema()

        # Register with CacheManager for coordinated invalidation
        get_cache_manager().register_db(self)

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    def close(self):
        """Close the database connection."""
        if self._direct_conn:
            self._direct_conn.close()
            self._direct_conn = None

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
        """Get a database connection."""
        if self._direct_conn:
            yield self._direct_conn
            self._direct_conn.commit()
        else:
            with self._db.connection() as conn:
                yield conn

    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            init_db(conn)

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
        Creates a new placeholder umbrella initialized with the problem's embedding.

        This is a thin wrapper around _create_signature_atomic with is_umbrella=True.

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
                is_umbrella=True,
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
    # Consolidated Routing Core
    # =========================================================================

    def _route_core(
        self,
        operation_embedding: np.ndarray,
        min_similarity: float = 0.85,
        max_depth: int = None,
        track_alternatives: bool = False,
        top_k: int = 3,
        use_ucb1: bool = True,
        epsilon_exploration: bool = False,
        dag_step_type: Optional[str] = None,
    ) -> RoutingResult:
        """Core DFS routing through signature hierarchy using graph embeddings.

        SINGLE PATHWAY for DFS routing. All variations use this function.
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

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
        min_similarity: float = 0.85,
        max_depth: int = None,
    ) -> tuple[Optional[StepSignature], list[StepSignature]]:
        """Route an operation embedding through the signature hierarchy.

        Thin wrapper around _route_core() for simple routing without alternatives.

        Args:
            operation_embedding: The operation embedding to route
            min_similarity: Minimum similarity threshold
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
        min_similarity: float = 0.85,
        max_depth: int = None,
        top_k: int = 3,
        dag_step_type: Optional[str] = None,
    ) -> RoutingResult:
        """Route with confidence scoring for MCTS multi-path exploration.

        Thin wrapper around _route_core() with UCB1 scoring, alternatives
        tracking, and epsilon-greedy exploration enabled.

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.

        Confidence interpretation:
        - High confidence (>0.8): Clear winner, single path likely sufficient
        - Medium confidence (0.5-0.8): Consider exploring 1-2 alternatives
        - Low confidence (<0.5): High uncertainty, explore multiple paths

        Args:
            operation_embedding: The operation embedding to route
            min_similarity: Minimum similarity threshold
            max_depth: Maximum depth to traverse
            top_k: Number of top alternatives to track at each level
            dag_step_type: Optional step type for step-node stats lookup

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
        )

    # =========================================================================
    # Core: Find or Create
    # =========================================================================

    def find_or_create(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: float = 0.85,
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

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            min_similarity: Minimum cosine similarity for matching
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
        min_similarity: float = 0.85,
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
        Routing uses graph_embedding exclusively. Text embedding is optional/legacy.

        Use this from async contexts to avoid blocking the event loop during
        database contention retries.

        Args:
            step_text: The step description text
            embedding: Optional text embedding (legacy, not used for routing)
            min_similarity: Minimum cosine similarity for matching
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
        """Create a new signature with global dedup check.

        Before creating, checks if a virtually identical signature already exists
        (by graph_embedding similarity). If so, returns existing to prevent duplicates.

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
            The newly created or matched StepSignature
        """
        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # GLOBAL DEDUP CHECK: Compute graph_embedding first, then check for duplicates
                # CRITICAL: Use embed_computation_graph_sync for consistency with stored embeddings
                step_graph_emb = None
                if dsl_hint:
                    op_graph = self._dsl_hint_to_graph(dsl_hint)
                    if op_graph:
                        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                        from mycelium.embedder import Embedder
                        embedder_instance = Embedder.get_instance()
                        step_graph_emb_list = embed_computation_graph_sync(embedder_instance, op_graph)
                        if step_graph_emb_list:
                            step_graph_emb = np.array(step_graph_emb_list)

                if step_graph_emb is not None:
                    global_match, global_sim = find_global_best_match(conn, step_graph_emb)
                    thresholds = get_adaptive_thresholds(conn)

                    if global_match is not None and global_sim >= thresholds.dedup_threshold:
                        if not global_match.is_semantic_umbrella:
                            # DEDUP: Found identical LEAF signature, return it
                            logger.info(
                                "[db] DEDUP (create): Found identical leaf (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                global_sim, thresholds.dedup_threshold, step_text[:40], global_match.id, global_match.step_type
                            )
                            update_similarity_stats(conn, 'match', global_sim)
                            self._update_centroid_atomic(conn, global_match.id, embedding, update_last_used=True)
                            conn.commit()
                            return global_match
                        else:
                            # Matched an UMBRELLA - search its children for matching leaf first
                            child_match, child_sim = find_best_child_match(
                                conn, global_match.id, step_graph_emb
                            )
                            if child_match is not None and child_sim >= thresholds.dedup_threshold:
                                # Found matching leaf under umbrella - DEDUP
                                logger.info(
                                    "[db] DEDUP (create): Found matching leaf under umbrella (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                    child_sim, thresholds.dedup_threshold, step_text[:40], child_match.id, child_match.step_type
                                )
                                update_similarity_stats(conn, 'match', child_sim)
                                self._update_centroid_atomic(conn, child_match.id, embedding, update_last_used=True)
                                conn.commit()
                                return child_match
                            else:
                                # No matching leaf found - create new leaf under umbrella
                                parent_id = global_match.id
                                logger.info(
                                    "[db] CLUSTER (create): No matching leaf in umbrella children (best=%.3f), creating under sig %d (%s)",
                                    child_sim, global_match.id, global_match.step_type
                                )
                                update_similarity_stats(conn, 'cluster', global_sim)

                    elif global_match is not None and global_sim >= thresholds.cluster_threshold:
                        if global_match.is_semantic_umbrella:
                            # Matched an UMBRELLA - use it as parent
                            parent_id = global_match.id
                            logger.info(
                                "[db] CLUSTER (create): Using umbrella as parent (sim=%.3f >= %.3f): sig %d (%s)",
                                global_sim, thresholds.cluster_threshold, global_match.id, global_match.step_type
                            )
                        else:
                            # Matched a LEAF - use same parent as the leaf
                            cursor = conn.execute(
                                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                                (global_match.id,)
                            )
                            row = cursor.fetchone()
                            if row:
                                parent_id = row[0]
                                logger.info(
                                    "[db] CLUSTER (create): Clustering with leaf (sim=%.3f >= %.3f), under parent %s",
                                    global_sim, thresholds.cluster_threshold, parent_id
                                )
                        update_similarity_stats(conn, 'cluster', global_sim)

                        # SIBLING DEDUP: Before creating, check if sibling already matches
                        if parent_id is not None and step_graph_emb is not None:
                            sibling_match, sibling_sim = find_best_child_match(
                                conn, parent_id, step_graph_emb
                            )
                            if sibling_match is not None and sibling_sim >= thresholds.dedup_threshold:
                                # Found matching sibling, return it instead of creating duplicate
                                logger.info(
                                    "[db] SIBLING DEDUP (sync): Found matching sibling (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                    sibling_sim, thresholds.dedup_threshold, step_text[:40], sibling_match.id, sibling_match.step_type
                                )
                                update_similarity_stats(conn, 'match', sibling_sim)
                                self._update_centroid_atomic(conn, sibling_match.id, embedding, update_last_used=True)
                                conn.commit()
                                return sibling_match

                # Create new signature (graph_embedding computed inside _create_signature_atomic)
                sig = self._create_signature_atomic(
                    conn, step_text, embedding, parent_problem, origin_depth,
                    extracted_values=extracted_values, parent_id=parent_id, dsl_hint=dsl_hint,
                    graph_embedding=step_graph_emb
                )
                conn.commit()
                logger.info(
                    "[db] Created signature: step='%s' type='%s' depth=%d",
                    step_text[:40], sig.step_type, origin_depth
                )

                return sig
            except Exception:
                conn.rollback()
                raise

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
                    # Empty DB - create root signature
                    sig = self._create_signature_atomic(
                        conn, step_text, embedding, parent_problem, origin_depth,
                        extracted_values=extracted_values, dsl_hint=dsl_hint
                    )
                    conn.commit()
                    logger.info(
                        "[db] Created ROOT signature: step='%s' type='%s'",
                        step_text[:40], sig.step_type
                    )
                    return sig, True

                # Route through hierarchy to find best match
                # Pass dsl_hint for graph-based routing (per CLAUDE.md: route by what operations DO)
                # Pass exclude_ids to prevent circular routing (e.g., child matching back to parent during decomposition)
                best_match, parent_for_new, best_sim = self._route_hierarchical(
                    conn, min_similarity, dsl_hint=dsl_hint, exclude_ids=exclude_ids
                )

                # Accept match from hierarchical routing
                # Simplified: no match_score, just use routing result
                # Global dedup check happens later if no routing match
                from mycelium.config import ALWAYS_ROUTE_TO_BEST
                similarity_ok = best_sim >= min_similarity if not ALWAYS_ROUTE_TO_BEST else True

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
                        from mycelium.data_layer.mcts import (
                            check_and_reject_if_low_similarity,
                            record_leaf_rejection,
                        )

                        # Use graph_embedding similarity if available (operational identity)
                        # Otherwise fall back to text similarity
                        rejection_sim = best_sim  # Default to text similarity
                        has_graph = best_match.graph_embedding is not None

                        if has_graph and dsl_hint:
                            # Convert dsl_hint to graph embedding for operational comparison
                            # Per CLAUDE.md: route by what operations DO, not what they SOUND LIKE
                            try:
                                op_graph = self._dsl_hint_to_graph(dsl_hint)
                                if op_graph:
                                    from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                                    from mycelium.embedder import Embedder
                                    embedder_inst = Embedder.get_instance()
                                    step_graph_emb_list = embed_computation_graph_sync(embedder_inst, op_graph)
                                    if step_graph_emb_list:
                                        step_graph_emb = np.array(step_graph_emb_list)
                                        leaf_graph_emb = np.array(best_match.graph_embedding)
                                        rejection_sim = cosine_similarity(step_graph_emb, leaf_graph_emb)
                                        logger.debug(
                                            "[routing] Leaf '%s' graph_sim=%.3f text_sim=%.3f",
                                            best_match.step_type, rejection_sim, best_sim
                                        )
                            except Exception as e:
                                logger.debug("[db] Graph embedding comparison failed: %s", e)

                        # Check 1: Low similarity rejection using adaptive Welford threshold
                        # Per CLAUDE.md: No magic numbers - threshold = mean - k*stddev
                        from mycelium.config import COLD_START_SIGNATURE_THRESHOLD
                        sig_count = self.count_signatures()
                        is_cold_start = sig_count < COLD_START_SIGNATURE_THRESHOLD

                        # Get adaptive threshold from this leaf's success similarity history
                        adaptive_threshold = best_match.get_adaptive_rejection_threshold(
                            k=1.5, min_samples=5, default_threshold=0.5
                        )

                        if rejection_sim < adaptive_threshold and not is_cold_start:
                            was_rejected, rejection_count = check_and_reject_if_low_similarity(
                                signature_id=best_match.id,
                                step_text=step_text,
                                similarity=rejection_sim,
                                problem_context=parent_problem,
                                conn=conn,  # Pass connection to avoid lock contention
                            )
                            if was_rejected:
                                logger.info(
                                    "[db] Leaf '%s' REJECTED step (sim=%.3f < adaptive=%.3f, n=%d, mean=%.3f), rejections=%d: '%s'",
                                    best_match.step_type, rejection_sim, adaptive_threshold,
                                    best_match.success_sim_count, best_match.success_sim_mean,
                                    rejection_count, step_text[:40]
                                )
                                # Step queued for decomposition - don't create new signature
                                conn.commit()
                                return None, False
                        elif rejection_sim < adaptive_threshold and is_cold_start:
                            logger.debug(
                                "[db] Cold start: skipping rejection (sig_count=%d < %d)",
                                sig_count, COLD_START_SIGNATURE_THRESHOLD
                            )

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

                                        # Record rejection (unified path for all leaf rejections)
                                        rejection_count = record_leaf_rejection(
                                            signature_id=best_match.id,
                                            step_text=step_text,
                                            similarity=max_atomic_sim,  # Use atomic similarity for tracking
                                            problem_context=parent_problem,
                                            conn=conn,  # Pass connection to avoid lock contention
                                        )

                                        logger.info(
                                            "[db] Leaf '%s' REJECTED %s step (sim=%.3f, gap=%.3f, best=%s, hint='%s', rejections=%d): '%s'",
                                            best_match.step_type, reason, max_atomic_sim, gap,
                                            best_atomic_op, dsl_hint, rejection_count, step_text[:50]
                                        )
                                        # record_leaf_rejection already queued for decomposition
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
                        op_graph = self._dsl_hint_to_graph(dsl_hint)
                        if op_graph:
                            from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                            from mycelium.embedder import Embedder
                            embedder_inst = Embedder.get_instance()
                            step_graph_emb_list = embed_computation_graph_sync(embedder_inst, op_graph)
                            if step_graph_emb_list:
                                step_graph_emb = np.array(step_graph_emb_list)
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
                # CRITICAL: Use embed_computation_graph_sync for consistency with stored embeddings
                step_graph_emb = None
                if dsl_hint:
                    op_graph = self._dsl_hint_to_graph(dsl_hint)
                    if op_graph:
                        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                        from mycelium.embedder import Embedder
                        embedder = Embedder.get_instance()
                        step_graph_emb_list = embed_computation_graph_sync(embedder, op_graph)
                        if step_graph_emb_list:
                            step_graph_emb = np.array(step_graph_emb_list)

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
                            conn.commit()
                            return global_match, False
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
                                conn.commit()
                                return child_match, False
                            else:
                                # No matching leaf found - create new leaf under umbrella
                                cluster_parent_id = global_match.id
                                logger.info(
                                    "[db] CLUSTER: No matching leaf in umbrella children (best=%.3f), creating under sig %d (%s)",
                                    child_sim, global_match.id, global_match.step_type
                                )
                                update_similarity_stats(conn, 'cluster', global_sim)

                                sig = self._create_signature_atomic(
                                    conn, step_text, embedding, parent_problem, origin_depth,
                                    extracted_values=extracted_values, dsl_hint=dsl_hint,
                                    parent_id=cluster_parent_id, graph_embedding=step_graph_emb
                                )
                                conn.commit()
                                logger.info(
                                    "[db] Created CLUSTERED signature (child of umbrella %s): step='%s' type='%s'",
                                    global_match.step_type, step_text[:40], sig.step_type
                                )
                                return sig, True

                    elif global_match is not None and global_sim >= thresholds.cluster_threshold:
                        # CLUSTER: Similar signature exists
                        if global_match.is_semantic_umbrella:
                            # Matched an UMBRELLA - use it as parent
                            cluster_parent_id = global_match.id
                            logger.info(
                                "[db] CLUSTER: Using umbrella as parent (sim=%.3f >= %.3f): sig %d (%s)",
                                global_sim, thresholds.cluster_threshold, global_match.id, global_match.step_type
                            )
                        else:
                            # Matched a LEAF - use same parent
                            cursor = conn.execute(
                                "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                                (global_match.id,)
                            )
                            row = cursor.fetchone()
                            cluster_parent_id = row[0] if row else None
                            logger.info(
                                "[db] CLUSTER: Similar leaf found (sim=%.3f >= %.3f), clustering under parent %s",
                                global_sim, thresholds.cluster_threshold, cluster_parent_id
                            )

                        update_similarity_stats(conn, 'cluster', global_sim)

                        # SIBLING DEDUP: Before creating, check if sibling already matches
                        if cluster_parent_id is not None and step_graph_emb is not None:
                            sibling_match, sibling_sim = find_best_child_match(
                                conn, cluster_parent_id, step_graph_emb
                            )
                            if sibling_match is not None and sibling_sim >= thresholds.dedup_threshold:
                                # Found matching sibling, return it instead of creating duplicate
                                logger.info(
                                    "[db] SIBLING DEDUP: Found matching sibling (sim=%.3f >= %.3f): '%s' → sig %d (%s)",
                                    sibling_sim, thresholds.dedup_threshold, step_text[:40], sibling_match.id, sibling_match.step_type
                                )
                                update_similarity_stats(conn, 'match', sibling_sim)
                                self._update_centroid_atomic(conn, sibling_match.id, embedding, update_last_used=True)
                                conn.commit()
                                return sibling_match, False

                        sig = self._create_signature_atomic(
                            conn, step_text, embedding, parent_problem, origin_depth,
                            extracted_values=extracted_values, dsl_hint=dsl_hint,
                            parent_id=cluster_parent_id, graph_embedding=step_graph_emb
                        )
                        conn.commit()
                        logger.info(
                            "[db] Created CLUSTERED signature (sibling of %s): step='%s' type='%s'",
                            global_match.step_type, step_text[:40], sig.step_type
                        )
                        return sig, True

                # No global match above thresholds - create new signature under routing parent
                actual_parent_id = parent_id if parent_id is not None else (parent_for_new.id if parent_for_new else None)

                sig = self._create_signature_atomic(
                    conn, step_text, embedding, parent_problem, origin_depth,
                    extracted_values=extracted_values, dsl_hint=dsl_hint,
                    parent_id=actual_parent_id, graph_embedding=step_graph_emb
                )
                conn.commit()
                parent_desc = f"id={parent_id}" if parent_id is not None else (parent_for_new.step_type if parent_for_new else "root")
                logger.info(
                    "[db] Created NEW signature (child of %s): step='%s' type='%s'",
                    parent_desc, step_text[:40], sig.step_type
                )
                return sig, True

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

        # Compute step's graph_embedding from dsl_hint for operational routing
        # Per CLAUDE.md: route by what operations DO, not what they SOUND LIKE
        # CRITICAL: Use embed_computation_graph_sync for consistency with stored embeddings
        step_graph_embedding = None
        if dsl_hint:
            op_graph = self._dsl_hint_to_graph(dsl_hint)
            if op_graph:
                from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync
                from mycelium.embedder import Embedder
                embedder_inst = Embedder.get_instance()
                step_graph_emb_list = embed_computation_graph_sync(embedder_inst, op_graph)
                if step_graph_emb_list:
                    step_graph_embedding = np.array(step_graph_emb_list)
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

        # Import once outside loop
        from mycelium.config import ALWAYS_ROUTE_TO_BEST

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
                # Umbrella with no children - return current as best match
                # sim already computed above using graph_embedding or centroid
                return current, current, sim

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
            for child, child_sim, used_graph in children_with_embeddings:
                if ALWAYS_ROUTE_TO_BEST or child_sim >= min_similarity:
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
                step_type, step_text, extracted_values=extracted_values, dsl_hint=dsl_hint
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
                from mycelium.embedder import Embedder
                embedder_inst = Embedder.get_instance()
                emb_list = embed_computation_graph_sync(embedder_inst, computation_graph)
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
        threshold: float = 0.7,
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

    def increment_signature_successes(
        self,
        signature_id: int,
        count: int = 1,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """Increment the successes count for a signature.

        Per beads mycelium-itkn: Used by amplitude credit propagation.
        Per CLAUDE.md: "Parent umbrellas get decay^depth credit (default 0.5 per level)"

        Args:
            signature_id: ID of the signature
            count: Amount to increment by (default 1)
            propagate_to_parents: If True, propagate credit up to parent routers with decay
            _depth: Internal recursion depth tracker
        """
        with self._connection() as conn:
            self._increment_signature_stat(
                conn, signature_id, "successes",
                amount=count, propagate_to_parents=propagate_to_parents, _depth=_depth
            )

    def increment_signature_failures(
        self,
        signature_id: int,
        count: int = 1,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """Increment the operational_failures count for a signature.

        Per beads mycelium-itkn: Used by amplitude credit propagation.
        Per CLAUDE.md: Failure signal also propagates up with decay.

        Args:
            signature_id: ID of the signature
            count: Amount to increment by (default 1)
            propagate_to_parents: If True, propagate failure up to parent routers with decay
            _depth: Internal recursion depth tracker
        """
        with self._connection() as conn:
            self._increment_signature_stat(
                conn, signature_id, "operational_failures",
                amount=count, propagate_to_parents=propagate_to_parents, _depth=_depth
            )

    def increment_signature_partial_success(
        self,
        signature_id: int,
        weight: float = 0.5,
        propagate_to_parents: bool = True,
        _depth: int = 0,
    ):
        """Increment successes with a fractional weight (partial credit).

        Per beads mycelium-7o8i: Used for correct steps in failed problems.
        Steps with high confidence in losing threads get partial credit rather
        than full blame - they were probably correct, just in a bad chain.

        Args:
            signature_id: ID of the signature
            weight: Fractional credit (default 0.5 = half a success)
            propagate_to_parents: If True, propagate partial credit up with decay
            _depth: Internal recursion depth tracker
        """
        with self._connection() as conn:
            self._increment_signature_stat(
                conn, signature_id, "successes",
                amount=weight, propagate_to_parents=propagate_to_parents, _depth=_depth
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
        min_similarity: float = 0.85,
        limit: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Find pairs of signatures that are candidates for merging.

        Candidates are signatures that:
        - Both have high success rates (operationally correct)
        - Have similar centroids (semantically similar)
        - Are not already archived

        Args:
            min_success_rate: Minimum success rate for both signatures
            min_uses: Minimum uses for both signatures (need data to trust)
            min_similarity: Minimum cosine similarity between centroids
            limit: Maximum number of pairs to return

        Returns:
            List of (sig1_id, sig2_id, similarity) tuples, ordered by similarity desc
        """
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
        min_similarity: float = 0.8,
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

            centroid = np.frombuffer(row[0], dtype=np.float32)

            # Check if alternative leaves could handle this operation type
            # Use permissive min_similarity - each candidate has its own adaptive threshold
            alternatives = self.match_step_to_leaves_mcts(
                embedding=centroid,
                dag_step_type=None,
                top_k=3,
                min_similarity=0.5,  # Permissive - adaptive threshold applied per-candidate
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
        """Archive a signature and reparent its children atomically.

        This handles the multi-step operation of re-parenting children to a
        grandparent and then archiving the signature, all within a single
        transaction to ensure consistency.

        Args:
            signature_id: ID of signature to archive
            parent_id: ID of parent to reparent children to
            child_ids: List of child signature IDs to reparent
            reason: Why it's being archived

        Returns:
            True if archived successfully
        """
        with self._connection() as conn:
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
            List of example dicts with step_text, result, success
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT step_text, result, success
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
                })
            return examples

    def update_example_result(
        self,
        signature_id: int,
        step_text: str,
        result: str,
        success: bool,
    ) -> bool:
        """Update an example with its execution result.

        Called after DSL execution to record the result for DSL regeneration.
        Finds the most recent example for this signature matching step_text.

        Args:
            signature_id: ID of the signature
            step_text: The step text to match
            result: The DSL execution result
            success: Whether execution succeeded

        Returns:
            True if an example was updated, False otherwise
        """
        with self._connection() as conn:
            # Update the most recent example for this signature
            # Match on first 100 chars of step_text to handle truncation
            cursor = conn.execute(
                """UPDATE step_examples
                   SET result = ?, success = ?
                   WHERE id = (
                       SELECT id FROM step_examples
                       WHERE signature_id = ?
                         AND substr(step_text, 1, 100) = substr(?, 1, 100)
                       ORDER BY created_at DESC
                       LIMIT 1
                   )""",
                (result, 1 if success else 0, signature_id, step_text),
            )
            updated = cursor.rowcount > 0
            if updated:
                logger.debug(
                    "[db] Updated example result for sig %d: success=%s",
                    signature_id, success
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
        increment_total_problems(self.db_path)

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
        min_similarity: float = 0.3,
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
        min_similarity: float = 0.5,
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

    def _promote_to_umbrella_internal(self, conn, signature_id: int) -> bool:
        """Internal: Mark signature as umbrella within existing transaction.

        This is the SINGLE PATHWAY for umbrella promotion. All code that needs
        to mark a signature as umbrella should call this function.

        Args:
            conn: Database connection (existing transaction)
            signature_id: ID of the signature to promote

        Returns:
            True if updated, False if signature not found
        """
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
        min_similarity: float = 0.75,
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
                # Create umbrella using unified pathway
                new_sig = self._create_signature_atomic(
                    conn=conn,
                    step_text=description,
                    embedding=new_centroid,
                    is_umbrella=True,
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

                return new_sig

            except sqlite3.IntegrityError as e:
                logger.warning("[db] Failed to create upward umbrella: %s", e)
                return None

    def remove_child(self, parent_id: int, child_id: int) -> bool:
        """Remove a parent-child relationship.

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
                    # Demote from umbrella
                    self._update_signature_fields(
                        conn, parent_id,
                        log_reason="demote_from_umbrella",
                        is_semantic_umbrella=0,
                    )
                # Invalidate caches (relationship change: child removed)
                self._invalidate_on_relationship_change(parent_id, child_id)
                logger.info("[db] Removed child relationship: parent=%d → child=%d", parent_id, child_id)
                return True
            return False

    def clear_all_data(self) -> dict:
        """Clear all signature data for a fresh start.

        Returns:
            Dict with counts of deleted rows
        """
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
        min_similarity: float = 0.75,
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
                # Router: if matches, explore children
                if sim >= min_similarity * 0.8:  # Slightly lower threshold for routers
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
