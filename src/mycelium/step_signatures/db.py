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
    CENTROID_PROPAGATION_BATCH_SIZE,
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

        # Batched centroid propagation: track pending updates per signature
        # Key: signature_id, Value: count of pending centroid updates
        self._pending_propagations: dict[int, int] = {}

        self._init_schema()

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

    def close(self):
        """Close the database, flushing any pending operations.

        Ensures all batched centroid propagations are written before closing.
        """
        self.flush_pending_propagations()
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
        """Initialize database schema and scaffold structure."""
        with self._connection() as conn:
            init_db(conn)

        # Initialize scaffold structure if enabled (creates placeholder umbrellas)
        self.initialize_scaffold()

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

    def initialize_scaffold(self) -> bool:
        """Initialize the pre-allocated scaffold structure for the universal tree.

        Creates placeholder umbrella levels that give the tree "room to grow".
        Domains emerge as problem traffic flows through and centroids specialize.

        Structure created:
            Level 0: ROOT (single entry point)
            Level 1-N: Placeholder umbrellas (SCAFFOLD_LEVELS)
            Level N+1+: Where actual leaf signatures will be created

        The placeholders start with null centroids. As problems route through,
        centroids get initialized and refined via averaging.

        Returns:
            True if scaffold was created, False if already exists or disabled
        """
        from mycelium.config import SCAFFOLD_ENABLED, SCAFFOLD_LEVELS

        if not SCAFFOLD_ENABLED:
            logger.debug("[db] Scaffold disabled, skipping initialization")
            return False

        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            # Use BEGIN IMMEDIATE for atomic check-and-create to prevent race condition
            # when multiple workers start simultaneously
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Check if scaffold already exists (has root with children)
                root_row = conn.execute(
                    "SELECT id FROM step_signatures WHERE is_root = 1 LIMIT 1"
                ).fetchone()

                if root_row is not None:
                    # Root exists, check if it has children (scaffold already created)
                    child_count = conn.execute(
                        "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                        (root_row[0],)
                    ).fetchone()[0]
                    if child_count > 0:
                        logger.debug("[db] Scaffold already exists (%d root children)", child_count)
                        conn.rollback()
                        return False

                logger.info("[db] Initializing scaffold: %d levels (single chain, no horizontal scaling)", SCAFFOLD_LEVELS)

                if root_row is None:
                    # Create the root signature
                    root_sig_id = f"root_{uuid.uuid4().hex[:8]}"
                    cursor = conn.execute(
                        """INSERT INTO step_signatures (
                            signature_id, centroid, centroid_bucket,
                            step_type, description, dsl_type,
                            is_semantic_umbrella, is_root, depth, created_at
                        ) VALUES (?, NULL, NULL, ?, ?, ?, 1, 1, 0, ?)""",
                        (root_sig_id, "root", "Universal math problem router", "router", now)
                    )
                    root_id = cursor.lastrowid
                    logger.info("[db] Created scaffold root: id=%d", root_id)
                else:
                    root_id = root_row[0]
                    # Ensure root is an umbrella - routers don't execute DSL
                    conn.execute(
                        "UPDATE step_signatures SET is_semantic_umbrella = 1, dsl_type = 'router', dsl_script = NULL WHERE id = ?",
                        (root_id,)
                    )

                # Create single-chain scaffold: ROOT → L1 → L2 → ... → LN
                # NO horizontal scaling - branches fork dynamically at runtime
                current_parent_id = root_id

                for level in range(1, SCAFFOLD_LEVELS + 1):
                    placeholder_id = f"scaffold_L{level}_{uuid.uuid4().hex[:8]}"
                    cursor = conn.execute(
                        """INSERT INTO step_signatures (
                            signature_id, centroid, centroid_bucket,
                            step_type, description, dsl_type,
                            is_semantic_umbrella, is_root, depth, created_at
                        ) VALUES (?, NULL, NULL, ?, ?, ?, 1, 0, ?, ?)""",
                        (
                            placeholder_id,
                            f"abstract_L{level}",
                            f"Abstract routing level {level}",
                            "router",
                            level,
                            now,
                        )
                    )
                    child_id = cursor.lastrowid

                    # Create parent-child relationship
                    conn.execute(
                        """INSERT INTO signature_relationships (parent_id, child_id, condition, created_at)
                           VALUES (?, ?, ?, ?)""",
                        (current_parent_id, child_id, f"scaffold_chain_L{level}", now)
                    )

                    logger.debug("[db] Created scaffold level %d: id=%d", level, child_id)
                    current_parent_id = child_id

                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Invalidate caches
        self._cached_root = None
        self.invalidate_centroid_matrix()

        logger.info("[db] Scaffold initialized: single chain of %d levels (branches fork dynamically)", SCAFFOLD_LEVELS)
        return True

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

        Args:
            conn: Database connection (within transaction)
            parent_id: ID of parent umbrella to branch from
            embedding: Embedding of the divergent problem
            depth: Depth of the new branch

        Returns:
            The new branch signature, or None on failure
        """
        now = datetime.now(timezone.utc).isoformat()
        branch_id = f"branch_L{depth}_{uuid.uuid4().hex[:8]}"
        centroid_json = json.dumps(embedding.tolist())
        centroid_bucket = compute_centroid_bucket(embedding)

        try:
            cursor = conn.execute(
                """INSERT INTO step_signatures (
                    signature_id, centroid, centroid_bucket, embedding_sum, embedding_count,
                    step_type, description, dsl_type,
                    is_semantic_umbrella, is_root, depth, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)""",
                (
                    branch_id,
                    centroid_json,
                    centroid_bucket,
                    centroid_json,  # embedding_sum = centroid initially
                    1,
                    f"branch_L{depth}",
                    f"Dynamic branch at level {depth}",
                    "router",
                    depth,
                    now,
                )
            )
            new_id = cursor.lastrowid

            # Create parent-child relationship
            conn.execute(
                """INSERT INTO signature_relationships (parent_id, child_id, condition, created_at)
                   VALUES (?, ?, ?, ?)""",
                (parent_id, new_id, f"dynamic_fork_L{depth}", now)
            )

            # Invalidate caches
            invalidate_centroid_cache(new_id)
            self.invalidate_centroid_matrix()

            # Return the new branch as a StepSignature
            return StepSignature(
                id=new_id,
                signature_id=branch_id,
                centroid=embedding,
                embedding_count=1,
                step_type=f"branch_L{depth}",
                description=f"Dynamic branch at level {depth}",
                dsl_type="router",
                is_semantic_umbrella=True,
                depth=depth,
                created_at=now,
            )

        except sqlite3.IntegrityError as e:
            logger.warning("[db] Failed to create scaffold branch: %s", e)
            return None

    def _maybe_propagate_centroid(
        self,
        conn,
        child_id: int,
        force: bool = False,
    ) -> bool:
        """Conditionally propagate centroid to parents using batching.

        Accumulates centroid updates and only propagates when batch size is reached.
        This reduces overhead from calling propagate_centroid_to_parents on every match.

        Args:
            conn: Database connection (within transaction)
            child_id: ID of the signature whose centroid was updated
            force: If True, propagate immediately regardless of batch size

        Returns:
            True if propagation was performed, False if deferred
        """
        if force or CENTROID_PROPAGATION_BATCH_SIZE <= 1:
            # Immediate propagation requested or batching disabled
            self.propagate_centroid_to_parents(conn, child_id)
            # Clear any pending count for this signature
            self._pending_propagations.pop(child_id, None)
            return True

        # Increment pending count for this signature
        pending = self._pending_propagations.get(child_id, 0) + 1
        self._pending_propagations[child_id] = pending

        if pending >= CENTROID_PROPAGATION_BATCH_SIZE:
            # Threshold reached, propagate now
            self.propagate_centroid_to_parents(conn, child_id)
            self._pending_propagations.pop(child_id, None)
            logger.debug(
                "[db] Batch propagation triggered for sig %d after %d updates",
                child_id, pending
            )
            return True

        logger.debug(
            "[db] Deferred centroid propagation for sig %d (%d/%d)",
            child_id, pending, CENTROID_PROPAGATION_BATCH_SIZE
        )
        return False

    def flush_pending_propagations(self):
        """Flush all pending centroid propagations.

        Call this at the end of a batch of operations to ensure all
        centroid updates are propagated to parent umbrellas.

        Uses batch propagation to avoid N+1 queries when flushing multiple signatures.
        """
        if not self._pending_propagations:
            return

        pending_ids = list(self._pending_propagations.keys())
        self._pending_propagations.clear()

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._batch_propagate_centroids_to_parents(conn, pending_ids)
                conn.commit()
                logger.debug(
                    "[db] Flushed %d pending centroid propagations (batched)",
                    len(pending_ids)
                )
            except Exception:
                conn.rollback()
                raise

    def _batch_propagate_centroids_to_parents(
        self,
        conn,
        child_ids: list[int],
    ) -> None:
        """Batch propagate centroid changes for multiple children at once.

        Optimized version that collects all ancestors for all children in one query,
        avoiding N+1 pattern. Reduces overhead from O(N*3) queries to O(3) queries.

        Args:
            conn: Database connection (within transaction)
            child_ids: List of signature IDs whose centroids were updated
        """
        if not child_ids:
            return

        max_depth = CENTROID_PROPAGATION_MAX_DEPTH

        # SQLite has a default limit of 999 parameters. Chunk if needed.
        # Reserve 1 slot for max_depth parameter.
        SQLITE_MAX_PARAMS = 998
        if len(child_ids) > SQLITE_MAX_PARAMS:
            # Process in chunks, collecting all ancestors
            all_ancestors: dict[int, int] = {}  # parent_id -> min_depth
            for i in range(0, len(child_ids), SQLITE_MAX_PARAMS):
                chunk = child_ids[i:i + SQLITE_MAX_PARAMS]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"""
                    WITH RECURSIVE ancestors AS (
                        SELECT parent_id, child_id, 1 as depth
                        FROM signature_relationships
                        WHERE child_id IN ({placeholders})

                        UNION ALL

                        SELECT r.parent_id, a.child_id, a.depth + 1
                        FROM signature_relationships r
                        JOIN ancestors a ON r.child_id = a.parent_id
                        WHERE a.depth < ?
                    )
                    SELECT DISTINCT parent_id, MIN(depth) as depth
                    FROM ancestors
                    GROUP BY parent_id
                    """,
                    (*chunk, max_depth),
                )
                for row in cursor.fetchall():
                    pid, depth = row[0], row[1]
                    if pid not in all_ancestors or depth < all_ancestors[pid]:
                        all_ancestors[pid] = depth
            # Convert to sorted list by depth
            ancestors = sorted(all_ancestors.items(), key=lambda x: x[1])
        else:
            # 1. Batch-fetch all ancestors for ALL child_ids using recursive CTE
            placeholders = ",".join("?" * len(child_ids))
            cursor = conn.execute(
                f"""
                WITH RECURSIVE ancestors AS (
                    SELECT parent_id, child_id, 1 as depth
                    FROM signature_relationships
                    WHERE child_id IN ({placeholders})

                    UNION ALL

                    SELECT r.parent_id, a.child_id, a.depth + 1
                    FROM signature_relationships r
                    JOIN ancestors a ON r.child_id = a.parent_id
                    WHERE a.depth < ?
                )
                SELECT DISTINCT parent_id, MIN(depth) as depth
                FROM ancestors
                GROUP BY parent_id
                ORDER BY depth
                """,
                (*child_ids, max_depth),
            )
            ancestors = cursor.fetchall()

        if not ancestors:
            return  # No parents (all are root nodes)

        ancestor_ids = [row[0] for row in ancestors]

        # 2. Batch-fetch all children data for ALL ancestors
        # Also chunk this query to respect SQLite parameter limits
        children_by_parent: dict[int, list[tuple]] = {}
        for i in range(0, len(ancestor_ids), SQLITE_MAX_PARAMS):
            chunk = ancestor_ids[i:i + SQLITE_MAX_PARAMS]
            ancestor_placeholders = ",".join("?" * len(chunk))
            cursor = conn.execute(
                f"""
                SELECT r.parent_id, s.centroid, s.embedding_count
                FROM signature_relationships r
                JOIN step_signatures s ON r.child_id = s.id
                WHERE r.parent_id IN ({ancestor_placeholders})
                """,
                chunk,
            )
            for parent_id, centroid, count in cursor.fetchall():
                if parent_id not in children_by_parent:
                    children_by_parent[parent_id] = []
                children_by_parent[parent_id].append((centroid, count))

        # 3. Compute new centroids for each ancestor (process in depth order)
        updates = []
        for parent_id, _ in ancestors:
            children = children_by_parent.get(parent_id, [])
            if not children:
                continue

            total_weight = 0
            centroid_sum = None

            for child_centroid_packed, child_count in children:
                child_centroid = unpack_embedding(child_centroid_packed)
                weight = child_count or 1
                if child_centroid is None:
                    continue

                if centroid_sum is None:
                    centroid_sum = child_centroid * weight
                else:
                    centroid_sum = centroid_sum + (child_centroid * weight)
                total_weight += weight

            if centroid_sum is not None and total_weight > 0:
                new_centroid = centroid_sum / total_weight
                updates.append((
                    pack_embedding(new_centroid),
                    pack_embedding(centroid_sum),
                    total_weight,
                    parent_id,
                ))

        # 4. Batch update all ancestors
        for packed_centroid, packed_sum, weight, parent_id in updates:
            try:
                conn.execute(
                    """UPDATE step_signatures
                       SET centroid = ?, embedding_sum = ?, embedding_count = ?
                       WHERE id = ?""",
                    (packed_centroid, packed_sum, weight, parent_id),
                )
                invalidate_centroid_cache(parent_id)
                invalidate_signature_cache(parent_id)
            except sqlite3.IntegrityError:
                logger.debug(
                    "[db] Skipped centroid propagation to parent %d (collision)",
                    parent_id
                )

        if updates:
            logger.debug(
                "[db] Batch propagated centroids to %d ancestors from %d children",
                len(updates), len(child_ids)
            )

    def propagate_centroid_to_parents(
        self,
        conn,
        child_id: int,
        visited: set[int] = None,
    ):
        """Propagate centroid changes up to parent umbrellas (batch approach).

        Uses recursive CTE to fetch all ancestors up to CENTROID_PROPAGATION_MAX_DEPTH
        in a single query, then batch-updates all centroids. Reduces N+1 queries to 3.

        Args:
            conn: Database connection (within transaction)
            child_id: ID of the signature whose centroid was updated
            visited: Unused, kept for API compatibility
        """
        max_depth = CENTROID_PROPAGATION_MAX_DEPTH

        # 1. Batch-fetch all ancestors up to max_depth using recursive CTE
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
            (child_id, max_depth),
        )
        ancestors = cursor.fetchall()

        if not ancestors:
            return  # No parents (root node)

        ancestor_ids = [row[0] for row in ancestors]

        # 2. Batch-fetch all children data for all ancestors in one query
        placeholders = ",".join("?" * len(ancestor_ids))
        cursor = conn.execute(
            f"""
            SELECT r.parent_id, s.centroid, s.embedding_count
            FROM signature_relationships r
            JOIN step_signatures s ON r.child_id = s.id
            WHERE r.parent_id IN ({placeholders})
            """,
            ancestor_ids,
        )
        children_data = cursor.fetchall()

        # Group children by parent_id
        children_by_parent: dict[int, list[tuple]] = {}
        for parent_id, centroid, count in children_data:
            if parent_id not in children_by_parent:
                children_by_parent[parent_id] = []
            children_by_parent[parent_id].append((centroid, count))

        # 3. Compute new centroids for each ancestor (process in depth order)
        updates = []
        for parent_id, _ in ancestors:
            children = children_by_parent.get(parent_id, [])
            if not children:
                continue

            total_weight = 0
            centroid_sum = None

            for child_centroid_packed, child_count in children:
                child_centroid = unpack_embedding(child_centroid_packed)
                weight = child_count or 1
                if child_centroid is None:
                    continue

                if centroid_sum is None:
                    centroid_sum = child_centroid * weight
                else:
                    centroid_sum = centroid_sum + (child_centroid * weight)
                total_weight += weight

            if centroid_sum is not None and total_weight > 0:
                new_centroid = centroid_sum / total_weight
                updates.append((
                    pack_embedding(new_centroid),
                    pack_embedding(centroid_sum),
                    total_weight,
                    parent_id,
                ))

        # 4. Batch update all ancestors
        for packed_centroid, packed_sum, weight, parent_id in updates:
            try:
                conn.execute(
                    """UPDATE step_signatures
                       SET centroid = ?, embedding_sum = ?, embedding_count = ?
                       WHERE id = ?""",
                    (packed_centroid, packed_sum, weight, parent_id),
                )
                invalidate_centroid_cache(parent_id)
                invalidate_signature_cache(parent_id)
                logger.debug(
                    "[db] Propagated centroid to parent %d (weight=%d)",
                    parent_id, weight
                )
            except sqlite3.IntegrityError:
                logger.debug(
                    "[db] Skipped centroid propagation to parent %d (collision)",
                    parent_id
                )

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
        child_id: int,
    ):
        """Propagate graph_centroid changes up to parent routers.

        When a child's graph_embedding changes, recompute parents' graph_centroids.
        Similar to text centroid propagation but for graph embeddings.

        Args:
            conn: Database connection
            child_id: ID of the signature whose graph_embedding changed
        """
        max_depth = CENTROID_PROPAGATION_MAX_DEPTH

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
            (child_id, max_depth),
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

    def route_through_hierarchy(
        self,
        embedding: np.ndarray,
        min_similarity: float = 0.85,
        max_depth: int = None,
    ) -> tuple[Optional[StepSignature], list[StepSignature]]:
        """Route an embedding through the signature hierarchy.

        Starting from the root, traverse down through umbrella signatures
        by picking the best-matching child at each level until reaching
        a leaf signature or no match is found.

        Args:
            embedding: The query embedding to route
            min_similarity: Minimum similarity threshold to follow a route
            max_depth: Maximum depth to traverse (default from config)

        Returns:
            Tuple of (best_leaf_signature, path_taken) where:
            - best_leaf_signature: The leaf signature matched, or None if no match
            - path_taken: List of signatures traversed from root to leaf
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

        # Validate max_depth to prevent unbounded recursion
        if max_depth is None:
            max_depth = UMBRELLA_MAX_DEPTH
        max_depth = max(1, min(int(max_depth), 100))  # Hard cap at 100

        root = self.get_root()
        if root is None:
            return None, []

        path = [root]
        current = root
        depth = 0

        while depth < max_depth:
            # If current is not an umbrella, it's a leaf - we're done
            if not current.is_semantic_umbrella:
                return current, path

            # Get children of current umbrella (fast routing mode - skip JSON parsing)
            children = self.get_children(current.id, for_routing=True)
            if not children:
                # Umbrella with no children - treat as leaf
                return current, path

            # Find best matching child
            best_child = None
            best_score = 0.0

            for child_sig, _condition in children:
                # Capture centroid once to avoid TOCTOU race condition
                centroid = child_sig.centroid
                if centroid is None:
                    continue
                sim = cosine_similarity(embedding, centroid)
                if sim >= min_similarity:
                    # Use routing score for tiebreaking
                    score = compute_routing_score(
                        sim, child_sig.uses, child_sig.successes, child_sig.last_used_at
                    )
                    if score > best_score:
                        best_child = child_sig
                        best_score = score

            if best_child is None:
                # No child matches - return current as "best effort"
                # (caller will decide whether to create new child)
                return current, path

            # Move to best child
            path.append(best_child)
            current = best_child
            depth += 1

        # Hit max depth - return current node
        logger.warning(
            "[db] Hit max routing depth %d at sig %d",
            max_depth, current.id
        )
        return current, path

    def route_with_confidence(
        self,
        embedding: np.ndarray,
        min_similarity: float = 0.85,
        max_depth: int = None,
        top_k: int = 3,
        dag_step_type: Optional[str] = None,
    ) -> RoutingResult:
        """Route with confidence scoring for MCTS multi-path exploration.

        Enhanced version of route_through_hierarchy that computes confidence
        signals based on UCB1 score gaps between top-k children at each level.

        Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning."
        If dag_step_type is provided, step-specific performance stats from
        dag_step_node_stats are used to improve routing decisions.

        Confidence interpretation:
        - High confidence (>0.8): Clear winner, single path likely sufficient
        - Medium confidence (0.5-0.8): Consider exploring 1-2 alternatives
        - Low confidence (<0.5): High uncertainty, explore multiple paths

        Args:
            embedding: The query embedding to route
            min_similarity: Minimum similarity threshold to follow a route
            max_depth: Maximum depth to traverse (default from config)
            top_k: Number of top alternatives to track at each level
            dag_step_type: Optional step type for step-node stats lookup
                (e.g., "compute_sum", "compute_product")

        Returns:
            RoutingResult with signature, path, confidence, and alternatives
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

        # Validate max_depth
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
        best_similarity = None  # Track best cosine similarity at final level

        while depth < max_depth:
            # If current is not an umbrella, it's a leaf - we're done
            if not current.is_semantic_umbrella:
                break

            # Get children of current umbrella
            children = self.get_children(current.id, for_routing=True)
            if not children:
                # Umbrella with no children - treat as leaf
                break

            # Score all children with UCB1
            parent_uses = current.uses or 1
            scored_children = []

            # Fetch step-node stats for all children if dag_step_type provided
            # This enables routing to use (dag_step_type, node_id) pair performance
            step_stats_map = {}
            if dag_step_type:
                from mycelium.data_layer.mcts import get_dag_step_node_stats_batch
                child_ids = [c.id for c, _ in children if c.id is not None]
                step_stats_map = get_dag_step_node_stats_batch(dag_step_type, child_ids)
                # Debug: Log step-node stats retrieval
                if step_stats_map:
                    logger.debug(
                        "[routing] Step-node stats for '%s': %d/%d children have stats",
                        dag_step_type[:40], len(step_stats_map), len(child_ids)
                    )
                    for sig_id, stats in list(step_stats_map.items())[:3]:  # Log first 3
                        logger.debug(
                            "[routing]   sig=%d: uses=%d win_rate=%.2f avg_amp=%.2f",
                            sig_id, stats.get("uses", 0), stats.get("win_rate", 0),
                            stats.get("avg_amplitude_post", 1.0)
                        )

            for child_sig, _condition in children:
                centroid = child_sig.centroid
                if centroid is None:
                    continue
                sim = cosine_similarity(embedding, centroid)
                if sim >= min_similarity * 0.7:  # Lower threshold to capture alternatives
                    # Get step-node stats for this child (if available)
                    child_step_stats = step_stats_map.get(child_sig.id)
                    ucb1 = compute_ucb1_score(
                        cosine_sim=sim,
                        uses=child_sig.uses,
                        successes=child_sig.successes,
                        parent_uses=parent_uses,
                        last_used_at=child_sig.last_used_at,
                        step_node_stats=child_step_stats,
                    )
                    scored_children.append((child_sig, ucb1, sim))

            if not scored_children:
                # No children match - return current as best effort
                break

            # Sort by UCB1 score (descending)
            scored_children.sort(key=lambda x: x[1], reverse=True)

            # Epsilon-greedy exploration: occasionally pick random child
            # This ensures under-visited signatures get attempts even when UCB1 favors exploitation
            from mycelium.config import EXPLORATION_EPSILON
            import random
            if EXPLORATION_EPSILON > 0 and random.random() < EXPLORATION_EPSILON and len(scored_children) > 1:
                # Pick random child (not necessarily the best)
                random_idx = random.randint(0, len(scored_children) - 1)
                # Move selected child to front so it becomes "best"
                selected = scored_children[random_idx]
                scored_children[random_idx] = scored_children[0]
                scored_children[0] = selected
                logger.debug(
                    "[routing] Epsilon exploration: picked random child %d (score=%.3f) instead of best (score=%.3f)",
                    selected[0].id, selected[1], scored_children[1][1] if len(scored_children) > 1 else 0
                )

            # Track top-k alternatives at this level
            level_alts = [(sig, score) for sig, score, _sim in scored_children[:top_k]]
            alternatives.append(level_alts)

            # Compute confidence from UCB1 gap
            best_score = scored_children[0][1]
            if len(scored_children) > 1:
                second_score = scored_children[1][1]
                gap = best_score - second_score
                # Normalize gap to 0-1 (empirically, gaps > 0.3 are very confident)
                level_confidence = min(1.0, gap / 0.3)
            else:
                # Only one option - high confidence by default
                gap = 1.0
                level_confidence = 1.0

            ucb1_gaps.append(gap)
            confidence_factors.append(level_confidence)

            # Check if best child meets similarity threshold
            best_child, _best_ucb1, best_sim = scored_children[0]
            best_similarity = best_sim  # Track for MCTS amplitude logging
            if best_sim < min_similarity:
                # Best child doesn't meet threshold - stop here
                break

            # Move to best child
            path.append(best_child)
            current = best_child
            depth += 1

        # Compute overall confidence as product of level confidences
        # (weakest link determines overall confidence)
        if confidence_factors:
            overall_confidence = min(confidence_factors)  # Weakest link
        else:
            overall_confidence = 1.0 if current is not None else 0.0

        return RoutingResult(
            signature=current if not current.is_semantic_umbrella else None,
            path=path,
            confidence=overall_confidence,
            ucb1_gaps=ucb1_gaps,
            alternatives=alternatives,
            best_similarity=best_similarity,
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
                    extracted_values=extracted_values, dsl_hint=dsl_hint, parent_id=parent_id
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
        embedding: np.ndarray,
        min_similarity: float = 0.85,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
        embedder=None,  # Optional sync embedder for graph embedding
    ) -> tuple[StepSignature, bool]:
        """Async version of find_or_create with non-blocking retry sleep.

        Use this from async contexts to avoid blocking the event loop during
        database contention retries.

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            min_similarity: Minimum cosine similarity for matching
            parent_problem: The parent problem this step came from
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication
            origin_depth: Decomposition depth at which this step was created
            extracted_values: Dict of semantic param names -> values from planner
            parent_id: Explicit parent ID for new signatures (overrides routing)
            embedder: Optional sync embedder for computing graph_embedding on new signatures

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
                    extracted_values=extracted_values, dsl_hint=dsl_hint, parent_id=parent_id
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
        """Force create a new signature (no matching, always creates new).

        Use this when you need a distinct child signature even if similar ones exist.

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
            The newly created StepSignature
        """
        from mycelium.step_signatures.graph_extractor import embed_computation_graph_sync

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                sig = self._create_signature_atomic(
                    conn, step_text, embedding, parent_problem, origin_depth,
                    extracted_values=extracted_values, parent_id=parent_id, dsl_hint=dsl_hint
                )
                # Propagate new child's centroid up to parent umbrellas
                if sig.id is not None:
                    self.propagate_centroid_to_parents(conn, sig.id)
                conn.commit()
                logger.info(
                    "[db] Force-created signature: step='%s' type='%s' depth=%d",
                    step_text[:40], sig.step_type, origin_depth
                )

                # Compute graph embedding for new signature (per CLAUDE.md: route by what ops DO)
                if sig and sig.computation_graph and embedder is not None:
                    try:
                        graph_emb = embed_computation_graph_sync(embedder, sig.computation_graph)
                        if graph_emb:
                            self.update_graph_embedding(sig.id, graph_emb)
                            logger.debug(
                                "[db] Embedded graph for new child sig %d: %s",
                                sig.id, sig.computation_graph[:30]
                            )
                    except Exception as e:
                        logger.warning("[db] Failed to embed graph for sig %d: %s", sig.id, e)

                return sig
            except Exception:
                conn.rollback()
                raise

    def _find_or_create_atomic(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: float,
        parent_problem: str,
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
    ) -> tuple[StepSignature, bool]:
        """Internal atomic find-or-create with hierarchical routing.

        Routes through the signature hierarchy starting from root:
        1. If DB is empty → create root signature
        2. Route from root → best matching child → recurse until leaf
        3. If leaf matches above threshold → update centroid and return
        4. If no match → create new child under where routing stopped (or explicit parent_id)

        Args:
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication
            parent_id: Explicit parent ID for new signatures (overrides routing)
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
                best_match, parent_for_new, best_sim = self._route_hierarchical(
                    conn, embedding, min_similarity, dsl_hint=dsl_hint
                )

                # ALWAYS_ROUTE_TO_BEST mode: accept any match, let failures drive learning
                # Per CLAUDE.md: "Let signatures fail. This is how the system learns."
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
                    # Per CLAUDE.md: leaves use graph_embedding (operational), not centroid (semantic)
                    if not best_match.is_semantic_umbrella:
                        from mycelium.data_layer.mcts import (
                            check_and_reject_if_low_similarity,
                            REJECTION_SIM_THRESHOLD,
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
                                    from mycelium.embedding_cache import cached_embed
                                    step_graph_emb = cached_embed(op_graph)  # Use singleton embedder
                                    if step_graph_emb is not None:
                                        leaf_graph_emb = np.array(best_match.graph_embedding)
                                        rejection_sim = cosine_similarity(step_graph_emb, leaf_graph_emb)
                                        logger.debug(
                                            "[routing] Leaf '%s' graph_sim=%.3f text_sim=%.3f",
                                            best_match.step_type, rejection_sim, best_sim
                                        )
                            except Exception as e:
                                logger.debug("[db] Graph embedding comparison failed: %s", e)

                        if rejection_sim < REJECTION_SIM_THRESHOLD:
                            was_rejected, rejection_count = check_and_reject_if_low_similarity(
                                signature_id=best_match.id,
                                step_text=step_text,
                                similarity=rejection_sim,
                                problem_context=parent_problem,
                            )
                            if was_rejected:
                                logger.info(
                                    "[db] Leaf '%s' REJECTED step (sim=%.3f < %.3f), rejections=%d: '%s'",
                                    best_match.step_type, rejection_sim, REJECTION_SIM_THRESHOLD,
                                    rejection_count, step_text[:40]
                                )
                                # Fall through to create new signature
                                best_match = None

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

                    # Batch propagate centroid change up to parent umbrellas
                    # Uses batching to reduce overhead on high-traffic matches
                    self._maybe_propagate_centroid(conn, best_match.id)

                    conn.commit()
                    logger.debug(
                        "[db] Matched signature (hierarchical): step='%s' sig='%s' sim=%.3f count=%d",
                        step_text[:40], best_match.step_type, best_sim, new_count or 0
                    )
                    return best_match, False

                # No match found - create new child
                # Use explicit parent_id if provided (e.g., from decomposition), else use routing result
                actual_parent_id = parent_id if parent_id is not None else (parent_for_new.id if parent_for_new else None)

                # Check if step is too complex - queue for batch decomposition
                # Per beads mycelium-mm08: Queue complex steps instead of creating many similar decompose-type sigs
                from mycelium.data_layer.mcts import is_step_complex, queue_for_decomposition
                is_complex, complexity_reason = is_step_complex(step_text)
                if is_complex:
                    try:
                        from mycelium.step_signatures.utils import pack_embedding
                        queue_for_decomposition(
                            step_text=step_text,
                            complexity_reason=complexity_reason,
                            embedding=embedding,
                            problem_context=parent_problem,
                        )
                        logger.info(
                            "[db] Queued complex step for decomposition: reason=%s step='%s'",
                            complexity_reason, step_text[:40]
                        )
                    except Exception as e:
                        logger.warning("[db] Failed to queue for decomposition: %s", e)

                sig = self._create_signature_atomic(
                    conn, step_text, embedding, parent_problem, origin_depth,
                    extracted_values=extracted_values, dsl_hint=dsl_hint,
                    parent_id=actual_parent_id
                )

                # Propagate new child's centroid up to parent umbrellas
                if sig.id is not None:
                    self.propagate_centroid_to_parents(conn, sig.id)

                conn.commit()
                parent_desc = f"id={parent_id}" if parent_id is not None else (parent_for_new.step_type if parent_for_new else "root")
                logger.info(
                    "[db] Created new signature (child of %s): step='%s' type='%s'%s",
                    parent_desc, step_text[:40], sig.step_type,
                    " [queued for decomp]" if is_complex else ""
                )
                return sig, True

            except Exception:
                conn.rollback()
                raise

    def _route_hierarchical(
        self,
        conn,
        embedding: np.ndarray,
        min_similarity: float,
        dsl_hint: str = None,
    ) -> tuple[Optional[StepSignature], Optional[StepSignature], float]:
        """Route through hierarchy using graph_embedding (operational similarity).

        Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.
        - Routers: use graph_centroid (avg of descendants' graph_embeddings)
        - Leaves: use graph_embedding (fixed operational identity)

        Falls back to text centroid if graph_embedding not available.

        Uses UCB1 scoring to balance exploitation (high-similarity, high-success)
        with exploration (under-visited signatures that might be better).

        Args:
            embedding: Text embedding of the step (fallback)
            min_similarity: Minimum similarity threshold
            dsl_hint: Operation hint from planner (+, -, *, /) for graph routing

        Returns:
            (best_match, parent_for_new, best_similarity)
            - best_match: Leaf signature if found above threshold
            - parent_for_new: Umbrella where routing stopped (for creating new child)
            - best_similarity: Similarity of best_match
        """
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
        step_graph_embedding = None
        if dsl_hint:
            op_graph = self._dsl_hint_to_graph(dsl_hint)
            if op_graph:
                from mycelium.embedding_cache import cached_embed
                step_graph_embedding = cached_embed(op_graph)
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
            # Fallback to centroid (semantic) if graph_embedding unavailable
            from mycelium.config import ALWAYS_ROUTE_TO_BEST

            # Get graph_embedding for current node (leaves have fixed, routers have centroid of children)
            current_graph_emb = current.graph_embedding
            if current_graph_emb is not None and not isinstance(current_graph_emb, np.ndarray):
                current_graph_emb = np.array(current_graph_emb)

            # Determine which embedding to compare against
            if step_graph_embedding is not None and current_graph_emb is not None:
                # Prefer graph_embedding routing (operational similarity)
                sim = cosine_similarity(step_graph_embedding, current_graph_emb)
                used_graph = True
            else:
                # Fallback to text centroid routing (semantic similarity)
                current_centroid = current.centroid
                sim = cosine_similarity(embedding, current_centroid) if current_centroid is not None else 0.0
                used_graph = False

            # If current is a leaf, return it
            if not current.is_semantic_umbrella:
                if used_graph:
                    logger.debug("[db] Leaf %d routed by graph_embedding (sim=%.3f)", current.id, sim)
                if ALWAYS_ROUTE_TO_BEST or sim >= min_similarity:
                    return current, parent_for_new, sim
                # Still return the leaf even if below threshold
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
                # Use graph_embedding if available, fallback to centroid
                if step_graph_embedding is not None and current_graph_emb is not None:
                    sim = cosine_similarity(step_graph_embedding, current_graph_emb)
                else:
                    empty_umbrella_centroid = current.centroid
                    sim = cosine_similarity(embedding, empty_umbrella_centroid) if empty_umbrella_centroid is not None else 0.0
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
                # Prefer graph_embedding for routing
                child_graph_emb = child.graph_embedding
                if child_graph_emb is not None and not isinstance(child_graph_emb, np.ndarray):
                    child_graph_emb = np.array(child_graph_emb)

                if step_graph_embedding is not None and child_graph_emb is not None:
                    # Route by graph_embedding (operational similarity)
                    child_sim = cosine_similarity(step_graph_embedding, child_graph_emb)
                    children_with_embeddings.append((child, child_sim, True))  # True = used graph
                elif child.centroid is not None:
                    # Fallback to centroid (semantic similarity)
                    child_sim = cosine_similarity(embedding, child.centroid)
                    children_with_embeddings.append((child, child_sim, False))  # False = used text
                else:
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
        embedding: np.ndarray,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        parent_id: int = None,
        dsl_hint: str = None,
    ) -> StepSignature:
        """Create a new signature within an existing transaction.

        Auto-assigns DSL based on step_type, description, extracted_values, and dsl_hint.

        Hierarchical routing:
        - First signature becomes THE root (is_root=1, is_semantic_umbrella=1)
        - Subsequent signatures become children of specified parent (or root if not specified)

        Args:
            parent_id: ID of parent signature. If None, defaults to root.
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication.
        """
        sig_id = str(uuid.uuid4())
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
        centroid_packed = pack_embedding(embedding)
        centroid_bucket = compute_centroid_bucket(embedding)

        # Auto-assign DSL based on step_type, description, planner's extracted_values, and dsl_hint
        # dsl_hint enables bidirectional LLM-signature communication
        dsl_script, dsl_type = infer_dsl_for_signature(
            step_type, step_text, extracted_values=extracted_values, dsl_hint=dsl_hint
        )

        # Auto-generate NL interface from extracted_values if we created a math DSL
        # The param names ARE the semantic descriptions - use them!
        clarifying_questions = []
        param_descriptions = {}
        if extracted_values and dsl_type == "math":
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

        # Set flags based on whether this is the root
        is_root_flag = 1 if is_first_signature else 0
        # Root is an umbrella (routes to children), others start as leaves
        is_umbrella = 1 if is_first_signature else 0

        # Calculate depth based on parent
        if is_first_signature:
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
                    is_root, is_semantic_umbrella, computation_graph, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, centroid_bucket, embedding_sum_packed, 1, step_type, step_text,
                 dsl_script, dsl_type, clarifying_json, params_json, actual_depth,
                 is_root_flag, is_umbrella, computation_graph, now),
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
            if not is_first_signature and actual_parent_id is not None:
                # Add parent-child relationship
                conn.execute(
                    """INSERT OR IGNORE INTO signature_relationships
                       (parent_id, child_id, condition, routing_order, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (actual_parent_id, row_id, step_type, 0, now),
                )
                # Mark parent as umbrella - routers don't execute DSL, they route
                # Clear dsl_script to avoid mismatch between dsl_type='router' and script type='math'
                conn.execute(
                    """UPDATE step_signatures
                       SET is_semantic_umbrella = 1, dsl_type = 'router', dsl_script = NULL
                       WHERE id = ?""",
                    (actual_parent_id,),
                )
                # Invalidate parent's children cache since we added a new child
                invalidate_children_cache(actual_parent_id)
                invalidate_signature_cache(actual_parent_id)
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

        # Also add as first example
        conn.execute(
            """INSERT INTO step_examples
               (signature_id, step_text, embedding, parent_problem, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (row_id, step_text, pack_embedding(embedding), parent_problem, now),
        )

        if is_first_signature:
            logger.info("[db] Created ROOT signature: type=%s (first in DB)", step_type)
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
            examples=[],
            uses=0,
            successes=0,
            depth=actual_depth,
            is_root=is_first_signature,
            is_semantic_umbrella=is_first_signature,
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

    def print_structure_stats(self) -> None:
        """Print a formatted summary of database structure stats."""
        stats = self.get_structure_stats()

        print("\n" + "=" * 50)
        print("DATABASE STRUCTURE STATS")
        print("=" * 50)
        print(f"Total signatures: {stats['total']}")
        print(f"Routers: {stats['by_role']['router']} | Leaves: {stats['by_role']['leaf']}")
        if stats['total'] > 0:
            ratio = stats['by_role']['router'] / stats['total']
            print(f"Router ratio: {ratio:.1%}")
        print(f"Umbrellas: {stats['umbrella_count']} (orphans: {stats['orphan_umbrellas']})")
        print(f"Overall success rate: {stats['success_rate']:.1%}")

        print("\nBy DSL type:")
        for dsl_type, count in sorted(stats['by_type'].items()):
            success = stats['success_by_type'].get(dsl_type, 0)
            print(f"  {dsl_type:15s} {count:4d} ({success:.0%} success)")

        print("\nDepth histogram:")
        for depth in sorted(stats['depth_histogram'].keys()):
            count = stats['depth_histogram'][depth]
            bar = "█" * min(count, 40)
            print(f"  {depth:2d}: {count:4d} {bar}")

        print(f"\nAvg depth: {stats['avg_depth']:.1f} | Max depth: {stats['max_depth']}")
        print("=" * 50)

    # =========================================================================
    # DSL Rewriter Support
    # =========================================================================

    def get_total_signature_uses(self) -> int:
        """Get total uses across all signatures."""
        with self._connection() as conn:
            row = conn.execute("SELECT SUM(uses) FROM step_signatures").fetchone()
            return row[0] if row and row[0] else 0

    def get_underperforming_signatures(
        self,
        min_uses: int = 10,
        max_success_rate: float = 0.40,
        limit: int = 20,
    ) -> list:
        """Find signatures with low success rate that need DSL rewriting.

        Args:
            min_uses: Minimum uses to be considered
            max_success_rate: Maximum success rate to be considered underperforming
            limit: Maximum results to return

        Returns:
            List of StepSignature objects
        """
        with self._connection() as conn:
            # Only consider leaf nodes with math DSLs (not decompose/router)
            rows = conn.execute(
                """SELECT * FROM step_signatures
                   WHERE uses >= ?
                   AND dsl_type = 'math'
                   AND is_semantic_umbrella = 0
                   AND CAST(successes AS REAL) / uses <= ?
                   ORDER BY uses DESC
                   LIMIT ?""",
                (min_uses, max_success_rate, limit)
            ).fetchall()

            return [self._row_to_signature(row) for row in rows]

    def reset_signature_stats(self, signature_id: int) -> None:
        """Reset uses and successes for a signature after DSL rewrite."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE step_signatures SET uses = 0, successes = 0 WHERE id = ?",
                (signature_id,)
            )
            logger.debug("[db] Reset stats for signature %d", signature_id)

    def mark_signature_rewritten(self, signature_id: int) -> None:
        """Mark a signature as recently rewritten (for cooldown tracking)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            # Store in last_used_at for now (could add dedicated column later)
            conn.execute(
                "UPDATE step_signatures SET last_used_at = ? WHERE id = ?",
                (now, signature_id)
            )

    def count_recently_rewritten(self, hours: int = 24) -> int:
        """Count signatures rewritten within the given time period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._connection() as conn:
            # This is approximate - uses last_used_at as proxy
            row = conn.execute(
                """SELECT COUNT(*) FROM step_signatures
                   WHERE last_used_at >= ?
                   AND dsl_type = 'math'""",
                (cutoff,)
            ).fetchone()
            return row[0] if row else 0

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
        """Update signature centroid within an existing transaction.

        Internal helper that performs the centroid update without managing
        its own transaction. Use this within code that already has an open
        transaction.

        Includes bounded drift monitoring: logs warning if centroid drift exceeds
        confidence bounds (more examples = tighter bounds).

        Formula: new_sum = old_sum + new_embedding
                 new_count = old_count + 1
                 new_centroid = new_sum / new_count

        Args:
            conn: Database connection (within transaction)
            signature_id: ID of the signature to update
            new_embedding: The new embedding to add to the running average
            update_last_used: If True, also update last_used_at timestamp

        Returns:
            New embedding count, or None if signature not found
        """
        row = conn.execute(
            """SELECT embedding_sum, embedding_count, centroid, centroid_bucket,
                      similarity_count, similarity_mean, similarity_m2
               FROM step_signatures WHERE id = ?""",
            (signature_id,)
        ).fetchone()

        if not row:
            logger.warning("[db] Cannot update centroid: signature %d not found", signature_id)
            return None

        # Parse current state
        if row["embedding_sum"]:
            current_sum = unpack_embedding(row["embedding_sum"])
            current_count = row["embedding_count"] or 1
        else:
            # Initialize from fresh centroid if no sum yet (migration case)
            fresh_centroid = unpack_embedding(row["centroid"])
            current_sum = fresh_centroid.copy() if fresh_centroid is not None else new_embedding.copy()
            current_count = 1

        old_centroid = unpack_embedding(row["centroid"])
        old_bucket = row["centroid_bucket"]

        # Variance tracking state (Welford's algorithm)
        sim_count = row["similarity_count"] or 0
        sim_mean = row["similarity_mean"] or 0.0
        sim_m2 = row["similarity_m2"] or 0.0

        # Update running sum and count
        new_sum = current_sum + new_embedding
        new_count = current_count + 1

        # Compute new centroid
        new_centroid = new_sum / new_count

        # Update variance tracking using Welford's online algorithm
        # Measures how diverse the embeddings routed to this signature are
        # High variance = too generic, should decompose
        if old_centroid is not None:
            # Compute similarity of new embedding to OLD centroid (before update)
            similarity = cosine_similarity(new_embedding, old_centroid)

            # Welford's algorithm for online variance computation
            sim_count += 1
            delta = similarity - sim_mean
            sim_mean += delta / sim_count
            delta2 = similarity - sim_mean
            sim_m2 += delta * delta2

            # Log high variance signatures (potential decomposition candidates)
            if sim_count >= 5:
                variance = sim_m2 / sim_count
                if variance > 0.01:  # High variance threshold
                    logger.debug(
                        "[db] High variance sig %d: count=%d mean=%.3f variance=%.4f",
                        signature_id, sim_count, sim_mean, variance
                    )

        # Check drift bounds (monitoring - don't reject, just warn)
        if old_centroid is not None:
            drift = 1.0 - cosine_similarity(old_centroid, new_centroid)
            # Adaptive threshold: tighter bounds with more examples
            # max_drift * decay^log2(count) - e.g., at count=8: 0.15 * 0.9^3 = 0.109
            adaptive_threshold = CENTROID_MAX_DRIFT * (CENTROID_DRIFT_DECAY ** math.log2(max(1, current_count)))
            if drift > adaptive_threshold:
                logger.warning(
                    "[db] Centroid drift %.4f exceeds bound %.4f for sig %d (count=%d)",
                    drift, adaptive_threshold, signature_id, current_count
                )

        # Compute new bucket and check if it changed
        new_bucket = compute_centroid_bucket(new_centroid)
        bucket_changed = new_bucket != old_bucket

        # Pack and store
        new_sum_packed = pack_embedding(new_sum)
        new_centroid_packed = pack_embedding(new_centroid)

        # Try updating with new bucket if changed, fall back to keeping old bucket on collision
        # All paths include variance tracking fields (similarity_count, similarity_mean, similarity_m2)
        if update_last_used:
            now = datetime.now(timezone.utc).isoformat()
            if bucket_changed:
                try:
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, centroid_bucket = ?,
                               similarity_count = ?, similarity_mean = ?, similarity_m2 = ?, last_used_at = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, new_bucket,
                         sim_count, sim_mean, sim_m2, now, signature_id),
                    )
                except sqlite3.IntegrityError:
                    # New bucket collides with existing signature - keep old bucket
                    logger.debug("[db] Bucket collision on update for sig %d, keeping old bucket", signature_id)
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?,
                               similarity_count = ?, similarity_mean = ?, similarity_m2 = ?, last_used_at = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed,
                         sim_count, sim_mean, sim_m2, now, signature_id),
                    )
            else:
                conn.execute(
                    """UPDATE step_signatures
                       SET embedding_sum = ?, embedding_count = ?, centroid = ?,
                           similarity_count = ?, similarity_mean = ?, similarity_m2 = ?, last_used_at = ?
                       WHERE id = ?""",
                    (new_sum_packed, new_count, new_centroid_packed,
                     sim_count, sim_mean, sim_m2, now, signature_id),
                )
        else:
            if bucket_changed:
                try:
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, centroid_bucket = ?,
                               similarity_count = ?, similarity_mean = ?, similarity_m2 = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, new_bucket,
                         sim_count, sim_mean, sim_m2, signature_id),
                    )
                except sqlite3.IntegrityError:
                    # New bucket collides with existing signature - keep old bucket
                    logger.debug("[db] Bucket collision on update for sig %d, keeping old bucket", signature_id)
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?,
                               similarity_count = ?, similarity_mean = ?, similarity_m2 = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed,
                         sim_count, sim_mean, sim_m2, signature_id),
                    )
            else:
                conn.execute(
                    """UPDATE step_signatures
                       SET embedding_sum = ?, embedding_count = ?, centroid = ?,
                           similarity_count = ?, similarity_mean = ?, similarity_m2 = ?
                       WHERE id = ?""",
                    (new_sum_packed, new_count, new_centroid_packed,
                     sim_count, sim_mean, sim_m2, signature_id),
                )

        if bucket_changed:
            logger.debug(
                "[db] Centroid bucket changed for sig %d: %s -> %s",
                signature_id, old_bucket, new_bucket
            )

        # Invalidate caches since centroid changed
        invalidate_centroid_cache(signature_id)
        invalidate_signature_cache(signature_id)  # Also invalidate signature cache
        self.invalidate_centroid_matrix()

        return new_count

    def update_centroid(
        self,
        signature_id: int,
        new_embedding: np.ndarray,
        propagate_to_parents: bool = True,
    ):
        """Update signature centroid with a new embedding (running average).

        This is called each time a step matches a signature. The centroid
        becomes more stable and representative over time as more examples
        are added.

        Formula: new_sum = old_sum + new_embedding
                 new_count = old_count + 1
                 new_centroid = new_sum / new_count

        Args:
            signature_id: ID of the signature to update
            new_embedding: The new embedding to add to the running average
            propagate_to_parents: If True, propagate centroid change up the tree
        """
        with self._connection() as conn:
            # Use BEGIN IMMEDIATE for atomic read-modify-write
            conn.execute("BEGIN IMMEDIATE")
            try:
                new_count = self._update_centroid_atomic(conn, signature_id, new_embedding)

                if new_count is None:
                    conn.rollback()
                    return

                # Propagate centroid change up to parent umbrellas (immediate, as caller expects)
                if propagate_to_parents:
                    self.propagate_centroid_to_parents(conn, signature_id)

                conn.commit()
                invalidate_signature_cache(signature_id)
                logger.debug(
                    "[db] Updated centroid for sig %d: count=%d",
                    signature_id, new_count
                )
            except Exception:
                conn.rollback()
                raise

    def update_centroid_on_operational_outcome(
        self,
        signature_id: int,
        embedding: np.ndarray,
        was_correct: bool,
        confidence: float = 1.0,
    ):
        """BIPOLAR centroid update based on operational outcome (Scorpion fix).

        This is the key mechanism for splitting vocab-based clusters into
        operation-based clusters:
        - SUCCESS: PULL centroid toward embedding (attract)
        - FAILURE: PUSH centroid away from embedding (repel)

        Key insight from AlphaGo/MCTS:
        - High confidence + failure = STRONG negative signal (push hard)
        - High confidence + success = STRONG positive signal (pull hard)
        - Centroids drift toward operational meaning, not vocabulary

        Both success and failure signals propagate to parent signatures,
        reinforcing good routing paths and weakening bad ones.

        Args:
            signature_id: ID of the signature to update
            embedding: The routing embedding from the step (None to skip centroid update)
            was_correct: Whether this path produced the correct answer
            confidence: How confident we were in this route (0.0-1.0)
        """
        # Skip centroid update if no embedding provided
        if embedding is None:
            return

        from mycelium.config import SCORPION_REPULSION_WEIGHT, SCORPION_ATTRACTION_WEIGHT

        if was_correct:
            # SUCCESS: Pull centroid toward this embedding (attract)
            # Weighted by confidence: high confidence = stronger pull
            attraction_strength = confidence * SCORPION_ATTRACTION_WEIGHT
            self._attract_centroid(signature_id, embedding, attraction_strength, propagate_to_parents=True)
            logger.debug(
                "[db] SCORPION PULL: sig %d centroid toward embedding (correct, conf=%.2f, strength=%.2f)",
                signature_id, confidence, attraction_strength
            )
        else:
            # FAILURE: Push centroid away from this embedding (repel)
            # Weighted by confidence: high confidence + failure = strong repulsion
            repulsion_strength = confidence * SCORPION_REPULSION_WEIGHT
            self._repel_centroid(signature_id, embedding, repulsion_strength, propagate_to_parents=True)
            logger.debug(
                "[db] SCORPION PUSH: sig %d centroid away from embedding (incorrect, conf=%.2f, strength=%.2f)",
                signature_id, confidence, repulsion_strength
            )

    def _attract_centroid(
        self,
        signature_id: int,
        embedding: np.ndarray,
        strength: float = 0.1,
        propagate_to_parents: bool = False,
    ):
        """Pull centroid TOWARD an embedding (success attraction).

        This strengthens operational clusters by pulling toward successful examples.
        The centroid moves in the direction of the embedding.

        Args:
            signature_id: Signature to update
            embedding: Embedding to attract toward
            strength: How strongly to pull (0.0-1.0)
            propagate_to_parents: If True, also update parent centroids (with decay)
        """
        with self._connection() as conn:
            row = conn.execute(
                """SELECT s.centroid, s.embedding_count, r.parent_id
                   FROM step_signatures s
                   LEFT JOIN signature_relationships r ON r.child_id = s.id
                   WHERE s.id = ?""",
                (signature_id,)
            ).fetchone()

            if row is None or row[0] is None:
                return  # No centroid to attract

            # Centroid is stored as JSON string, not bytes
            centroid_data = row[0]
            if isinstance(centroid_data, str):
                centroid = np.array(json.loads(centroid_data), dtype=np.float32)
            else:
                centroid = _parse_centroid_data(centroid_data)
            count = row[1] or 1
            parent_id = row[2]

            # Attraction: move centroid toward embedding
            # direction = embedding - centroid (opposite of repulsion)
            direction = embedding - centroid
            direction_norm = np.linalg.norm(direction)

            # Edge case: if centroid == embedding, direction is zero vector
            # This is correct behavior - no movement needed when already aligned
            if direction_norm > 0:
                direction = direction / direction_norm  # Normalize direction

            # Scale attraction by strength and inverse of count (mature signatures resist change)
            attraction_scale = strength / np.sqrt(count + 1)
            new_centroid = centroid + attraction_scale * direction

            # Normalize to unit length
            centroid_norm = np.linalg.norm(new_centroid)
            if centroid_norm > 0:
                new_centroid = new_centroid / centroid_norm

            # Update in DB
            conn.execute(
                "UPDATE step_signatures SET centroid = ? WHERE id = ?",
                (new_centroid.tobytes(), signature_id)
            )
            conn.commit()

            # Propagate to parents with decay
            if propagate_to_parents and parent_id is not None:
                decayed_strength = strength * PARENT_CREDIT_DECAY
                if decayed_strength >= PARENT_CREDIT_MIN:
                    self._attract_centroid(parent_id, embedding, decayed_strength, propagate_to_parents=True)

    def _repel_centroid(
        self,
        signature_id: int,
        embedding: np.ndarray,
        strength: float = 0.1,
        propagate_to_parents: bool = False,
    ):
        """Push centroid AWAY from an embedding (failure repulsion).

        This creates separation between operationally different clusters.
        The centroid moves in the opposite direction from the embedding.

        Args:
            signature_id: Signature to update
            embedding: Embedding to repel from
            strength: How strongly to push (0.0-1.0)
            propagate_to_parents: If True, also update parent centroids (with decay)
        """
        with self._connection() as conn:
            row = conn.execute(
                """SELECT s.centroid, s.embedding_count, r.parent_id
                   FROM step_signatures s
                   LEFT JOIN signature_relationships r ON r.child_id = s.id
                   WHERE s.id = ?""",
                (signature_id,)
            ).fetchone()

            if row is None or row[0] is None:
                return  # No centroid to repel from

            # Centroid is stored as JSON string, not bytes
            centroid_data = row[0]
            if isinstance(centroid_data, str):
                centroid = np.array(json.loads(centroid_data), dtype=np.float32)
            else:
                centroid = _parse_centroid_data(centroid_data)
            count = row[1] or 1
            parent_id = row[2]

            # Repulsion: move centroid away from embedding
            # new_centroid = centroid + strength * (centroid - embedding)
            # This pushes in the opposite direction of the embedding
            direction = centroid - embedding
            direction_norm = np.linalg.norm(direction)

            # Edge case: if centroid == embedding, direction is zero vector
            # This means the centroid is exactly at the failed embedding's position.
            # We can't determine which direction to push, so we skip the update.
            # This is rare but possible if a signature was created from this exact embedding.
            if direction_norm > 0:
                direction = direction / direction_norm  # Normalize direction
            else:
                logger.debug(
                    "[db] SCORPION: skipping repulsion for sig %d (centroid == embedding)",
                    signature_id
                )
                return

            # Scale repulsion by strength and inverse of count (mature signatures resist change)
            repulsion_scale = strength / np.sqrt(count + 1)
            new_centroid = centroid + repulsion_scale * direction

            # Normalize to unit length
            centroid_norm = np.linalg.norm(new_centroid)
            if centroid_norm > 0:
                new_centroid = new_centroid / centroid_norm

            # Update in DB
            conn.execute(
                "UPDATE step_signatures SET centroid = ? WHERE id = ?",
                (new_centroid.tobytes(), signature_id)
            )
            conn.commit()

            # Propagate to parents with decay
            if propagate_to_parents and parent_id is not None:
                decayed_strength = strength * PARENT_CREDIT_DECAY
                if decayed_strength >= PARENT_CREDIT_MIN:
                    self._repel_centroid(parent_id, embedding, decayed_strength, propagate_to_parents=True)

    # Backward compatibility alias
    def update_centroid_on_operational_success(
        self,
        signature_id: int,
        embedding: np.ndarray,
        was_correct: bool,
    ):
        """Backward-compatible alias for update_centroid_on_operational_outcome."""
        self.update_centroid_on_operational_outcome(signature_id, embedding, was_correct)

    def record_operational_failure(
        self,
        signature_id: int,
        produced_answer: str,
        expected_answer: str,
    ):
        """Record an operational failure for a signature path.

        Per CLAUDE.md: "Record every failure—it feeds the refinement loop"

        This tracks when a signature produces a different answer than ground truth,
        providing signal for potential cluster splitting. Over time, if a signature
        accumulates many operational failures, it may need to be decomposed into
        more specific sub-signatures.

        Args:
            signature_id: ID of the signature that failed operationally
            produced_answer: What the signature produced
            expected_answer: The ground truth answer
        """
        with self._connection() as conn:
            # Increment operational failure count
            # This is separate from regular "uses" - tracks semantic mismatches
            conn.execute(
                """UPDATE step_signatures
                   SET operational_failures = COALESCE(operational_failures, 0) + 1
                   WHERE id = ?""",
                (signature_id,)
            )

            # Log for analysis (could be expanded to store in separate table)
            logger.debug(
                "[db] Recorded operational failure for sig %d: produced=%s, expected=%s",
                signature_id,
                produced_answer[:30] if produced_answer else "None",
                expected_answer[:30] if expected_answer else "None",
            )

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
                conn.execute(
                    """UPDATE step_signatures
                       SET successes = COALESCE(successes, 0) + ?
                       WHERE id = ?""",
                    (thread_count, signature_id)
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
                conn.execute(
                    """UPDATE step_signatures
                       SET operational_failures = COALESCE(operational_failures, 0) + ?
                       WHERE id = ?""",
                    (failure_count, signature_id)
                )
                logger.debug(
                    "[db] Destructive interference: sig %d recorded %d failures "
                    "(%d/%d threads failed)",
                    signature_id, failure_count, failure_count, thread_count,
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
        from mycelium.config import PARENT_CREDIT_DECAY, PARENT_CREDIT_MAX_DEPTH, PARENT_CREDIT_MIN

        with self._connection() as conn:
            # Update this signature
            conn.execute(
                """UPDATE step_signatures
                   SET successes = COALESCE(successes, 0) + ?
                   WHERE id = ?""",
                (count, signature_id)
            )

            # Propagate to parent with decay
            if propagate_to_parents and _depth < PARENT_CREDIT_MAX_DEPTH:
                parent_row = conn.execute(
                    "SELECT parent_id FROM signature_relationships WHERE child_id = ? LIMIT 1",
                    (signature_id,)
                ).fetchone()

                if parent_row and parent_row[0]:
                    decayed_count = count * PARENT_CREDIT_DECAY
                    if decayed_count >= PARENT_CREDIT_MIN:
                        self.increment_signature_successes(
                            parent_row[0],
                            count=decayed_count,
                            propagate_to_parents=True,
                            _depth=_depth + 1,
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
        from mycelium.config import PARENT_CREDIT_DECAY, PARENT_CREDIT_MAX_DEPTH, PARENT_CREDIT_MIN

        with self._connection() as conn:
            # Update this signature
            conn.execute(
                """UPDATE step_signatures
                   SET operational_failures = COALESCE(operational_failures, 0) + ?
                   WHERE id = ?""",
                (count, signature_id)
            )

            # Propagate to parent with decay
            if propagate_to_parents and _depth < PARENT_CREDIT_MAX_DEPTH:
                parent_row = conn.execute(
                    "SELECT parent_id FROM signature_relationships WHERE child_id = ? LIMIT 1",
                    (signature_id,)
                ).fetchone()

                if parent_row and parent_row[0]:
                    decayed_count = count * PARENT_CREDIT_DECAY
                    if decayed_count >= PARENT_CREDIT_MIN:
                        self.increment_signature_failures(
                            parent_row[0],
                            count=decayed_count,
                            propagate_to_parents=True,
                            _depth=_depth + 1,
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
        from mycelium.config import PARENT_CREDIT_DECAY, PARENT_CREDIT_MAX_DEPTH, PARENT_CREDIT_MIN

        with self._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET successes = COALESCE(successes, 0) + ?
                   WHERE id = ?""",
                (weight, signature_id)
            )

            # Propagate to parent with decay
            if propagate_to_parents and _depth < PARENT_CREDIT_MAX_DEPTH:
                parent_row = conn.execute(
                    "SELECT parent_id FROM signature_relationships WHERE child_id = ? LIMIT 1",
                    (signature_id,)
                ).fetchone()

                if parent_row and parent_row[0]:
                    decayed_weight = weight * PARENT_CREDIT_DECAY
                    if decayed_weight >= PARENT_CREDIT_MIN:
                        self.increment_signature_partial_success(
                            parent_row[0],
                            weight=decayed_weight,
                            propagate_to_parents=True,
                            _depth=_depth + 1,
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
            conn.execute(
                "UPDATE step_signatures SET is_archived = 1 WHERE id = ?",
                (absorbed_id,)
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

    def flag_for_split(self, signature_id: int, reason: str = "destructive_interference") -> bool:
        """Flag a signature for potential decomposition/split.

        This marks a signature as needing attention due to mixed interference
        results. The actual decomposition is triggered by umbrella_learner.

        This increments operational_failures which gates decomposition in umbrella_learner.
        Per CLAUDE.md: only signatures with operational_failures > 0 are decomposition candidates.

        Args:
            signature_id: ID of signature to flag
            reason: Why it's being flagged

        Returns:
            True if flagged successfully
        """
        with self._connection() as conn:
            # Increment operational failures to trigger decomposition consideration
            # The umbrella learner checks success rate and will decompose low performers
            conn.execute(
                """UPDATE step_signatures
                   SET operational_failures = COALESCE(operational_failures, 0) + 1
                   WHERE id = ?""",
                (signature_id,)
            )

            logger.info(
                "[db] Flagged signature %d for split (reason: %s)",
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
            conn.execute(
                """UPDATE step_signatures
                   SET is_archived = 1
                   WHERE id = ?""",
                (signature_id,)
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
            conn.execute(
                """UPDATE step_signatures
                   SET is_archived = 1
                   WHERE id = ?""",
                (signature_id,)
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
            conn.execute(
                """UPDATE step_signatures
                   SET uses = uses + 1
                   WHERE id = ?""",
                (signature_id,),
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
                    # Only demote if:
                    # 1. This step failed (step_completed=False), OR
                    # 2. We have graded history (successes > 0) showing poor success rate
                    should_demote = (
                        (not step_completed) or  # This step failed
                        (successes > 0 and success_rate < AUTO_DEMOTE_MAX_SUCCESS_RATE)  # Historical failures
                    )
                    if should_demote:
                        # Promote to umbrella: clear DSL, set type to router
                        # Clear dsl_script to avoid type mismatch
                        conn.execute(
                            """UPDATE step_signatures
                               SET is_semantic_umbrella = 1,
                                   dsl_type = 'router',
                                   dsl_script = NULL
                               WHERE id = ?""",
                            (signature_id,),
                        )
                        logger.info(
                            "[db] Auto-demoted sig %d to umbrella/router (%.0f%% after %d uses, step_ok=%s, min=%d)",
                            signature_id, success_rate * 100, uses, step_completed, min_uses
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
        conn.execute(
            """UPDATE step_signatures
               SET difficulty_stats = ?, max_difficulty_solved = ?
               WHERE id = ?""",
            (json.dumps(stats), max_diff, signature_id),
        )

    def record_failure(
        self,
        step_text: str,
        failure_type: str,
        error_message: str = None,
        signature_id: int = None,
        context: dict = None,
    ) -> int:
        """Record a step failure for pattern learning.

        Per CLAUDE.md: "Failures Are Valuable Data Points"
        - Record every failure—it feeds the refinement loop
        - Failed signatures get decomposed
        - Success/failure stats drive routing decisions

        Args:
            step_text: The step that failed
            failure_type: Category of failure:
                - 'dsl_error': DSL execution failed
                - 'no_match': No signature matched
                - 'llm_error': LLM call failed
                - 'timeout': Operation timed out
                - 'validation': Result validation failed
                - 'routing': Umbrella routing failed
            error_message: The actual error text
            signature_id: ID of signature that failed (None if no match)
            context: Additional context dict (params, expected, problem, etc.)

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

            logger.debug(
                "[db] Recorded failure: type=%s sig=%s step='%s'",
                failure_type, signature_id, step_text[:50]
            )

            return failure_id

    def get_failure_patterns(
        self,
        signature_id: int = None,
        failure_type: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get failure patterns for analysis.

        Used to:
        1. Identify signatures that need decomposition
        2. Feed planner hints about common failure patterns
        3. Inform DSL rewriting decisions

        Args:
            signature_id: Filter by specific signature (None for all)
            failure_type: Filter by failure type (None for all)
            limit: Maximum records to return

        Returns:
            List of failure records with counts grouped by pattern
        """
        with self._connection() as conn:
            if signature_id is not None:
                cursor = conn.execute(
                    """SELECT signature_id, failure_type, COUNT(*) as count,
                              GROUP_CONCAT(DISTINCT error_message) as errors
                       FROM step_failures
                       WHERE signature_id = ?
                       GROUP BY signature_id, failure_type
                       ORDER BY count DESC
                       LIMIT ?""",
                    (signature_id, limit),
                )
            elif failure_type is not None:
                cursor = conn.execute(
                    """SELECT signature_id, failure_type, COUNT(*) as count,
                              GROUP_CONCAT(DISTINCT error_message) as errors
                       FROM step_failures
                       WHERE failure_type = ?
                       GROUP BY signature_id, failure_type
                       ORDER BY count DESC
                       LIMIT ?""",
                    (failure_type, limit),
                )
            else:
                cursor = conn.execute(
                    """SELECT signature_id, failure_type, COUNT(*) as count,
                              GROUP_CONCAT(DISTINCT error_message) as errors
                       FROM step_failures
                       GROUP BY signature_id, failure_type
                       ORDER BY count DESC
                       LIMIT ?""",
                    (limit,),
                )

            return [
                {
                    "signature_id": row["signature_id"],
                    "failure_type": row["failure_type"],
                    "count": row["count"],
                    "errors": row["errors"],
                }
                for row in cursor.fetchall()
            ]

    def get_signatures_needing_decomposition(
        self,
        min_failures: int = 3,
        failure_types: list[str] = None,
    ) -> list[int]:
        """Get signatures with repeated failures that may need decomposition.

        Per CLAUDE.md: "Failed signatures get decomposed"

        Args:
            min_failures: Minimum failure count to consider
            failure_types: Filter by failure types (default: dsl_error, validation)

        Returns:
            List of signature IDs that should be considered for decomposition
        """
        if failure_types is None:
            failure_types = ["dsl_error", "validation"]

        placeholders = ",".join("?" * len(failure_types))

        with self._connection() as conn:
            cursor = conn.execute(
                f"""SELECT signature_id, COUNT(*) as fail_count
                    FROM step_failures
                    WHERE signature_id IS NOT NULL
                      AND failure_type IN ({placeholders})
                    GROUP BY signature_id
                    HAVING fail_count >= ?
                    ORDER BY fail_count DESC""",
                (*failure_types, min_failures),
            )
            return [row["signature_id"] for row in cursor.fetchall()]

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
            invalidate_signature_cache(signature_id)

    # =========================================================================
    # Helpers
    # =========================================================================

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
        HINT_TO_GRAPH = {
            "+": "ADD(a, b)",
            "add": "ADD(a, b)",
            "sum": "ADD(a, b)",
            "-": "SUB(a, b)",
            "subtract": "SUB(a, b)",
            "difference": "SUB(a, b)",
            "*": "MUL(a, b)",
            "multiply": "MUL(a, b)",
            "product": "MUL(a, b)",
            "/": "DIV(a, b)",
            "divide": "DIV(a, b)",
            "quotient": "DIV(a, b)",
        }

        return HINT_TO_GRAPH.get(hint)

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
            # Mark parent as umbrella - routers don't execute DSL, they route
            # Clear dsl_script to avoid mismatch between dsl_type='router' and script type='math'
            conn.execute(
                """UPDATE step_signatures
                   SET is_semantic_umbrella = 1, dsl_type = 'router', dsl_script = NULL
                   WHERE id = ?""",
                (parent_id,),
            )
            # Set child's depth = parent_depth + 1
            child_depth = parent_depth + 1
            conn.execute(
                "UPDATE step_signatures SET depth = ? WHERE id = ?",
                (child_depth, child_id),
            )
            # Invalidate caches (new child affects routing and lookups)
            invalidate_centroid_cache(parent_id)
            invalidate_children_cache(parent_id)
            invalidate_signature_cache(parent_id)
            invalidate_signature_cache(child_id)
            self.invalidate_centroid_matrix()

            # Propagate graph_centroid to parents (Option B: routers use graph_centroid)
            self.propagate_graph_centroid_to_parents(conn, child_id)

            logger.info(
                "[db] Added child: parent=%d (depth=%d) → child=%d (depth=%d) (condition='%s')",
                parent_id, parent_depth, child_id, child_depth, condition[:30]
            )
            return True

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
            # Clear DSL and set type to router - umbrellas don't execute, they route
            # Clear dsl_script to avoid mismatch between dsl_type='router' and script type='math'
            cursor = conn.execute(
                """UPDATE step_signatures
                   SET is_semantic_umbrella = 1,
                       dsl_type = 'router',
                       dsl_script = NULL
                   WHERE id = ?""",
                (signature_id,),
            )
            if cursor.rowcount > 0:
                invalidate_signature_cache(signature_id)

                # Compute graph_centroid from children (Option B: routers use graph_centroid)
                graph_centroid = self.compute_graph_centroid_from_children(conn, signature_id)
                if graph_centroid is not None:
                    conn.execute(
                        "UPDATE step_signatures SET graph_embedding = ? WHERE id = ?",
                        (json.dumps(graph_centroid.tolist()), signature_id),
                    )
                    logger.info(
                        "[db] Promoted signature %d to umbrella with graph_centroid",
                        signature_id
                    )
                else:
                    logger.info(
                        "[db] Promoted signature %d to umbrella (no children with graph_embeddings yet)",
                        signature_id
                    )
                return True
            return False

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

    def detect_upward_restructuring(
        self,
        embedding: np.ndarray,
        difficulty: float,
        min_similarity: float = 0.6,
        difficulty_gap_threshold: float = 0.2,
    ) -> Optional[tuple[StepSignature, float]]:
        """Detect if a new problem should trigger upward restructuring.

        Upward restructuring occurs when a new problem represents a HIGHER
        abstraction than existing signatures. This happens when:
        1. Problem is semantically similar to existing signatures
        2. Problem difficulty significantly exceeds their max_difficulty_solved

        Key insight from CLAUDE.md: "if we start with gsm8k and increase to
        MATH L1-L2 that might represent a new node higher up in the tree"

        When detected, the caller should:
        1. Create a new umbrella signature at the higher abstraction level
        2. Make the existing signature a child of the new umbrella
        3. The new umbrella represents the harder class of problems

        Args:
            embedding: Query embedding of the new problem
            difficulty: Estimated difficulty of the new problem (0.0-1.0)
            min_similarity: Minimum similarity to consider signatures
            difficulty_gap_threshold: Min gap between problem and sig difficulty

        Returns:
            Tuple of (matched_signature, difficulty_gap) if restructuring needed,
            None otherwise.
        """
        from mycelium.step_signatures.utils import cosine_similarity

        with self._connection() as conn:
            # Find non-root signatures (we don't restructure above the root)
            cursor = conn.execute(
                """SELECT * FROM step_signatures
                   WHERE is_root = 0
                   AND is_archived = 0
                   ORDER BY max_difficulty_solved DESC"""
            )
            rows = cursor.fetchall()

        best_candidate = None
        best_gap = 0.0

        for row in rows:
            sig_id = row["id"]
            centroid = get_cached_centroid(sig_id, row.get("centroid"))
            if centroid is None:
                continue

            sim = cosine_similarity(embedding, centroid)
            if sim < min_similarity:
                continue

            # Check difficulty gap
            max_diff = row.get("max_difficulty_solved") or 0.0
            gap = difficulty - max_diff

            if gap >= difficulty_gap_threshold and gap > best_gap:
                best_candidate = StepSignature.from_row_fast(row)
                best_gap = gap

        if best_candidate:
            logger.info(
                "[db] Upward restructuring detected: sig=%d (%s) max_diff=%.2f, "
                "problem_diff=%.2f, gap=%.2f",
                best_candidate.id,
                best_candidate.step_type[:30],
                best_candidate.max_difficulty_solved,
                difficulty,
                best_gap,
            )

        return (best_candidate, best_gap) if best_candidate else None

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

        # Generate step_type from description
        step_type = normalize_step_text(description)[:50]

        with self._connection() as conn:
            now = datetime.now(timezone.utc).isoformat()
            sig_id = f"umbrella_{uuid.uuid4().hex[:8]}"
            centroid_json = json.dumps(new_centroid.tolist())
            centroid_bucket = compute_centroid_bucket(new_centroid)

            try:
                cursor = conn.execute(
                    """INSERT INTO step_signatures (
                        signature_id, centroid, centroid_bucket, embedding_sum, embedding_count,
                        step_type, description, dsl_type,
                        is_semantic_umbrella, depth, max_difficulty_solved, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sig_id,
                        centroid_json,
                        centroid_bucket,
                        centroid_json,  # embedding_sum starts as centroid
                        2,  # count: child centroid + new problem
                        step_type,
                        description,
                        "router",  # umbrellas route, don't execute
                        1,  # is_semantic_umbrella
                        max(0, child_signature.depth - 1),  # one level above child
                        difficulty,  # starts with the higher difficulty
                        now,
                    ),
                )
                new_id = cursor.lastrowid

                # Create parent-child relationship
                conn.execute(
                    """INSERT INTO signature_relationships (parent_id, child_id, condition, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (new_id, child_signature.id, f"difficulty <= {child_signature.max_difficulty_solved}", now),
                )

                # Invalidate caches
                self.invalidate_centroid_matrix()

                logger.info(
                    "[db] Created upward umbrella: id=%d, child=%d, depth=%d, max_diff=%.2f",
                    new_id, child_signature.id, max(0, child_signature.depth - 1), difficulty
                )

                # Fetch and return the new signature
                cursor = conn.execute("SELECT * FROM step_signatures WHERE id = ?", (new_id,))
                row = cursor.fetchone()
                return StepSignature.from_row_fast(row) if row else None

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
                    conn.execute(
                        "UPDATE step_signatures SET is_semantic_umbrella = 0 WHERE id = ?",
                        (parent_id,),
                    )
                # Invalidate parent's centroid cache (child removal affects routing)
                invalidate_centroid_cache(parent_id)
                self.invalidate_centroid_matrix()
                logger.info("[db] Removed child relationship: parent=%d → child=%d", parent_id, child_id)
                return True
            return False

    def get_umbrella_signatures(self) -> list[StepSignature]:
        """Get all signatures that are semantic umbrellas (fast variant)."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT id, signature_id, centroid, embedding_count, step_type,
                       description, param_descriptions, dsl_script, dsl_type,
                       uses, successes, is_semantic_umbrella, is_root, depth,
                       created_at, last_used_at
                FROM step_signatures WHERE is_semantic_umbrella = 1
            """)
            return [self._row_to_signature_fast(row) for row in cursor.fetchall()]

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
                invalidate_signature_cache(signature_id)
                logger.debug("[db] Updated graph_embedding for sig %d", signature_id)
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

    def get_signatures_with_graph_embeddings(
        self, for_routing: bool = True
    ) -> list[StepSignature]:
        """Get all signatures that have graph_embeddings for routing.

        Returns signatures optimized for routing (minimal parsing).

        Args:
            for_routing: If True, use fast parsing (skip most JSON fields)

        Returns:
            List of signatures with graph_embedding populated
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT *
                   FROM step_signatures
                   WHERE graph_embedding IS NOT NULL
                     AND graph_embedding != ''
                     AND is_semantic_umbrella = 0"""  # Only leaf nodes for execution
            ).fetchall()

            if for_routing:
                return [self._row_to_signature_for_routing(dict(row)) for row in rows]
            return [self._row_to_signature(dict(row)) for row in rows]

    def route_by_graph_embedding(
        self,
        operation_embedding: np.ndarray,
        min_similarity: float = 0.75,
        top_k: int = 5,
        hierarchical: bool = True,
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
            hierarchical: If True, traverse hierarchy; if False, flat search

        Returns:
            List of (signature, similarity) tuples, sorted by similarity descending
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

        if hierarchical:
            return self._route_by_graph_hierarchical(
                operation_embedding, min_similarity, top_k, UMBRELLA_MAX_DEPTH
            )
        else:
            return self._route_by_graph_flat(operation_embedding, min_similarity, top_k)

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

    def _route_by_graph_flat(
        self,
        operation_embedding: np.ndarray,
        min_similarity: float,
        top_k: int,
    ) -> list[tuple[StepSignature, float]]:
        """Flat graph routing - search all leaves directly (legacy mode)."""
        matches = []

        with self._connection() as conn:
            # Get all signatures with graph embeddings (leaf nodes only)
            rows = conn.execute(
                """SELECT id, graph_embedding, step_type, description,
                          dsl_script, dsl_type, computation_graph,
                          uses, successes, depth, is_semantic_umbrella
                   FROM step_signatures
                   WHERE graph_embedding IS NOT NULL
                     AND graph_embedding != ''
                     AND is_semantic_umbrella = 0"""
            ).fetchall()

            for row in rows:
                try:
                    graph_emb = np.array(json.loads(row["graph_embedding"]))
                    sim = cosine_similarity(operation_embedding, graph_emb)

                    if sim >= min_similarity:
                        # Create minimal signature for routing
                        sig = StepSignature(
                            id=row["id"],
                            step_type=row["step_type"],
                            description=row["description"],
                            dsl_script=row["dsl_script"],
                            dsl_type=row["dsl_type"],
                            computation_graph=row["computation_graph"],
                            uses=row["uses"] or 0,
                            successes=row["successes"] or 0,
                            depth=row["depth"] or 0,
                            is_semantic_umbrella=bool(row["is_semantic_umbrella"]),
                        )
                        matches.append((sig, sim))

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("[db] Invalid graph_embedding for sig %d: %s", row["id"], e)
                    continue

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
