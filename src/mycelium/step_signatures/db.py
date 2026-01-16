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
from typing import Optional, Literal

import numpy as np

# Version for centroid matrix cache (increment to invalidate old caches)
_CENTROID_CACHE_VERSION = 1

from mycelium.config import (
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
)

logger = logging.getLogger(__name__)


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


MatchMode = Literal["cosine", "auto"]


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

        self._init_schema()

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

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

                # Quick staleness check: compare signature count
                with self._connection() as conn:
                    row = conn.execute("SELECT COUNT(*) FROM step_signatures WHERE centroid IS NOT NULL").fetchone()
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
        from mycelium.config import (
            SCAFFOLD_ENABLED,
            SCAFFOLD_LEVELS,
            SCAFFOLD_BRANCHES_PER_LEVEL,
        )

        if not SCAFFOLD_ENABLED:
            logger.debug("[db] Scaffold disabled, skipping initialization")
            return False

        # Check if scaffold already exists (has root with children)
        if self.has_root():
            root = self.get_root()
            if root:
                with self._connection() as conn:
                    child_count = conn.execute(
                        "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                        (root.id,)
                    ).fetchone()[0]
                    if child_count > 0:
                        logger.debug("[db] Scaffold already exists (%d root children)", child_count)
                        return False

        logger.info(
            "[db] Initializing scaffold: %d levels, %d branches/level",
            SCAFFOLD_LEVELS, SCAFFOLD_BRANCHES_PER_LEVEL
        )

        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            # Create root if it doesn't exist
            root_row = conn.execute(
                "SELECT id FROM step_signatures WHERE is_root = 1 LIMIT 1"
            ).fetchone()

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
                # Ensure root is an umbrella
                conn.execute(
                    "UPDATE step_signatures SET is_semantic_umbrella = 1, dsl_type = 'router' WHERE id = ?",
                    (root_id,)
                )

            # Create placeholder umbrellas for each level
            current_level_parents = [root_id]

            for level in range(1, SCAFFOLD_LEVELS + 1):
                next_level_parents = []

                for parent_idx, parent_id in enumerate(current_level_parents):
                    for branch in range(SCAFFOLD_BRANCHES_PER_LEVEL):
                        # Create placeholder signature
                        placeholder_id = f"scaffold_L{level}_{parent_idx}_{branch}_{uuid.uuid4().hex[:6]}"
                        cursor = conn.execute(
                            """INSERT INTO step_signatures (
                                signature_id, centroid, centroid_bucket,
                                step_type, description, dsl_type,
                                is_semantic_umbrella, is_root, depth, created_at
                            ) VALUES (?, NULL, NULL, ?, ?, ?, 1, 0, ?, ?)""",
                            (
                                placeholder_id,
                                f"placeholder_L{level}_{branch}",
                                f"Scaffold placeholder at level {level}",
                                "router",
                                level,
                                now,
                            )
                        )
                        child_id = cursor.lastrowid
                        next_level_parents.append(child_id)

                        # Create parent-child relationship
                        conn.execute(
                            """INSERT INTO signature_relationships (parent_id, child_id, condition, created_at)
                               VALUES (?, ?, ?, ?)""",
                            (parent_id, child_id, f"scaffold_level_{level}", now)
                        )

                logger.info(
                    "[db] Created scaffold level %d: %d placeholders",
                    level, len(next_level_parents)
                )
                current_level_parents = next_level_parents

            conn.commit()

        # Invalidate caches
        self._cached_root = None
        self.invalidate_centroid_matrix()

        total_placeholders = sum(
            SCAFFOLD_BRANCHES_PER_LEVEL ** i for i in range(1, SCAFFOLD_LEVELS + 1)
        )
        logger.info(
            "[db] Scaffold initialized: %d total placeholders across %d levels",
            total_placeholders, SCAFFOLD_LEVELS
        )
        return True

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
                logger.debug(
                    "[db] Propagated centroid to parent %d (weight=%d)",
                    parent_id, weight
                )
            except sqlite3.IntegrityError:
                logger.debug(
                    "[db] Skipped centroid propagation to parent %d (collision)",
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
    ) -> RoutingResult:
        """Route with confidence scoring for MCTS multi-path exploration.

        Enhanced version of route_through_hierarchy that computes confidence
        signals based on UCB1 score gaps between top-k children at each level.

        Confidence interpretation:
        - High confidence (>0.8): Clear winner, single path likely sufficient
        - Medium confidence (0.5-0.8): Consider exploring 1-2 alternatives
        - Low confidence (<0.5): High uncertainty, explore multiple paths

        Args:
            embedding: The query embedding to route
            min_similarity: Minimum similarity threshold to follow a route
            max_depth: Maximum depth to traverse (default from config)
            top_k: Number of top alternatives to track at each level

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

            for child_sig, _condition in children:
                centroid = child_sig.centroid
                if centroid is None:
                    continue
                sim = cosine_similarity(embedding, centroid)
                if sim >= min_similarity * 0.7:  # Lower threshold to capture alternatives
                    ucb1 = compute_ucb1_score(
                        cosine_sim=sim,
                        uses=child_sig.uses,
                        successes=child_sig.successes,
                        parent_uses=parent_uses,
                        last_used_at=child_sig.last_used_at,
                    )
                    scored_children.append((child_sig, ucb1, sim))

            if not scored_children:
                # No children match - return current as best effort
                break

            # Sort by UCB1 score (descending)
            scored_children.sort(key=lambda x: x[1], reverse=True)

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
        match_mode: MatchMode = "cosine",
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
            match_mode: Matching algorithm (cosine or auto)
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
        match_mode: MatchMode = "cosine",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
    ) -> tuple[StepSignature, bool]:
        """Async version of find_or_create with non-blocking retry sleep.

        Use this from async contexts to avoid blocking the event loop during
        database contention retries.

        Args:
            step_text: The step description text
            embedding: Embedding vector for the step
            min_similarity: Minimum cosine similarity for matching
            parent_problem: The parent problem this step came from
            match_mode: Matching algorithm (cosine or auto)
            dsl_hint: Explicit operation hint from planner (+, -, *, /) for bidirectional communication
            origin_depth: Decomposition depth at which this step was created
            extracted_values: Dict of semantic param names -> values from planner
            parent_id: Explicit parent ID for new signatures (overrides routing)

        Returns:
            Tuple of (signature, is_new) where is_new=True if newly created
        """
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
                    await asyncio.sleep(delay + jitter)
                    logger.debug(
                        "[db] Retry %d/%d after OperationalError: %s (delay=%.3fs)",
                        attempt + 1, DB_MAX_RETRIES, str(e)[:50], delay + jitter
                    )
                    continue
                raise

    def create_signature(
        self,
        step_text: str,
        embedding: np.ndarray,
        parent_problem: str = "",
        origin_depth: int = 0,
        extracted_values: dict = None,
        dsl_hint: str = None,
        parent_id: int = None,
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

        Returns:
            The newly created StepSignature
        """
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
                best_match, parent_for_new, best_sim = self._route_hierarchical(
                    conn, embedding, min_similarity
                )

                if best_match is not None and best_sim >= min_similarity:
                    # Found a match - update centroid using shared helper
                    new_count = self._update_centroid_atomic(
                        conn, best_match.id, embedding, update_last_used=True
                    )

                    # Propagate centroid change up to parent umbrellas
                    self.propagate_centroid_to_parents(conn, best_match.id)

                    conn.commit()
                    logger.debug(
                        "[db] Matched signature (hierarchical): step='%s' sig='%s' sim=%.3f count=%d",
                        step_text[:40], best_match.step_type, best_sim, new_count or 0
                    )
                    return best_match, False

                # No match found - create new child
                # Use explicit parent_id if provided (e.g., from decomposition), else use routing result
                actual_parent_id = parent_id if parent_id is not None else (parent_for_new.id if parent_for_new else None)
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
                    "[db] Created new signature (child of %s): step='%s' type='%s'",
                    parent_desc, step_text[:40], sig.step_type
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
    ) -> tuple[Optional[StepSignature], Optional[StepSignature], float]:
        """Route through hierarchy using MCTS-style UCB1 selection.

        Uses UCB1 scoring to balance exploitation (high-similarity, high-success)
        with exploration (under-visited signatures that might be better).

        SCAFFOLD SUPPORT: Handles null-centroid placeholders by:
        1. Picking least-used placeholder when all children have null centroids
        2. Initializing placeholder centroid with first problem that routes through
        3. Continuing to route until MIN_SIGNATURE_DEPTH for proper tree structure

        Returns:
            (best_match, parent_for_new, best_similarity)
            - best_match: Leaf signature if found above threshold
            - parent_for_new: Umbrella where routing stopped (for creating new child)
            - best_similarity: Similarity of best_match
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH, SCAFFOLD_ENABLED, MIN_SIGNATURE_DEPTH

        # Validate max depth to prevent unbounded recursion
        max_depth = max(1, min(int(UMBRELLA_MAX_DEPTH or 10), 100))  # Hard cap at 100

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
            # Check similarity to current node
            # Capture centroid once to avoid TOCTOU race condition
            current_centroid = current.centroid
            if current_centroid is not None:
                sim = cosine_similarity(embedding, current_centroid)
                # If current is a leaf and matches, return it
                if not current.is_semantic_umbrella and sim >= min_similarity:
                    return current, parent_for_new, sim

            # If current is not an umbrella, it's a leaf - return similarity result
            if not current.is_semantic_umbrella:
                # Capture centroid once to avoid TOCTOU race condition
                leaf_centroid = current.centroid
                sim = cosine_similarity(embedding, leaf_centroid) if leaf_centroid is not None else 0.0
                return current, parent_for_new, sim

            # Get children of current umbrella
            cursor = conn.execute(
                """SELECT s.* FROM signature_relationships r
                   JOIN step_signatures s ON r.child_id = s.id
                   WHERE r.parent_id = ?
                   ORDER BY r.routing_order ASC""",
                (current.id,)
            )
            children = [self._row_to_signature(row) for row in cursor.fetchall()]

            if not children:
                # Umbrella with no children - return current as best match
                # Capture centroid once to avoid TOCTOU race condition
                empty_umbrella_centroid = current.centroid
                sim = cosine_similarity(embedding, empty_umbrella_centroid) if empty_umbrella_centroid is not None else 0.0
                return current, current, sim

            # MCTS UCB1 Selection: balance exploitation vs exploration
            # parent_uses = current node's uses (N in UCB1 formula)
            parent_uses = current.uses or 1

            best_child = None
            best_child_sim = 0.0
            best_child_score = 0.0

            # Separate children with centroids from null-centroid placeholders
            children_with_centroids = []
            null_centroid_children = []

            for child in children:
                centroid = child.centroid
                if centroid is None:
                    null_centroid_children.append(child)
                else:
                    child_sim = cosine_similarity(embedding, centroid)
                    children_with_centroids.append((child, child_sim))

            # Try children with centroids first (standard UCB1 selection)
            for child, child_sim in children_with_centroids:
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

            # SCAFFOLD: If no match found but we have null-centroid placeholders,
            # and we haven't reached MIN_SIGNATURE_DEPTH yet, route through one
            if best_child is None and null_centroid_children and SCAFFOLD_ENABLED:
                if depth < MIN_SIGNATURE_DEPTH - 1:
                    # Pick least-used placeholder (exploration) or random if all equal
                    null_centroid_children.sort(key=lambda c: c.uses or 0)
                    placeholder = null_centroid_children[0]

                    # Initialize placeholder's centroid with this embedding
                    logger.info(
                        "[db] Initializing scaffold placeholder: id=%d depth=%d",
                        placeholder.id, depth + 1
                    )
                    self._update_centroid_atomic(conn, placeholder.id, embedding, update_last_used=False)

                    # Update our local object to reflect the change
                    placeholder.centroid = embedding
                    best_child = placeholder
                    best_child_sim = 1.0  # Perfect match since we just set it

            if best_child is None:
                # No child matches above threshold
                # SCAFFOLD: If we haven't reached MIN_SIGNATURE_DEPTH, keep routing
                if SCAFFOLD_ENABLED and depth < MIN_SIGNATURE_DEPTH - 1:
                    # Must continue deeper - pick best below-threshold or any placeholder
                    best_below = None
                    best_below_sim = 0.0
                    best_below_score = -float('inf')

                    # First try children with centroids
                    for child, child_sim in children_with_centroids:
                        score = compute_ucb1_score(
                            child_sim, child.uses, child.successes,
                            parent_uses, child.last_used_at
                        )
                        if score > best_below_score:
                            best_below = child
                            best_below_sim = child_sim
                            best_below_score = score

                    # If no children with centroids, use a placeholder
                    if best_below is None and null_centroid_children:
                        null_centroid_children.sort(key=lambda c: c.uses or 0)
                        placeholder = null_centroid_children[0]
                        # Initialize centroid
                        self._update_centroid_atomic(conn, placeholder.id, embedding, update_last_used=False)
                        placeholder.centroid = embedding
                        best_below = placeholder
                        best_below_sim = 1.0
                        logger.info(
                            "[db] Initializing scaffold placeholder (below threshold): id=%d depth=%d",
                            placeholder.id, depth + 1
                        )

                    if best_below:
                        # Continue routing deeper
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
                for child, child_sim in children_with_centroids:
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

        step_type = self._infer_step_type(step_text)
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
                    is_root, is_semantic_umbrella, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, centroid_bucket, embedding_sum_packed, 1, step_type, step_text,
                 dsl_script, dsl_type, clarifying_json, params_json, actual_depth,
                 is_root_flag, is_umbrella, now),
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
                # Mark parent as umbrella (it now has children)
                conn.execute(
                    "UPDATE step_signatures SET is_semantic_umbrella = 1 WHERE id = ?",
                    (actual_parent_id,),
                )
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
        """Get a signature by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM step_signatures WHERE id = ?",
                (signature_id,)
            ).fetchone()
            if row:
                return self._row_to_signature(row)
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
        # Skips: centroid_bucket, embedding_sum, clarifying_questions, examples, is_archived, last_rewrite_at
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT id, signature_id, centroid, embedding_count, step_type,
                       description, param_descriptions, dsl_script, dsl_type,
                       uses, successes, is_semantic_umbrella, is_root, depth,
                       created_at, last_used_at
                FROM step_signatures
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
            self._centroid_matrix = np.array([], dtype=np.float32).reshape(0, 768)
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
            "SELECT embedding_sum, embedding_count, centroid, centroid_bucket FROM step_signatures WHERE id = ?",
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

        # Update running sum and count
        new_sum = current_sum + new_embedding
        new_count = current_count + 1

        # Compute new centroid
        new_centroid = new_sum / new_count

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
        if update_last_used:
            now = datetime.now(timezone.utc).isoformat()
            if bucket_changed:
                try:
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, centroid_bucket = ?, last_used_at = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, new_bucket, now, signature_id),
                    )
                except sqlite3.IntegrityError:
                    # New bucket collides with existing signature - keep old bucket
                    logger.debug("[db] Bucket collision on update for sig %d, keeping old bucket", signature_id)
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, last_used_at = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, now, signature_id),
                    )
            else:
                conn.execute(
                    """UPDATE step_signatures
                       SET embedding_sum = ?, embedding_count = ?, centroid = ?, last_used_at = ?
                       WHERE id = ?""",
                    (new_sum_packed, new_count, new_centroid_packed, now, signature_id),
                )
        else:
            if bucket_changed:
                try:
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, centroid_bucket = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, new_bucket, signature_id),
                    )
                except sqlite3.IntegrityError:
                    # New bucket collides with existing signature - keep old bucket
                    logger.debug("[db] Bucket collision on update for sig %d, keeping old bucket", signature_id)
                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?
                           WHERE id = ?""",
                        (new_sum_packed, new_count, new_centroid_packed, signature_id),
                    )
            else:
                conn.execute(
                    """UPDATE step_signatures
                       SET embedding_sum = ?, embedding_count = ?, centroid = ?
                       WHERE id = ?""",
                    (new_sum_packed, new_count, new_centroid_packed, signature_id),
                )

        if bucket_changed:
            logger.debug(
                "[db] Centroid bucket changed for sig %d: %s -> %s",
                signature_id, old_bucket, new_bucket
            )

        # Invalidate caches since centroid changed
        invalidate_centroid_cache(signature_id)
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

                # Propagate centroid change up to parent umbrellas
                if propagate_to_parents:
                    self.propagate_centroid_to_parents(conn, signature_id)

                conn.commit()
                logger.debug(
                    "[db] Updated centroid for sig %d: count=%d",
                    signature_id, new_count
                )
            except Exception:
                conn.rollback()
                raise

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
                        conn.execute(
                            """UPDATE step_signatures
                               SET is_semantic_umbrella = 1,
                                   dsl_type = 'router'
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

    def _infer_step_type(self, step_text: str) -> str:
        """Infer a step type from step text.

        Uses keyword matching to categorize common math operations.
        """
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
                       uses, successes, is_semantic_umbrella, is_root, depth,
                       created_at, last_used_at
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

        Returns a mix of:
        1. Cluster hints (umbrellas with children) - shows operation categories
        2. Leaf hints (specific operations) - shows concrete patterns

        This gives the planner a hierarchical view of available operations.

        Args:
            limit: Maximum number of top-level hints to return
            problem_embedding: Optional embedding to filter by semantic similarity
            min_similarity: Minimum cosine similarity to include hint (default 0.3)

        Returns:
            List of SignatureHint objects (some with children for clusters)
        """
        from mycelium.planner import SignatureHint

        hints = []
        seen_ids = set()

        with self._connection() as conn:
            # First, get level-1 clusters (umbrellas that are children of root)
            root = self.get_root()
            if root is not None:
                # Get root's children (level-1 clusters)
                cursor = conn.execute(
                    """SELECT s.* FROM signature_relationships r
                       JOIN step_signatures s ON r.child_id = s.id
                       WHERE r.parent_id = ?
                       ORDER BY s.successes DESC, s.uses DESC
                       LIMIT ?""",
                    (root.id, limit)
                )
                level1_sigs = [self._row_to_signature(row) for row in cursor.fetchall()]

                # Filter by embedding similarity if provided
                if problem_embedding is not None:
                    scored = []
                    for sig in level1_sigs:
                        # Capture centroid once to avoid TOCTOU race condition
                        centroid = sig.centroid
                        if centroid is not None:
                            sim = cosine_similarity(problem_embedding, centroid)
                            if sim >= min_similarity:
                                scored.append((sig, sim))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    level1_sigs = [sig for sig, _ in scored]

                # Build cluster hints
                for sig in level1_sigs:
                    seen_ids.add(sig.id)

                    # Get children of this cluster
                    child_hints = []
                    if sig.is_semantic_umbrella:
                        child_cursor = conn.execute(
                            """SELECT s.* FROM signature_relationships r
                               JOIN step_signatures s ON r.child_id = s.id
                               WHERE r.parent_id = ?
                               ORDER BY s.successes DESC
                               LIMIT 5""",
                            (sig.id,)
                        )
                        for child_row in child_cursor.fetchall():
                            child_sig = self._row_to_signature(child_row)
                            seen_ids.add(child_sig.id)
                            child_hints.append(SignatureHint(
                                step_type=child_sig.step_type,
                                description=child_sig.description,
                                param_names=self._extract_param_names(child_sig),
                                param_descriptions=child_sig.param_descriptions or {},
                                clarifying_questions=child_sig.clarifying_questions or [],
                                is_cluster=False,
                                children=[],
                            ))

                    hint = SignatureHint(
                        step_type=sig.step_type,
                        description=sig.description,
                        param_names=self._extract_param_names(sig),
                        param_descriptions=sig.param_descriptions or {},
                        clarifying_questions=sig.clarifying_questions or [],
                        is_cluster=sig.is_semantic_umbrella and len(child_hints) > 0,
                        children=child_hints,
                    )
                    hints.append(hint)

            # Fill remaining slots with high-quality leaf signatures not already included
            remaining = limit - len(hints)
            if remaining > 0:
                placeholders = ",".join("?" * len(seen_ids)) if seen_ids else "0"
                cursor = conn.execute(
                    f"""SELECT * FROM step_signatures
                       WHERE id NOT IN ({placeholders})
                       AND (clarifying_questions IS NOT NULL AND clarifying_questions != '[]'
                            OR param_descriptions IS NOT NULL AND param_descriptions != '{{}}')
                       AND is_semantic_umbrella = 0
                       ORDER BY successes DESC, uses DESC
                       LIMIT ?""",
                    list(seen_ids) + [remaining * 2]
                )
                leaf_sigs = [self._row_to_signature(row) for row in cursor.fetchall()]

                # Filter by embedding if provided
                if problem_embedding is not None:
                    scored = []
                    for sig in leaf_sigs:
                        # Capture centroid once to avoid TOCTOU race condition
                        centroid = sig.centroid
                        if centroid is not None:
                            sim = cosine_similarity(problem_embedding, centroid)
                            if sim >= min_similarity:
                                scored.append((sig, sim))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    leaf_sigs = [sig for sig, _ in scored[:remaining]]
                else:
                    leaf_sigs = leaf_sigs[:remaining]

                for sig in leaf_sigs:
                    hints.append(SignatureHint(
                        step_type=sig.step_type,
                        description=sig.description,
                        param_names=self._extract_param_names(sig),
                        param_descriptions=sig.param_descriptions or {},
                        clarifying_questions=sig.clarifying_questions or [],
                        is_cluster=False,
                        children=[],
                    ))

        logger.debug(
            "[db] Retrieved %d hierarchical hints (%d clusters)",
            len(hints), sum(1 for h in hints if h.is_cluster)
        )
        return hints

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
        self, parent_id: int, for_routing: bool = False
    ) -> list[tuple[StepSignature, str]]:
        """Get child signatures for an umbrella parent.

        Args:
            parent_id: ID of the parent signature
            for_routing: If True, use fast parsing (centroid only, skip JSON).
                        Per CLAUDE.md: "Umbrella routing should not require LLM call"

        Returns:
            List of (child_signature, condition) tuples, ordered by routing_order
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT s.*, r.condition
                   FROM signature_relationships r
                   JOIN step_signatures s ON r.child_id = s.id
                   WHERE r.parent_id = ?
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
            # Mark parent as umbrella (keep DSL as fallback if routing fails)
            conn.execute(
                """UPDATE step_signatures
                   SET is_semantic_umbrella = 1, dsl_type = 'router'
                   WHERE id = ?""",
                (parent_id,),
            )
            # Set child's depth = parent_depth + 1
            child_depth = parent_depth + 1
            conn.execute(
                "UPDATE step_signatures SET depth = ? WHERE id = ?",
                (child_depth, child_id),
            )
            # Invalidate parent's centroid cache (new child affects routing)
            invalidate_centroid_cache(parent_id)
            self.invalidate_centroid_matrix()
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

        Args:
            signature_id: ID of the signature to promote

        Returns:
            True if updated, False if signature not found
        """
        with self._connection() as conn:
            # Clear DSL and set type to router - umbrellas don't execute, they route
            cursor = conn.execute(
                """UPDATE step_signatures
                   SET is_semantic_umbrella = 1,
                       dsl_type = 'router'
                   WHERE id = ?""",
                (signature_id,),
            )
            if cursor.rowcount > 0:
                logger.info("[db] Promoted signature %d to umbrella (DSL kept as fallback)", signature_id)
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
