"""StepSignatureDB V2: SQLite-backed database with Natural Language Interface.

Lazy NL approach:
- New signatures start with just step_type + description (= step text)
- clarifying_questions, param_descriptions, dsl_script are empty initially
- These get filled in later as the system learns
"""

import json
import logging
import random
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Literal

import numpy as np

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
)

# Import from focused modules (scoring and DSL templates)
from mycelium.step_signatures.scoring import (
    compute_routing_score,
    compute_ucb1_score,
    normalize_step_text,
    increment_total_problems,
)
from mycelium.step_signatures.dsl_templates import infer_dsl_for_signature

from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import (
    cosine_similarity,
    batch_cosine_similarity,
    pack_embedding,
    unpack_embedding,
    get_cached_centroid,
    invalidate_centroid_cache,
)

logger = logging.getLogger(__name__)


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
            self._direct_conn.row_factory = sqlite3.Row
            self._direct_conn.execute("PRAGMA journal_mode = WAL")
            self._direct_conn.execute("PRAGMA busy_timeout = 30000")
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

        self._init_schema()

    @property
    def db_path(self) -> str:
        """Get the database path."""
        return self._db_path

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
        All problems route through the root first.

        Returns:
            The root signature, or None if database is empty
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM step_signatures WHERE is_root = 1 LIMIT 1"
            ).fetchone()
            if row:
                return self._row_to_signature(row)
            return None

    def has_root(self) -> bool:
        """Check if a root signature exists."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM step_signatures WHERE is_root = 1 LIMIT 1"
            ).fetchone()
            return row is not None

    def propagate_centroid_to_parents(
        self,
        conn,
        child_id: int,
        visited: set[int] = None,
    ):
        """Propagate centroid changes up to parent umbrella (tree structure).

        When a child's centroid is updated, this recomputes the parent's centroid
        as the average of its children's centroids, recursively up the tree.

        Tree structure: each child has exactly one parent.

        Args:
            conn: Database connection (within transaction)
            child_id: ID of the signature whose centroid was updated
            visited: Set of already-visited parent IDs (prevents infinite loops)
        """
        if visited is None:
            visited = set()

        # Get parent of this child (single parent in tree structure)
        cursor = conn.execute(
            "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
            (child_id,)
        )
        row = cursor.fetchone()
        if not row:
            return  # No parent (root node)

        parent_id = row[0]
        if parent_id in visited:
            return
        visited.add(parent_id)

        # Get all children of this parent
        cursor = conn.execute(
            """SELECT s.centroid, s.embedding_count
               FROM signature_relationships r
               JOIN step_signatures s ON r.child_id = s.id
               WHERE r.parent_id = ?""",
            (parent_id,)
        )
        children_data = cursor.fetchall()

        if not children_data:
            return

        # Compute parent centroid as weighted average of children
        # Weight by embedding_count (more examples = more weight)
        total_weight = 0
        centroid_sum = None

        for child_row in children_data:
            child_centroid = unpack_embedding(child_row[0])
            child_count = child_row[1] or 1
            if child_centroid is None:
                continue

            if centroid_sum is None:
                centroid_sum = child_centroid * child_count
            else:
                centroid_sum = centroid_sum + (child_centroid * child_count)
            total_weight += child_count

        if centroid_sum is not None and total_weight > 0:
            new_centroid = centroid_sum / total_weight
            try:
                conn.execute(
                    """UPDATE step_signatures
                       SET centroid = ?, embedding_sum = ?, embedding_count = ?
                       WHERE id = ?""",
                    (pack_embedding(new_centroid), pack_embedding(centroid_sum),
                     total_weight, parent_id),
                )
                # Invalidate cache since centroid changed
                invalidate_centroid_cache(parent_id)
                logger.debug(
                    "[db] Propagated centroid to parent %d (weight=%d)",
                    parent_id, total_weight
                )
            except sqlite3.IntegrityError:
                # Centroid collision with another signature - skip this update
                logger.debug(
                    "[db] Skipped centroid propagation to parent %d (collision)",
                    parent_id
                )
                return

            # Recurse to grandparent
            self.propagate_centroid_to_parents(conn, parent_id, visited)

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

            # Get children of current umbrella
            children = self.get_children(current.id)
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
        """
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            try:
                return self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, origin_depth,
                    extracted_values=extracted_values, dsl_hint=dsl_hint, parent_id=parent_id
                )
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter to avoid thundering herd
                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, delay * 0.5)
                    time.sleep(delay + jitter)
                    logger.debug(
                        "[db] Retry %d/%d after OperationalError: %s (delay=%.3fs)",
                        attempt + 1, max_retries, str(e)[:50], delay + jitter
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

        Returns:
            (best_match, parent_for_new, best_similarity)
            - best_match: Leaf signature if found above threshold
            - parent_for_new: Umbrella where routing stopped (for creating new child)
            - best_similarity: Similarity of best_match
        """
        from mycelium.config import UMBRELLA_MAX_DEPTH

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
                sim = cosine_similarity(embedding, current.centroid) if current.centroid is not None else 0.0
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
                sim = cosine_similarity(embedding, current.centroid) if current.centroid is not None else 0.0
                return current, current, sim

            # MCTS UCB1 Selection: balance exploitation vs exploration
            # parent_uses = current node's uses (N in UCB1 formula)
            parent_uses = current.uses or 1

            best_child = None
            best_child_sim = 0.0
            best_child_score = 0.0

            for child in children:
                # Capture centroid once to avoid TOCTOU race condition
                centroid = child.centroid
                if centroid is None:
                    continue
                child_sim = cosine_similarity(embedding, centroid)
                if child_sim >= min_similarity:
                    # UCB1 score: exploit (sim * success_rate) + explore (bonus for under-visited)
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

            if best_child is None:
                # No child matches above threshold - current is where we'd add new child
                parent_for_new = current
                # Return best child below threshold as "best effort" (or None)
                best_below = None
                best_below_sim = 0.0
                best_below_score = 0.0
                for child in children:
                    # Capture centroid once to avoid TOCTOU race condition
                    centroid = child.centroid
                    if centroid is not None:
                        child_sim = cosine_similarity(embedding, centroid)
                        # Still use UCB1 for below-threshold exploration
                        score = compute_ucb1_score(
                            child_sim,
                            child.uses,
                            child.successes,
                            parent_uses,
                            child.last_used_at
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
        sim = cosine_similarity(embedding, current.centroid) if current.centroid is not None else 0.0
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
        now = datetime.utcnow().isoformat()

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
                   (signature_id, centroid, embedding_sum, embedding_count, step_type, description,
                    dsl_script, dsl_type, clarifying_questions, param_descriptions, depth,
                    is_root, is_semantic_umbrella, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, embedding_sum_packed, 1, step_type, step_text,
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
        except sqlite3.IntegrityError:
            # Centroid collision - find existing signature
            row = conn.execute(
                "SELECT * FROM step_signatures WHERE centroid = ?",
                (centroid_packed,)
            ).fetchone()
            if row:
                return self._row_to_signature(row)
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
        """Build or refresh the cached centroid matrix for fast similarity search."""
        if self._centroid_matrix is not None:
            return  # Already loaded

        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM step_signatures")
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
        else:
            self._centroid_matrix = np.array([], dtype=np.float32).reshape(0, 768)
            self._centroid_sig_ids = []
            self._centroid_rows = []

    def invalidate_centroid_matrix(self):
        """Invalidate cached centroid matrix (call when signatures change)."""
        self._centroid_matrix = None
        self._centroid_sig_ids = None
        self._centroid_rows = None

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
        for i in sorted_indices:
            sig = self._row_to_signature_fast(self._centroid_rows[i])
            results.append((sig, float(scores[i])))

        return results

    def count_signatures(self) -> int:
        """Get total number of signatures."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
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
            "SELECT embedding_sum, embedding_count, centroid FROM step_signatures WHERE id = ?",
            (signature_id,)
        ).fetchone()

        if not row:
            logger.warning("[db] Cannot update centroid: signature %d not found", signature_id)
            return None

        # Parse current sum (or initialize from fresh centroid)
        if row["embedding_sum"]:
            current_sum = unpack_embedding(row["embedding_sum"])
            current_count = row["embedding_count"] or 1
        else:
            # Initialize from fresh centroid if no sum yet (migration case)
            fresh_centroid = unpack_embedding(row["centroid"])
            current_sum = fresh_centroid.copy() if fresh_centroid is not None else new_embedding.copy()
            current_count = 1

        # Update running sum and count
        new_sum = current_sum + new_embedding
        new_count = current_count + 1

        # Compute new centroid
        new_centroid = new_sum / new_count

        # Pack and store
        new_sum_packed = pack_embedding(new_sum)
        new_centroid_packed = pack_embedding(new_centroid)

        if update_last_used:
            now = datetime.utcnow().isoformat()
            conn.execute(
                """UPDATE step_signatures
                   SET embedding_sum = ?, embedding_count = ?, centroid = ?, last_used_at = ?
                   WHERE id = ?""",
                (new_sum_packed, new_count, new_centroid_packed, now, signature_id),
            )
        else:
            conn.execute(
                """UPDATE step_signatures
                   SET embedding_sum = ?, embedding_count = ?, centroid = ?
                   WHERE id = ?""",
                (new_sum_packed, new_count, new_centroid_packed, signature_id),
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

        Returns:
            New uses count (for triggering DSL regeneration on mod 10)
        """
        now = datetime.utcnow().isoformat()

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
    ):
        """Update signature success counts based on problem outcome.

        Call this after grading a problem to propagate correctness back to
        all signatures that were used. Also propagates credit up to parent
        umbrella signatures with decay.

        Args:
            signature_ids: IDs of signatures used in the solved problem
            problem_correct: Whether the final answer was correct
            decay_factor: Credit decay per level (default from config)
        """
        # Increment global problem counter (for traffic-based decay)
        increment_total_problems(self.db_path)

        if decay_factor is None:
            decay_factor = PARENT_CREDIT_DECAY
        if not signature_ids:
            return

        now = datetime.now().isoformat()

        with self._connection() as conn:
            if problem_correct:
                # Increment success count and update last_used_at for all signatures used
                # (last_used_at only updates on success to properly penalize stale failures)
                placeholders = ",".join("?" * len(signature_ids))
                conn.execute(
                    f"""UPDATE step_signatures
                       SET successes = successes + 1, last_used_at = ?
                       WHERE id IN ({placeholders})""",
                    [now] + signature_ids,
                )
                logger.debug(
                    "[db] Problem correct: incremented successes for %d signatures",
                    len(signature_ids)
                )

                # Propagate credit to parent umbrellas with decay
                # Track credits per parent to avoid double-counting
                parent_credits: dict[int, float] = {}

                for sig_id in signature_ids:
                    self._collect_parent_credits(
                        conn, sig_id, decay_factor, 1, parent_credits
                    )

                # Apply accumulated credits
                for parent_id, credit in parent_credits.items():
                    if credit >= PARENT_CREDIT_MIN:
                        conn.execute(
                            """UPDATE step_signatures
                               SET successes = successes + ?
                               WHERE id = ? AND is_semantic_umbrella = 1""",
                            (credit, parent_id),
                        )

                if parent_credits:
                    logger.debug(
                        "[db] Propagated credit to %d parent umbrellas",
                        len(parent_credits)
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
        """Recursively collect credits for parent umbrellas (tree structure).

        Args:
            conn: Database connection
            signature_id: Current signature ID
            decay_factor: Credit multiplier per level (e.g., 0.7^depth)
            current_depth: Current depth in traversal
            credits: Dict accumulating {parent_id: total_credit}
            max_depth: Max depth to traverse (default from config)
        """
        if max_depth is None:
            max_depth = PARENT_CREDIT_MAX_DEPTH
        if current_depth > max_depth:
            return

        # Get parent umbrella (single parent in tree structure)
        row = conn.execute(
            """SELECT r.parent_id
               FROM signature_relationships r
               JOIN step_signatures s ON r.parent_id = s.id
               WHERE r.child_id = ? AND s.is_semantic_umbrella = 1""",
            (signature_id,)
        ).fetchone()

        if not row:
            return

        parent_id = row[0]
        credit = decay_factor ** current_depth
        credits[parent_id] = credit

        # Recurse to grandparent
        self._collect_parent_credits(
            conn, parent_id, decay_factor, current_depth + 1, credits, max_depth
        )

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
        """Get all signatures in the database."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM step_signatures")
            return [self._row_to_signature(row) for row in cursor.fetchall()]

    def get_signatures_with_dsl(self) -> list[StepSignature]:
        """Get all signatures that have DSL scripts."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM step_signatures WHERE dsl_script IS NOT NULL AND dsl_script != ''"
            )
            return [self._row_to_signature(row) for row in cursor.fetchall()]

    def get_signatures_needing_nl(self) -> list[StepSignature]:
        """Get signatures that need NL interface filled in.

        These are signatures with empty clarifying_questions that have been
        used successfully multiple times.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM step_signatures
                   WHERE (clarifying_questions IS NULL OR clarifying_questions = '[]')
                   AND successes >= 3
                   ORDER BY successes DESC"""
            )
            return [self._row_to_signature(row) for row in cursor.fetchall()]

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
        except (json.JSONDecodeError, TypeError):
            return []

    # =========================================================================
    # Umbrella Routing (DAG of DAGs)
    # =========================================================================

    def get_children(self, parent_id: int) -> list[tuple[StepSignature, str]]:
        """Get child signatures for an umbrella parent.

        Args:
            parent_id: ID of the parent signature

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
            for row in cursor.fetchall():
                row_dict = dict(row)
                condition = row_dict.pop("condition")
                sig = self._row_to_signature(row_dict)
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

        from datetime import datetime
        now = datetime.utcnow().isoformat()

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
                logger.info("[db] Removed child relationship: parent=%d → child=%d", parent_id, child_id)
                return True
            return False

    def get_umbrella_signatures(self) -> list[StepSignature]:
        """Get all signatures that are semantic umbrellas."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM step_signatures WHERE is_semantic_umbrella = 1"
            )
            return [self._row_to_signature(dict(row)) for row in cursor.fetchall()]

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
    # Duplicate Management
    # =========================================================================

    def merge_duplicates(self, threshold: float = 0.85, dry_run: bool = False) -> dict:
        """Merge duplicate signatures above similarity threshold.

        Keeps the signature with more uses, merges stats, deletes the other.

        Args:
            threshold: Minimum cosine similarity to consider as duplicate
            dry_run: If True, just report what would be merged without doing it

        Returns:
            Dict with merge statistics
        """
        sigs = self.get_all_signatures()

        # Build list of (id, centroid) for comparison
        sig_data = []
        for s in sigs:
            # Capture centroid once to avoid TOCTOU race condition
            centroid = s.centroid
            if centroid is not None:
                sig_data.append((s.id, s.uses, s.successes, s.description, np.array(centroid)))

        # Find pairs to merge
        to_merge = []  # (keep_id, delete_id, similarity)
        merged_ids = set()

        for i, (id1, uses1, succ1, desc1, c1) in enumerate(sig_data):
            if id1 in merged_ids:
                continue
            for j, (id2, uses2, succ2, desc2, c2) in enumerate(sig_data[i+1:], i+1):
                if id2 in merged_ids:
                    continue
                sim = cosine_similarity(c1, c2)
                if sim >= threshold:
                    # Keep the one with more uses
                    if uses1 >= uses2:
                        to_merge.append((id1, id2, sim, desc1, desc2))
                        merged_ids.add(id2)
                    else:
                        to_merge.append((id2, id1, sim, desc2, desc1))
                        merged_ids.add(id1)
                        break  # id1 is being deleted, stop comparing it

        if dry_run:
            return {
                "duplicates_found": len(to_merge),
                "would_merge": [(keep, delete, f"{sim:.3f}", d1[:40], d2[:40])
                               for keep, delete, sim, d1, d2 in to_merge],
            }

        # Perform merges
        merged_count = 0
        with self._connection() as conn:
            for keep_id, delete_id, sim, _, _ in to_merge:
                # Get stats from both
                keep_row = conn.execute(
                    "SELECT uses, successes FROM step_signatures WHERE id = ?", (keep_id,)
                ).fetchone()
                delete_row = conn.execute(
                    "SELECT uses, successes FROM step_signatures WHERE id = ?", (delete_id,)
                ).fetchone()

                if keep_row and delete_row:
                    # Merge stats
                    new_uses = keep_row[0] + delete_row[0]
                    new_successes = keep_row[1] + delete_row[1]

                    conn.execute(
                        "UPDATE step_signatures SET uses = ?, successes = ? WHERE id = ?",
                        (new_uses, new_successes, keep_id),
                    )

                    # Update foreign key references to point to kept signature
                    conn.execute(
                        "UPDATE step_examples SET signature_id = ? WHERE signature_id = ?",
                        (keep_id, delete_id),
                    )
                    conn.execute(
                        "UPDATE step_usage_log SET signature_id = ? WHERE signature_id = ?",
                        (keep_id, delete_id),
                    )

                    # Delete the duplicate
                    conn.execute("DELETE FROM step_signatures WHERE id = ?", (delete_id,))
                    merged_count += 1
                    logger.info(f"[db] Merged sig {delete_id} into {keep_id} (sim={sim:.3f})")

        return {
            "duplicates_found": len(to_merge),
            "merged": merged_count,
            "remaining_signatures": len(sigs) - merged_count,
        }
