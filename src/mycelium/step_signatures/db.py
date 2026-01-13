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
    normalize_step_text,
    increment_total_problems,
)
from mycelium.step_signatures.dsl_templates import (
    DSL_TEMPLATES,
    DSL_INFERENCE_PATTERNS,
    infer_dsl_for_signature,
)

from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import cosine_similarity, pack_embedding, unpack_embedding

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
            origin_depth: Decomposition depth at which this step was created

        Returns:
            Tuple of (signature, is_new) where is_new=True if newly created
        """
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            try:
                return self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, origin_depth
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

    def _find_or_create_atomic(
        self,
        step_text: str,
        embedding: np.ndarray,
        min_similarity: float,
        parent_problem: str,
        origin_depth: int = 0,
    ) -> tuple[StepSignature, bool]:
        """Internal atomic find-or-create with transaction locking."""
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Find best matching signature
                cursor = conn.execute("SELECT * FROM step_signatures")
                rows = cursor.fetchall()

                best_match = None
                best_score = 0.0

                for row in rows:
                    centroid = unpack_embedding(row["centroid"])
                    if centroid is None:
                        continue

                    cosine_sim = cosine_similarity(embedding, centroid)
                    uses = row["uses"] or 0
                    successes = row["successes"] or 0
                    last_used_at = row["last_used_at"]
                    score = compute_routing_score(cosine_sim, uses, successes, last_used_at)

                    if cosine_sim >= min_similarity and score > best_score:
                        best_match = self._row_to_signature(row)
                        best_score = score

                if best_match:
                    now = datetime.utcnow().isoformat()

                    # Update centroid with new embedding (running average)
                    # Do this inline to stay within the transaction
                    # IMPORTANT: Fetch centroid too to avoid stale fallback race condition
                    row = conn.execute(
                        "SELECT embedding_sum, embedding_count, centroid FROM step_signatures WHERE id = ?",
                        (best_match.id,)
                    ).fetchone()

                    if row and row["embedding_sum"]:
                        current_sum = unpack_embedding(row["embedding_sum"])
                        current_count = row["embedding_count"] or 1
                    else:
                        # Initialize from fresh centroid if no sum yet (migration case)
                        # Use freshly-read centroid, not stale best_match.centroid
                        fresh_centroid = unpack_embedding(row["centroid"]) if row else None
                        current_sum = fresh_centroid.copy() if fresh_centroid is not None else embedding.copy()
                        current_count = 1

                    new_sum = current_sum + embedding
                    new_count = current_count + 1
                    new_centroid = new_sum / new_count

                    conn.execute(
                        """UPDATE step_signatures
                           SET embedding_sum = ?, embedding_count = ?, centroid = ?, last_used_at = ?
                           WHERE id = ?""",
                        (pack_embedding(new_sum), new_count, pack_embedding(new_centroid), now, best_match.id),
                    )
                    conn.commit()
                    logger.debug(
                        "[db] Matched signature: step='%s' sig='%s' score=%.3f count=%d",
                        step_text[:40], best_match.step_type, best_score, new_count
                    )
                    return best_match, False

                # Create new signature (Lazy NL: empty clarifying_questions, etc.)
                sig = self._create_signature_atomic(conn, step_text, embedding, parent_problem, origin_depth)
                conn.commit()
                logger.info(
                    "[db] Created new signature: step='%s' type='%s'",
                    step_text[:40], sig.step_type
                )
                return sig, True

            except Exception:
                conn.rollback()
                raise

    def _create_signature_atomic(
        self,
        conn,
        step_text: str,
        embedding: np.ndarray,
        parent_problem: str = "",
        origin_depth: int = 0,
    ) -> StepSignature:
        """Create a new signature within an existing transaction.

        Auto-assigns DSL based on step_type and description.
        """
        sig_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        step_type = self._infer_step_type(step_text)
        centroid_packed = pack_embedding(embedding)

        # Auto-assign DSL based on step_type and description
        dsl_script, dsl_type = infer_dsl_for_signature(step_type, step_text)

        # Initialize embedding_sum = embedding, embedding_count = 1
        embedding_sum_packed = centroid_packed  # Same as centroid initially

        try:
            cursor = conn.execute(
                """INSERT INTO step_signatures
                   (signature_id, centroid, embedding_sum, embedding_count, step_type, description, dsl_script, dsl_type, depth, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, embedding_sum_packed, 1, step_type, step_text, dsl_script, dsl_type, origin_depth, now),
            )
            row_id = cursor.lastrowid
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

        logger.debug("[db] Auto-assigned DSL type=%s for step_type=%s", dsl_type, step_type)

        return StepSignature(
            id=row_id,
            signature_id=sig_id,
            centroid=embedding,
            step_type=step_type,
            description=step_text,
            clarifying_questions=[],
            param_descriptions={},
            dsl_script=dsl_script,
            dsl_type=dsl_type,
            examples=[],
            uses=0,
            successes=0,
            depth=origin_depth,
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
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM step_signatures")
            rows = cursor.fetchall()

        results = []
        for row in rows:
            centroid = unpack_embedding(row["centroid"])
            if centroid is None:
                continue

            score = cosine_similarity(embedding, centroid)
            if score >= threshold:
                sig = self._row_to_signature(row)
                results.append((sig, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def count_signatures(self) -> int:
        """Get total number of signatures."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
            return row[0]

    # =========================================================================
    # Centroid Management (Running Average Embeddings)
    # =========================================================================

    def update_centroid(
        self,
        signature_id: int,
        new_embedding: np.ndarray,
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
        """
        with self._connection() as conn:
            # Get current sum and count
            row = conn.execute(
                "SELECT embedding_sum, embedding_count FROM step_signatures WHERE id = ?",
                (signature_id,)
            ).fetchone()

            if not row:
                logger.warning("[db] Cannot update centroid: signature %d not found", signature_id)
                return

            # Parse current sum (or initialize from None)
            current_sum = None
            if row["embedding_sum"]:
                current_sum = unpack_embedding(row["embedding_sum"])

            current_count = row["embedding_count"] or 1

            # If no sum exists, initialize from new embedding
            if current_sum is None:
                current_sum = new_embedding.copy()
                current_count = 0  # Will be incremented to 1

            # Update running sum and count
            new_sum = current_sum + new_embedding
            new_count = current_count + 1

            # Compute new centroid
            new_centroid = new_sum / new_count

            # Pack and store
            new_sum_packed = pack_embedding(new_sum)
            new_centroid_packed = pack_embedding(new_centroid)

            conn.execute(
                """UPDATE step_signatures
                   SET embedding_sum = ?, embedding_count = ?, centroid = ?
                   WHERE id = ?""",
                (new_sum_packed, new_count, new_centroid_packed, signature_id),
            )

            logger.debug(
                "[db] Updated centroid for sig %d: count=%d",
                signature_id, new_count
            )

    # =========================================================================
    # Usage Recording
    # =========================================================================

    def record_usage(
        self,
        signature_id: int,
        step_text: str,
        success: bool,
        was_injected: bool = False,
        params_extracted: dict = None,
    ) -> int:
        """Record usage of a signature (step-level, not problem-level).

        Note: This tracks whether a step returned a result, not whether the
        final problem answer was correct. Use update_problem_outcome() after
        grading to track actual correctness.

        Args:
            signature_id: ID of the signature used
            step_text: The step text that was executed
            success: Whether the step returned a result
            was_injected: Whether DSL was injected
            params_extracted: Parameters that were extracted (for learning)

        Returns:
            New uses count (for triggering DSL regeneration on mod 10)
        """
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Insert usage log (success here = step returned result, not problem correct)
            conn.execute(
                """INSERT INTO step_usage_log
                   (signature_id, step_text, success, was_injected, params_extracted, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    signature_id,
                    step_text,
                    1 if success else 0,
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
            if AUTO_DEMOTE_ENABLED and not is_umbrella and dsl_type not in AUTO_DEMOTE_EXCLUDED_TYPES:
                sig_count = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()[0]
                min_uses = min(
                    AUTO_DEMOTE_MIN_USES_FLOOR + sig_count // AUTO_DEMOTE_RAMP_DIVISOR,
                    AUTO_DEMOTE_MIN_USES_CAP
                )

                if uses >= min_uses:
                    success_rate = successes / uses if uses > 0 else 0
                    if success_rate < AUTO_DEMOTE_MAX_SUCCESS_RATE:
                        conn.execute(
                            """UPDATE step_signatures
                               SET is_semantic_umbrella = 1
                               WHERE id = ?""",
                            (signature_id,),
                        )
                        logger.info(
                            "[db] Auto-demoted sig %d to umbrella (%.0f%% after %d uses, min=%d, %d sigs)",
                            signature_id, success_rate * 100, uses, min_uses, sig_count
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
        """Recursively collect credits for parent umbrellas.

        Args:
            conn: Database connection
            signature_id: Current signature ID
            decay_factor: Credit multiplier per level
            current_depth: Current depth in traversal
            credits: Dict accumulating {parent_id: total_credit}
            max_depth: Max depth to traverse (default from config)
        """
        if max_depth is None:
            max_depth = PARENT_CREDIT_MAX_DEPTH
        if current_depth > max_depth:
            return

        # Get parent umbrella signatures
        cursor = conn.execute(
            """SELECT r.parent_id
               FROM signature_relationships r
               JOIN step_signatures s ON r.parent_id = s.id
               WHERE r.child_id = ? AND s.is_semantic_umbrella = 1""",
            (signature_id,)
        )

        for row in cursor.fetchall():
            parent_id = row[0]
            credit = decay_factor ** current_depth

            # Accumulate (take max if parent reached via multiple paths)
            if parent_id not in credits or credits[parent_id] < credit:
                credits[parent_id] = credit

            # Recurse to grandparents
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
        """Get top signatures as hints for the decomposer.

        Returns signatures with their NL interface info so the decomposer
        knows what operations are available and what parameters they need.

        Args:
            limit: Maximum number of hints to return
            problem_embedding: Optional embedding of the problem to filter hints
                              by semantic similarity (quick win for relevance)
            min_similarity: Minimum cosine similarity to include hint (default 0.3)

        Returns:
            List of SignatureHint objects
        """
        from mycelium.planner import SignatureHint
        from mycelium.step_signatures.utils import cosine_similarity, unpack_embedding

        # Get signatures with NL interface populated
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM step_signatures
                   WHERE dsl_type != 'decompose'
                   AND (clarifying_questions IS NOT NULL AND clarifying_questions != '[]'
                        OR param_descriptions IS NOT NULL AND param_descriptions != '{}')
                   ORDER BY successes DESC, uses DESC
                   LIMIT ?""",
                (limit * 3,)  # Fetch more, filter by similarity
            )
            signatures = [self._row_to_signature(row) for row in cursor.fetchall()]

        # If problem embedding provided, filter by similarity
        if problem_embedding is not None:
            scored = []
            for sig in signatures:
                if sig.centroid is not None:
                    sim = cosine_similarity(problem_embedding, sig.centroid)
                    if sim >= min_similarity:
                        scored.append((sig, sim))
            # Sort by similarity, take top N
            scored.sort(key=lambda x: x[1], reverse=True)
            signatures = [sig for sig, _ in scored[:limit]]
            logger.debug(
                "[db] Filtered hints by embedding: %d → %d (min_sim=%.2f)",
                len(scored) + (limit * 3 - len(scored)), len(signatures), min_similarity
            )

        hints = []
        for sig in signatures[:limit]:
            # Get param names from DSL spec if available
            param_names = []
            if sig.dsl_script:
                try:
                    import json
                    dsl_data = json.loads(sig.dsl_script) if sig.dsl_script.startswith('{') else {}
                    param_names = dsl_data.get('params', [])
                except (json.JSONDecodeError, TypeError):
                    pass

            hint = SignatureHint(
                step_type=sig.step_type,
                description=sig.description,
                param_names=param_names,
                param_descriptions=sig.param_descriptions or {},
                clarifying_questions=sig.clarifying_questions or [],
            )
            hints.append(hint)

        logger.debug("[db] Retrieved %d signature hints with NL interface for decomposer", len(hints))
        return hints

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

    def get_parents(self, child_id: int) -> list[StepSignature]:
        """Get parent signatures for a child (DAG traversal).

        Args:
            child_id: ID of the child signature

        Returns:
            List of parent signatures
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT s.*
                   FROM signature_relationships r
                   JOIN step_signatures s ON r.parent_id = s.id
                   WHERE r.child_id = ?""",
                (child_id,)
            )
            return [self._row_to_signature(dict(row)) for row in cursor.fetchall()]

    def add_child(
        self,
        parent_id: int,
        child_id: int,
        condition: str,
        routing_order: int = 0,
    ) -> bool:
        """Add a parent-child relationship.

        Args:
            parent_id: ID of the parent signature
            child_id: ID of the child signature
            condition: Routing condition (e.g., "counting outcomes")
            routing_order: Priority (lower = higher priority)

        Returns:
            True if relationship was created, False if already exists
        """
        # Prevent self-references
        if parent_id == child_id:
            logger.warning("[db] Rejecting self-reference: parent_id=%d == child_id=%d", parent_id, child_id)
            return False

        from datetime import datetime
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            # Cycle prevention: check if parent is already a descendant of child
            # (adding parent -> child would create child -> ... -> parent -> child cycle)
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

            try:
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
                # Mark parent as umbrella
                conn.execute(
                    "UPDATE step_signatures SET is_semantic_umbrella = 1 WHERE id = ?",
                    (parent_id,),
                )
                # Set child's depth = parent_depth + 1 (only if deeper than current)
                child_depth = parent_depth + 1
                conn.execute(
                    "UPDATE step_signatures SET depth = MAX(depth, ?) WHERE id = ?",
                    (child_depth, child_id),
                )
                logger.info(
                    "[db] Added child relationship: parent=%d (depth=%d) → child=%d (depth=%d) (condition='%s')",
                    parent_id, parent_depth, child_id, child_depth, condition[:30]
                )
                return True
            except Exception as e:
                if "UNIQUE constraint" in str(e):
                    logger.debug("[db] Relationship already exists: parent=%d → child=%d", parent_id, child_id)
                    return False
                raise

    def promote_to_umbrella(self, signature_id: int) -> bool:
        """Mark a signature as a semantic umbrella.

        Args:
            signature_id: ID of the signature to promote

        Returns:
            True if updated, False if signature not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE step_signatures SET is_semantic_umbrella = 1 WHERE id = ?",
                (signature_id,),
            )
            if cursor.rowcount > 0:
                logger.info("[db] Promoted signature %d to umbrella", signature_id)
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

            centroid = unpack_embedding(row["centroid"])
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
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM signature_relationships WHERE parent_id = ?",
                    (parent_id,),
                ).fetchone()[0]
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
            # Get counts before deletion
            sig_count = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()[0]
            ex_count = conn.execute("SELECT COUNT(*) FROM step_examples").fetchone()[0]
            log_count = conn.execute("SELECT COUNT(*) FROM step_usage_log").fetchone()[0]
            rel_count = conn.execute("SELECT COUNT(*) FROM signature_relationships").fetchone()[0] if self._table_exists(conn, "signature_relationships") else 0

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
            if s.centroid is not None:
                sig_data.append((s.id, s.uses, s.successes, s.description, np.array(s.centroid)))

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
