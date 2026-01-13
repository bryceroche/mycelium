"""StepSignatureDB V2: SQLite-backed database with Natural Language Interface.

Lazy NL approach:
- New signatures start with just step_type + description (= step text)
- clarifying_questions, param_descriptions, dsl_script are empty initially
- These get filled in later as the system learns
"""

import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Literal

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DSL Templates for Auto-Assignment
# =============================================================================

DSL_TEMPLATES = {
    "compute_sum": {"type": "math", "script": "a + b", "params": ["a", "b"], "purpose": "Add two numbers"},
    "compute_product": {"type": "math", "script": "a * b", "params": ["a", "b"], "purpose": "Multiply two numbers"},
    "compute_difference": {"type": "math", "script": "a - b", "params": ["a", "b"], "purpose": "Subtract b from a"},
    "compute_quotient": {"type": "math", "script": "a / b", "params": ["a", "b"], "purpose": "Divide a by b"},
    "compute_power": {"type": "math", "script": "base ** exponent", "params": ["base", "exponent"], "purpose": "Raise base to power"},
    "compute_factorial": {"type": "math", "script": "factorial(n)", "params": ["n"], "purpose": "Calculate n!"},
    "compute_sqrt": {"type": "math", "script": "sqrt(x)", "params": ["x"], "purpose": "Square root"},
    "compute_modulo": {"type": "math", "script": "a % b", "params": ["a", "b"], "purpose": "Remainder"},
    "compute_gcd": {"type": "math", "script": "gcd(a, b)", "params": ["a", "b"], "purpose": "Greatest common divisor"},
    "compute_lcm": {"type": "math", "script": "lcm(a, b)", "params": ["a", "b"], "purpose": "Least common multiple"},
    "compute_area": {"type": "math", "script": "length * width", "params": ["length", "width"], "purpose": "Calculate area"},
    "compute_average": {"type": "math", "script": "(a + b) / 2", "params": ["a", "b"], "purpose": "Calculate average"},
    "compute_probability": {"type": "decompose", "script": "compute_probability", "params": ["favorable", "total"], "purpose": "Calculate probability"},
    "simplify_expression": {"type": "sympy", "script": "simplify(expr)", "params": ["expr"], "purpose": "Simplify expression"},
    "solve_equation": {"type": "sympy", "script": "solve(equation, x)", "params": ["equation"], "purpose": "Solve equation"},
    "factor_expression": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "purpose": "Factor expression"},
    "evaluate_expression": {"type": "math", "script": "eval(expr)", "params": ["expr"], "purpose": "Evaluate expression"},
    "compute_angle": {"type": "math", "script": "degrees", "params": ["degrees"], "purpose": "Angle calculation"},
    "count_combinations": {"type": "math", "script": "factorial(n) / (factorial(r) * factorial(n - r))", "params": ["n", "r"], "purpose": "n choose r"},
    "count_permutations": {"type": "math", "script": "factorial(n) / factorial(n - r)", "params": ["n", "r"], "purpose": "P(n,r)"},
}

# Patterns for inferring DSL from description
DSL_INFERENCE_PATTERNS = [
    (r"combine.*result|final.*answer|synthesize", {"type": "decompose", "script": "synthesize_results", "params": ["results"], "purpose": "Combine results"}),
    (r"coordinate|point.*\(|define.*point", {"type": "sympy", "script": "Point(x, y)", "params": ["x", "y"], "purpose": "Coordinate point"}),
    (r"substitut|plug.*in|replace.*with", {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "purpose": "Substitution"}),
    (r"find.*minimum|find.*maximum|minimize|maximize|min.*value|max.*value", {"type": "sympy", "script": "solve(diff(expr, x), x)", "params": ["expr"], "purpose": "Find min/max"}),
    (r"define.*constraint|constraint|given.*condition", {"type": "decompose", "script": "extract_constraints", "params": ["problem"], "purpose": "Extract constraints"}),
    (r"express.*in terms|write.*as|rewrite", {"type": "sympy", "script": "solve(eq, var)", "params": ["eq", "var"], "purpose": "Express in terms of"}),
    (r"find.*equation|equation of|derive.*equation", {"type": "sympy", "script": "Eq(lhs, rhs)", "params": ["lhs", "rhs"], "purpose": "Find equation"}),
    (r"identify|extract|determine.*value|find.*value", {"type": "decompose", "script": "extract_values", "params": ["text"], "purpose": "Extract values"}),
    (r"magnitude|absolute|modulus", {"type": "sympy", "script": "Abs(z)", "params": ["z"], "purpose": "Magnitude"}),
    (r"argument|angle.*of|arg\(", {"type": "sympy", "script": "arg(z)", "params": ["z"], "purpose": "Argument/angle"}),
    (r"range|interval|bounds|between", {"type": "decompose", "script": "find_range", "params": ["expr", "var"], "purpose": "Find range"}),
    (r"solve for|find.*n\b|find.*x\b", {"type": "sympy", "script": "solve(eq, var)", "params": ["eq", "var"], "purpose": "Solve for variable"}),
    (r"critical point|derivative.*zero", {"type": "sympy", "script": "solve(diff(f, x), x)", "params": ["f"], "purpose": "Critical points"}),
    (r"relationship|connection|relate", {"type": "decompose", "script": "find_relationship", "params": ["a", "b"], "purpose": "Find relationship"}),
]


def normalize_step_text(text: str) -> str:
    """Normalize step text for embedding by replacing specific numbers with placeholders.

    This helps match similar operations regardless of specific values:
    - "Calculate 15 factorial" → "Calculate N factorial"
    - "Raise 5 to power 3" → "Raise N to power N"
    """
    # Replace standalone numbers (not part of words) with N
    # Keep decimal points for now
    normalized = re.sub(r'\b\d+\.?\d*\b', 'N', text)
    return normalized


def infer_dsl_for_signature(step_type: str, description: str) -> tuple[Optional[str], str]:
    """Infer DSL script and type for a new signature.

    Returns (dsl_script_json, dsl_type) or (None, "math") if no DSL.
    """
    # Try template first
    if step_type in DSL_TEMPLATES:
        dsl = DSL_TEMPLATES[step_type]
        return json.dumps(dsl), dsl["type"]

    # Infer from description
    desc_lower = description.lower()
    for pattern, dsl in DSL_INFERENCE_PATTERNS:
        if re.search(pattern, desc_lower):
            return json.dumps(dsl), dsl["type"]

    # Default fallback: guidance DSL
    fallback = {
        "type": "decompose",
        "script": "reason_step",
        "params": ["context"],
        "purpose": f"Execute: {description[:50]}",
    }
    return json.dumps(fallback), "decompose"

from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import cosine_similarity, pack_embedding, unpack_embedding


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
            self._db_path = None
        self._init_schema()

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
        import time
        max_retries = 5

        for attempt in range(max_retries):
            try:
                return self._find_or_create_atomic(
                    step_text, embedding, min_similarity, parent_problem, origin_depth
                )
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))
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

                    score = cosine_similarity(embedding, centroid)

                    if score >= min_similarity and score > best_score:
                        best_match = self._row_to_signature(row)
                        best_score = score

                if best_match:
                    # Update last_used_at
                    now = datetime.utcnow().isoformat()
                    conn.execute(
                        "UPDATE step_signatures SET last_used_at = ? WHERE id = ?",
                        (now, best_match.id),
                    )
                    conn.commit()
                    logger.debug(
                        "[db] Matched signature: step='%s' sig='%s' score=%.3f",
                        step_text[:40], best_match.step_type, best_score
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

        try:
            cursor = conn.execute(
                """INSERT INTO step_signatures
                   (signature_id, centroid, step_type, description, dsl_script, dsl_type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sig_id, centroid_packed, step_type, step_text, dsl_script, dsl_type, now),
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
    # Usage Recording
    # =========================================================================

    def record_usage(
        self,
        signature_id: int,
        step_text: str,
        success: bool,
        was_injected: bool = False,
        params_extracted: dict = None,
    ):
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
            # Successes are updated by update_problem_outcome() after grading
            conn.execute(
                """UPDATE step_signatures
                   SET uses = uses + 1, last_used_at = ?
                   WHERE id = ?""",
                (now, signature_id),
            )

    def update_problem_outcome(
        self,
        signature_ids: list[int],
        problem_correct: bool,
    ):
        """Update signature success counts based on problem outcome.

        Call this after grading a problem to propagate correctness back to
        all signatures that were used. This is how we track real lift.

        Args:
            signature_ids: IDs of signatures used in the solved problem
            problem_correct: Whether the final answer was correct
        """
        if not signature_ids:
            return

        with self._connection() as conn:
            if problem_correct:
                # Increment success count for all signatures used
                placeholders = ",".join("?" * len(signature_ids))
                conn.execute(
                    f"""UPDATE step_signatures
                       SET successes = successes + 1
                       WHERE id IN ({placeholders})""",
                    signature_ids,
                )
                logger.debug(
                    "[db] Problem correct: incremented successes for %d signatures",
                    len(signature_ids)
                )
            else:
                # Problem failed - signatures don't get success credit
                # This is how we detect negative lift
                logger.debug(
                    "[db] Problem incorrect: %d signatures get no success credit",
                    len(signature_ids)
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
        from datetime import datetime
        now = datetime.utcnow().isoformat()

        with self._connection() as conn:
            try:
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
                logger.info(
                    "[db] Added child relationship: parent=%d → child=%d (condition='%s')",
                    parent_id, child_id, condition[:30]
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
