"""Operational Alignment Outcome Recording.

Records routing outcomes for potential future analysis.
Per CLAUDE.md: MCTS rollouts provide ground truth for operational equivalence.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from mycelium.data_layer import configure_connection

logger = logging.getLogger(__name__)


class OperationalAlignmentTracker:
    """Records routing outcomes to database."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS operational_alignment_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signature_id INTEGER NOT NULL,
        step_text TEXT NOT NULL,
        embedding_similarity REAL NOT NULL,
        was_correct INTEGER NOT NULL,
        dsl_type TEXT,
        problem_id TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_oao_signature ON operational_alignment_outcomes(signature_id);
    CREATE INDEX IF NOT EXISTS idx_oao_created ON operational_alignment_outcomes(created_at);
    CREATE INDEX IF NOT EXISTS idx_oao_correct ON operational_alignment_outcomes(was_correct);
    """

    def __init__(self, db_path: str):
        """Initialize tracker with database path."""
        self.db_path = db_path
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        configure_connection(conn, enable_foreign_keys=False)
        return conn

    def _ensure_schema(self):
        """Ensure tracking tables exist."""
        conn = self._get_connection()
        try:
            conn.executescript(self.SCHEMA)
            conn.commit()
        except Exception as e:
            logger.warning("[alignment] Schema creation failed: %s", e)
        finally:
            conn.close()

    def record_outcome(
        self,
        signature_id: int,
        step_text: str,
        embedding_similarity: float,
        was_correct: bool,
        dsl_type: str = "unknown",
        problem_id: Optional[str] = None,
    ) -> int:
        """Record a routing outcome.

        Args:
            signature_id: ID of the signature that was routed to
            step_text: The step text that was processed
            embedding_similarity: Cosine similarity to signature centroid
            was_correct: Whether the path produced the correct answer
            dsl_type: Type of DSL operation ('math', 'decompose', etc)
            problem_id: Optional problem identifier for grouping

        Returns:
            Row ID of the inserted record
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO operational_alignment_outcomes
                   (signature_id, step_text, embedding_similarity, was_correct,
                    dsl_type, problem_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (signature_id, step_text[:500], embedding_similarity,
                 1 if was_correct else 0, dsl_type, problem_id, now)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()


def record_routing_outcome(
    db_path: str,
    signature_id: int,
    step_text: str,
    embedding_similarity: float,
    was_correct: bool,
    dsl_type: str = "unknown",
    problem_id: Optional[str] = None,
) -> int:
    """Convenience function to record a routing outcome.

    Args:
        db_path: Path to SQLite database
        signature_id: ID of signature routed to
        step_text: The step that was processed
        embedding_similarity: Similarity score from routing
        was_correct: Whether final answer was correct
        dsl_type: Type of operation
        problem_id: Optional problem identifier

    Returns:
        Row ID of recorded outcome
    """
    tracker = OperationalAlignmentTracker(db_path)
    return tracker.record_outcome(
        signature_id=signature_id,
        step_text=step_text,
        embedding_similarity=embedding_similarity,
        was_correct=was_correct,
        dsl_type=dsl_type,
        problem_id=problem_id,
    )
