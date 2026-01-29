"""StepSignatureDB - Minimal implementation for local decomposition architecture.

Core functionality only:
- Store/retrieve signatures
- Record success/failure stats
- Route steps to signatures via embedding similarity
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np

from mycelium.config import EMBEDDING_DIM, DB_PATH
from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import (
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)

logger = logging.getLogger(__name__)


def normalize_step_text(text: str) -> str:
    """Normalize step text by replacing numbers with N."""
    return re.sub(r'\b\d+\.?\d*\b', 'N', text)


@dataclass
class RoutingResult:
    """Result of routing a step to a signature."""
    signature: Optional[StepSignature]
    similarity: float
    path: List[int] = None

    def __post_init__(self):
        if self.path is None:
            self.path = []

    @property
    def is_match(self) -> bool:
        return self.signature is not None


class StepSignatureDB:
    """Minimal signature database for local decomposition."""

    def __init__(self, db_path: str = None, embedder=None):
        """Initialize database."""
        self.db_path = db_path or DB_PATH
        self._embedder = embedder
        with get_db().connection() as conn:
            init_db(conn)

    def _connection(self):
        """Get database connection manager."""
        return get_db()

    # =========================================================================
    # CORE QUERIES
    # =========================================================================

    def count_signatures(self) -> int:
        """Count total signatures."""
        conn = self._connection()
        row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
        return row[0] if row else 0

    def get_signature_count(self) -> int:
        """Alias for count_signatures."""
        return self.count_signatures()

    def get_signature(self, signature_id: int) -> Optional[StepSignature]:
        """Get signature by ID."""
        conn = self._connection()
        row = conn.execute(
            "SELECT * FROM step_signatures WHERE id = ?",
            (signature_id,),
        ).fetchone()

        if row is None:
            return None

        return StepSignature.from_row(dict(row))

    def get_all_leaves(self) -> List[StepSignature]:
        """Get all leaf signatures (non-umbrellas)."""
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT * FROM step_signatures
            WHERE is_semantic_umbrella = 0 OR is_semantic_umbrella IS NULL
            """
        ).fetchall()

        return [StepSignature.from_row_for_routing(dict(row)) for row in rows]

    def get_all_signatures(self) -> List[StepSignature]:
        """Get all signatures."""
        conn = self._connection()
        rows = conn.execute("SELECT * FROM step_signatures").fetchall()
        return [StepSignature.from_row(dict(row)) for row in rows]

    # =========================================================================
    # STATS RECORDING
    # =========================================================================

    def record_success(self, signature_id: int, similarity: float = None) -> None:
        """Record a successful execution with optional similarity."""
        conn = self._connection()
        if similarity is not None:
            # Update success stats with Welford algorithm
            conn.execute(
                """
                UPDATE step_signatures
                SET successes = COALESCE(successes, 0) + 1,
                    uses = COALESCE(uses, 0) + 1,
                    success_sim_count = COALESCE(success_sim_count, 0) + 1,
                    success_sim_mean = COALESCE(success_sim_mean, 0) +
                        (? - COALESCE(success_sim_mean, 0)) / (COALESCE(success_sim_count, 0) + 1),
                    success_sim_m2 = COALESCE(success_sim_m2, 0) +
                        (? - COALESCE(success_sim_mean, 0)) *
                        (? - (COALESCE(success_sim_mean, 0) + (? - COALESCE(success_sim_mean, 0)) / (COALESCE(success_sim_count, 0) + 1)))
                WHERE id = ?
                """,
                (similarity, similarity, similarity, similarity, signature_id),
            )
        else:
            conn.execute(
                """
                UPDATE step_signatures
                SET successes = COALESCE(successes, 0) + 1,
                    uses = COALESCE(uses, 0) + 1
                WHERE id = ?
                """,
                (signature_id,),
            )

    def record_failure(self, signature_id: int) -> None:
        """Record a failed execution."""
        conn = self._connection()
        conn.execute(
            """
            UPDATE step_signatures
            SET operational_failures = COALESCE(operational_failures, 0) + 1,
                uses = COALESCE(uses, 0) + 1
            WHERE id = ?
            """,
            (signature_id,),
        )

    def record_similarity(self, signature_id: int, similarity: float) -> None:
        """Record similarity observation for Welford tracking."""
        conn = self._connection()
        conn.execute(
            """
            UPDATE step_signatures
            SET similarity_count = COALESCE(similarity_count, 0) + 1,
                similarity_mean = COALESCE(similarity_mean, 0) +
                    (? - COALESCE(similarity_mean, 0)) / (COALESCE(similarity_count, 0) + 1),
                similarity_m2 = COALESCE(similarity_m2, 0) +
                    (? - COALESCE(similarity_mean, 0)) *
                    (? - (COALESCE(similarity_mean, 0) + (? - COALESCE(similarity_mean, 0)) / (COALESCE(similarity_count, 0) + 1)))
            WHERE id = ?
            """,
            (similarity, similarity, similarity, similarity, signature_id),
        )

    # =========================================================================
    # WELFORD-ADAPTIVE THRESHOLDS
    # =========================================================================

    def get_global_similarity_stats(self) -> tuple[float, float, int]:
        """Get aggregate Welford stats across all signatures.

        Returns:
            (mean, stddev, count) tuple for successful similarity observations.
        """
        conn = self._connection()
        row = conn.execute(
            """
            SELECT
                SUM(success_sim_count) as total_count,
                SUM(success_sim_mean * success_sim_count) as weighted_sum,
                SUM(success_sim_m2) as total_m2
            FROM step_signatures
            WHERE success_sim_count > 0
            """
        ).fetchone()

        if row is None or row[0] is None or row[0] == 0:
            return 0.0, 0.0, 0

        total_count = row[0]
        weighted_mean = row[1] / total_count if total_count > 0 else 0.0
        total_m2 = row[2] or 0.0

        # Variance from combined M2
        variance = total_m2 / total_count if total_count > 0 else 0.0
        stddev = variance ** 0.5

        return weighted_mean, stddev, total_count

    def get_adaptive_threshold(self, fallback: float = 0.85) -> float:
        """Get Welford-adaptive similarity threshold.

        Per CLAUDE.md "The Flow": DB Stats → Welford → Tree Structure.

        Formula: mean - k * stddev (captures ~93% of good matches at k=1.5)
        Clamped to [0.70, 0.95] range.

        Returns:
            Adaptive threshold, or fallback if insufficient data.
        """
        from mycelium.config import (
            ADAPTIVE_THRESHOLD_MIN_SAMPLES,
            ADAPTIVE_THRESHOLD_K,
            ADAPTIVE_THRESHOLD_MIN,
            ADAPTIVE_THRESHOLD_MAX,
        )

        mean, stddev, count = self.get_global_similarity_stats()

        # Need sufficient samples for reliable estimate
        if count < ADAPTIVE_THRESHOLD_MIN_SAMPLES:
            logger.debug(
                "[db] Adaptive threshold: insufficient samples (%d < %d), using fallback %.3f",
                count, ADAPTIVE_THRESHOLD_MIN_SAMPLES, fallback
            )
            return fallback

        # Adaptive: mean - k * std
        adaptive = mean - ADAPTIVE_THRESHOLD_K * stddev

        # Clamp to reasonable range
        clamped = max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, adaptive))

        logger.debug(
            "[db] Adaptive threshold: mean=%.3f, std=%.3f, raw=%.3f, clamped=%.3f (n=%d)",
            mean, stddev, adaptive, clamped, count
        )

        return clamped

    # =========================================================================
    # ROUTING
    # =========================================================================

    def route_to_best(self, embedding: List[float]) -> RoutingResult:
        """Route to best matching signature via cosine similarity.

        Returns best match regardless of threshold - let caller decide.
        """
        leaves = self.get_all_leaves()
        best_sig = None
        best_sim = 0.0

        for sig in leaves:
            if sig.centroid is None:
                continue
            sim = cosine_similarity(embedding, sig.centroid)
            if sim > best_sim:
                best_sim = sim
                best_sig = sig

        return RoutingResult(signature=best_sig, similarity=best_sim)

    # =========================================================================
    # SIGNATURE CREATION
    # =========================================================================

    def find_or_create(
        self,
        step_text: str,
        embedding: List[float],
        dsl_hint: str = None,
        **kwargs,
    ) -> tuple[StepSignature, bool]:
        """Find existing signature or create new one.

        Args:
            step_text: Description of the step
            embedding: Step embedding vector
            dsl_hint: Optional DSL hint (+, -, *, /)

        Returns:
            (signature, created) tuple
        """
        # Find best match among leaves
        leaves = self.get_all_leaves()
        best_sig = None
        best_sim = 0.0

        for sig in leaves:
            if sig.centroid is None:
                continue
            sim = cosine_similarity(embedding, sig.centroid)
            if sim > best_sim:
                best_sim = sim
                best_sig = sig

        # If good match found, return it
        if best_sig and best_sim >= 0.85:
            return best_sig, False

        # Create new signature
        conn = self._connection()
        step_type = normalize_step_text(step_text)[:100]
        sig_id = str(uuid.uuid4())

        # Infer DSL from hint
        dsl_script = None
        if dsl_hint:
            dsl_map = {"+": "a + b", "-": "a - b", "*": "a * b", "/": "a / b"}
            dsl_script = dsl_map.get(dsl_hint, f"a {dsl_hint} b")

        now = datetime.now(timezone.utc).isoformat()

        with conn.connection() as raw_conn:
            cursor = raw_conn.execute(
                """
                INSERT INTO step_signatures (
                    signature_id, step_type, description, dsl_script,
                    centroid, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sig_id, step_type, step_text, dsl_script, pack_embedding(embedding), now),
            )
            last_id = cursor.lastrowid

        new_sig = self.get_signature(last_id)
        return new_sig, True

    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================

    def clear_all_data(self, force: bool = False) -> dict:
        """Clear all signature data."""
        if not force:
            return {"error": "Use force=True to confirm"}

        conn = self._connection()
        with conn.connection() as raw_conn:
            raw_conn.execute("DELETE FROM step_signatures")
            raw_conn.execute("DELETE FROM signature_relationships")

        return {"cleared": True}


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_step_db: Optional[StepSignatureDB] = None


def get_step_db() -> StepSignatureDB:
    """Get the singleton StepSignatureDB instance."""
    global _step_db
    if _step_db is None:
        _step_db = StepSignatureDB()
    return _step_db


def reset_step_db() -> None:
    """Reset the singleton instance."""
    global _step_db
    _step_db = None
