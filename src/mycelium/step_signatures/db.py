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
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import numpy as np

from mycelium.config import EMBEDDING_DIM, DB_PATH, MIN_MATCH_THRESHOLD
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
    # COVERAGE TRACKING: Track how well signatures cover step descriptions
    # =========================================================================

    def record_coverage(self, signature_id: int, similarity: float, threshold: float = None) -> None:
        """Record a coverage observation for a signature.

        Tracks similarity scores using Welford algorithm.
        Also counts low-coverage observations (below threshold).

        Args:
            signature_id: The signature that was matched
            similarity: The cosine similarity score
            threshold: Optional threshold for "low coverage" (uses adaptive if None)
        """
        if threshold is None:
            threshold = self.get_adaptive_threshold()

        is_low = 1 if similarity < threshold else 0

        conn = self._connection()
        conn.execute(
            """
            UPDATE step_signatures
            SET coverage_sim_count = COALESCE(coverage_sim_count, 0) + 1,
                coverage_sim_mean = COALESCE(coverage_sim_mean, 0) +
                    (? - COALESCE(coverage_sim_mean, 0)) / (COALESCE(coverage_sim_count, 0) + 1),
                coverage_sim_m2 = COALESCE(coverage_sim_m2, 0) +
                    (? - COALESCE(coverage_sim_mean, 0)) *
                    (? - (COALESCE(coverage_sim_mean, 0) + (? - COALESCE(coverage_sim_mean, 0)) / (COALESCE(coverage_sim_count, 0) + 1))),
                low_coverage_count = COALESCE(low_coverage_count, 0) + ?
            WHERE id = ?
            """,
            (similarity, similarity, similarity, similarity, is_low, signature_id),
        )

    def get_coverage_stats(self, func_name: str = None) -> dict:
        """Get coverage statistics.

        Args:
            func_name: Optional filter by function

        Returns:
            Dict with mean, std, count, low_coverage_rate per function
        """
        conn = self._connection()

        if func_name:
            where = "WHERE func_name = ?"
            params = (func_name,)
        else:
            where = "WHERE func_name IS NOT NULL"
            params = ()

        rows = conn.execute(
            f"""
            SELECT
                func_name,
                SUM(coverage_sim_count) as total_count,
                SUM(coverage_sim_mean * coverage_sim_count) / NULLIF(SUM(coverage_sim_count), 0) as mean_sim,
                SUM(coverage_sim_m2) as total_m2,
                SUM(low_coverage_count) as low_count
            FROM step_signatures
            {where}
            GROUP BY func_name
            """,
            params,
        ).fetchall()

        stats = {}
        for row in rows:
            func = row[0]
            count = row[1] or 0
            mean = row[2] or 0.0
            m2 = row[3] or 0.0
            low = row[4] or 0

            variance = m2 / count if count > 0 else 0.0
            std = variance ** 0.5
            low_rate = low / count if count > 0 else 0.0

            stats[func] = {
                "count": count,
                "mean_similarity": mean,
                "std_similarity": std,
                "low_coverage_count": low,
                "low_coverage_rate": low_rate,
            }

        return stats

    def get_functions_needing_coverage(
        self, min_observations: int = 10, max_low_rate: float = 0.3
    ) -> List[str]:
        """Get functions with poor coverage (high rate of low-similarity matches).

        These are candidates for adding more signature variants.

        Args:
            min_observations: Minimum observations to consider a function
            max_low_rate: Functions with low_coverage_rate > this are returned

        Returns:
            List of function names sorted by low coverage rate (worst first)
        """
        stats = self.get_coverage_stats()

        needs_coverage = []
        for func, s in stats.items():
            if s["count"] >= min_observations and s["low_coverage_rate"] > max_low_rate:
                needs_coverage.append((func, s["low_coverage_rate"], s["count"]))

        # Sort by low coverage rate (worst first)
        needs_coverage.sort(key=lambda x: -x[1])
        return [func for func, rate, count in needs_coverage]

    # =========================================================================
    # FAILURE TRACKING: Record failures for periodic review
    # =========================================================================

    def record_execution_failure(
        self,
        step_description: str,
        func_name: str = None,
        signature_id: int = None,
        similarity: float = None,
        error_type: str = "execution_error",
        error_message: str = None,
        problem_id: str = None,
    ) -> None:
        """Record a failed execution for later review.

        Args:
            step_description: The step that failed
            func_name: Function that was attempted (if any)
            signature_id: Signature that was matched (if any)
            similarity: Similarity score of match (if any)
            error_type: 'wrong_answer', 'execution_error', 'no_match'
            error_message: Optional error details
            problem_id: Optional problem reference
        """
        conn = self._connection()
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO execution_failures
            (step_description, func_name, signature_id, similarity,
             error_type, error_message, problem_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_description,
                func_name,
                signature_id,
                similarity,
                error_type,
                error_message,
                problem_id,
                now,
            ),
        )

    def get_failure_stats(self) -> dict:
        """Get summary statistics of failures."""
        conn = self._connection()

        # By error type
        rows = conn.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM execution_failures
            GROUP BY error_type
            """
        ).fetchall()
        by_type = {row[0]: row[1] for row in rows}

        # By function
        rows = conn.execute(
            """
            SELECT func_name, COUNT(*) as count
            FROM execution_failures
            WHERE func_name IS NOT NULL
            GROUP BY func_name
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()
        by_func = {row[0]: row[1] for row in rows}

        # Total
        total = conn.execute(
            "SELECT COUNT(*) FROM execution_failures"
        ).fetchone()[0]

        return {
            "total": total,
            "by_type": by_type,
            "top_failing_functions": by_func,
        }

    def get_recent_failures(
        self, limit: int = 50, error_type: str = None
    ) -> List[dict]:
        """Get recent failures for review.

        Args:
            limit: Max failures to return
            error_type: Optional filter by error type

        Returns:
            List of failure dicts
        """
        conn = self._connection()

        if error_type:
            rows = conn.execute(
                """
                SELECT * FROM execution_failures
                WHERE error_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (error_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM execution_failures
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_failure_patterns(self, min_occurrences: int = 3) -> List[dict]:
        """Find repeated failure patterns.

        Groups similar step descriptions and returns patterns
        that occur frequently - these are candidates for new signatures.
        """
        conn = self._connection()

        # Group by func_name and look for repeated failures
        rows = conn.execute(
            """
            SELECT
                func_name,
                step_description,
                COUNT(*) as occurrences,
                AVG(similarity) as avg_similarity
            FROM execution_failures
            WHERE func_name IS NOT NULL
            GROUP BY func_name, step_description
            HAVING COUNT(*) >= ?
            ORDER BY occurrences DESC
            """,
            (min_occurrences,),
        ).fetchall()

        patterns = []
        for row in rows:
            patterns.append(
                {
                    "func_name": row[0],
                    "step_description": row[1],
                    "occurrences": row[2],
                    "avg_similarity": row[3],
                }
            )

        return patterns

    def clear_old_failures(self, days: int = 30) -> int:
        """Clear failures older than specified days.

        Returns number of deleted records.
        """
        conn = self._connection()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor = conn.execute(
            """
            DELETE FROM execution_failures
            WHERE created_at < ?
            """,
            (cutoff,),
        )

        return cursor.rowcount

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

    def get_adaptive_threshold(self, fallback: float = None) -> float:
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
            MIN_MATCH_THRESHOLD,
        )

        if fallback is None:
            fallback = MIN_MATCH_THRESHOLD

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
    # SIGNATURE MERGING - Welford-guided duplicate prevention
    # =========================================================================

    def get_nearest_same_func_signature(
        self, embedding: np.ndarray, func_name: str
    ) -> tuple[Optional[StepSignature], float]:
        """Find nearest signature with same func_name.

        Args:
            embedding: The query embedding vector
            func_name: Function name to filter by

        Returns:
            (signature, distance) or (None, inf) if no signatures for this func
        """
        sigs = self.get_signatures_by_func(func_name)
        if not sigs:
            return None, float("inf")

        # Normalize query
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None, float("inf")
        query = query / query_norm

        best_sig = None
        best_dist = float("inf")

        for sig in sigs:
            if sig.centroid is None:
                continue

            # Cosine distance = 1 - cosine_similarity
            centroid = sig.centroid
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                continue
            centroid = centroid / centroid_norm

            similarity = float(np.dot(query, centroid))
            distance = 1.0 - similarity

            if distance < best_dist:
                best_dist = distance
                best_sig = sig

        return best_sig, best_dist

    def get_merge_threshold(self, func_name: str, default: float = 0.15) -> float:
        """Get Welford-adaptive merge threshold for a function.

        Formula: mean + 1.5 * std of observed merge distances
        Returns default if insufficient data (<10 merges).

        Args:
            func_name: Function name to get threshold for
            default: Default threshold if insufficient data

        Returns:
            Adaptive merge threshold (cosine distance)
        """
        conn = self._connection()
        row = conn.execute(
            """
            SELECT
                SUM(merge_dist_count) as total_count,
                SUM(merge_dist_mean * merge_dist_count) as weighted_sum,
                SUM(merge_dist_m2) as total_m2
            FROM step_signatures
            WHERE func_name = ? AND merge_dist_count > 0
            """,
            (func_name,),
        ).fetchone()

        if row is None or row[0] is None or row[0] < 10:
            logger.debug(
                "[db] Merge threshold for %s: insufficient data (%s), using default %.3f",
                func_name,
                row[0] if row else 0,
                default,
            )
            return default

        total_count = row[0]
        weighted_mean = row[1] / total_count if total_count > 0 else 0.0
        total_m2 = row[2] or 0.0

        # Variance from combined M2
        variance = total_m2 / total_count if total_count > 0 else 0.0
        stddev = variance ** 0.5

        # Adaptive threshold: mean + 1.5 * std
        adaptive = weighted_mean + 1.5 * stddev

        # Clamp to reasonable range [0.05, 0.30]
        clamped = max(0.05, min(0.30, adaptive))

        logger.debug(
            "[db] Merge threshold for %s: mean=%.3f, std=%.3f, raw=%.3f, clamped=%.3f (n=%d)",
            func_name,
            weighted_mean,
            stddev,
            adaptive,
            clamped,
            total_count,
        )

        return clamped

    def merge_into_signature(
        self, sig_id: int, embedding: np.ndarray, description: str
    ) -> None:
        """Merge a new description into an existing signature.

        Updates:
        - centroid via running average: (embedding_sum + new) / (count + 1)
        - embedding_sum and embedding_count
        - description_variants (append new description)
        - Welford stats for merge distance

        Args:
            sig_id: The signature ID to merge into
            embedding: The new embedding to merge
            description: The new description to add as variant
        """
        from mycelium.step_signatures.utils import invalidate_centroid_cache

        conn = self._connection()
        sig = self.get_signature(sig_id)
        if sig is None:
            logger.warning("[db] Cannot merge into non-existent signature %d", sig_id)
            return

        # Compute distance for Welford update
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query_normalized = query / query_norm
        else:
            query_normalized = query

        if sig.centroid is not None:
            centroid = sig.centroid
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid_normalized = centroid / centroid_norm
                similarity = float(np.dot(query_normalized, centroid_normalized))
                distance = 1.0 - similarity
            else:
                distance = 1.0
        else:
            distance = 1.0

        # Get current embedding_sum or initialize from centroid
        if sig.embedding_sum is not None:
            current_sum = sig.embedding_sum
        elif sig.centroid is not None:
            current_sum = sig.centroid * sig.embedding_count
        else:
            current_sum = np.zeros_like(query)

        # Update centroid via running average
        new_count = sig.embedding_count + 1
        new_sum = current_sum + query
        new_centroid = new_sum / new_count

        # Update description_variants
        variants = sig.description_variants.copy()
        if description not in variants and description != sig.description:
            variants.append(description)
        variants_json = json.dumps(variants)

        # Pack embeddings for storage
        new_centroid_json = pack_embedding(new_centroid)
        new_sum_json = pack_embedding(new_sum)

        # Update with Welford algorithm for merge distance
        # Use convenience method - single statement, no cursor properties needed
        conn.execute(
            """
            UPDATE step_signatures
            SET centroid = ?,
                embedding_sum = ?,
                embedding_count = ?,
                description_variants = ?,
                merge_dist_count = COALESCE(merge_dist_count, 0) + 1,
                merge_dist_mean = COALESCE(merge_dist_mean, 0) +
                    (? - COALESCE(merge_dist_mean, 0)) / (COALESCE(merge_dist_count, 0) + 1),
                merge_dist_m2 = COALESCE(merge_dist_m2, 0) +
                    (? - COALESCE(merge_dist_mean, 0)) *
                    (? - (COALESCE(merge_dist_mean, 0) + (? - COALESCE(merge_dist_mean, 0)) / (COALESCE(merge_dist_count, 0) + 1)))
            WHERE id = ?
            """,
            (
                new_centroid_json,
                new_sum_json,
                new_count,
                variants_json,
                distance,
                distance,
                distance,
                distance,
                sig_id,
            ),
        )

        # Invalidate centroid cache for this signature
        invalidate_centroid_cache(sig_id)

        logger.debug(
            "[db] Merged into signature %d: new_count=%d, distance=%.3f, variants=%d",
            sig_id,
            new_count,
            distance,
            len(variants),
        )

    def should_merge_or_create(
        self, embedding: np.ndarray, func_name: str
    ) -> tuple[str, Optional[int]]:
        """Decide whether to merge into existing or create new signature.

        Args:
            embedding: The query embedding vector
            func_name: Function name for the signature

        Returns:
            ("merge", sig_id) if close to existing
            ("create", None) if should create new
        """
        nearest, distance = self.get_nearest_same_func_signature(embedding, func_name)

        if nearest is None:
            logger.debug("[db] No existing signatures for func=%s, create new", func_name)
            return "create", None

        threshold = self.get_merge_threshold(func_name)

        if distance <= threshold:
            logger.debug(
                "[db] Merge decision: distance=%.3f <= threshold=%.3f, merge into sig=%d",
                distance,
                threshold,
                nearest.id,
            )
            return "merge", nearest.id
        else:
            logger.debug(
                "[db] Merge decision: distance=%.3f > threshold=%.3f, create new",
                distance,
                threshold,
            )
            return "create", None

    # =========================================================================
    # FLAT PROTOTYPE STORE - k-NN Classification
    # =========================================================================

    def get_all_prototypes(self) -> List[StepSignature]:
        """Get all signatures as prototypes (alias for get_all_leaves).

        In the flat architecture, all signatures are prototypes for classification.
        """
        return self.get_all_leaves()

    def get_signatures_by_func(self, func_name: str, min_successes: int = 0) -> List[StepSignature]:
        """Get all signatures that map to a given function.

        Args:
            func_name: The function name to filter by
            min_successes: Minimum success count (for quality filtering)

        Returns:
            List of signatures sorted by success rate (descending)
        """
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT * FROM step_signatures
            WHERE func_name = ?
              AND COALESCE(successes, 0) >= ?
            ORDER BY
                CASE WHEN uses > 0 THEN CAST(successes AS REAL) / uses ELSE 0 END DESC,
                successes DESC
            """,
            (func_name, min_successes),
        ).fetchall()

        return [StepSignature.from_row(dict(row)) for row in rows]

    def get_all_func_names(self) -> List[str]:
        """Get all unique function names in the signature store."""
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT DISTINCT func_name FROM step_signatures
            WHERE func_name IS NOT NULL
            ORDER BY func_name
            """
        ).fetchall()
        return [row[0] for row in rows]

    def classify(self, embedding: np.ndarray) -> tuple[Optional[str], float, Optional[StepSignature]]:
        """Classify a step embedding to a function name via k-NN.

        This is the core 200-class classification:
        - Input: step embedding
        - Output: function name (or None if no prototypes)

        Args:
            embedding: The step embedding to classify

        Returns:
            (func_name, similarity, signature) tuple
            - func_name: The function this step maps to (or None)
            - similarity: Cosine similarity to best match
            - signature: The matched prototype signature
        """
        result = self.route_to_best_vectorized(embedding)
        if result.signature is None:
            return None, 0.0, None
        return result.signature.func_name, result.similarity, result.signature

    def route_to_best_vectorized(self, embedding: np.ndarray) -> RoutingResult:
        """Route to best signature using vectorized numpy (fast k-NN).

        At 5k prototypes, this is ~0.5ms.

        Args:
            embedding: The query embedding (numpy array)

        Returns:
            RoutingResult with best matching signature
        """
        prototypes = self.get_all_prototypes()
        if not prototypes:
            return RoutingResult(signature=None, similarity=0.0)

        # Filter to signatures with centroids
        valid_prototypes = [p for p in prototypes if p.centroid is not None]
        if not valid_prototypes:
            return RoutingResult(signature=None, similarity=0.0)

        # Vectorized cosine similarity
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return RoutingResult(signature=None, similarity=0.0)
        query = query / query_norm

        # Stack all centroids
        centroids = np.stack([p.centroid for p in valid_prototypes])
        # Normalize centroids (they should already be normalized, but ensure)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        centroids = centroids / norms

        # Compute all similarities at once
        similarities = centroids @ query

        # Find best
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])
        best_sig = valid_prototypes[best_idx]

        return RoutingResult(signature=best_sig, similarity=best_sim)

    def route_to_best(self, embedding: List[float]) -> RoutingResult:
        """Route to best matching signature via cosine similarity.

        Returns best match regardless of threshold - let caller decide.
        Note: Use route_to_best_vectorized() for better performance.
        """
        return self.route_to_best_vectorized(np.asarray(embedding, dtype=np.float32))

    def get_top_k(self, embedding: np.ndarray, k: int = 5) -> List[tuple[StepSignature, float]]:
        """Get top-k matching prototypes.

        Useful for showing the LLM nearby options or for debugging.

        Args:
            embedding: Query embedding
            k: Number of results to return

        Returns:
            List of (signature, similarity) tuples sorted by similarity (descending)
        """
        prototypes = self.get_all_prototypes()
        if not prototypes:
            return []

        valid_prototypes = [p for p in prototypes if p.centroid is not None]
        if not valid_prototypes:
            return []

        # Vectorized similarity
        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        centroids = np.stack([p.centroid for p in valid_prototypes])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids = centroids / norms

        similarities = centroids @ query

        # Get top-k indices
        if len(similarities) <= k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(valid_prototypes[i], float(similarities[i])) for i in top_indices]

    # =========================================================================
    # LLM PROMPT BUILDING - Signature Menu
    # =========================================================================

    def select_diverse_signatures(
        self, func_name: str, k: int = 10, quality_weight: float = 0.3
    ) -> List[StepSignature]:
        """Select k diverse signatures using quality-weighted farthest-point sampling.

        Algorithm:
        1. Start with highest success_rate signature
        2. For each remaining slot:
           - For each candidate, compute:
             - distance = min distance to any selected signature
             - quality = success_rate
             - score = (1 - quality_weight) * distance + quality_weight * quality
           - Select candidate with highest score

        Args:
            func_name: Function to get signatures for
            k: Number of signatures to select
            quality_weight: Balance between diversity (0) and quality (1)

        Returns:
            List of diverse, high-quality signatures
        """
        # Handle edge case of k <= 0
        if k <= 0:
            return []

        # Get all signatures for this function
        all_sigs = self.get_signatures_by_func(func_name, min_successes=0)

        # Filter to those with centroids
        candidates = [s for s in all_sigs if s.centroid is not None]

        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates

        # Start with highest success_rate signature
        candidates.sort(key=lambda s: s.success_rate, reverse=True)
        selected = [candidates[0]]
        remaining = candidates[1:]

        # Pre-compute normalized centroids for efficiency
        def normalize(v: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        selected_centroids = [normalize(selected[0].centroid)]

        while len(selected) < k and remaining:
            best_score = -1.0
            best_idx = 0

            for i, cand in enumerate(remaining):
                cand_centroid = normalize(cand.centroid)

                # Compute min distance to any selected signature (cosine distance = 1 - similarity)
                min_distance = float("inf")
                for sel_centroid in selected_centroids:
                    sim = float(np.dot(cand_centroid, sel_centroid))
                    distance = 1.0 - sim
                    if distance < min_distance:
                        min_distance = distance

                # Normalize distance to [0, 1] range (max cosine distance is 2 for opposite vectors)
                # In practice, for similar domain vectors, distance rarely exceeds 1
                normalized_distance = min(min_distance, 1.0)

                # Quality score (success_rate is already in [0, 1])
                quality = cand.success_rate

                # Combined score
                score = (1 - quality_weight) * normalized_distance + quality_weight * quality

                if score > best_score:
                    best_score = score
                    best_idx = i

            # Add best candidate to selected
            best_cand = remaining.pop(best_idx)
            selected.append(best_cand)
            selected_centroids.append(normalize(best_cand.centroid))

        return selected

    def build_diverse_menu(
        self, max_examples: int = 10, quality_weight: float = 0.3
    ) -> dict[str, List[dict]]:
        """Build menu with evenly-spaced examples per function.

        Uses quality-weighted farthest-point sampling to select diverse,
        high-quality signature examples for each function.

        Args:
            max_examples: Maximum examples per function
            quality_weight: Balance between diversity (0) and quality (1)

        Returns:
            Dict mapping func_name -> list of {description, successes, success_rate}
        """
        func_names = self.get_all_func_names()
        menu = {}

        for func_name in func_names:
            diverse_sigs = self.select_diverse_signatures(
                func_name, k=max_examples, quality_weight=quality_weight
            )

            if diverse_sigs:
                menu[func_name] = [
                    {
                        "description": sig.description,
                        "successes": sig.successes,
                        "success_rate": sig.success_rate,
                    }
                    for sig in diverse_sigs
                ]

        return menu

    def build_signature_menu(self, min_successes: int = 3, max_examples_per_func: int = 3) -> dict[str, List[dict]]:
        """Build a menu of functions with their proven signature examples.

        This is the "learned vocabulary" that guides LLM decomposition.
        High-success signatures become few-shot examples.

        Args:
            min_successes: Minimum successes to include a signature
            max_examples_per_func: Max examples per function

        Returns:
            Dict mapping func_name -> list of {description, successes, success_rate}
        """
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT func_name, description, successes, uses
            FROM step_signatures
            WHERE func_name IS NOT NULL
              AND COALESCE(successes, 0) >= ?
            ORDER BY func_name,
                CASE WHEN uses > 0 THEN CAST(successes AS REAL) / uses ELSE 0 END DESC,
                successes DESC
            """,
            (min_successes,),
        ).fetchall()

        menu = {}
        for row in rows:
            func_name = row[0]
            if func_name not in menu:
                menu[func_name] = []

            if len(menu[func_name]) < max_examples_per_func:
                uses = row[3] or 0
                successes = row[2] or 0
                menu[func_name].append({
                    "description": row[1],
                    "successes": successes,
                    "success_rate": successes / uses if uses > 0 else 0.0,
                })

        return menu

    def format_signature_menu(
        self,
        max_examples_per_func: int = 10,
        quality_weight: float = 0.3,
        use_diverse: bool = True,
    ) -> str:
        """Format the signature menu as a string for LLM prompts.

        Returns text like:
            add:
              - "combine two prices" (47 successes)
              - "sum the quantities" (38 successes)

            multiply:
              - "calculate total cost" (52 successes)

        Args:
            max_examples_per_func: Maximum examples per function
            quality_weight: Balance between diversity (0) and quality (1)
            use_diverse: If True, use quality-weighted farthest-point sampling
        """
        if use_diverse:
            menu = self.build_diverse_menu(max_examples_per_func, quality_weight)
        else:
            # Fallback to original behavior (kept for compatibility)
            menu = self.build_signature_menu(min_successes=0, max_examples_per_func=max_examples_per_func)

        if not menu:
            return "No proven patterns yet. Decompose into basic operations."

        lines = []
        for func_name in sorted(menu.keys()):
            examples = menu[func_name]
            lines.append(f"{func_name}:")
            for ex in examples:
                lines.append(f'  - "{ex["description"]}" ({ex["successes"]} successes)')
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # SIGNATURE CREATION
    # =========================================================================

    def find_or_create(
        self,
        step_text: str,
        embedding: List[float],
        func_name: str = None,
        **kwargs,
    ) -> tuple[StepSignature, bool]:
        """Find existing signature or create new one.

        Args:
            step_text: Description of the step
            embedding: Step embedding vector
            func_name: Optional function name (add, sub, mul, etc.)

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
        if best_sig and best_sim >= MIN_MATCH_THRESHOLD:
            return best_sig, False

        # Create new signature
        conn = self._connection()
        step_type = normalize_step_text(step_text)[:100]
        sig_id = str(uuid.uuid4())

        # Use func_name for function registry lookup
        final_func_name = func_name

        # Get arity from function registry if available
        func_arity = 2  # default
        if final_func_name:
            try:
                from mycelium.function_registry import get_function_info
                info = get_function_info(final_func_name)
                if info:
                    func_arity = info.get("arity", 2)
            except ImportError:
                pass

        now = datetime.now(timezone.utc).isoformat()

        # Use explicit context manager to access cursor.lastrowid within transaction
        with conn.connection() as raw_conn:
            cursor = raw_conn.execute(
                """
                INSERT INTO step_signatures (
                    signature_id, step_type, description, func_name, func_arity,
                    centroid, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sig_id, step_type, step_text, final_func_name, func_arity, pack_embedding(embedding), now),
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
        # Use explicit context manager for atomic multi-statement transaction
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
