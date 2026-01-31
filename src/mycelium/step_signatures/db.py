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

    def format_signature_menu(self, min_successes: int = 3, max_examples_per_func: int = 3) -> str:
        """Format the signature menu as a string for LLM prompts.

        Returns text like:
            add:
              - "combine two prices" (47 successes)
              - "sum the quantities" (38 successes)

            multiply:
              - "calculate total cost" (52 successes)
        """
        menu = self.build_signature_menu(min_successes, max_examples_per_func)

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
        dsl_hint: str = None,
        func_name: str = None,
        **kwargs,
    ) -> tuple[StepSignature, bool]:
        """Find existing signature or create new one.

        Args:
            step_text: Description of the step
            embedding: Step embedding vector
            dsl_hint: Optional function name hint (add, sub, mul, etc.)
            func_name: Explicit function name (takes precedence over dsl_hint)

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

        # Determine function name (func_name takes precedence over dsl_hint)
        final_func_name = func_name or dsl_hint

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
