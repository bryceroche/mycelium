"""Operational Alignment Validation Metrics.

Per CLAUDE.md insight:
- Standard embedding models conflate "looks similar" with "means similar"
- MCTS rollouts provide ground truth for operational equivalence
- Use rollout outcomes to validate that embeddings are learning operations, not vocabulary

This module measures whether MCTS training is actually helping:
1. Operational Alignment Score - correlation(embedding_similarity, operational_outcome_match)
2. Centroid Drift Ratio - same-op distance decreasing vs different-op distance increasing
3. Routing Misfire Rate - high-sim-wrong-op vs low-sim-right-op

If MCTS is working, these metrics should improve over training.
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from mycelium.data_layer import configure_connection

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OperationalAlignmentScore:
    """Snapshot of operational alignment at a point in time.

    Higher values = embeddings better reflect operational equivalence.
    """
    # Core metric: correlation between embedding similarity and operational match
    # Range: -1.0 to 1.0 (higher = better alignment)
    correlation: float = 0.0

    # Component metrics
    same_op_avg_similarity: float = 0.0  # Avg similarity for operationally equivalent pairs
    diff_op_avg_similarity: float = 0.0  # Avg similarity for operationally different pairs
    separation_gap: float = 0.0  # same_op - diff_op (positive = good)

    # Misfire rates
    high_sim_wrong_op_count: int = 0  # High similarity but different operation (false positive)
    low_sim_right_op_count: int = 0   # Low similarity but same operation (false negative)
    misfire_ratio: float = 0.0  # false_pos / (false_pos + true_pos)

    # Sample sizes
    total_pairs_evaluated: int = 0
    same_op_pairs: int = 0
    diff_op_pairs: int = 0

    # Timestamp
    computed_at: str = ""

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_aligned(self) -> bool:
        """Check if embeddings show operational alignment."""
        return self.correlation > 0.3 and self.separation_gap > 0.1

    @property
    def health_status(self) -> str:
        """Get health assessment of operational alignment."""
        if self.total_pairs_evaluated < 10:
            return "insufficient_data"
        if self.correlation > 0.5 and self.separation_gap > 0.15:
            return "excellent"
        if self.correlation > 0.3 and self.separation_gap > 0.1:
            return "good"
        if self.correlation > 0.1:
            return "learning"
        if self.correlation > -0.1:
            return "neutral"
        return "misaligned"


@dataclass
class CentroidDriftMetrics:
    """Track how centroids evolve relative to operational equivalence.

    If MCTS is working:
    - same_op_distance_trend should be NEGATIVE (converging)
    - diff_op_distance_trend should be POSITIVE (diverging)
    """
    # Current state
    avg_same_op_centroid_distance: float = 0.0  # Distance between operationally equivalent sig centroids
    avg_diff_op_centroid_distance: float = 0.0  # Distance between operationally different sig centroids

    # Trends (positive = increasing, negative = decreasing)
    same_op_distance_trend: float = 0.0  # Should be NEGATIVE if learning
    diff_op_distance_trend: float = 0.0  # Should be POSITIVE if learning

    # Ratio: same_op / diff_op (should DECREASE over training)
    distance_ratio: float = 0.0
    distance_ratio_trend: float = 0.0  # Should be NEGATIVE

    # Sample sizes
    same_op_pairs_measured: int = 0
    diff_op_pairs_measured: int = 0

    computed_at: str = ""

    @property
    def is_converging_correctly(self) -> bool:
        """Check if centroids are moving in the right direction."""
        return self.same_op_distance_trend < 0 and self.diff_op_distance_trend > 0


@dataclass
class RoutingOutcome:
    """Record of a routing decision and its operational outcome."""
    signature_id: int
    step_text: str
    embedding_similarity: float  # Cosine similarity to signature centroid
    was_correct: bool  # Did this produce the correct answer?
    dsl_type: str  # What operation was attempted
    problem_id: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# ALIGNMENT TRACKER
# =============================================================================


class OperationalAlignmentTracker:
    """Tracks and computes operational alignment metrics over time.

    Usage:
        tracker = OperationalAlignmentTracker(db_path)

        # Record routing outcomes as they happen
        tracker.record_outcome(sig_id, step_text, similarity, was_correct, dsl_type)

        # Compute current alignment score
        score = tracker.compute_alignment_score()

        # Get trend over time
        history = tracker.get_alignment_history(days=7)
    """

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

    CREATE TABLE IF NOT EXISTS operational_alignment_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        correlation REAL NOT NULL,
        same_op_avg_similarity REAL,
        diff_op_avg_similarity REAL,
        separation_gap REAL,
        high_sim_wrong_op_count INTEGER,
        low_sim_right_op_count INTEGER,
        misfire_ratio REAL,
        total_pairs_evaluated INTEGER,
        same_op_pairs INTEGER,
        diff_op_pairs INTEGER,
        computed_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_oas_computed ON operational_alignment_snapshots(computed_at);
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
            logger.debug("[alignment] Schema ensured")
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
        """Record a routing outcome for alignment tracking.

        Call this after every routing decision that gets an operational result.

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
            row_id = cursor.lastrowid
            logger.debug(
                "[alignment] Recorded: sig=%d sim=%.3f correct=%s",
                signature_id, embedding_similarity, was_correct
            )
            return row_id
        finally:
            conn.close()

    def compute_alignment_score(
        self,
        window_hours: int = 24,
        similarity_threshold: float = 0.7,
    ) -> OperationalAlignmentScore:
        """Compute operational alignment score from recent outcomes.

        This is the core validation metric. It measures:
        1. Correlation between embedding similarity and operational correctness
        2. Separation gap between same-op and different-op similarities
        3. Misfire rates (high-sim wrong vs low-sim right)

        Args:
            window_hours: How far back to look for outcomes
            similarity_threshold: Threshold for "high similarity" classification

        Returns:
            OperationalAlignmentScore with computed metrics
        """
        conn = self._get_connection()
        try:
            # Get outcomes grouped by signature to find operational patterns
            # Two steps routed to same signature = same operation
            # Two steps routed to different signatures = potentially different operation

            cutoff = datetime.now(timezone.utc)
            cutoff_iso = cutoff.isoformat()

            # Fetch recent outcomes
            cursor = conn.execute(
                """SELECT signature_id, embedding_similarity, was_correct, dsl_type
                   FROM operational_alignment_outcomes
                   WHERE created_at >= datetime(?, '-' || ? || ' hours')
                   ORDER BY signature_id, created_at""",
                (cutoff_iso, window_hours)
            )
            rows = cursor.fetchall()

            if len(rows) < 10:
                return OperationalAlignmentScore(
                    total_pairs_evaluated=len(rows),
                    computed_at=cutoff_iso,
                )

            # Group by signature
            sig_outcomes: dict[int, list[tuple[float, bool]]] = {}
            for row in rows:
                sig_id = row["signature_id"]
                sim = row["embedding_similarity"]
                correct = bool(row["was_correct"])
                if sig_id not in sig_outcomes:
                    sig_outcomes[sig_id] = []
                sig_outcomes[sig_id].append((sim, correct))

            # Compute metrics
            same_op_sims = []  # Similarities within same signature (same operation)
            same_op_correct = []
            diff_op_sims = []  # Similarities across different signatures

            high_sim_wrong = 0
            low_sim_right = 0
            high_sim_right = 0

            # Within-signature analysis (same operation)
            for sig_id, outcomes in sig_outcomes.items():
                for sim, correct in outcomes:
                    same_op_sims.append(sim)
                    same_op_correct.append(correct)

                    # Misfire tracking
                    if sim >= similarity_threshold:
                        if correct:
                            high_sim_right += 1
                        else:
                            high_sim_wrong += 1
                    else:
                        if correct:
                            low_sim_right += 1

            # Cross-signature analysis (different operations)
            # Compare average similarity of correct vs incorrect outcomes
            sig_ids = list(sig_outcomes.keys())
            for i, sig1 in enumerate(sig_ids):
                for sig2 in sig_ids[i+1:]:
                    # Different signatures = different operations
                    # Average the similarities
                    sims1 = [s for s, _ in sig_outcomes[sig1]]
                    sims2 = [s for s, _ in sig_outcomes[sig2]]
                    if sims1 and sims2:
                        avg_sim = (sum(sims1)/len(sims1) + sum(sims2)/len(sims2)) / 2
                        diff_op_sims.append(avg_sim)

            # Compute correlation between similarity and correctness
            # Using point-biserial correlation approximation
            correlation = self._compute_point_biserial(same_op_sims, same_op_correct)

            # Compute averages
            same_op_avg = sum(same_op_sims) / len(same_op_sims) if same_op_sims else 0
            diff_op_avg = sum(diff_op_sims) / len(diff_op_sims) if diff_op_sims else 0
            separation_gap = same_op_avg - diff_op_avg

            # Compute misfire ratio
            total_high_sim = high_sim_wrong + high_sim_right
            misfire_ratio = high_sim_wrong / total_high_sim if total_high_sim > 0 else 0

            score = OperationalAlignmentScore(
                correlation=correlation,
                same_op_avg_similarity=same_op_avg,
                diff_op_avg_similarity=diff_op_avg,
                separation_gap=separation_gap,
                high_sim_wrong_op_count=high_sim_wrong,
                low_sim_right_op_count=low_sim_right,
                misfire_ratio=misfire_ratio,
                total_pairs_evaluated=len(rows),
                same_op_pairs=len(same_op_sims),
                diff_op_pairs=len(diff_op_sims),
                computed_at=cutoff_iso,
            )

            # Save snapshot for trend analysis
            self._save_snapshot(conn, score)

            return score

        finally:
            conn.close()

    def _compute_point_biserial(
        self,
        similarities: list[float],
        correctness: list[bool]
    ) -> float:
        """Compute point-biserial correlation between similarity and correctness.

        This measures how well embedding similarity predicts operational correctness.
        Range: -1 to 1, higher = better alignment.
        """
        if len(similarities) < 2:
            return 0.0

        n = len(similarities)
        correct_sims = [s for s, c in zip(similarities, correctness) if c]
        incorrect_sims = [s for s, c in zip(similarities, correctness) if not c]

        if not correct_sims or not incorrect_sims:
            return 0.0

        n1 = len(correct_sims)
        n0 = len(incorrect_sims)

        m1 = sum(correct_sims) / n1
        m0 = sum(incorrect_sims) / n0

        # Overall std dev
        mean_all = sum(similarities) / n
        var_all = sum((s - mean_all) ** 2 for s in similarities) / n
        std_all = math.sqrt(var_all) if var_all > 0 else 1e-10

        # Point-biserial formula
        r_pb = ((m1 - m0) / std_all) * math.sqrt((n1 * n0) / (n * n))

        return max(-1.0, min(1.0, r_pb))  # Clamp to valid range

    def _save_snapshot(self, conn: sqlite3.Connection, score: OperationalAlignmentScore):
        """Save alignment snapshot for historical tracking."""
        try:
            conn.execute(
                """INSERT INTO operational_alignment_snapshots
                   (correlation, same_op_avg_similarity, diff_op_avg_similarity,
                    separation_gap, high_sim_wrong_op_count, low_sim_right_op_count,
                    misfire_ratio, total_pairs_evaluated, same_op_pairs,
                    diff_op_pairs, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (score.correlation, score.same_op_avg_similarity,
                 score.diff_op_avg_similarity, score.separation_gap,
                 score.high_sim_wrong_op_count, score.low_sim_right_op_count,
                 score.misfire_ratio, score.total_pairs_evaluated,
                 score.same_op_pairs, score.diff_op_pairs, score.computed_at)
            )
            conn.commit()
        except Exception as e:
            logger.warning("[alignment] Failed to save snapshot: %s", e)

    def get_alignment_history(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> list[OperationalAlignmentScore]:
        """Get historical alignment scores for trend analysis.

        Args:
            days: How many days of history to retrieve
            limit: Maximum number of snapshots

        Returns:
            List of OperationalAlignmentScore ordered by time (oldest first)
        """
        conn = self._get_connection()
        try:
            cutoff = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """SELECT * FROM operational_alignment_snapshots
                   WHERE computed_at >= datetime(?, '-' || ? || ' days')
                   ORDER BY computed_at ASC
                   LIMIT ?""",
                (cutoff, days, limit)
            )

            history = []
            for row in cursor.fetchall():
                history.append(OperationalAlignmentScore(
                    correlation=row["correlation"],
                    same_op_avg_similarity=row["same_op_avg_similarity"] or 0,
                    diff_op_avg_similarity=row["diff_op_avg_similarity"] or 0,
                    separation_gap=row["separation_gap"] or 0,
                    high_sim_wrong_op_count=row["high_sim_wrong_op_count"] or 0,
                    low_sim_right_op_count=row["low_sim_right_op_count"] or 0,
                    misfire_ratio=row["misfire_ratio"] or 0,
                    total_pairs_evaluated=row["total_pairs_evaluated"] or 0,
                    same_op_pairs=row["same_op_pairs"] or 0,
                    diff_op_pairs=row["diff_op_pairs"] or 0,
                    computed_at=row["computed_at"],
                ))

            return history
        finally:
            conn.close()

    def compute_alignment_trend(self, days: int = 7) -> dict:
        """Compute trend in alignment metrics over time.

        Args:
            days: Window for trend calculation

        Returns:
            Dict with trend metrics:
            - correlation_trend: Change in correlation (positive = improving)
            - separation_trend: Change in separation gap (positive = improving)
            - misfire_trend: Change in misfire ratio (negative = improving)
            - is_improving: Overall assessment
        """
        history = self.get_alignment_history(days=days)

        if len(history) < 2:
            return {
                "correlation_trend": 0.0,
                "separation_trend": 0.0,
                "misfire_trend": 0.0,
                "is_improving": None,
                "data_points": len(history),
            }

        # Simple linear regression slope for each metric
        n = len(history)
        x = list(range(n))
        x_mean = sum(x) / n

        def compute_slope(values: list[float]) -> float:
            if not values or len(values) < 2:
                return 0.0
            y_mean = sum(values) / len(values)
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
            denominator = sum((xi - x_mean) ** 2 for xi in x)
            return numerator / denominator if denominator > 0 else 0.0

        correlation_trend = compute_slope([h.correlation for h in history])
        separation_trend = compute_slope([h.separation_gap for h in history])
        misfire_trend = compute_slope([h.misfire_ratio for h in history])

        # Improving if correlation increasing AND misfire decreasing
        is_improving = correlation_trend > 0.01 and misfire_trend < 0

        return {
            "correlation_trend": correlation_trend,
            "separation_trend": separation_trend,
            "misfire_trend": misfire_trend,
            "is_improving": is_improving,
            "data_points": n,
            "first_correlation": history[0].correlation,
            "last_correlation": history[-1].correlation,
            "correlation_change": history[-1].correlation - history[0].correlation,
        }


# =============================================================================
# CENTROID DRIFT ANALYZER
# =============================================================================


class CentroidDriftAnalyzer:
    """Analyze how centroids drift relative to operational equivalence.

    Tracks whether:
    - Same-operation signatures are converging (good)
    - Different-operation signatures are diverging (good)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        configure_connection(conn, enable_foreign_keys=False)
        return conn

    def compute_centroid_distances(self) -> CentroidDriftMetrics:
        """Compute current centroid distance metrics.

        Groups signatures by their operational success patterns to determine
        which signatures are "operationally equivalent" even if they have
        different DSL scripts.

        Returns:
            CentroidDriftMetrics with current state
        """
        conn = self._get_connection()
        try:
            # Get signatures with centroids and success patterns
            cursor = conn.execute(
                """SELECT s.id, s.centroid, s.dsl_type, s.uses, s.successes,
                          s.is_semantic_umbrella
                   FROM step_signatures s
                   WHERE s.centroid IS NOT NULL
                     AND s.uses >= 3
                     AND s.is_semantic_umbrella = 0"""
            )

            signatures = []
            for row in cursor.fetchall():
                try:
                    centroid = np.array(json.loads(row["centroid"]))
                    signatures.append({
                        "id": row["id"],
                        "centroid": centroid,
                        "dsl_type": row["dsl_type"],
                        "success_rate": row["successes"] / row["uses"] if row["uses"] > 0 else 0,
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

            if len(signatures) < 2:
                return CentroidDriftMetrics()

            # Group by dsl_type (proxy for operation type)
            by_dsl_type: dict[str, list[dict]] = {}
            for sig in signatures:
                dsl = sig["dsl_type"] or "unknown"
                if dsl not in by_dsl_type:
                    by_dsl_type[dsl] = []
                by_dsl_type[dsl].append(sig)

            # Compute same-operation distances (within same dsl_type)
            same_op_distances = []
            for dsl_type, sigs in by_dsl_type.items():
                if len(sigs) < 2:
                    continue
                for i, sig1 in enumerate(sigs):
                    for sig2 in sigs[i+1:]:
                        dist = np.linalg.norm(sig1["centroid"] - sig2["centroid"])
                        same_op_distances.append(dist)

            # Compute different-operation distances (across dsl_types)
            diff_op_distances = []
            dsl_types = list(by_dsl_type.keys())
            for i, dsl1 in enumerate(dsl_types):
                for dsl2 in dsl_types[i+1:]:
                    for sig1 in by_dsl_type[dsl1]:
                        for sig2 in by_dsl_type[dsl2]:
                            dist = np.linalg.norm(sig1["centroid"] - sig2["centroid"])
                            diff_op_distances.append(dist)

            avg_same = sum(same_op_distances) / len(same_op_distances) if same_op_distances else 0
            avg_diff = sum(diff_op_distances) / len(diff_op_distances) if diff_op_distances else 0

            return CentroidDriftMetrics(
                avg_same_op_centroid_distance=avg_same,
                avg_diff_op_centroid_distance=avg_diff,
                distance_ratio=avg_same / avg_diff if avg_diff > 0 else 0,
                same_op_pairs_measured=len(same_op_distances),
                diff_op_pairs_measured=len(diff_op_distances),
                computed_at=datetime.now(timezone.utc).isoformat(),
            )

        finally:
            conn.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_operational_alignment_score(db_path: str, window_hours: int = 24) -> OperationalAlignmentScore:
    """Convenience function to compute alignment score.

    Args:
        db_path: Path to SQLite database
        window_hours: How far back to look

    Returns:
        OperationalAlignmentScore
    """
    tracker = OperationalAlignmentTracker(db_path)
    return tracker.compute_alignment_score(window_hours=window_hours)


def get_alignment_trend(db_path: str, days: int = 7) -> dict:
    """Convenience function to get alignment trend.

    Args:
        db_path: Path to SQLite database
        days: Window for trend calculation

    Returns:
        Dict with trend metrics
    """
    tracker = OperationalAlignmentTracker(db_path)
    return tracker.compute_alignment_trend(days=days)


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

    Call this from solver.py after each routing decision.

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


def print_alignment_report(db_path: str, window_hours: int = 24, trend_days: int = 7):
    """Print a human-readable alignment report.

    Args:
        db_path: Path to SQLite database
        window_hours: Window for current score
        trend_days: Window for trend analysis
    """
    tracker = OperationalAlignmentTracker(db_path)
    score = tracker.compute_alignment_score(window_hours=window_hours)
    trend = tracker.compute_alignment_trend(days=trend_days)

    drift_analyzer = CentroidDriftAnalyzer(db_path)
    drift = drift_analyzer.compute_centroid_distances()

    print("\n" + "=" * 60)
    print("OPERATIONAL ALIGNMENT REPORT")
    print("=" * 60)

    print(f"\n[Current Score - Last {window_hours}h]")
    print(f"  Correlation (sim vs correctness): {score.correlation:.3f}")
    print(f"  Same-op avg similarity:           {score.same_op_avg_similarity:.3f}")
    print(f"  Diff-op avg similarity:           {score.diff_op_avg_similarity:.3f}")
    print(f"  Separation gap:                   {score.separation_gap:.3f}")
    print(f"  Misfire ratio:                    {score.misfire_ratio:.3f}")
    print(f"  Health status:                    {score.health_status}")
    print(f"  Pairs evaluated:                  {score.total_pairs_evaluated}")

    print(f"\n[Trend - Last {trend_days} days]")
    print(f"  Correlation trend:    {trend['correlation_trend']:+.4f}/snapshot")
    print(f"  Separation trend:     {trend['separation_trend']:+.4f}/snapshot")
    print(f"  Misfire trend:        {trend['misfire_trend']:+.4f}/snapshot")
    print(f"  Is improving:         {trend['is_improving']}")
    print(f"  Correlation change:   {trend.get('correlation_change', 0):+.3f}")

    print(f"\n[Centroid Drift]")
    print(f"  Same-op avg distance: {drift.avg_same_op_centroid_distance:.4f}")
    print(f"  Diff-op avg distance: {drift.avg_diff_op_centroid_distance:.4f}")
    print(f"  Distance ratio:       {drift.distance_ratio:.4f} (lower = better)")

    print("\n" + "=" * 60)

    # Interpretation
    if score.health_status == "excellent":
        print("MCTS rollouts are working well. Embeddings reflect operations.")
    elif score.health_status == "good":
        print("MCTS is learning. Embeddings are starting to reflect operations.")
    elif score.health_status == "learning":
        print("Early stages. Need more training data for embeddings to align.")
    elif score.health_status == "neutral":
        print("No clear signal yet. Embeddings not yet distinguishing operations.")
    elif score.health_status == "misaligned":
        print("WARNING: Embeddings may be clustering by vocabulary, not operations.")
    else:
        print("Insufficient data. Need more routing outcomes to evaluate.")

    print("=" * 60 + "\n")
