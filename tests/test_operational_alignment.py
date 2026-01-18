"""Tests for operational alignment validation metrics.

Tests that MCTS rollouts are actually helping distinguish operations from vocabulary.
"""

import os
import tempfile
import pytest
from datetime import datetime, timezone

from mycelium.step_signatures.operational_alignment import (
    OperationalAlignmentScore,
    CentroidDriftMetrics,
    OperationalAlignmentTracker,
    CentroidDriftAnalyzer,
    compute_operational_alignment_score,
    get_alignment_trend,
    record_routing_outcome,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestOperationalAlignmentScore:
    """Tests for the OperationalAlignmentScore dataclass."""

    def test_default_values(self):
        """Test default score values."""
        score = OperationalAlignmentScore()
        assert score.correlation == 0.0
        assert score.separation_gap == 0.0
        assert score.total_pairs_evaluated == 0
        assert score.computed_at != ""

    def test_is_aligned_property(self):
        """Test is_aligned property."""
        # Not aligned: low correlation and gap
        score = OperationalAlignmentScore(correlation=0.1, separation_gap=0.05)
        assert not score.is_aligned

        # Aligned: high correlation and gap
        score = OperationalAlignmentScore(correlation=0.5, separation_gap=0.15)
        assert score.is_aligned

    def test_health_status_insufficient_data(self):
        """Test health status with insufficient data."""
        score = OperationalAlignmentScore(total_pairs_evaluated=5)
        assert score.health_status == "insufficient_data"

    def test_health_status_excellent(self):
        """Test excellent health status."""
        score = OperationalAlignmentScore(
            correlation=0.6,
            separation_gap=0.2,
            total_pairs_evaluated=100,
        )
        assert score.health_status == "excellent"

    def test_health_status_good(self):
        """Test good health status."""
        score = OperationalAlignmentScore(
            correlation=0.35,
            separation_gap=0.12,
            total_pairs_evaluated=100,
        )
        assert score.health_status == "good"

    def test_health_status_learning(self):
        """Test learning health status."""
        score = OperationalAlignmentScore(
            correlation=0.2,
            separation_gap=0.05,
            total_pairs_evaluated=100,
        )
        assert score.health_status == "learning"

    def test_health_status_neutral(self):
        """Test neutral health status."""
        score = OperationalAlignmentScore(
            correlation=0.0,
            separation_gap=0.0,
            total_pairs_evaluated=100,
        )
        assert score.health_status == "neutral"

    def test_health_status_misaligned(self):
        """Test misaligned health status."""
        score = OperationalAlignmentScore(
            correlation=-0.3,
            separation_gap=-0.1,
            total_pairs_evaluated=100,
        )
        assert score.health_status == "misaligned"


class TestCentroidDriftMetrics:
    """Tests for CentroidDriftMetrics dataclass."""

    def test_is_converging_correctly(self):
        """Test the convergence check."""
        # Correct: same_op decreasing, diff_op increasing
        metrics = CentroidDriftMetrics(
            same_op_distance_trend=-0.1,
            diff_op_distance_trend=0.1,
        )
        assert metrics.is_converging_correctly

        # Wrong: both increasing
        metrics = CentroidDriftMetrics(
            same_op_distance_trend=0.1,
            diff_op_distance_trend=0.1,
        )
        assert not metrics.is_converging_correctly


class TestOperationalAlignmentTracker:
    """Tests for the OperationalAlignmentTracker class."""

    def test_schema_creation(self, temp_db):
        """Test that schema is created on init."""
        tracker = OperationalAlignmentTracker(temp_db)
        conn = tracker._get_connection()
        try:
            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'operational_alignment%'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert "operational_alignment_outcomes" in tables
            assert "operational_alignment_snapshots" in tables
        finally:
            conn.close()

    def test_record_outcome(self, temp_db):
        """Test recording a routing outcome."""
        tracker = OperationalAlignmentTracker(temp_db)
        row_id = tracker.record_outcome(
            signature_id=1,
            step_text="Calculate x + y",
            embedding_similarity=0.85,
            was_correct=True,
            dsl_type="math",
            problem_id="test_problem_1",
        )
        assert row_id is not None
        assert row_id > 0

    def test_compute_alignment_score_insufficient_data(self, temp_db):
        """Test alignment score with insufficient data."""
        tracker = OperationalAlignmentTracker(temp_db)

        # Record only a few outcomes
        for i in range(5):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"step {i}",
                embedding_similarity=0.8,
                was_correct=True,
                dsl_type="math",
            )

        score = tracker.compute_alignment_score()
        assert score.total_pairs_evaluated == 5
        # With insufficient data, correlation should be 0
        assert score.correlation == 0.0

    def test_compute_alignment_score_with_data(self, temp_db):
        """Test alignment score with sufficient data."""
        tracker = OperationalAlignmentTracker(temp_db)

        # Record outcomes that should show correlation
        # High similarity + correct
        for i in range(10):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"high_sim_correct_{i}",
                embedding_similarity=0.9,
                was_correct=True,
                dsl_type="math",
            )

        # Low similarity + incorrect
        for i in range(10):
            tracker.record_outcome(
                signature_id=2,
                step_text=f"low_sim_incorrect_{i}",
                embedding_similarity=0.3,
                was_correct=False,
                dsl_type="math",
            )

        score = tracker.compute_alignment_score()
        assert score.total_pairs_evaluated == 20

        # With this pattern, correlation should be positive
        # (high sim correlates with correct)
        assert score.correlation > 0

    def test_get_alignment_history_empty(self, temp_db):
        """Test history retrieval with no data."""
        tracker = OperationalAlignmentTracker(temp_db)
        history = tracker.get_alignment_history()
        assert history == []

    def test_compute_alignment_trend_insufficient_data(self, temp_db):
        """Test trend calculation with insufficient data."""
        tracker = OperationalAlignmentTracker(temp_db)
        trend = tracker.compute_alignment_trend()
        assert trend["data_points"] == 0
        assert trend["is_improving"] is None

    def test_misfire_tracking(self, temp_db):
        """Test that misfires are tracked correctly."""
        tracker = OperationalAlignmentTracker(temp_db)

        # High similarity but wrong (misfire)
        for i in range(5):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"high_sim_wrong_{i}",
                embedding_similarity=0.9,
                was_correct=False,
                dsl_type="math",
            )

        # High similarity and correct
        for i in range(5):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"high_sim_right_{i}",
                embedding_similarity=0.85,
                was_correct=True,
                dsl_type="math",
            )

        score = tracker.compute_alignment_score(similarity_threshold=0.7)
        assert score.high_sim_wrong_op_count == 5
        # Misfire ratio should be 0.5 (5 wrong out of 10 high-sim)
        assert score.misfire_ratio == pytest.approx(0.5, rel=0.1)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_compute_operational_alignment_score(self, temp_db):
        """Test convenience function."""
        # Record some data first
        tracker = OperationalAlignmentTracker(temp_db)
        for i in range(15):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"step_{i}",
                embedding_similarity=0.8,
                was_correct=True,
                dsl_type="math",
            )

        score = compute_operational_alignment_score(temp_db)
        assert isinstance(score, OperationalAlignmentScore)
        assert score.total_pairs_evaluated == 15

    def test_get_alignment_trend(self, temp_db):
        """Test trend convenience function."""
        trend = get_alignment_trend(temp_db)
        assert isinstance(trend, dict)
        assert "correlation_trend" in trend
        assert "is_improving" in trend

    def test_record_routing_outcome(self, temp_db):
        """Test recording convenience function."""
        row_id = record_routing_outcome(
            db_path=temp_db,
            signature_id=1,
            step_text="test step",
            embedding_similarity=0.75,
            was_correct=True,
            dsl_type="math",
        )
        assert row_id is not None


class TestPointBiserialCorrelation:
    """Tests for the point-biserial correlation calculation."""

    def test_perfect_positive_correlation(self, temp_db):
        """Test with perfect positive correlation."""
        tracker = OperationalAlignmentTracker(temp_db)

        # High similarity always correct
        for i in range(20):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"step_{i}",
                embedding_similarity=0.9 + i * 0.001,
                was_correct=True,
                dsl_type="math",
            )

        # Low similarity always incorrect
        for i in range(20):
            tracker.record_outcome(
                signature_id=2,
                step_text=f"step_low_{i}",
                embedding_similarity=0.1 + i * 0.001,
                was_correct=False,
                dsl_type="math",
            )

        score = tracker.compute_alignment_score()
        # Should have strong positive correlation
        assert score.correlation > 0.5

    def test_negative_correlation(self, temp_db):
        """Test with negative correlation (opposite pattern)."""
        tracker = OperationalAlignmentTracker(temp_db)

        # High similarity always incorrect (misaligned embeddings)
        for i in range(20):
            tracker.record_outcome(
                signature_id=1,
                step_text=f"step_{i}",
                embedding_similarity=0.9,
                was_correct=False,
                dsl_type="math",
            )

        # Low similarity always correct
        for i in range(20):
            tracker.record_outcome(
                signature_id=2,
                step_text=f"step_low_{i}",
                embedding_similarity=0.2,
                was_correct=True,
                dsl_type="math",
            )

        score = tracker.compute_alignment_score()
        # Should have negative correlation (embeddings are misleading)
        assert score.correlation < 0


class TestCentroidDriftAnalyzer:
    """Tests for the CentroidDriftAnalyzer class."""

    def test_compute_distances_empty_db(self, temp_db):
        """Test with empty database."""
        # Need to create the step_signatures table first
        import sqlite3
        conn = sqlite3.connect(temp_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS step_signatures (
                id INTEGER PRIMARY KEY,
                centroid TEXT,
                dsl_type TEXT,
                uses INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                is_semantic_umbrella INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        analyzer = CentroidDriftAnalyzer(temp_db)
        metrics = analyzer.compute_centroid_distances()
        assert metrics.same_op_pairs_measured == 0
        assert metrics.diff_op_pairs_measured == 0
