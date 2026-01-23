"""Tests for step-node stats feedback loop.

Per mycelium-wev3: Unit tests for the (dag_step_type, node_id) stats tracking
that closes the feedback loop between post-mortem analysis and routing decisions.
"""

import os
import tempfile
import pytest
import sqlite3
from unittest.mock import patch

from mycelium.data_layer.schema import init_db
from mycelium.data_layer.connection import get_db, reset_db, configure_connection
from mycelium.data_layer.mcts import (
    update_dag_step_node_stats,
    get_dag_step_node_stats_batch,
    get_dag_step_node_stats_single,
    propagate_step_node_stats,
    create_dag,
    create_dag_steps,
    create_thread,
    log_thread_step,
    complete_thread,
    grade_thread,
    run_postmortem,
)


@pytest.fixture
def step_node_db():
    """Create a temp database with schema for step-node stats tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    os.environ["MYCELIUM_DB_PATH"] = path
    reset_db()

    # Initialize the schema
    conn = get_db()
    import sqlite3 as sq
    raw_conn = sq.connect(path)
    init_db(raw_conn)
    raw_conn.close()

    # Re-get connection after schema init
    reset_db()
    conn = get_db()

    # Create a test signature
    conn.execute(
        """
        INSERT INTO step_signatures (signature_id, centroid, step_type, description, created_at)
        VALUES ('sig_001', '[]', 'compute_sum', 'Add two numbers', datetime('now'))
        """
    )
    conn.execute(
        """
        INSERT INTO step_signatures (signature_id, centroid, step_type, description, created_at)
        VALUES ('sig_002', '[]', 'compute_product', 'Multiply two numbers', datetime('now'))
        """
    )

    yield path

    reset_db()
    try:
        os.unlink(path)
    except Exception:
        pass


class TestUpdateDagStepNodeStats:
    """Tests for update_dag_step_node_stats function."""

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_initial_insert_win(self, step_node_db):
        """Test initial insert with a win."""
        from mycelium.config import STEP_NODE_STATS_PRIOR_WINS, STEP_NODE_STATS_PRIOR_USES

        update_dag_step_node_stats(
            dag_step_type="compute_sum",
            node_id=1,
            won=True,
            amplitude_post=0.9,
        )

        conn = get_db()
        cursor = conn.execute(
            "SELECT uses, wins, losses, win_rate, avg_amplitude_post FROM dag_step_node_stats WHERE dag_step_type = ? AND node_id = ?",
            ("compute_sum", 1),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 1  # uses
        assert row[1] == 1  # wins
        assert row[2] == 0  # losses
        assert row[4] == 0.9  # avg_amplitude_post

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_initial_insert_loss(self, step_node_db):
        """Test initial insert with a loss."""
        update_dag_step_node_stats(
            dag_step_type="compute_product",
            node_id=2,
            won=False,
            amplitude_post=0.3,
        )

        conn = get_db()
        cursor = conn.execute(
            "SELECT uses, wins, losses, avg_amplitude_post FROM dag_step_node_stats WHERE dag_step_type = ? AND node_id = ?",
            ("compute_product", 2),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 1  # uses
        assert row[1] == 0  # wins
        assert row[2] == 1  # losses
        assert row[3] == 0.3  # avg_amplitude_post

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_update_increments_counters(self, step_node_db):
        """Test that subsequent updates increment counters correctly."""
        # First insert
        update_dag_step_node_stats(
            dag_step_type="compute_sum",
            node_id=1,
            won=True,
            amplitude_post=0.8,
        )

        # Second update - another win
        update_dag_step_node_stats(
            dag_step_type="compute_sum",
            node_id=1,
            won=True,
            amplitude_post=0.9,
        )

        conn = get_db()
        cursor = conn.execute(
            "SELECT uses, wins, losses FROM dag_step_node_stats WHERE dag_step_type = ? AND node_id = ?",
            ("compute_sum", 1),
        )
        row = cursor.fetchone()

        assert row[0] == 2  # uses
        assert row[1] == 2  # wins
        assert row[2] == 0  # losses

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_update_mixed_outcomes(self, step_node_db):
        """Test updates with mixed win/loss outcomes."""
        # Win
        update_dag_step_node_stats("compute_sum", 1, won=True, amplitude_post=0.9)
        # Loss
        update_dag_step_node_stats("compute_sum", 1, won=False, amplitude_post=0.2)
        # Win
        update_dag_step_node_stats("compute_sum", 1, won=True, amplitude_post=0.85)

        conn = get_db()
        cursor = conn.execute(
            "SELECT uses, wins, losses FROM dag_step_node_stats WHERE dag_step_type = ? AND node_id = ?",
            ("compute_sum", 1),
        )
        row = cursor.fetchone()

        assert row[0] == 3  # uses
        assert row[1] == 2  # wins
        assert row[2] == 1  # losses

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", False)
    def test_disabled_does_nothing(self, step_node_db):
        """Test that disabled stats tracking does nothing."""
        update_dag_step_node_stats(
            dag_step_type="compute_sum",
            node_id=1,
            won=True,
            amplitude_post=0.9,
        )

        conn = get_db()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM dag_step_node_stats"
        )
        assert cursor.fetchone()[0] == 0


class TestGetDagStepNodeStatsBatch:
    """Tests for get_dag_step_node_stats_batch function."""

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_batch_retrieval(self, step_node_db):
        """Test batch retrieval of stats."""
        # Insert stats for multiple nodes
        update_dag_step_node_stats("compute_sum", 1, won=True, amplitude_post=0.9)
        update_dag_step_node_stats("compute_sum", 2, won=False, amplitude_post=0.3)

        result = get_dag_step_node_stats_batch("compute_sum", [1, 2, 3])

        assert 1 in result
        assert 2 in result
        assert 3 not in result  # No stats for node 3

        assert result[1]["wins"] == 1
        assert result[2]["losses"] == 1

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    def test_batch_empty_node_ids(self, step_node_db):
        """Test batch retrieval with empty node_ids list."""
        result = get_dag_step_node_stats_batch("compute_sum", [])
        assert result == {}

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", False)
    def test_batch_disabled(self, step_node_db):
        """Test batch retrieval when disabled returns empty dict."""
        result = get_dag_step_node_stats_batch("compute_sum", [1, 2])
        assert result == {}

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_batch_different_step_types(self, step_node_db):
        """Test that batch only returns stats for the specified step type."""
        update_dag_step_node_stats("compute_sum", 1, won=True, amplitude_post=0.9)
        update_dag_step_node_stats("compute_product", 1, won=False, amplitude_post=0.3)

        sum_stats = get_dag_step_node_stats_batch("compute_sum", [1])
        product_stats = get_dag_step_node_stats_batch("compute_product", [1])

        assert sum_stats[1]["wins"] == 1
        assert sum_stats[1]["losses"] == 0

        assert product_stats[1]["wins"] == 0
        assert product_stats[1]["losses"] == 1


class TestGetDagStepNodeStatsSingle:
    """Tests for get_dag_step_node_stats_single function."""

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_single_retrieval(self, step_node_db):
        """Test single retrieval of stats."""
        update_dag_step_node_stats("compute_sum", 1, won=True, amplitude_post=0.9)

        result = get_dag_step_node_stats_single("compute_sum", 1)

        assert result is not None
        assert result["uses"] == 1
        assert result["wins"] == 1
        assert result["avg_amplitude_post"] == 0.9

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    def test_single_not_found(self, step_node_db):
        """Test single retrieval when no stats exist."""
        result = get_dag_step_node_stats_single("compute_sum", 999)
        assert result is None


class TestPropagateStepNodeStats:
    """Tests for propagate_step_node_stats function."""

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_propagation_end_to_end(self, step_node_db):
        """Test end-to-end propagation from thread_steps to dag_step_node_stats."""
        conn = get_db()

        # Create a DAG (returns dag_id string)
        dag_id = create_dag(
            problem_id="test_problem_001",
            problem_desc="Test problem",
            benchmark="test",
            ground_truth="42",
        )

        # Create DAG steps - format is (step_desc, step_num, branch_num, is_atomic, dsl_hint)
        steps = [
            ("compute_sum", 1, 1, True, None),
            ("compute_product", 2, 1, True, None),
        ]
        create_dag_steps(dag_id, steps)

        # Get dag_step_ids
        cursor = conn.execute(
            "SELECT dag_step_id, step_desc FROM mcts_dag_steps WHERE dag_id = ? ORDER BY step_num",
            (dag_id,)
        )
        dag_steps = cursor.fetchall()

        # Create a thread (returns thread_id string)
        thread_id = create_thread(dag_id, parent_thread_id=None, fork_at_step=None)

        # Log thread steps
        log_thread_step(
            thread_id=thread_id,
            dag_id=dag_id,
            dag_step_id=dag_steps[0][0],
            node_id=1,
            amplitude=0.9,
            similarity_score=0.85,
        )
        log_thread_step(
            thread_id=thread_id,
            dag_id=dag_id,
            dag_step_id=dag_steps[1][0],
            node_id=2,
            amplitude=0.8,
            similarity_score=0.82,
        )

        # Complete and grade the thread
        complete_thread(thread_id, final_answer="42")
        grade_thread(thread_id, success=True)

        # Run post-mortem to compute amplitude_post
        run_postmortem(dag_id)

        # Propagate to step-node stats
        result = propagate_step_node_stats(dag_id)

        assert result["pairs_updated"] == 2

        # Verify stats were created
        sum_stats = get_dag_step_node_stats_single("compute_sum", 1)
        product_stats = get_dag_step_node_stats_single("compute_product", 2)

        assert sum_stats is not None
        assert sum_stats["uses"] == 1
        assert sum_stats["wins"] == 1  # Thread won

        assert product_stats is not None
        assert product_stats["uses"] == 1
        assert product_stats["wins"] == 1  # Thread won

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", False)
    def test_propagation_disabled(self, step_node_db):
        """Test propagation when disabled returns skipped."""
        result = propagate_step_node_stats("any_dag_id")

        assert result["pairs_updated"] == 0
        assert result["skipped"] is True


class TestStepNodeStatsIntegration:
    """Integration tests for the full feedback loop."""

    @patch("mycelium.config.STEP_NODE_STATS_ENABLED", True)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_WINS", 1)
    @patch("mycelium.config.STEP_NODE_STATS_PRIOR_USES", 2)
    def test_multiple_threads_same_step(self, step_node_db):
        """Test that stats accumulate correctly across multiple threads."""
        conn = get_db()

        # Create a DAG (returns dag_id string)
        dag_id = create_dag(
            problem_id="test_problem_002",
            problem_desc="Test problem 2",
            benchmark="test",
            ground_truth="100",
        )

        # Create DAG step - format is (step_desc, step_num, branch_num, is_atomic, dsl_hint)
        steps = [("compute_multiply", 1, 1, True, None)]
        create_dag_steps(dag_id, steps)

        cursor = conn.execute(
            "SELECT dag_step_id FROM mcts_dag_steps WHERE dag_id = ?",
            (dag_id,)
        )
        dag_step_id = cursor.fetchone()[0]

        # Create first thread (wins)
        thread1_id = create_thread(dag_id, parent_thread_id=None, fork_at_step=None)
        log_thread_step(thread1_id, dag_id, dag_step_id, node_id=1, amplitude=0.9, similarity_score=0.85)
        complete_thread(thread1_id, final_answer="100")
        grade_thread(thread1_id, success=True)

        # Create second thread (loses)
        thread2_id = create_thread(dag_id, parent_thread_id=None, fork_at_step=None)
        log_thread_step(thread2_id, dag_id, dag_step_id, node_id=1, amplitude=0.7, similarity_score=0.80)
        complete_thread(thread2_id, final_answer="50")
        grade_thread(thread2_id, success=False)

        # Run post-mortem and propagate
        run_postmortem(dag_id)
        propagate_step_node_stats(dag_id)

        # Check accumulated stats
        stats = get_dag_step_node_stats_single("compute_multiply", 1)

        assert stats is not None
        assert stats["uses"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
