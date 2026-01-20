"""MCTS Wave Function Data Access Layer.

Data access functions for the MCTS tables:
- mcts_dags: Problem-level tracking
- mcts_dag_steps: Individual plan steps
- mcts_threads: MCTS rollout paths
- mcts_thread_steps: Fact table with amplitude for post-mortem analysis

Per ideas.md: "The combination of dag_step_id and node_id is what we're learning"
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from mycelium.data_layer import get_db

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MCTSDag:
    """A problem and its decomposition plan."""
    dag_id: str
    problem_id: str
    problem_desc: Optional[str] = None
    benchmark: Optional[str] = None
    difficulty_level: Optional[float] = None
    success: Optional[int] = None  # NULL until graded
    ground_truth: Optional[str] = None
    created_at: Optional[str] = None
    graded_at: Optional[str] = None


@dataclass
class MCTSDagStep:
    """Individual step in a decomposition plan."""
    dag_step_id: str
    dag_id: str
    step_desc: str
    step_num: int
    branch_num: int = 1
    is_atomic: int = 0
    created_at: Optional[str] = None


@dataclass
class MCTSThread:
    """A single MCTS rollout path."""
    thread_id: str
    dag_id: str
    parent_thread_id: Optional[str] = None
    fork_at_step: Optional[str] = None
    fork_reason: Optional[str] = None  # 'undecided', 'explore', 'top_k'
    final_answer: Optional[str] = None
    success: Optional[int] = None  # NULL until graded
    created_at: Optional[str] = None
    graded_at: Optional[str] = None


@dataclass
class MCTSThreadStep:
    """Fact table entry for MCTS rollouts with wave function amplitude."""
    thread_step_id: str
    thread_id: str
    dag_id: str
    dag_step_id: str
    node_id: int  # step_signatures.id
    amplitude: float = 1.0
    amplitude_post: Optional[float] = None
    similarity_score: Optional[float] = None
    was_undecided: int = 0
    ucb1_gap: Optional[float] = None
    alternatives_considered: int = 1
    step_result: Optional[str] = None
    step_success: Optional[int] = None
    created_at: Optional[str] = None


# =============================================================================
# DAG FUNCTIONS
# =============================================================================


def create_dag(
    problem_id: str,
    problem_desc: Optional[str] = None,
    benchmark: Optional[str] = None,
    difficulty_level: Optional[float] = None,
    ground_truth: Optional[str] = None,
) -> str:
    """Create a new MCTS DAG for a problem.

    Returns the dag_id.
    """
    dag_id = f"dag-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO mcts_dags (dag_id, problem_id, problem_desc, benchmark,
                               difficulty_level, ground_truth, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (dag_id, problem_id, problem_desc, benchmark, difficulty_level, ground_truth, now),
    )
    conn.commit()

    logger.debug("[mcts] Created DAG %s for problem %s", dag_id, problem_id[:30])
    return dag_id


def grade_dag(dag_id: str, success: bool) -> None:
    """Update DAG with grading result."""
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        UPDATE mcts_dags SET success = ?, graded_at = ? WHERE dag_id = ?
        """,
        (1 if success else 0, now, dag_id),
    )
    conn.commit()


# =============================================================================
# DAG STEP FUNCTIONS
# =============================================================================


def create_dag_steps(dag_id: str, steps: list[tuple[str, int, int, bool]]) -> list[str]:
    """Create DAG steps for a plan.

    Args:
        dag_id: Parent DAG ID
        steps: List of (step_desc, step_num, branch_num, is_atomic)

    Returns:
        List of dag_step_ids
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()

    step_ids = []
    for step_desc, step_num, branch_num, is_atomic in steps:
        dag_step_id = f"step-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO mcts_dag_steps (dag_step_id, dag_id, step_desc, step_num,
                                        branch_num, is_atomic, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (dag_step_id, dag_id, step_desc, step_num, branch_num, 1 if is_atomic else 0, now),
        )
        step_ids.append(dag_step_id)

    conn.commit()
    logger.debug("[mcts] Created %d DAG steps for %s", len(steps), dag_id)
    return step_ids


# =============================================================================
# THREAD FUNCTIONS
# =============================================================================


def create_thread(
    dag_id: str,
    parent_thread_id: Optional[str] = None,
    fork_at_step: Optional[str] = None,
    fork_reason: Optional[str] = None,
) -> str:
    """Create a new MCTS thread.

    Returns the thread_id.
    """
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO mcts_threads (thread_id, dag_id, parent_thread_id,
                                  fork_at_step, fork_reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, dag_id, parent_thread_id, fork_at_step, fork_reason, now),
    )
    conn.commit()

    logger.debug("[mcts] Created thread %s (parent=%s)", thread_id, parent_thread_id)
    return thread_id


def complete_thread(thread_id: str, final_answer: str, success: Optional[bool] = None) -> None:
    """Update thread with final answer and optional grading."""
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    if success is not None:
        conn.execute(
            """
            UPDATE mcts_threads
            SET final_answer = ?, success = ?, graded_at = ?
            WHERE thread_id = ?
            """,
            (final_answer, 1 if success else 0, now, thread_id),
        )
    else:
        conn.execute(
            """
            UPDATE mcts_threads SET final_answer = ? WHERE thread_id = ?
            """,
            (final_answer, thread_id),
        )
    conn.commit()


def grade_thread(thread_id: str, success: bool) -> None:
    """Update thread with grading result."""
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        UPDATE mcts_threads SET success = ?, graded_at = ? WHERE thread_id = ?
        """,
        (1 if success else 0, now, thread_id),
    )
    conn.commit()


# =============================================================================
# THREAD STEP FUNCTIONS (WAVE FUNCTION AMPLITUDE)
# =============================================================================


def log_thread_step(
    thread_id: str,
    dag_id: str,
    dag_step_id: str,
    node_id: int,
    amplitude: float = 1.0,
    similarity_score: Optional[float] = None,
    was_undecided: bool = False,
    ucb1_gap: Optional[float] = None,
    alternatives_considered: int = 1,
    step_result: Optional[str] = None,
    step_success: Optional[bool] = None,
) -> str:
    """Log a thread step execution with wave function amplitude.

    This is the core logging function for MCTS post-mortem analysis.
    The (dag_step_id, node_id) combination is what we're learning.

    Returns the thread_step_id.
    """
    thread_step_id = f"tstep-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO mcts_thread_steps (
            thread_step_id, thread_id, dag_id, dag_step_id, node_id,
            amplitude, similarity_score, was_undecided, ucb1_gap,
            alternatives_considered, step_result, step_success, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_step_id, thread_id, dag_id, dag_step_id, node_id,
            amplitude, similarity_score, 1 if was_undecided else 0, ucb1_gap,
            alternatives_considered, step_result,
            (1 if step_success else 0) if step_success is not None else None,
            now,
        ),
    )
    conn.commit()

    return thread_step_id


def update_amplitude_post(thread_step_id: str, amplitude_post: float) -> None:
    """Update the post-observation amplitude for a thread step.

    Called after wave function collapse (grading).
    """
    conn = get_db()
    conn.execute(
        """
        UPDATE mcts_thread_steps SET amplitude_post = ? WHERE thread_step_id = ?
        """,
        (amplitude_post, thread_step_id),
    )
    conn.commit()


def batch_update_amplitudes(updates: list[tuple[str, float]]) -> None:
    """Batch update amplitude_post values.

    Args:
        updates: List of (thread_step_id, amplitude_post) tuples
    """
    if not updates:
        return

    conn = get_db()
    conn.executemany(
        """
        UPDATE mcts_thread_steps SET amplitude_post = ? WHERE thread_step_id = ?
        """,
        [(amp, tsid) for tsid, amp in updates],
    )
    conn.commit()
    logger.debug("[mcts] Batch updated %d amplitudes", len(updates))


# =============================================================================
# POST-MORTEM ANALYSIS QUERIES
# =============================================================================


def get_thread_steps_for_dag(dag_id: str) -> list[MCTSThreadStep]:
    """Get all thread steps for a DAG (for post-mortem analysis)."""
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT thread_step_id, thread_id, dag_id, dag_step_id, node_id,
               amplitude, amplitude_post, similarity_score, was_undecided,
               ucb1_gap, alternatives_considered, step_result, step_success, created_at
        FROM mcts_thread_steps
        WHERE dag_id = ?
        ORDER BY created_at
        """,
        (dag_id,),
    )

    return [
        MCTSThreadStep(
            thread_step_id=row[0],
            thread_id=row[1],
            dag_id=row[2],
            dag_step_id=row[3],
            node_id=row[4],
            amplitude=row[5],
            amplitude_post=row[6],
            similarity_score=row[7],
            was_undecided=row[8],
            ucb1_gap=row[9],
            alternatives_considered=row[10],
            step_result=row[11],
            step_success=row[12],
            created_at=row[13],
        )
        for row in cursor.fetchall()
    ]


def get_node_step_stats(node_id: int) -> dict:
    """Get aggregate statistics for a node across all (dag_step, node) combinations.

    Returns:
        Dict with success_rate, avg_amplitude, total_uses, etc.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_uses,
            SUM(CASE WHEN ts.step_success = 1 THEN 1 ELSE 0 END) as step_successes,
            AVG(ts.amplitude) as avg_amplitude,
            AVG(ts.amplitude_post) as avg_amplitude_post,
            SUM(ts.was_undecided) as undecided_count
        FROM mcts_thread_steps ts
        WHERE ts.node_id = ?
        """,
        (node_id,),
    )

    row = cursor.fetchone()
    if not row or row[0] == 0:
        return {"total_uses": 0, "success_rate": 0.0, "avg_amplitude": 1.0}

    return {
        "total_uses": row[0],
        "step_successes": row[1] or 0,
        "success_rate": (row[1] or 0) / row[0] if row[0] > 0 else 0.0,
        "avg_amplitude": row[2] or 1.0,
        "avg_amplitude_post": row[3],
        "undecided_count": row[4] or 0,
    }


def get_dag_step_node_performance(dag_step_id: str, node_id: int) -> dict:
    """Get performance for a specific (dag_step_id, node_id) combination.

    This is "what we're learning" - how well does this node perform at this step?
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as uses,
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as thread_wins,
            SUM(CASE WHEN t.success = 0 THEN 1 ELSE 0 END) as thread_losses,
            AVG(ts.amplitude) as avg_amplitude,
            AVG(ts.amplitude_post) as avg_amplitude_post
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE ts.dag_step_id = ? AND ts.node_id = ?
        """,
        (dag_step_id, node_id),
    )

    row = cursor.fetchone()
    if not row or row[0] == 0:
        return {"uses": 0, "win_rate": 0.0}

    total = row[0]
    wins = row[1] or 0
    losses = row[2] or 0
    graded = wins + losses

    return {
        "uses": total,
        "thread_wins": wins,
        "thread_losses": losses,
        "win_rate": wins / graded if graded > 0 else 0.0,
        "avg_amplitude": row[3] or 1.0,
        "avg_amplitude_post": row[4],
    }
