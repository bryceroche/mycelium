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
from mycelium.config import (
    POSTMORTEM_ENABLED,
    POSTMORTEM_HIGH_CONF_THRESHOLD,
    POSTMORTEM_REINFORCE_MULT,
    POSTMORTEM_BOOST_MULT,
    POSTMORTEM_MILD_PENALTY_MULT,
    POSTMORTEM_STRONG_PENALTY_MULT,
    POSTMORTEM_AMPLITUDE_MIN,
    POSTMORTEM_AMPLITUDE_MAX,
    INTERFERENCE_ENABLED,
    INTERFERENCE_MIN_CONSTRUCTIVE,
    INTERFERENCE_MIN_DESTRUCTIVE,
    INTERFERENCE_CONSTRUCTIVE_BOOST,
    INTERFERENCE_DESTRUCTIVE_PENALTY,
    POSTMORTEM_DSL_REGEN_ENABLED,
    POSTMORTEM_DSL_REGEN_MIN_HIGH_CONF_WRONG,
    POSTMORTEM_DSL_REGEN_BATCH_SIZE,
)

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
    thread_id: Optional[str] = None,
) -> str:
    """Create a new MCTS thread.

    Args:
        dag_id: Parent DAG being solved
        parent_thread_id: NULL for root thread, else forked from
        fork_at_step: dag_step_id where this thread forked
        fork_reason: Why we branched: 'undecided', 'explore', 'top_k'
        thread_id: Optional thread ID (generated if not provided)

    Returns the thread_id.
    """
    if thread_id is None:
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
    node_depth: Optional[int] = None,
) -> str:
    """Log a thread step execution with wave function amplitude.

    This is the core logging function for MCTS post-mortem analysis.
    The (dag_step_id, node_id) combination is what we're learning.

    Args:
        node_depth: Depth of the signature node in the tree (for post-mortem analysis)

    Returns the thread_step_id.
    """
    thread_step_id = f"tstep-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO mcts_thread_steps (
            thread_step_id, thread_id, dag_id, dag_step_id, node_id, node_depth,
            amplitude, similarity_score, was_undecided, ucb1_gap,
            alternatives_considered, step_result, step_success, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_step_id, thread_id, dag_id, dag_step_id, node_id, node_depth,
            amplitude, similarity_score, 1 if was_undecided else 0, ucb1_gap,
            alternatives_considered, step_result,
            (1 if step_success else 0) if step_success is not None else None,
            now,
        ),
    )

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


def batch_update_amplitudes(updates: list[tuple[str, float]]) -> None:
    """Batch update amplitude_post values.

    Args:
        updates: List of (thread_step_id, amplitude_post) tuples
    """
    if not updates:
        return

    conn = get_db()
    with conn.connection() as raw_conn:
        raw_conn.executemany(
            """
            UPDATE mcts_thread_steps SET amplitude_post = ? WHERE thread_step_id = ?
            """,
            [(amp, tsid) for tsid, amp in updates],
        )
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


# =============================================================================
# POST-MORTEM ANALYSIS (amplitude_post computation)
# =============================================================================


def run_postmortem(dag_id: str) -> dict:
    """Run post-mortem analysis on a completed DAG.

    Computes amplitude_post for each thread_step based on thread outcomes.
    Uses config values for thresholds and multipliers.

    Returns:
        Dict with summary statistics:
        - total_steps: Number of thread_steps processed
        - threads_won: Number of winning threads
        - threads_lost: Number of losing threads
        - high_conf_wrong: Count of high-confidence wrong decisions (red flag)
        - low_conf_right: Count of low-confidence right decisions (opportunity)
    """
    if not POSTMORTEM_ENABLED:
        return {"total_steps": 0, "threads_won": 0, "threads_lost": 0, "skipped": True}

    conn = get_db()

    # Get all thread outcomes for this DAG
    cursor = conn.execute(
        """
        SELECT thread_id, success FROM mcts_threads WHERE dag_id = ?
        """,
        (dag_id,),
    )
    thread_outcomes = {row[0]: row[1] for row in cursor.fetchall()}

    if not thread_outcomes:
        logger.debug("[mcts] No threads found for DAG %s", dag_id)
        return {"total_steps": 0, "threads_won": 0, "threads_lost": 0}

    # Get all thread_steps for this DAG
    cursor = conn.execute(
        """
        SELECT thread_step_id, thread_id, amplitude
        FROM mcts_thread_steps
        WHERE dag_id = ?
        """,
        (dag_id,),
    )
    thread_steps = cursor.fetchall()

    if not thread_steps:
        logger.debug("[mcts] No thread_steps found for DAG %s", dag_id)
        return {"total_steps": 0, "threads_won": 0, "threads_lost": 0}

    # Compute amplitude_post for each step
    updates = []
    stats = {
        "total_steps": len(thread_steps),
        "threads_won": sum(1 for s in thread_outcomes.values() if s == 1),
        "threads_lost": sum(1 for s in thread_outcomes.values() if s == 0),
        "high_conf_wrong": 0,
        "low_conf_right": 0,
        "total_high_conf": 0,  # For UCB1 adjustment (mycelium-nirq)
        "total_low_conf": 0,   # For UCB1 adjustment (mycelium-nirq)
    }

    for thread_step_id, thread_id, amplitude in thread_steps:
        thread_success = thread_outcomes.get(thread_id)

        if thread_success is None:
            # Thread not graded yet, skip
            continue

        amp = amplitude if amplitude is not None else 1.0
        is_high_conf = amp >= POSTMORTEM_HIGH_CONF_THRESHOLD
        won = thread_success == 1

        # Track totals for UCB1 adjustment hit/miss rates
        if is_high_conf:
            stats["total_high_conf"] += 1
        else:
            stats["total_low_conf"] += 1

        # Compute amplitude_post based on outcome × confidence (using config multipliers)
        if won and is_high_conf:
            # Reinforce: confident and right
            amplitude_post = amp * POSTMORTEM_REINFORCE_MULT
        elif won and not is_high_conf:
            # Boost: discovered something (low confidence but right)
            amplitude_post = amp * POSTMORTEM_BOOST_MULT
            stats["low_conf_right"] += 1
        elif not won and not is_high_conf:
            # Mild penalty: uncertain and wrong (expected)
            amplitude_post = amp * POSTMORTEM_MILD_PENALTY_MULT
        else:
            # Strong penalty: confident and wrong (bad signal)
            amplitude_post = amp * POSTMORTEM_STRONG_PENALTY_MULT
            stats["high_conf_wrong"] += 1

        # Clamp to configured range
        amplitude_post = max(POSTMORTEM_AMPLITUDE_MIN, min(POSTMORTEM_AMPLITUDE_MAX, amplitude_post))
        updates.append((thread_step_id, amplitude_post))

    # Batch update
    if updates:
        batch_update_amplitudes(updates)
        logger.info(
            "[mcts] Post-mortem for DAG %s: %d steps, %d won, %d lost, "
            "%d high-conf-wrong, %d low-conf-right",
            dag_id, stats["total_steps"], stats["threads_won"], stats["threads_lost"],
            stats["high_conf_wrong"], stats["low_conf_right"],
        )

    return stats


def propagate_amplitude_to_signature_stats(dag_id: str, step_db) -> dict:
    """Propagate amplitude_post values to signature stats with partial credit.

    Per beads mycelium-itkn + mycelium-7o8i: Close the loop from post-mortem to
    signature learning, with partial credit for correct steps in failed problems.

    Key insight: In a failed problem, not all steps are wrong. Steps with high
    confidence (amplitude) in a failed thread were probably correct - only the
    step(s) that caused the failure should be blamed.

    Credit logic:
    1. Thread won + any amplitude → full credit (success)
    2. Thread lost + high amplitude (≥ 0.7) → PARTIAL credit (benefit of doubt)
    3. Thread lost + low amplitude (< 0.7) → blame (uncertain and thread failed)

    This prevents good signatures from being punished for a single bad step.

    Args:
        dag_id: The DAG to process
        step_db: StepSignatureDB instance for stat updates

    Returns:
        Dict with propagation statistics
    """
    from mycelium.config import (
        CREDIT_PROPAGATION_ENABLED,
        PARTIAL_CREDIT_HIGH_CONF_THRESHOLD,
        PARTIAL_CREDIT_WEIGHT,
    )

    if not CREDIT_PROPAGATION_ENABLED:
        return {"nodes_processed": 0, "successes_credited": 0, "failures_credited": 0,
                "partial_credits": 0, "skipped": True}

    conn = get_db()

    # Get per-node stats with thread outcome context
    # Per beads mycelium-7o8i: Need to know if step was in winning or losing thread
    cursor = conn.execute(
        """
        SELECT
            ts.node_id,
            COUNT(*) as total_steps,
            -- Steps in winning threads (eligible for full credit)
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as winning_steps,
            -- Steps in losing threads with high confidence (eligible for partial credit)
            SUM(CASE WHEN t.success = 0 AND ts.amplitude >= ? THEN 1 ELSE 0 END) as high_conf_losing_steps,
            -- Steps in losing threads with low confidence (eligible for blame)
            SUM(CASE WHEN t.success = 0 AND ts.amplitude < ? THEN 1 ELSE 0 END) as low_conf_losing_steps,
            -- Average amplitude_post for reference
            AVG(ts.amplitude_post) as avg_amplitude_post
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE ts.dag_id = ? AND ts.amplitude_post IS NOT NULL
        GROUP BY ts.node_id
        """,
        (PARTIAL_CREDIT_HIGH_CONF_THRESHOLD, PARTIAL_CREDIT_HIGH_CONF_THRESHOLD, dag_id),
    )

    stats = {
        "nodes_processed": 0,
        "successes_credited": 0,
        "partial_credits": 0,
        "failures_credited": 0,
    }

    for row in cursor.fetchall():
        node_id, total_steps, winning_steps, high_conf_losing, low_conf_losing, avg_amp = row

        if total_steps == 0:
            continue

        stats["nodes_processed"] += 1

        # 1. Full credit for winning thread steps
        if winning_steps > 0:
            step_db.increment_signature_successes(node_id, count=1)
            stats["successes_credited"] += 1
            logger.debug(
                "[mcts] Full credit to node %d (%d winning steps)",
                node_id, winning_steps
            )

        # 2. Partial credit for high-confidence steps in losing threads
        # Per mycelium-7o8i: These steps were probably correct, just in a bad chain
        elif high_conf_losing > 0:
            # Give partial credit - benefit of the doubt
            step_db.increment_signature_partial_success(node_id, weight=PARTIAL_CREDIT_WEIGHT)
            stats["partial_credits"] += 1
            logger.debug(
                "[mcts] Partial credit to node %d (%d high-conf losing steps, avg_amp=%.2f)",
                node_id, high_conf_losing, avg_amp or 0
            )

        # 3. Blame for low-confidence steps in losing threads
        # These were uncertain AND the thread failed - likely the problem
        elif low_conf_losing > 0:
            step_db.increment_signature_failures(node_id, count=1)
            stats["failures_credited"] += 1
            logger.debug(
                "[mcts] Blamed node %d (%d low-conf losing steps, avg_amp=%.2f)",
                node_id, low_conf_losing, avg_amp or 0
            )

    if stats["nodes_processed"] > 0:
        logger.info(
            "[mcts] Credit propagation for DAG %s: %d nodes, +%d full, +%d partial, +%d failures",
            dag_id, stats["nodes_processed"],
            stats["successes_credited"], stats["partial_credits"], stats["failures_credited"],
        )

    return stats


# =============================================================================
# DIVERGENCE-POINT ANALYSIS
# =============================================================================
# Per beads mycelium-2rss: Compare winning vs losing thread paths to find
# exactly where the problem occurred. Not just the divergence point, but also
# downstream nodes in the losing path that might be the actual culprit.


@dataclass
class ThreadPath:
    """Ordered sequence of (dag_step_id, node_id) for a thread."""
    thread_id: str
    success: Optional[int]  # 1=won, 0=lost, None=ungraded
    steps: list[tuple[str, int]]  # [(dag_step_id, node_id), ...]


@dataclass
class DivergencePoint:
    """Where a winning and losing thread diverged."""
    winning_thread_id: str
    losing_thread_id: str
    shared_prefix_len: int  # Number of identical steps before divergence
    divergence_step_idx: int  # Index where paths diverged
    divergence_dag_step_id: Optional[str]  # dag_step_id where they diverged
    # The actual nodes chosen at divergence (winning chose one, losing chose another)
    winning_node_at_divergence: Optional[int]
    losing_node_at_divergence: Optional[int]
    # The losing thread's suffix (nodes after divergence that might be problematic)
    losing_suffix: list[tuple[str, int]]  # [(dag_step_id, node_id), ...]


def get_thread_paths(dag_id: str) -> list[ThreadPath]:
    """Get ordered step paths for all threads in a DAG.

    Returns threads with their sequence of (dag_step_id, node_id) pairs,
    ordered by step execution (created_at).

    Args:
        dag_id: The DAG to analyze

    Returns:
        List of ThreadPath objects with ordered steps
    """
    conn = get_db()

    # Get thread outcomes
    thread_cursor = conn.execute(
        """
        SELECT thread_id, success FROM mcts_threads WHERE dag_id = ?
        """,
        (dag_id,),
    )
    thread_outcomes = {row[0]: row[1] for row in thread_cursor.fetchall()}

    if not thread_outcomes:
        return []

    # Get all steps for all threads, ordered by created_at within each thread
    steps_cursor = conn.execute(
        """
        SELECT thread_id, dag_step_id, node_id
        FROM mcts_thread_steps
        WHERE dag_id = ?
        ORDER BY thread_id, created_at
        """,
        (dag_id,),
    )

    # Group steps by thread
    thread_steps: dict[str, list[tuple[str, int]]] = {}
    for row in steps_cursor.fetchall():
        thread_id, dag_step_id, node_id = row
        if thread_id not in thread_steps:
            thread_steps[thread_id] = []
        thread_steps[thread_id].append((dag_step_id, node_id))

    # Build ThreadPath objects
    paths = []
    for thread_id, steps in thread_steps.items():
        paths.append(ThreadPath(
            thread_id=thread_id,
            success=thread_outcomes.get(thread_id),
            steps=steps,
        ))

    return paths


def find_divergence_points(dag_id: str) -> list[DivergencePoint]:
    """Find divergence points between winning and losing threads.

    Compares each winning thread against each losing thread to find where
    they share a common prefix but then diverge. This helps identify
    exactly which routing decision led to failure.

    Args:
        dag_id: The DAG to analyze

    Returns:
        List of DivergencePoint objects describing where paths split
    """
    paths = get_thread_paths(dag_id)

    if not paths:
        return []

    # Separate winning and losing threads
    winning = [p for p in paths if p.success == 1]
    losing = [p for p in paths if p.success == 0]

    if not winning or not losing:
        # Need both to find divergence
        logger.debug(
            "[mcts] No divergence analysis for DAG %s: %d winning, %d losing threads",
            dag_id, len(winning), len(losing)
        )
        return []

    divergence_points = []

    for win_path in winning:
        for lose_path in losing:
            # Find common prefix length
            shared_len = 0
            min_len = min(len(win_path.steps), len(lose_path.steps))

            for i in range(min_len):
                if win_path.steps[i] == lose_path.steps[i]:
                    shared_len += 1
                else:
                    break

            # Determine divergence details
            divergence_idx = shared_len
            divergence_dag_step = None
            winning_node = None
            losing_node = None
            losing_suffix = []

            if divergence_idx < len(lose_path.steps):
                # There's a divergence point
                divergence_dag_step = lose_path.steps[divergence_idx][0]
                losing_node = lose_path.steps[divergence_idx][1]

                if divergence_idx < len(win_path.steps):
                    winning_node = win_path.steps[divergence_idx][1]

                # Capture the losing thread's suffix (divergence point + downstream)
                losing_suffix = lose_path.steps[divergence_idx:]

            divergence_points.append(DivergencePoint(
                winning_thread_id=win_path.thread_id,
                losing_thread_id=lose_path.thread_id,
                shared_prefix_len=shared_len,
                divergence_step_idx=divergence_idx,
                divergence_dag_step_id=divergence_dag_step,
                winning_node_at_divergence=winning_node,
                losing_node_at_divergence=losing_node,
                losing_suffix=losing_suffix,
            ))

    logger.debug(
        "[mcts] Found %d divergence points for DAG %s (%d winning × %d losing)",
        len(divergence_points), dag_id, len(winning), len(losing)
    )

    return divergence_points


def assign_divergence_blame(dag_id: str, step_db) -> dict:
    """Assign targeted blame/credit based on divergence analysis.

    Per beads mycelium-2rss: Uses divergence points to assign more precise
    credit/blame:

    1. Shared prefix nodes in losing thread: PARTIAL credit (they were correct)
    2. Divergence point node: PRIMARY blame (this is where it went wrong)
    3. Downstream nodes in losing suffix: SECONDARY blame (might be the real problem)

    The insight is that the divergence point isn't always the root cause -
    sometimes a downstream step in the losing path is the actual problem.

    Args:
        dag_id: The DAG to analyze
        step_db: StepSignatureDB for stat updates

    Returns:
        Dict with divergence blame statistics
    """
    from mycelium.config import PARTIAL_CREDIT_WEIGHT

    divergence_points = find_divergence_points(dag_id)

    if not divergence_points:
        return {
            "divergence_points_found": 0,
            "divergence_blame_assigned": 0,
            "suffix_blame_assigned": 0,
            "shared_prefix_credit": 0,
        }

    stats = {
        "divergence_points_found": len(divergence_points),
        "divergence_blame_assigned": 0,
        "suffix_blame_assigned": 0,
        "shared_prefix_credit": 0,
    }

    # Track which nodes have been credited/blamed to avoid double-counting
    blamed_nodes: set[int] = set()
    credited_nodes: set[int] = set()

    # Pre-fetch all thread paths once (instead of inside loop)
    all_paths = get_thread_paths(dag_id)
    paths_by_thread = {p.thread_id: p for p in all_paths}

    for dp in divergence_points:
        # 1. Give partial credit to shared prefix nodes (they were correct)
        losing_path = paths_by_thread.get(dp.losing_thread_id)

        if losing_path and dp.shared_prefix_len > 0:
            for i in range(dp.shared_prefix_len):
                if i < len(losing_path.steps):
                    node_id = losing_path.steps[i][1]
                    if node_id not in credited_nodes:
                        step_db.increment_signature_partial_success(node_id, weight=PARTIAL_CREDIT_WEIGHT)
                        credited_nodes.add(node_id)
                        stats["shared_prefix_credit"] += 1

        # 2. Primary blame to divergence point
        if dp.losing_node_at_divergence is not None:
            node_id = dp.losing_node_at_divergence
            if node_id not in blamed_nodes:
                step_db.increment_signature_failures(node_id, count=1)
                blamed_nodes.add(node_id)
                stats["divergence_blame_assigned"] += 1
                logger.debug(
                    "[mcts] Divergence blame: node %d at step %s (winning chose %s)",
                    node_id, dp.divergence_dag_step_id,
                    dp.winning_node_at_divergence
                )

        # 3. Secondary blame to downstream suffix nodes
        # Skip first element (that's the divergence point, already blamed)
        if len(dp.losing_suffix) > 1:
            for dag_step_id, node_id in dp.losing_suffix[1:]:
                if node_id not in blamed_nodes and node_id not in credited_nodes:
                    # Give partial blame (0.5 weight) since these might not be
                    # the root cause - they might just be victims of early bad routing
                    step_db.increment_signature_failures(node_id, count=1)
                    blamed_nodes.add(node_id)
                    stats["suffix_blame_assigned"] += 1
                    logger.debug(
                        "[mcts] Suffix blame: node %d at step %s (downstream of divergence)",
                        node_id, dag_step_id
                    )

    if stats["divergence_blame_assigned"] > 0 or stats["suffix_blame_assigned"] > 0:
        logger.info(
            "[mcts] Divergence blame for DAG %s: %d divergence points, "
            "%d primary blame, %d suffix blame, %d prefix credit",
            dag_id, stats["divergence_points_found"],
            stats["divergence_blame_assigned"],
            stats["suffix_blame_assigned"],
            stats["shared_prefix_credit"],
        )

    return stats


def get_problem_nodes_needing_attention(dag_id: str) -> list[dict]:
    """Identify nodes that performed poorly in this DAG.

    Returns nodes where:
    - High confidence but thread lost (amplitude >= threshold, success = 0)

    These are candidates for decomposition or centroid adjustment.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT
            ts.node_id,
            ts.dag_step_id,
            ts.amplitude,
            ts.amplitude_post,
            t.success as thread_success
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE ts.dag_id = ?
          AND ts.amplitude >= ?
          AND t.success = 0
        """,
        (dag_id, POSTMORTEM_HIGH_CONF_THRESHOLD),
    )

    return [
        {
            "node_id": row[0],
            "dag_step_id": row[1],
            "amplitude": row[2],
            "amplitude_post": row[3],
            "thread_success": row[4],
        }
        for row in cursor.fetchall()
    ]


# =============================================================================
# INTERFERENCE PATTERN DETECTION
# =============================================================================
# Per CLAUDE.md: When multiple threads visit the same (dag_step_id, node_id):
# - Constructive interference (both succeed): Reinforce, consider MERGE centroids
# - Destructive interference (mixed results): Signal to SPLIT the cluster


@dataclass
class InterferencePattern:
    """Represents interference when multiple threads visit the same (dag_step_id, node_id)."""
    dag_id: str
    dag_step_id: str
    node_id: int
    thread_count: int  # How many threads visited this combination
    successes: int     # How many threads succeeded
    failures: int      # How many threads failed
    interference_type: str  # 'constructive', 'destructive', or 'neutral'
    avg_amplitude: float  # Average amplitude across threads


def detect_interference_patterns(dag_id: str) -> list[InterferencePattern]:
    """Find interference patterns where multiple threads visited the same (dag_step_id, node_id).

    Per CLAUDE.md:
    - Constructive (all succeed): Reinforce, consider merge
    - Destructive (mixed): Signal to split cluster

    Only returns patterns where thread_count >= 2 (actual interference).

    Returns:
        List of InterferencePattern objects for this DAG
    """
    conn = get_db()

    # Find (dag_step_id, node_id) combinations visited by multiple threads
    cursor = conn.execute(
        """
        SELECT
            ts.dag_step_id,
            ts.node_id,
            COUNT(DISTINCT ts.thread_id) as thread_count,
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as successes,
            SUM(CASE WHEN t.success = 0 THEN 1 ELSE 0 END) as failures,
            AVG(ts.amplitude) as avg_amplitude
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE ts.dag_id = ?
          AND t.success IS NOT NULL  -- Only graded threads
        GROUP BY ts.dag_step_id, ts.node_id
        HAVING COUNT(DISTINCT ts.thread_id) >= 2
        """,
        (dag_id,),
    )

    patterns = []
    for row in cursor.fetchall():
        dag_step_id, node_id, thread_count, successes, failures, avg_amplitude = row
        successes = successes or 0
        failures = failures or 0

        # Classify interference type
        if failures == 0 and successes > 0:
            interference_type = "constructive"  # All succeeded
        elif successes == 0 and failures > 0:
            interference_type = "destructive"   # All failed (coherent failure)
        elif successes > 0 and failures > 0:
            interference_type = "destructive"   # Mixed results = cluster too generic
        else:
            interference_type = "neutral"       # No graded results

        patterns.append(InterferencePattern(
            dag_id=dag_id,
            dag_step_id=dag_step_id,
            node_id=node_id,
            thread_count=thread_count,
            successes=successes,
            failures=failures,
            interference_type=interference_type,
            avg_amplitude=avg_amplitude or 1.0,
        ))

    logger.debug(
        "[mcts] Detected %d interference patterns for DAG %s: %d constructive, %d destructive",
        len(patterns), dag_id,
        sum(1 for p in patterns if p.interference_type == "constructive"),
        sum(1 for p in patterns if p.interference_type == "destructive"),
    )

    return patterns


def get_nodes_for_merge_consideration(min_constructive_count: int = None) -> list[dict]:
    """Find nodes that consistently appear in constructive interference patterns.

    These nodes might be candidates for merging - they represent operations
    that consistently succeed together, suggesting they're operationally similar.

    Args:
        min_constructive_count: Minimum number of constructive interference occurrences
            (defaults to INTERFERENCE_MIN_CONSTRUCTIVE from config)

    Returns:
        List of dicts with node_id and constructive interference stats
    """
    if min_constructive_count is None:
        min_constructive_count = INTERFERENCE_MIN_CONSTRUCTIVE

    conn = get_db()

    # Find nodes that appear together in successful thread runs
    cursor = conn.execute(
        """
        SELECT
            ts.node_id,
            COUNT(*) as total_appearances,
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as success_appearances,
            AVG(ts.amplitude) as avg_amplitude
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE t.success IS NOT NULL
        GROUP BY ts.node_id
        HAVING SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) >= ?
        ORDER BY success_appearances DESC
        """,
        (min_constructive_count,),
    )

    return [
        {
            "node_id": row[0],
            "total_appearances": row[1],
            "success_appearances": row[2],
            "success_rate": row[2] / row[1] if row[1] > 0 else 0.0,
            "avg_amplitude": row[3] or 1.0,
        }
        for row in cursor.fetchall()
    ]


def get_nodes_for_split_consideration(min_destructive_count: int = None) -> list[dict]:
    """Find nodes that consistently appear in destructive interference patterns.

    These nodes are candidates for cluster splitting - they represent operations
    that produce mixed results, suggesting the cluster is too generic.

    Args:
        min_destructive_count: Minimum number of destructive interference occurrences
            (defaults to INTERFERENCE_MIN_DESTRUCTIVE from config)

    Returns:
        List of dicts with node_id, destructive count, and stats
    """
    if min_destructive_count is None:
        min_destructive_count = INTERFERENCE_MIN_DESTRUCTIVE

    conn = get_db()

    # Find nodes with high variance in outcomes (mixed success/failure)
    cursor = conn.execute(
        """
        SELECT
            ts.node_id,
            COUNT(*) as total_appearances,
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as successes,
            SUM(CASE WHEN t.success = 0 THEN 1 ELSE 0 END) as failures,
            AVG(ts.amplitude) as avg_amplitude
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        WHERE t.success IS NOT NULL
        GROUP BY ts.node_id
        HAVING
            SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) >= 1
            AND SUM(CASE WHEN t.success = 0 THEN 1 ELSE 0 END) >= ?
        ORDER BY failures DESC
        """,
        (min_destructive_count,),
    )

    return [
        {
            "node_id": row[0],
            "total_appearances": row[1],
            "successes": row[2],
            "failures": row[3],
            "mixed_ratio": min(row[2], row[3]) / max(row[2], row[3]) if max(row[2], row[3]) > 0 else 0.0,
            "avg_amplitude": row[4] or 1.0,
        }
        for row in cursor.fetchall()
    ]


@dataclass
class InterferenceResult:
    """Result of applying interference effects."""
    patterns_processed: int
    constructive_count: int
    destructive_count: int
    nodes_reinforced: list[int]   # Node IDs that got constructive boost
    nodes_flagged_split: list[int]  # Node IDs flagged for potential split


def apply_interference_effects(
    dag_id: str,
    step_db,  # StepSignatureDB instance
    constructive_boost: float = None,
    destructive_penalty: float = None,
) -> InterferenceResult:
    """Apply centroid updates based on interference patterns detected in a DAG.

    Per CLAUDE.md:
    - Constructive interference (all succeed): REINFORCE centroid, flag for potential merge
    - Destructive interference (mixed): WEAKEN centroid, flag for potential split

    The key insight: Interference patterns reveal operational equivalence/difference
    that pure embedding similarity cannot capture.

    Args:
        dag_id: The DAG to analyze
        step_db: StepSignatureDB instance for centroid updates
        constructive_boost: Strength multiplier for constructive interference
            (defaults to INTERFERENCE_CONSTRUCTIVE_BOOST from config)
        destructive_penalty: Strength multiplier for destructive interference
            (defaults to INTERFERENCE_DESTRUCTIVE_PENALTY from config)

    Returns:
        InterferenceResult with counts and affected node IDs
    """
    if not INTERFERENCE_ENABLED:
        return InterferenceResult(
            patterns_processed=0,
            constructive_count=0,
            destructive_count=0,
            nodes_reinforced=[],
            nodes_flagged_split=[],
        )

    if constructive_boost is None:
        constructive_boost = INTERFERENCE_CONSTRUCTIVE_BOOST
    if destructive_penalty is None:
        destructive_penalty = INTERFERENCE_DESTRUCTIVE_PENALTY

    patterns = detect_interference_patterns(dag_id)

    if not patterns:
        return InterferenceResult(
            patterns_processed=0,
            constructive_count=0,
            destructive_count=0,
            nodes_reinforced=[],
            nodes_flagged_split=[],
        )

    nodes_reinforced = []
    nodes_flagged_split = []

    for pattern in patterns:
        if pattern.interference_type == "constructive":
            # All threads succeeded at this (dag_step, node) combination
            # This is strong evidence the node is operationally correct for this step type
            # Boost the centroid stability (don't move it, just record success)
            step_db.record_interference_outcome(
                signature_id=pattern.node_id,
                interference_type="constructive",
                thread_count=pattern.thread_count,
                success_count=pattern.successes,
            )
            nodes_reinforced.append(pattern.node_id)
            logger.debug(
                "[mcts] Constructive interference: node %d at step %s "
                "(%d threads, all succeeded, avg_amp=%.2f)",
                pattern.node_id, pattern.dag_step_id[:12],
                pattern.thread_count, pattern.avg_amplitude,
            )

        elif pattern.interference_type == "destructive":
            # Mixed results: some threads succeeded, some failed
            # This suggests the cluster is too generic - needs splitting
            step_db.record_interference_outcome(
                signature_id=pattern.node_id,
                interference_type="destructive",
                thread_count=pattern.thread_count,
                success_count=pattern.successes,
            )
            nodes_flagged_split.append(pattern.node_id)
            logger.debug(
                "[mcts] Destructive interference: node %d at step %s "
                "(%d threads, %d succeeded, %d failed, avg_amp=%.2f)",
                pattern.node_id, pattern.dag_step_id[:12],
                pattern.thread_count, pattern.successes, pattern.failures,
                pattern.avg_amplitude,
            )

    result = InterferenceResult(
        patterns_processed=len(patterns),
        constructive_count=len(nodes_reinforced),
        destructive_count=len(nodes_flagged_split),
        nodes_reinforced=list(set(nodes_reinforced)),  # Dedupe
        nodes_flagged_split=list(set(nodes_flagged_split)),  # Dedupe
    )

    logger.info(
        "[mcts] Interference effects for DAG %s: %d patterns, "
        "%d constructive (reinforced), %d destructive (flagged for split)",
        dag_id, result.patterns_processed,
        result.constructive_count, result.destructive_count,
    )

    return result


# =============================================================================
# POSTMORTEM STATE MANAGEMENT (Thread-safe singleton)
# =============================================================================


@dataclass
class PostmortemState:
    """Encapsulates mutable state for post-mortem batching.

    Replaces module-level globals for thread safety and testability.
    """
    problem_count: int = 0
    nodes_for_split: list[int] = field(default_factory=list)
    high_conf_wrong_nodes: list[int] = field(default_factory=list)
    dsl_regen_problem_count: int = 0

    def reset_merge_split(self) -> None:
        """Reset state after merge/split batch processing."""
        self.problem_count = 0
        self.nodes_for_split = []

    def reset_dsl_regen(self) -> None:
        """Reset state after DSL regeneration batch."""
        self.high_conf_wrong_nodes = []
        self.dsl_regen_problem_count = 0

    def accumulate_split_nodes(self, node_ids: list[int]) -> None:
        """Add nodes flagged for split."""
        self.nodes_for_split.extend(node_ids)
        self.problem_count += 1

    def accumulate_high_conf_wrong(self, nodes: list[dict]) -> None:
        """Add nodes with high-confidence wrong decisions."""
        for node in nodes:
            node_id = node.get("node_id")
            if node_id is not None and node_id not in self.high_conf_wrong_nodes:
                self.high_conf_wrong_nodes.append(node_id)
        self.dsl_regen_problem_count += 1


# Singleton instance
_postmortem_state: Optional[PostmortemState] = None


def get_postmortem_state() -> PostmortemState:
    """Get or create the singleton PostmortemState instance."""
    global _postmortem_state
    if _postmortem_state is None:
        _postmortem_state = PostmortemState()
    return _postmortem_state


def reset_postmortem_state() -> None:
    """Reset the singleton state (useful for testing)."""
    global _postmortem_state
    _postmortem_state = PostmortemState()


def run_postmortem_with_interference(dag_id: str, step_db) -> dict:
    """Run full post-mortem including interference pattern analysis.

    This is the main entry point for post-mortem analysis that includes:
    1. Standard amplitude_post computation (run_postmortem) - ALWAYS runs
    2. Interference pattern detection and centroid effects - ALWAYS runs
    3. Merge/split operations - BATCHED (runs every N problems per config)
    4. Retirement processing - BATCHED (runs at same interval as merge/split)
    5. DSL regeneration accumulation - tracks high-conf-wrong for batch regen

    Args:
        dag_id: The DAG to analyze
        step_db: StepSignatureDB instance for centroid updates

    Returns:
        Combined dict with amplitude stats and interference results
    """
    from mycelium.config import MERGE_SPLIT_BATCH_SIZE, RETIREMENT_ENABLED
    from mycelium.mcts.adaptive import AdaptiveExploration

    # Get singleton state (thread-safe)
    state = get_postmortem_state()

    # First run standard postmortem (amplitude_post computation) - cheap, always run
    amplitude_stats = run_postmortem(dag_id)

    # Record hit/miss stats for UCB1 adjustment (per mycelium-nirq)
    # This feeds into AdaptiveExploration to tune the exploration constant
    adaptive = AdaptiveExploration.get_instance()
    adaptive.record_postmortem_stats(
        high_conf_wrong=amplitude_stats.get("high_conf_wrong", 0),
        low_conf_right=amplitude_stats.get("low_conf_right", 0),
        total_high_conf=amplitude_stats.get("total_high_conf", 0),
        total_low_conf=amplitude_stats.get("total_low_conf", 0),
    )

    # Then detect and apply interference effects - cheap, always run
    interference_result = apply_interference_effects(dag_id, step_db)

    # Per beads mycelium-itkn: Propagate amplitude_post to signature stats
    # This closes the loop from post-mortem analysis to signature learning
    credit_stats = propagate_amplitude_to_signature_stats(dag_id, step_db)

    # Per beads mycelium-2rss: Divergence-point analysis for targeted blame
    # Compare winning vs losing thread paths to find exactly where failure occurred
    divergence_stats = assign_divergence_blame(dag_id, step_db)

    # Accumulate nodes flagged for split (using state object)
    state.accumulate_split_nodes(interference_result.nodes_flagged_split)

    # Per beads mycelium-flbq: Accumulate high-conf-wrong nodes for DSL regen
    # This tracks nodes that had high confidence but produced wrong answers
    if amplitude_stats.get("high_conf_wrong", 0) >= POSTMORTEM_DSL_REGEN_MIN_HIGH_CONF_WRONG:
        problem_nodes = get_problem_nodes_needing_attention(dag_id)
        state.accumulate_high_conf_wrong(problem_nodes)
        logger.debug(
            "[mcts] Accumulated %d high-conf-wrong nodes for DSL regen (total: %d)",
            len(problem_nodes), len(state.high_conf_wrong_nodes)
        )

    # Check if we should run merge/split/retirement (batched for performance)
    merge_split_result = {"merges_succeeded": 0, "merged_pairs": [], "splits_flagged": 0}
    retirement_result = {"demoted": 0, "pruned": 0, "merged_up": 0}

    should_run_batch_ops = (
        MERGE_SPLIT_BATCH_SIZE > 0 and
        state.problem_count >= MERGE_SPLIT_BATCH_SIZE
    )

    if should_run_batch_ops:
        # Run merge/split with accumulated data
        merge_split_result = run_merge_split_from_interference(
            step_db,
            nodes_reinforced=interference_result.nodes_reinforced,
            nodes_flagged_split=list(set(state.nodes_for_split)),  # Dedupe
        )

        # Run retirement check (per beads mycelium-x0mt)
        if RETIREMENT_ENABLED:
            retirement_result = run_retirement_check(step_db)
            if retirement_result.get("demoted", 0) or retirement_result.get("pruned", 0):
                logger.info(
                    "[mcts] Retirement: %d demoted, %d pruned, %d merged_up",
                    retirement_result.get("demoted", 0),
                    retirement_result.get("pruned", 0),
                    retirement_result.get("merged_up", 0),
                )

        # Reset merge/split state
        state.reset_merge_split()

        logger.info(
            "[mcts] Batch operations complete: %d merges, %d splits flagged",
            merge_split_result["merges_succeeded"],
            merge_split_result["splits_flagged"],
        )

    return {
        **amplitude_stats,
        "interference_patterns": interference_result.patterns_processed,
        "constructive_interference": interference_result.constructive_count,
        "destructive_interference": interference_result.destructive_count,
        "nodes_reinforced": interference_result.nodes_reinforced,
        "nodes_flagged_split": interference_result.nodes_flagged_split,
        "merges_succeeded": merge_split_result["merges_succeeded"],
        "merged_pairs": merge_split_result["merged_pairs"],
        "splits_flagged": merge_split_result["splits_flagged"],
        # Retirement stats (per beads mycelium-x0mt)
        "retirement_demoted": retirement_result.get("demoted", 0),
        "retirement_pruned": retirement_result.get("pruned", 0),
        "retirement_merged_up": retirement_result.get("merged_up", 0),
        "batch_counter": state.problem_count,
        "next_merge_split_in": MERGE_SPLIT_BATCH_SIZE - state.problem_count if MERGE_SPLIT_BATCH_SIZE > 0 else -1,
        # DSL regeneration info (per beads mycelium-flbq)
        "dsl_regen_nodes_accumulated": len(state.high_conf_wrong_nodes),
        "dsl_regen_ready": should_trigger_dsl_regen(),
        # Amplitude credit propagation (per beads mycelium-itkn)
        "credit_nodes_processed": credit_stats.get("nodes_processed", 0),
        "credit_successes": credit_stats.get("successes_credited", 0),
        "credit_failures": credit_stats.get("failures_credited", 0),
        # Divergence-point analysis (per beads mycelium-2rss)
        "divergence_points_found": divergence_stats.get("divergence_points_found", 0),
        "divergence_blame_assigned": divergence_stats.get("divergence_blame_assigned", 0),
        "suffix_blame_assigned": divergence_stats.get("suffix_blame_assigned", 0),
        "shared_prefix_credit": divergence_stats.get("shared_prefix_credit", 0),
    }


# =============================================================================
# MERGE/SPLIT OPERATIONS (structural tree changes from interference)
# =============================================================================


@dataclass
class MergeSplitResult:
    """Result of merge/split operations."""
    merges_attempted: int
    merges_succeeded: int
    merged_pairs: list[tuple[int, int]]  # (survivor_id, absorbed_id)
    splits_flagged: int
    split_node_ids: list[int]


def process_merge_candidates(
    step_db,
    min_success_rate: float = None,
    min_uses: int = None,
    min_similarity: float = None,
    max_merges_per_run: int = None,
) -> MergeSplitResult:
    """Process merge candidates from constructive interference patterns.

    Finds pairs of signatures that consistently succeed together with similar
    centroids and merges them. This consolidates operationally equivalent
    signatures that were split by vocabulary differences.

    Args:
        step_db: StepSignatureDB instance
        min_success_rate: Minimum success rate (default from config)
        min_uses: Minimum uses to trust (default from config)
        min_similarity: Minimum centroid similarity (default from config)
        max_merges_per_run: Limit merges per call (default from config)

    Returns:
        MergeSplitResult with merge statistics
    """
    from mycelium.config import (
        MERGE_MIN_SUCCESS_RATE,
        MERGE_MIN_USES,
        MERGE_MIN_SIMILARITY,
        MERGE_MAX_PER_BATCH,
    )

    # Use config defaults if not specified
    if min_success_rate is None:
        min_success_rate = MERGE_MIN_SUCCESS_RATE
    if min_uses is None:
        min_uses = MERGE_MIN_USES
    if min_similarity is None:
        min_similarity = MERGE_MIN_SIMILARITY
    if max_merges_per_run is None:
        max_merges_per_run = MERGE_MAX_PER_BATCH

    candidates = step_db.find_merge_candidates(
        min_success_rate=min_success_rate,
        min_uses=min_uses,
        min_similarity=min_similarity,
        limit=max_merges_per_run * 2,  # Get extra in case some fail
    )

    merged_pairs = []
    merges_attempted = 0

    for sig1_id, sig2_id, similarity in candidates:
        if len(merged_pairs) >= max_merges_per_run:
            break

        merges_attempted += 1

        # Determine survivor (more uses = more mature)
        sig1 = step_db.get_signature(sig1_id)
        sig2 = step_db.get_signature(sig2_id)

        if sig1 is None or sig2 is None:
            continue

        # Survivor is the one with more uses (more established)
        if (sig1.uses or 0) >= (sig2.uses or 0):
            survivor_id, absorbed_id = sig1_id, sig2_id
        else:
            survivor_id, absorbed_id = sig2_id, sig1_id

        # Attempt merge
        success = step_db.merge_signatures(survivor_id, absorbed_id)
        if success:
            merged_pairs.append((survivor_id, absorbed_id))
            logger.info(
                "[mcts] Merged signatures: %d absorbed into %d (similarity=%.3f)",
                absorbed_id, survivor_id, similarity
            )

    result = MergeSplitResult(
        merges_attempted=merges_attempted,
        merges_succeeded=len(merged_pairs),
        merged_pairs=merged_pairs,
        splits_flagged=0,
        split_node_ids=[],
    )

    if merged_pairs:
        logger.info(
            "[mcts] Merge processing complete: %d/%d succeeded",
            result.merges_succeeded, result.merges_attempted
        )

    return result


def process_split_candidates(
    step_db,
    node_ids: list[int],
) -> MergeSplitResult:
    """Flag nodes for split/decomposition based on destructive interference.

    Nodes with destructive interference (mixed success/failure) are flagged
    for decomposition. The actual split is handled by umbrella_learner when
    the signature's success rate drops low enough.

    Args:
        step_db: StepSignatureDB instance
        node_ids: List of node IDs flagged from destructive interference

    Returns:
        MergeSplitResult with split statistics
    """
    split_node_ids = []

    for node_id in node_ids:
        success = step_db.flag_for_split(node_id, reason="destructive_interference")
        if success:
            split_node_ids.append(node_id)

    result = MergeSplitResult(
        merges_attempted=0,
        merges_succeeded=0,
        merged_pairs=[],
        splits_flagged=len(split_node_ids),
        split_node_ids=split_node_ids,
    )

    if split_node_ids:
        logger.info(
            "[mcts] Split flagging complete: %d nodes flagged for decomposition",
            result.splits_flagged
        )

    return result


def run_merge_split_from_interference(
    step_db,
    nodes_reinforced: list[int] = None,
    nodes_flagged_split: list[int] = None,
    enable_merges: bool = True,
    enable_splits: bool = True,
) -> dict:
    """Run merge/split operations based on interference analysis.

    This is called after apply_interference_effects to perform structural
    changes to the signature tree:

    - MERGE: Consolidate operationally equivalent signatures
    - SPLIT: Flag generic clusters for decomposition

    Args:
        step_db: StepSignatureDB instance
        nodes_reinforced: Node IDs from constructive interference (for merge consideration)
        nodes_flagged_split: Node IDs from destructive interference (for split)
        enable_merges: Whether to attempt merges
        enable_splits: Whether to flag splits

    Returns:
        Dict with merge and split statistics
    """
    merge_result = MergeSplitResult(0, 0, [], 0, [])
    split_result = MergeSplitResult(0, 0, [], 0, [])

    # Process merges (uses global candidates, not just reinforced nodes)
    # Reinforced nodes inform us that merging is safe, but we find candidates
    # based on centroid similarity and success rates
    if enable_merges:
        merge_result = process_merge_candidates(step_db)

    # Process splits (directly uses flagged nodes)
    if enable_splits and nodes_flagged_split:
        split_result = process_split_candidates(step_db, nodes_flagged_split)

    return {
        "merges_attempted": merge_result.merges_attempted,
        "merges_succeeded": merge_result.merges_succeeded,
        "merged_pairs": merge_result.merged_pairs,
        "splits_flagged": split_result.splits_flagged,
        "split_node_ids": split_result.split_node_ids,
    }


# =============================================================================
# POST-MORTEM TRIGGERED DSL REGENERATION
# =============================================================================
# Per beads mycelium-flbq: When post-mortem detects high failure rate at a
# specific (dag_step_id, node_id) pair, trigger DSL regeneration.

def accumulate_high_conf_wrong_nodes(nodes: list[dict]) -> None:
    """Accumulate nodes with high-confidence wrong decisions for batch DSL regen.

    Args:
        nodes: List of dicts with node_id from get_problem_nodes_needing_attention
    """
    state = get_postmortem_state()
    state.accumulate_high_conf_wrong(nodes)


def get_accumulated_failing_nodes() -> list[int]:
    """Get the current list of accumulated failing node IDs."""
    state = get_postmortem_state()
    return list(state.high_conf_wrong_nodes)


def clear_accumulated_failing_nodes() -> None:
    """Clear the accumulated failing nodes after processing."""
    state = get_postmortem_state()
    state.reset_dsl_regen()


async def trigger_dsl_regeneration_for_nodes(
    node_ids: list[int],
    step_db,
    client,
) -> dict:
    """Trigger DSL regeneration for specific failing nodes.

    This is called when post-mortem has accumulated enough evidence that
    certain signatures have incorrect DSLs.

    Args:
        node_ids: List of signature IDs (node_id in MCTS tables = signature id)
        step_db: StepSignatureDB instance
        client: LLM client for DSL generation

    Returns:
        Dict with regeneration statistics
    """
    from mycelium.step_signatures.dsl_rewriter import (
        RewriteCandidate,
        generate_improved_dsl,
    )

    if not POSTMORTEM_DSL_REGEN_ENABLED:
        return {"skipped": True, "reason": "POSTMORTEM_DSL_REGEN_ENABLED is False"}

    if not node_ids:
        return {"regenerated": 0, "failed": 0, "skipped": 0}

    # Deduplicate and get unique node IDs
    unique_nodes = list(set(node_ids))

    regenerated = 0
    failed = 0
    skipped = 0

    for node_id in unique_nodes:
        try:
            # Get signature details
            sig = step_db.get_signature_by_id(node_id)
            if sig is None:
                logger.warning("[mcts] Node %d not found in signatures, skipping", node_id)
                skipped += 1
                continue

            # Skip non-math DSLs (decompose, router, etc)
            if sig.dsl_type not in ("math", "sympy", "python"):
                logger.debug("[mcts] Skipping non-math signature %d (type=%s)", node_id, sig.dsl_type)
                skipped += 1
                continue

            # Build rewrite candidate
            candidate = RewriteCandidate(
                signature_id=node_id,
                step_type=sig.step_type,
                description=sig.description,
                current_dsl=sig.dsl_script,
                uses=sig.uses,
                successes=sig.successes,
                success_rate=sig.success_rate,
            )

            # Get failure examples from recent MCTS data
            failure_examples = _get_recent_failures_for_node(node_id)

            # Generate improved DSL
            new_dsl = await generate_improved_dsl(candidate, client, failure_examples)

            if new_dsl:
                # Update signature with new DSL
                step_db.update_nl_interface(
                    signature_id=node_id,
                    dsl_script=new_dsl,
                )
                # Mark as rewritten for cooldown
                step_db.mark_signature_rewritten(node_id)
                regenerated += 1
                logger.info(
                    "[mcts] Regenerated DSL for sig %d '%s' (was %.1f%% success)",
                    node_id, sig.step_type, sig.success_rate * 100
                )
            else:
                failed += 1
                logger.warning(
                    "[mcts] Failed to generate new DSL for sig %d '%s'",
                    node_id, sig.step_type
                )

        except Exception as e:
            logger.error("[mcts] Error regenerating DSL for node %d: %s", node_id, e)
            failed += 1

    logger.info(
        "[mcts] DSL regeneration batch complete: %d regenerated, %d failed, %d skipped",
        regenerated, failed, skipped
    )

    return {
        "regenerated": regenerated,
        "failed": failed,
        "skipped": skipped,
        "total_nodes": len(unique_nodes),
    }


def _get_recent_failures_for_node(node_id: int, limit: int = 5) -> list[dict]:
    """Get recent failure examples for a node from MCTS thread_steps.

    Returns examples of what inputs/results led to failures.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT
            ts.step_result,
            ds.step_desc,
            t.final_answer
        FROM mcts_thread_steps ts
        JOIN mcts_threads t ON ts.thread_id = t.thread_id
        LEFT JOIN mcts_dag_steps ds ON ts.dag_step_id = ds.dag_step_id
        WHERE ts.node_id = ?
          AND t.success = 0
          AND ts.amplitude >= ?
        ORDER BY ts.created_at DESC
        LIMIT ?
        """,
        (node_id, POSTMORTEM_HIGH_CONF_THRESHOLD, limit),
    )

    examples = []
    for row in cursor.fetchall():
        examples.append({
            "result": row[0],
            "step_description": row[1],
            "thread_answer": row[2],
            "error": "High-confidence wrong decision",
        })

    return examples


def should_trigger_dsl_regen() -> bool:
    """Check if we should trigger batch DSL regeneration.

    Returns True if:
    - POSTMORTEM_DSL_REGEN_ENABLED is True
    - We've accumulated enough problems
    - We have nodes needing attention
    """
    if not POSTMORTEM_DSL_REGEN_ENABLED:
        return False
    state = get_postmortem_state()
    if state.dsl_regen_problem_count < POSTMORTEM_DSL_REGEN_BATCH_SIZE:
        return False
    if not state.high_conf_wrong_nodes:
        return False
    return True


def increment_dsl_regen_counter() -> int:
    """Increment the problem counter for DSL regen batching.

    Note: This is now handled internally by accumulate_high_conf_wrong_nodes.
    Kept for backward compatibility.
    """
    state = get_postmortem_state()
    state.dsl_regen_problem_count += 1
    return state.dsl_regen_problem_count


# =============================================================================
# SIGNATURE RETIREMENT (Prune consistently failing nodes)
# =============================================================================
# Per beads mycelium-x0mt: Signatures that consistently fail across multiple
# problems should be flagged for retirement. Post-mortem identifies "dead weight"
# nodes that hurt routing.


@dataclass
class RetirementResult:
    """Result of retirement processing."""
    candidates_found: int
    demoted: int
    pruned: int
    merged_up: int
    demoted_ids: list[int]
    pruned_ids: list[int]
    merged_up_ids: list[int]


def detect_retirement_candidates(step_db) -> list[tuple[int, str]]:
    """Find signatures that should be considered for retirement.

    Criteria:
    - uses >= RETIREMENT_MIN_USES (enough selections to trust accuracy)
    - success_rate <= RETIREMENT_MAX_SUCCESS_RATE (low accuracy)
    - operational_failures >= RETIREMENT_MIN_OPERATIONAL_FAILURES (flagged by post-mortem)

    Accuracy is inferred from MCTS post-mortem: for each leaf node, we track how
    many times it was selected (uses) vs how many times the thread it was in won
    (successes). This gives leaf_accuracy = successes / uses.

    Returns:
        List of (signature_id, recommended_action) tuples.
        Actions: "demote", "prune", "merge_up"
    """
    from mycelium.config import (
        RETIREMENT_ENABLED,
        RETIREMENT_MIN_USES,
        RETIREMENT_MAX_SUCCESS_RATE,
        RETIREMENT_MIN_OPERATIONAL_FAILURES,
        RETIREMENT_PRUNE_SUCCESS_RATE,
        RETIREMENT_PRUNE_MIN_USES,
    )

    if not RETIREMENT_ENABLED:
        return []

    candidates = []
    all_sigs = step_db.get_all_signatures()

    for sig in all_sigs:
        # Skip root signature (never retire)
        if sig.is_root:
            continue

        # Skip if not enough uses to trust accuracy
        if (sig.uses or 0) < RETIREMENT_MIN_USES:
            continue

        # Skip if accuracy is acceptable
        if sig.success_rate > RETIREMENT_MAX_SUCCESS_RATE:
            continue

        # Skip if not flagged enough by post-mortem
        if (sig.operational_failures or 0) < RETIREMENT_MIN_OPERATIONAL_FAILURES:
            continue

        # Determine recommended action based on severity
        if sig.success_rate <= RETIREMENT_PRUNE_SUCCESS_RATE and (sig.uses or 0) >= RETIREMENT_PRUNE_MIN_USES:
            # Very bad - prune entirely
            action = "prune"
        elif sig.is_semantic_umbrella:
            # Umbrella with poor performance - merge children up
            action = "merge_up"
        else:
            # Default - demote (add routing penalty)
            action = "demote"

        candidates.append((sig.id, action))
        logger.debug(
            "[mcts] Retirement candidate: sig=%d, action=%s, uses=%d, accuracy=%.1f%%, op_fail=%d",
            sig.id, action, sig.uses or 0, sig.success_rate * 100, sig.operational_failures or 0
        )

    return candidates


def process_retirement_candidates(
    step_db,
    candidates: list[tuple[int, str]],
    max_per_batch: int = None,
) -> RetirementResult:
    """Apply retirement actions to candidate signatures.

    Actions:
    - demote: Add routing penalty via operational_failures increment
    - prune: Archive signature (soft delete) and reparent children
    - merge_up: Absorb back into parent umbrella

    Args:
        step_db: StepSignatureDB instance
        candidates: List of (signature_id, action) tuples from detect_retirement_candidates
        max_per_batch: Max retirements to process (default from config)

    Returns:
        RetirementResult with statistics
    """
    from mycelium.config import RETIREMENT_MAX_PER_BATCH

    if max_per_batch is None:
        max_per_batch = RETIREMENT_MAX_PER_BATCH

    demoted_ids = []
    pruned_ids = []
    merged_up_ids = []
    failed_ids = []  # Track partial failures
    processed = 0

    for sig_id, action in candidates:
        if processed >= max_per_batch:
            break

        try:
            if action == "demote":
                # Add routing penalty by incrementing operational_failures further
                # This affects routing decisions via success_rate
                success = step_db.flag_for_split(sig_id, reason="retirement_demote")
                if success:
                    demoted_ids.append(sig_id)
                    processed += 1
                    logger.info("[mcts] Demoted signature %d (added routing penalty)", sig_id)

            elif action == "prune":
                # Soft delete - archive the signature
                # Use transaction for atomic re-parent + archive
                parent = step_db.get_parent(sig_id)
                children = step_db.get_children(sig_id, for_routing=True)

                # Perform re-parent and archive atomically via step_db method
                # Note: step_db.archive_signature_with_reparent handles transaction
                if parent is not None and children:
                    success = step_db.archive_signature_with_reparent(
                        sig_id,
                        parent_id=parent.id,
                        child_ids=[c.id for c in children],
                        reason="retirement_prune"
                    )
                    if success:
                        logger.info("[mcts] Reparented %d children of sig %d to parent %d", len(children), sig_id, parent.id)
                else:
                    # No children to reparent, just archive
                    success = step_db.archive_signature(sig_id, reason="retirement_prune")

                if success:
                    pruned_ids.append(sig_id)
                    processed += 1
                    logger.info("[mcts] Pruned (archived) signature %d", sig_id)

            elif action == "merge_up":
                # Merge this umbrella back into its parent
                parent = step_db.get_parent(sig_id)
                if parent is not None:
                    # Use existing merge_signatures (parent absorbs this sig)
                    success = step_db.merge_signatures(parent.id, sig_id)
                    if success:
                        merged_up_ids.append(sig_id)
                        processed += 1
                        logger.info("[mcts] Merged signature %d up into parent %d", sig_id, parent.id)
                else:
                    # No parent - just demote instead
                    success = step_db.flag_for_split(sig_id, reason="retirement_merge_failed")
                    if success:
                        demoted_ids.append(sig_id)
                        processed += 1
                        logger.info("[mcts] No parent for merge_up, demoted signature %d instead", sig_id)

        except Exception as e:
            logger.warning("[mcts] Failed to retire signature %d (action=%s): %s", sig_id, action, e)
            failed_ids.append(sig_id)

    result = RetirementResult(
        candidates_found=len(candidates),
        demoted=len(demoted_ids),
        pruned=len(pruned_ids),
        merged_up=len(merged_up_ids),
        demoted_ids=demoted_ids,
        pruned_ids=pruned_ids,
        merged_up_ids=merged_up_ids,
    )

    if demoted_ids or pruned_ids or merged_up_ids:
        logger.info(
            "[mcts] Retirement complete: %d demoted, %d pruned, %d merged_up, %d failed (of %d candidates)",
            result.demoted, result.pruned, result.merged_up, len(failed_ids), result.candidates_found
        )

    return result


def run_retirement_check(step_db) -> dict:
    """Run retirement check and process candidates.

    This should be called as part of the post-mortem pipeline,
    typically after merge/split processing.

    Args:
        step_db: StepSignatureDB instance

    Returns:
        Dict with retirement statistics
    """
    from mycelium.config import RETIREMENT_ENABLED

    if not RETIREMENT_ENABLED:
        return {
            "candidates_found": 0,
            "demoted": 0,
            "pruned": 0,
            "merged_up": 0,
        }

    # Detect candidates
    candidates = detect_retirement_candidates(step_db)

    if not candidates:
        return {
            "candidates_found": 0,
            "demoted": 0,
            "pruned": 0,
            "merged_up": 0,
        }

    # Process candidates
    result = process_retirement_candidates(step_db, candidates)

    return {
        "candidates_found": result.candidates_found,
        "demoted": result.demoted,
        "pruned": result.pruned,
        "merged_up": result.merged_up,
        "demoted_ids": result.demoted_ids,
        "pruned_ids": result.pruned_ids,
        "merged_up_ids": result.merged_up_ids,
    }
