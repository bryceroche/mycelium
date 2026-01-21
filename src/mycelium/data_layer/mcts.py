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
    conn.executemany(
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
    }

    for thread_step_id, thread_id, amplitude in thread_steps:
        thread_success = thread_outcomes.get(thread_id)

        if thread_success is None:
            # Thread not graded yet, skip
            continue

        amp = amplitude if amplitude is not None else 1.0
        is_high_conf = amp >= POSTMORTEM_HIGH_CONF_THRESHOLD
        won = thread_success == 1

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


def get_problem_nodes_needing_attention(dag_id: str) -> list[dict]:
    """Identify nodes that performed poorly in this DAG.

    Returns nodes where:
    - High confidence but thread lost (amplitude >= 0.7, success = 0)

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
          AND ts.amplitude >= 0.7
          AND t.success = 0
        """,
        (dag_id,),
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


# Module-level counter for batch merge/split scheduling
_postmortem_problem_count = 0
_accumulated_nodes_for_split: list[int] = []


def run_postmortem_with_interference(dag_id: str, step_db) -> dict:
    """Run full post-mortem including interference pattern analysis.

    This is the main entry point for post-mortem analysis that includes:
    1. Standard amplitude_post computation (run_postmortem) - ALWAYS runs
    2. Interference pattern detection and centroid effects - ALWAYS runs
    3. Merge/split operations - BATCHED (runs every N problems per config)

    Args:
        dag_id: The DAG to analyze
        step_db: StepSignatureDB instance for centroid updates

    Returns:
        Combined dict with amplitude stats and interference results
    """
    global _postmortem_problem_count, _accumulated_nodes_for_split

    from mycelium.config import MERGE_SPLIT_BATCH_SIZE

    # First run standard postmortem (amplitude_post computation) - cheap, always run
    amplitude_stats = run_postmortem(dag_id)

    # Then detect and apply interference effects - cheap, always run
    interference_result = apply_interference_effects(dag_id, step_db)

    # Accumulate nodes flagged for split
    _accumulated_nodes_for_split.extend(interference_result.nodes_flagged_split)
    _postmortem_problem_count += 1

    # Check if we should run merge/split (batched for performance)
    merge_split_result = {"merges_succeeded": 0, "merged_pairs": [], "splits_flagged": 0}

    should_run_merge_split = (
        MERGE_SPLIT_BATCH_SIZE > 0 and
        _postmortem_problem_count >= MERGE_SPLIT_BATCH_SIZE
    )

    if should_run_merge_split:
        # Run merge/split with accumulated data
        merge_split_result = run_merge_split_from_interference(
            step_db,
            nodes_reinforced=interference_result.nodes_reinforced,
            nodes_flagged_split=list(set(_accumulated_nodes_for_split)),  # Dedupe
        )

        # Reset counters
        _postmortem_problem_count = 0
        _accumulated_nodes_for_split = []

        logger.info(
            "[mcts] Batch merge/split complete: %d merges, %d splits flagged",
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
        "batch_counter": _postmortem_problem_count,
        "next_merge_split_in": MERGE_SPLIT_BATCH_SIZE - _postmortem_problem_count if MERGE_SPLIT_BATCH_SIZE > 0 else -1,
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
