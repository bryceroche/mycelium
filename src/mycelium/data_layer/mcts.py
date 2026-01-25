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
from typing import Any, Optional

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

    # Track plan success rate (per beads mycelium-ogo6)
    try:
        record_plan_outcome(dag_id, success)
    except Exception as e:
        logger.warning("[mcts] Failed to record plan outcome: %s", e)


# =============================================================================
# DAG PLAN STATS: Track success rates of (DAG plan, Thread) pairs
# =============================================================================
# Per beads mycelium-ogo6: Track which decomposition strategies work.
# A plan_signature = hash of (step tasks + dependency structure)


def compute_plan_signature(dag_id: str) -> tuple[str, int, str]:
    """Compute a hash signature for a DAG plan structure.

    The signature captures:
    - Number of steps
    - Normalized step descriptions (lowercase, numbers removed)
    - Step ordering and dependencies

    Returns:
        Tuple of (plan_signature, step_count, plan_structure_json)
    """
    import hashlib
    import json
    import re

    conn = get_db()
    cursor = conn.execute(
        """
        SELECT step_desc, step_num, branch_num, dsl_hint
        FROM mcts_dag_steps
        WHERE dag_id = ?
        ORDER BY step_num, branch_num
        """,
        (dag_id,),
    )
    rows = cursor.fetchall()

    if not rows:
        return ("empty", 0, "[]")

    # Normalize steps: remove numbers, lowercase, strip
    def normalize_step(desc: str) -> str:
        # Remove specific numbers but keep structure
        normalized = re.sub(r'\b\d+\.?\d*\b', 'N', desc.lower().strip())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized

    # Build canonical structure
    plan_structure = []
    for row in rows:
        step_desc, step_num, branch_num, dsl_hint = row
        plan_structure.append({
            "step_num": step_num,
            "branch_num": branch_num,
            "normalized": normalize_step(step_desc),
            "dsl_hint": dsl_hint or "",
        })

    # Create stable JSON representation
    plan_json = json.dumps(plan_structure, sort_keys=True)

    # Hash for quick lookup
    signature = hashlib.sha256(plan_json.encode()).hexdigest()[:16]

    return (signature, len(rows), plan_json)


def record_plan_outcome(dag_id: str, success: bool) -> None:
    """Record the outcome of a DAG plan for stats tracking.

    Called after grade_dag() to update plan success rates.

    Args:
        dag_id: The DAG that was graded
        success: Whether the DAG produced correct answer
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        signature, step_count, plan_json = compute_plan_signature(dag_id)
    except Exception as e:
        logger.warning("[mcts] Failed to compute plan signature for %s: %s", dag_id, e)
        return

    if signature == "empty":
        return  # Don't track empty plans

    conn = get_db()

    # Upsert with atomic update
    conn.execute(
        """
        INSERT INTO dag_plan_stats (
            plan_signature, step_count, plan_structure,
            uses, successes, success_rate,
            first_seen_at, last_used_at
        ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
        ON CONFLICT(plan_signature) DO UPDATE SET
            uses = uses + 1,
            successes = successes + excluded.successes,
            success_rate = CAST(successes + excluded.successes AS REAL) / (uses + 1),
            last_used_at = excluded.last_used_at
        """,
        (signature, step_count, plan_json, 1 if success else 0,
         0.5 if success else 0.0, now, now),
    )

    logger.debug(
        "[mcts] Recorded plan outcome: sig=%s steps=%d success=%s",
        signature[:8], step_count, success
    )


def get_plan_stats_summary() -> dict:
    """Get summary statistics for plan tracking.

    Returns:
        Dict with total_plans, total_uses, avg_success_rate, etc.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_plans,
            SUM(uses) as total_uses,
            AVG(success_rate) as avg_success_rate,
            MIN(success_rate) as min_success_rate,
            MAX(success_rate) as max_success_rate
        FROM dag_plan_stats
        """
    )
    row = cursor.fetchone()
    if row:
        return {
            "total_plans": row[0] or 0,
            "total_uses": row[1] or 0,
            "avg_success_rate": row[2] or 0.0,
            "min_success_rate": row[3] or 0.0,
            "max_success_rate": row[4] or 0.0,
        }
    return {"total_plans": 0, "total_uses": 0, "avg_success_rate": 0.0}


def get_top_plans(limit: int = 10, min_uses: int = 3) -> list[dict]:
    """Get top performing plans by success rate.

    Args:
        limit: Max plans to return
        min_uses: Minimum uses required (filter noise)

    Returns:
        List of plan stats dicts sorted by success_rate desc
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT plan_signature, step_count, uses, successes, success_rate
        FROM dag_plan_stats
        WHERE uses >= ?
        ORDER BY success_rate DESC, uses DESC
        LIMIT ?
        """,
        (min_uses, limit),
    )
    return [
        {
            "signature": row[0],
            "step_count": row[1],
            "uses": row[2],
            "successes": row[3],
            "success_rate": row[4],
        }
        for row in cursor.fetchall()
    ]


def get_worst_plans(limit: int = 10, min_uses: int = 3) -> list[dict]:
    """Get worst performing plans by success rate.

    Args:
        limit: Max plans to return
        min_uses: Minimum uses required (filter noise)

    Returns:
        List of plan stats dicts sorted by success_rate asc
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT plan_signature, step_count, uses, successes, success_rate
        FROM dag_plan_stats
        WHERE uses >= ?
        ORDER BY success_rate ASC, uses DESC
        LIMIT ?
        """,
        (min_uses, limit),
    )
    return [
        {
            "signature": row[0],
            "step_count": row[1],
            "uses": row[2],
            "successes": row[3],
            "success_rate": row[4],
        }
        for row in cursor.fetchall()
    ]


# =============================================================================
# DECOMPOSITION QUEUE: Batch complex steps for later decomposition
# =============================================================================
# Per beads mycelium-mm08: Queue complex steps instead of decomposing immediately.
# Batch LLM calls are more efficient than one-at-a-time.
#
# NOTE: is_step_complex() was removed - replaced by divergence-based splitting
# in step_signatures/divergence.py. Natural splitting happens based on observed
# success/failure divergence, not pre-execution complexity detection.


def check_substeps_match_existing(
    substeps: list[str],
    step_db,
    min_similarity: float = 0.70,
) -> tuple[bool, float, list[tuple[str, float]]]:
    """Check if decomposed sub-steps would match existing leaf signatures.

    This is the speculative decomposition probe - we check if breaking down
    a step produces sub-steps that match existing leaves well.

    Args:
        substeps: List of decomposed step texts
        step_db: StepSignatureDB instance for similarity checking
        min_similarity: Threshold for "good match"

    Returns:
        Tuple of (all_match, avg_similarity, details)
        - all_match: True if all sub-steps have good matches
        - avg_similarity: Average similarity across sub-steps
        - details: List of (substep, best_sim) for debugging
    """
    from mycelium.embedding_cache import cached_embed
    import numpy as np

    if not substeps:
        return False, 0.0, []

    total_sim = 0.0
    matches = 0
    details = []

    for substep in substeps:
        substep_emb = cached_embed(substep)
        if substep_emb is None:
            details.append((substep, 0.0))
            continue

        # Route through hierarchy to find best match (doesn't create anything)
        best_sig, path = step_db.route_through_hierarchy(
            embedding=np.array(substep_emb),
            min_similarity=0.0,  # Accept any match, we just want the similarity
        )

        # Calculate similarity to best match
        if best_sig and best_sig.centroid is not None:
            from mycelium.step_signatures.db import cosine_similarity
            best_sim = cosine_similarity(np.array(substep_emb), np.array(best_sig.centroid))
        else:
            best_sim = 0.0

        total_sim += best_sim
        details.append((substep[:40], best_sim))
        if best_sim >= min_similarity:
            matches += 1

    avg_sim = total_sim / len(substeps) if substeps else 0.0
    all_match = matches == len(substeps)

    logger.info(
        "[speculative_decomp] %d/%d sub-steps match existing leaves (avg_sim=%.3f)",
        matches, len(substeps), avg_sim
    )
    for substep, sim in details:
        logger.debug("[speculative_decomp]   %.3f: %s", sim, substep)

    return all_match, avg_sim, details


def queue_for_decomposition(
    step_text: str,
    complexity_reason: str,
    embedding=None,
    dag_step_id: str = None,
    problem_context: str = None,
    conn=None,
) -> int:
    """Add a complex step to the decomposition queue.

    Args:
        step_text: The step to decompose
        complexity_reason: Why it's being queued (e.g., "compound", "novel")
        embedding: Optional embedding for the step
        dag_step_id: Optional link to originating dag_step
        problem_context: Optional problem text for LLM context
        conn: Optional DB connection (reuse caller's connection to avoid locks)

    Returns:
        Queue entry ID (or existing ID if already queued)
    """
    import json
    from mycelium.step_signatures.utils import pack_embedding

    now = datetime.now(timezone.utc).isoformat()
    if conn is None:
        conn = get_db()

    # Check if already queued (pending) with same step_text - avoid duplicates
    cursor = conn.execute(
        """
        SELECT id FROM decomposition_queue
        WHERE step_text = ? AND processed_at IS NULL
        LIMIT 1
        """,
        (step_text,),
    )
    existing = cursor.fetchone()
    if existing:
        return existing[0]  # Return existing queue ID

    embedding_packed = pack_embedding(embedding) if embedding is not None else None

    cursor = conn.execute(
        """
        INSERT INTO decomposition_queue (
            step_text, embedding, dag_step_id, problem_context,
            complexity_reason, queued_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (step_text, embedding_packed, dag_step_id, problem_context,
         complexity_reason, now),
    )

    queue_id = cursor.lastrowid
    logger.info(
        "[decomp-queue] Queued step for decomposition: id=%d reason=%s step='%s'",
        queue_id, complexity_reason, step_text[:50]
    )
    return queue_id


def get_pending_decompositions(limit: int = 10) -> list[dict]:
    """Get steps waiting to be decomposed.

    Args:
        limit: Max entries to return

    Returns:
        List of queue entries with id, step_text, problem_context, etc.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT id, step_text, embedding, dag_step_id, problem_context,
               complexity_reason, queued_at
        FROM decomposition_queue
        WHERE processed_at IS NULL
        ORDER BY queued_at ASC
        LIMIT ?
        """,
        (limit,),
    )

    return [
        {
            "id": row[0],
            "step_text": row[1],
            "embedding": row[2],
            "dag_step_id": row[3],
            "problem_context": row[4],
            "complexity_reason": row[5],
            "queued_at": row[6],
        }
        for row in cursor.fetchall()
    ]


def get_decomposition_queue_size() -> int:
    """Get count of pending decompositions."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT COUNT(*) FROM decomposition_queue WHERE processed_at IS NULL"
    )
    row = cursor.fetchone()
    return row[0] if row else 0


def get_oldest_pending_age_seconds() -> float:
    """Get age in seconds of the oldest pending decomposition.

    Returns:
        Age in seconds, or 0.0 if queue is empty.
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT MIN(queued_at) FROM decomposition_queue
        WHERE processed_at IS NULL
        """
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return 0.0

    # Parse ISO timestamp
    oldest_time = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    age = (now - oldest_time).total_seconds()
    return max(0.0, age)


def mark_decomposition_processed(
    queue_id: int,
    result_signature_ids: list[int],
    decomposition_steps: list[str],
) -> None:
    """Mark a queue entry as processed with results.

    Args:
        queue_id: The queue entry ID
        result_signature_ids: IDs of created atomic signatures
        decomposition_steps: The atomic steps produced by LLM
    """
    import json

    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()

    conn.execute(
        """
        UPDATE decomposition_queue
        SET processed_at = ?,
            result_signature_ids = ?,
            decomposition_steps = ?
        WHERE id = ?
        """,
        (now, json.dumps(result_signature_ids), json.dumps(decomposition_steps), queue_id),
    )

    logger.info(
        "[decomp-queue] Marked queue entry %d as processed: %d signatures created",
        queue_id, len(result_signature_ids)
    )


def get_decomposition_queue_stats() -> dict:
    """Get statistics about the decomposition queue."""
    conn = get_db()

    # Count by status
    cursor = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE processed_at IS NULL) as pending,
            COUNT(*) FILTER (WHERE processed_at IS NOT NULL) as processed,
            COUNT(*) as total
        FROM decomposition_queue
    """)
    row = cursor.fetchone()

    # Count by reason (pending only)
    cursor = conn.execute("""
        SELECT complexity_reason, COUNT(*)
        FROM decomposition_queue
        WHERE processed_at IS NULL
        GROUP BY complexity_reason
    """)
    by_reason = {r[0]: r[1] for r in cursor.fetchall()}

    return {
        "pending": row[0] if row else 0,
        "processed": row[1] if row else 0,
        "total": row[2] if row else 0,
        "by_reason": by_reason,
    }


def get_decomposition_results(queue_ids: list[int]) -> dict[int, dict]:
    """Get decomposition results for specific queue entries.

    Args:
        queue_ids: List of queue entry IDs to retrieve

    Returns:
        Dict mapping queue_id to result dict with:
        - processed: bool
        - decomposition_steps: list[str] (if processed)
        - result_signature_ids: list[int] (if processed)
    """
    import json

    if not queue_ids:
        return {}

    conn = get_db()
    placeholders = ",".join("?" * len(queue_ids))
    cursor = conn.execute(
        f"""
        SELECT id, processed_at, decomposition_steps, result_signature_ids
        FROM decomposition_queue
        WHERE id IN ({placeholders})
        """,
        queue_ids,
    )

    results = {}
    for row in cursor.fetchall():
        queue_id = row[0]
        processed = row[1] is not None
        results[queue_id] = {
            "processed": processed,
            "decomposition_steps": json.loads(row[2]) if row[2] else [],
            "result_signature_ids": json.loads(row[3]) if row[3] else [],
        }

    return results


def are_decompositions_ready(queue_ids: list[int]) -> bool:
    """Check if all specified queue entries have been processed.

    Args:
        queue_ids: Queue entry IDs to check

    Returns:
        True if all are processed, False otherwise
    """
    if not queue_ids:
        return True

    conn = get_db()
    placeholders = ",".join("?" * len(queue_ids))
    cursor = conn.execute(
        f"""
        SELECT COUNT(*) FROM decomposition_queue
        WHERE id IN ({placeholders}) AND processed_at IS NULL
        """,
        queue_ids,
    )
    pending_count = cursor.fetchone()[0]
    return pending_count == 0


def get_pending_queue_ids() -> list[int]:
    """Get all pending (unprocessed) queue entry IDs.

    Returns:
        List of queue IDs waiting for decomposition
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT id FROM decomposition_queue
        WHERE processed_at IS NULL
        ORDER BY queued_at ASC
        """
    )
    return [row[0] for row in cursor.fetchall()]


# =============================================================================
# LEAF REJECTION TRACKING
# =============================================================================


# Rejection thresholds (per CLAUDE.md: leaves define their own boundaries)
REJECTION_SIM_THRESHOLD = 0.85  # Below this similarity, leaf rejects the step (lowered for latency)
REJECTION_COUNT_THRESHOLD = 10  # Min rejections before considering decomposition
REJECTION_RATE_THRESHOLD = 0.30  # 30% rejection rate triggers decomposition flag


def record_leaf_rejection(
    signature_id: int,
    step_text: str,
    similarity: float,
    dag_step_id: str = None,
    problem_context: str = None,
    conn=None,
) -> int:
    """Record that a leaf signature rejected a dag_step due to low similarity.

    Args:
        signature_id: The leaf signature that rejected
        step_text: The step that was rejected
        similarity: The similarity score that caused rejection
        dag_step_id: Optional link to the dag_step
        problem_context: Optional problem text for context
        conn: Optional DB connection (reuse caller's connection to avoid locks)

    Returns:
        Updated rejection_count for the signature
    """
    db = conn if conn is not None else get_db()
    rejection_count = 0

    try:
        # Increment rejection count
        db.execute(
            "UPDATE step_signatures SET rejection_count = rejection_count + 1 WHERE id = ?",
            (signature_id,),
        )

        # Get updated count
        cursor = db.execute(
            "SELECT rejection_count FROM step_signatures WHERE id = ?",
            (signature_id,),
        )
        row = cursor.fetchone()
        rejection_count = row[0] if row else 0
    except Exception as e:
        logger.warning("[rejection] DB error recording rejection: %s", e)

    # Queue the rejected step for decomposition (non-blocking)
    try:
        queue_for_decomposition(
            step_text=step_text,
            complexity_reason=f"rejected_by_leaf_{signature_id}_sim_{similarity:.3f}",
            dag_step_id=dag_step_id,
            problem_context=problem_context,
            conn=db,  # Pass connection to avoid lock contention
        )
    except Exception as e:
        logger.warning("[rejection] Failed to queue for decomposition: %s", e)

    logger.debug(
        "[rejection] Leaf %d rejected step (sim=%.3f), total rejections=%d",
        signature_id, similarity, rejection_count
    )

    return rejection_count


def get_leaf_rejection_stats(signature_id: int) -> dict:
    """Get rejection statistics for a leaf signature.

    Returns:
        Dict with rejection_count, uses, rejection_rate, should_decompose
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT rejection_count, uses, is_semantic_umbrella
        FROM step_signatures
        WHERE id = ?
        """,
        (signature_id,),
    )
    row = cursor.fetchone()

    if not row:
        return {"error": "signature not found"}

    rejection_count = row[0] or 0
    uses = row[1] or 0
    is_umbrella = row[2] or 0

    # Calculate rejection rate (rejections / total attempts)
    total_attempts = uses + rejection_count
    rejection_rate = rejection_count / total_attempts if total_attempts > 0 else 0.0

    # Determine if this leaf should be decomposed
    should_decompose = (
        not is_umbrella  # Only leaves, not umbrellas
        and rejection_count >= REJECTION_COUNT_THRESHOLD
        and rejection_rate >= REJECTION_RATE_THRESHOLD
    )

    return {
        "signature_id": signature_id,
        "rejection_count": rejection_count,
        "uses": uses,
        "total_attempts": total_attempts,
        "rejection_rate": rejection_rate,
        "should_decompose": should_decompose,
    }


def get_leaves_needing_decomposition(limit: int = 10) -> list[dict]:
    """Find leaf signatures with high rejection rates that need decomposition.

    Returns:
        List of leaf stats dicts for signatures that should be decomposed
    """
    conn = get_db()
    cursor = conn.execute(
        """
        SELECT id, rejection_count, uses
        FROM step_signatures
        WHERE is_semantic_umbrella = 0
          AND rejection_count >= ?
          AND (rejection_count * 1.0 / (uses + rejection_count + 0.001)) >= ?
        ORDER BY rejection_count DESC
        LIMIT ?
        """,
        (REJECTION_COUNT_THRESHOLD, REJECTION_RATE_THRESHOLD, limit),
    )

    results = []
    for row in cursor.fetchall():
        sig_id, rejection_count, uses = row
        total = uses + rejection_count
        results.append({
            "signature_id": sig_id,
            "rejection_count": rejection_count,
            "uses": uses,
            "rejection_rate": rejection_count / total if total > 0 else 0,
        })

    return results


def check_and_reject_if_low_similarity(
    signature_id: int,
    step_text: str,
    similarity: float,
    dag_step_id: str = None,
    problem_context: str = None,
    conn=None,
) -> tuple[bool, int]:
    """Check if similarity is below threshold and record rejection if so.

    Args:
        signature_id: The leaf signature being checked
        step_text: The step being routed
        similarity: Cosine similarity to the signature
        dag_step_id: Optional dag_step ID
        problem_context: Optional problem context
        conn: Optional DB connection (reuse caller's connection to avoid locks)

    Returns:
        Tuple of (was_rejected, rejection_count)
    """
    if similarity >= REJECTION_SIM_THRESHOLD:
        return False, 0

    rejection_count = record_leaf_rejection(
        signature_id=signature_id,
        step_text=step_text,
        similarity=similarity,
        dag_step_id=dag_step_id,
        problem_context=problem_context,
        conn=conn,
    )

    return True, rejection_count


def flag_high_rejection_leaves_for_decomposition(step_db=None) -> list[dict]:
    """Find and flag high-rejection leaves for decomposition.

    This should be called periodically (e.g., during batch operations).
    Uses the unified flag_for_split pathway which includes MCTS alternative check.

    Args:
        step_db: StepSignatureDB instance (required for MCTS check in flag_for_split)

    Returns:
        List of flagged signature stats
    """
    leaves = get_leaves_needing_decomposition()

    if not leaves:
        return []

    if step_db is None:
        logger.warning("[rejection] step_db required for MCTS-aware leaf decomposition")
        return []

    flagged = []

    for leaf in leaves:
        sig_id = leaf["signature_id"]

        # Use unified flag_for_split which includes MCTS alternative check
        was_flagged = step_db.flag_for_split(
            signature_id=sig_id,
            reason=f"high_rejection_rate_{leaf['rejection_rate']:.2f}",
            skip_mcts_check=False,  # Use MCTS check
        )

        if was_flagged:
            logger.info(
                "[rejection] Flagged leaf %d for decomposition: %d rejections, %.1f%% rate",
                sig_id, leaf["rejection_count"], leaf["rejection_rate"] * 100
            )
            flagged.append(leaf)

    return flagged


# =============================================================================
# DB MATURITY
# =============================================================================
# Maturity emerges from actual ground-truth performance.
# Used for various adaptive behaviors in the system.


def sigmoid(x: float, k: float = 1.0, midpoint: float = 0.0) -> float:
    """Sigmoid function for smooth transitions."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


def compute_db_maturity(rolling_window: int = 100) -> float:
    """Infer DB maturity from rolling window of recent problem outcomes.

    Maturity EMERGES from actual ground-truth performance:
    - Uses the last N graded problems (not leaf stats which can be stale)
    - 70% accuracy on GSM8K → ~0.70 maturity (intuitive!)

    This is much more accurate than leaf success rates because:
    1. It's ground truth (did we actually solve the problem?)
    2. It's recent (reflects current performance, not stale history)
    3. It's directly what we care about

    Args:
        rolling_window: Number of recent problems to consider (default 100)

    Returns:
        float: 0.0 (immature/struggling) to 1.0 (mature/working well)
    """
    conn = get_db()

    # Get the last N graded problems
    cursor = conn.execute(
        """
        SELECT success
        FROM mcts_dags
        WHERE success IS NOT NULL
        ORDER BY graded_at DESC
        LIMIT ?
        """,
        (rolling_window,),
    )
    outcomes = cursor.fetchall()

    if not outcomes:
        logger.debug("[maturity] No graded problems yet, returning 0.0 (cold start)")
        return 0.0  # Cold start → fully immature

    # Simple rolling accuracy
    total = len(outcomes)
    successes = sum(1 for row in outcomes if row["success"] == 1)
    maturity = successes / total

    logger.debug(
        "[maturity] DB maturity=%.3f from last %d problems (%d/%d correct)",
        maturity, total, successes, total
    )

    return maturity


# NOTE: DecompositionDecision, get_leaf_stats, and compute_decomposition_decision
# were removed - replaced by divergence-based splitting in step_signatures/divergence.py.
# Natural splitting happens based on observed success/failure divergence.


# =============================================================================
# DAG STEP FUNCTIONS
# =============================================================================


def create_dag_steps(dag_id: str, steps: list[tuple[str, int, int, bool, Optional[str]]]) -> list[str]:
    """Create DAG steps for a plan.

    Args:
        dag_id: Parent DAG ID
        steps: List of (step_desc, step_num, branch_num, is_atomic, dsl_hint)
            - step_desc: Natural language description of the step
            - step_num: Sequential order (1..n)
            - branch_num: Parallel branch ID
            - is_atomic: Whether step can be decomposed further
            - dsl_hint: Operation type hint (e.g., "compute_sum") for stats normalization

    Returns:
        List of dag_step_ids
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()

    step_ids = []
    for step_tuple in steps:
        # Support both old 4-tuple and new 5-tuple format for backward compatibility
        if len(step_tuple) == 5:
            step_desc, step_num, branch_num, is_atomic, dsl_hint = step_tuple
        else:
            step_desc, step_num, branch_num, is_atomic = step_tuple
            dsl_hint = None

        dag_step_id = f"step-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO mcts_dag_steps (dag_step_id, dag_id, step_desc, dsl_hint,
                                        step_num, branch_num, is_atomic, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dag_step_id, dag_id, step_desc, dsl_hint, step_num, branch_num, 1 if is_atomic else 0, now),
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
) -> tuple[str, int]:
    """Log a thread step execution with wave function amplitude.

    This is the core logging function for MCTS post-mortem analysis.
    The (dag_step_id, node_id) combination is what we're learning.

    Two-table strategy for efficiency:
    - mcts_step_summaries: Always stores minimal data for credit propagation
    - mcts_thread_steps: Only stores detailed records for failures (debugging)

    Args:
        node_depth: Depth of the signature node in the tree (for post-mortem analysis)

    Returns:
        Tuple of (thread_step_id, summary_id) for later amplitude_post updates.
    """
    from mycelium.config import LOG_DETAILED_STEPS_FAILURES_ONLY

    thread_step_id = f"tstep-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    step_success_int = (1 if step_success else 0) if step_success is not None else None

    conn = get_db()

    # Always insert into summaries table (minimal data for credit propagation)
    # Per mycelium-i601: Include similarity_score for adaptive rejection threshold learning
    cursor = conn.execute(
        """
        INSERT INTO mcts_step_summaries (
            thread_id, dag_id, dag_step_id, node_id,
            amplitude, step_success, similarity_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (thread_id, dag_id, dag_step_id, node_id, amplitude, step_success_int, similarity_score),
    )
    summary_id = cursor.lastrowid

    # Only insert detailed records for failures (or if optimization is disabled)
    should_log_details = (
        not LOG_DETAILED_STEPS_FAILURES_ONLY  # Optimization disabled
        or step_success is False              # Failure - always log details
        or step_success is None               # Unknown - log details for safety
    )

    if should_log_details:
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
                alternatives_considered, step_result, step_success_int, now,
            ),
        )

    return thread_step_id, summary_id


def update_amplitude_post(thread_step_id: str, amplitude_post: float) -> None:
    """Update the post-observation amplitude for a thread step (detailed table).

    Called after wave function collapse (grading).
    Note: For credit propagation, use update_summary_amplitude_post() instead.
    """
    conn = get_db()
    conn.execute(
        """
        UPDATE mcts_thread_steps SET amplitude_post = ? WHERE thread_step_id = ?
        """,
        (amplitude_post, thread_step_id),
    )


def update_summary_amplitude_post(summary_id: int, amplitude_post: float) -> None:
    """Update amplitude_post in the summaries table (used for credit propagation).

    Args:
        summary_id: The ID from mcts_step_summaries (returned by log_thread_step)
        amplitude_post: The post-observation amplitude value
    """
    conn = get_db()
    conn.execute(
        """
        UPDATE mcts_step_summaries SET amplitude_post = ? WHERE id = ?
        """,
        (amplitude_post, summary_id),
    )


def batch_update_amplitudes(updates: list[tuple[str, float]]) -> None:
    """Batch update amplitude_post values in detailed table.

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


def batch_update_summary_amplitudes(updates: list[tuple[int, float]]) -> None:
    """Batch update amplitude_post values in summaries table.

    Args:
        updates: List of (summary_id, amplitude_post) tuples
    """
    if not updates:
        return

    conn = get_db()
    with conn.connection() as raw_conn:
        raw_conn.executemany(
            """
            UPDATE mcts_step_summaries SET amplitude_post = ? WHERE id = ?
            """,
            [(amp, sid) for sid, amp in updates],
        )
    logger.debug("[mcts] Batch updated %d summary amplitudes", len(updates))


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
# POST-MORTEM ANALYSIS (Consolidated Single Pathway)
# =============================================================================


def _compute_amplitude_post(dag_id: str) -> dict:
    """Core amplitude_post computation for post-mortem analysis.

    Internal function - use run_postmortem() as the public entry point.

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

    # Get all step summaries for this DAG (lightweight table for credit propagation)
    cursor = conn.execute(
        """
        SELECT id, thread_id, amplitude
        FROM mcts_step_summaries
        WHERE dag_id = ?
        """,
        (dag_id,),
    )
    step_summaries = cursor.fetchall()

    if not step_summaries:
        logger.debug("[mcts] No step_summaries found for DAG %s", dag_id)
        return {"total_steps": 0, "threads_won": 0, "threads_lost": 0}

    # Compute amplitude_post for each step
    updates = []
    stats = {
        "total_steps": len(step_summaries),
        "threads_won": sum(1 for s in thread_outcomes.values() if s == 1),
        "threads_lost": sum(1 for s in thread_outcomes.values() if s == 0),
        "high_conf_wrong": 0,
        "low_conf_right": 0,
        "total_high_conf": 0,  # For UCB1 adjustment (mycelium-nirq)
        "total_low_conf": 0,   # For UCB1 adjustment (mycelium-nirq)
    }

    for summary_id, thread_id, amplitude in step_summaries:
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

        # Compute amplitude_post based on outcome × confidence (fixed multipliers)
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
        updates.append((summary_id, amplitude_post))

    # Batch update summaries table (used for credit propagation)
    if updates:
        batch_update_summary_amplitudes(updates)
        logger.info(
            "[mcts] Post-mortem for DAG %s: %d steps, %d won, %d lost, "
            "%d high-conf-wrong, %d low-conf-right",
            dag_id, stats["total_steps"], stats["threads_won"], stats["threads_lost"],
            stats["high_conf_wrong"], stats["low_conf_right"],
        )

    return stats


def run_postmortem(
    dag_id: str,
    step_db=None,
    step_embeddings: dict = None,
    include_interference: bool = True,
    include_diagnostics: bool = True,
) -> dict:
    """Single pathway for post-mortem analysis on a completed DAG.

    This is the main entry point for all post-mortem analysis. Features are
    enabled based on parameters provided:

    - Always: amplitude_post computation (confidence × outcome → amplitude adjustments)
    - If step_db provided: interference detection, credit propagation, merge/split batching
    - If step_embeddings provided + include_diagnostics: failure diagnosis with verdicts

    Args:
        dag_id: The DAG to analyze
        step_db: Optional StepSignatureDB instance. If provided, enables interference
            detection, credit propagation, and structural operations.
        step_embeddings: Optional dict mapping dag_step_id to embeddings. Enables
            enhanced diagnostic analysis with rerouting recommendations.
        include_interference: If True (default), run interference pattern analysis
            when step_db is provided.
        include_diagnostics: If True (default), run diagnostic analysis when step_db
            is provided.

    Returns:
        Combined dict with all analysis results. Keys depend on features enabled.
    """
    from mycelium.config import DIAGNOSTIC_POSTMORTEM_ENABLED

    # If no step_db, just compute amplitudes (fast path for tests)
    if step_db is None:
        return _compute_amplitude_post(dag_id)

    # Delegate to full implementation (avoids code duplication)
    # run_postmortem_with_interference contains all interference logic including variance decomposition
    result = run_postmortem_with_interference(dag_id, step_db)

    # Add diagnostic analysis if requested
    if include_diagnostics and DIAGNOSTIC_POSTMORTEM_ENABLED:
        diagnostic_result = _run_diagnostic_analysis(dag_id, step_db, step_embeddings)
        if not diagnostic_result.get("skipped"):
            result.update({
                "diagnostic_system_maturity": diagnostic_result.get("system_maturity", 0),
                "diagnostic_failure_threshold": diagnostic_result.get("failure_threshold", 0),
                "diagnostic_pairs_analyzed": diagnostic_result.get("pairs_analyzed", 0),
                "steps_to_decompose": diagnostic_result.get("steps_to_decompose", []),
                "signatures_to_decompose": diagnostic_result.get("signatures_to_decompose", []),
                "routing_misses": diagnostic_result.get("routing_misses", []),
                "diagnoses": diagnostic_result.get("diagnoses", []),
            })

    return result


def _run_diagnostic_analysis(dag_id: str, step_db, step_embeddings: dict = None) -> dict:
    """Internal diagnostic analysis for post-mortem.

    Analyzes failures to determine what should be decomposed.
    Called from run_postmortem() when include_diagnostics=True.
    """
    # Get system maturity (signature count)
    system_maturity = step_db.count_signatures() if step_db else 0
    failure_threshold = get_diagnostic_failure_threshold(system_maturity)

    # Get thread steps for this DAG
    thread_steps = get_thread_steps_for_dag(dag_id)

    # Group by (dag_step_id, node_id) to find repeated failures
    pair_failures = {}
    pair_embeddings = {}

    for ts in thread_steps:
        if ts.step_success == 0:
            key = (ts.dag_step_id, ts.node_id)
            pair_failures[key] = pair_failures.get(key, 0) + 1
            if step_embeddings and ts.dag_step_id in step_embeddings:
                pair_embeddings[key] = step_embeddings[ts.dag_step_id]

    # Diagnose pairs that exceed threshold
    diagnoses = []
    steps_to_decompose = []
    sigs_to_decompose = []
    routing_misses = []

    for (dag_step_id, node_id), failure_count in pair_failures.items():
        if failure_count >= failure_threshold:
            step_emb = pair_embeddings.get((dag_step_id, node_id))
            diagnosis = diagnose_failure(node_id, step_emb, step_db)

            diagnoses.append({
                "dag_step_id": dag_step_id,
                "node_id": node_id,
                "failure_count": failure_count,
                "verdict": diagnosis.verdict,
                "scores": {
                    "decompose_step": diagnosis.decompose_step_score,
                    "decompose_sig": diagnosis.decompose_sig_score,
                    "reroute": diagnosis.reroute_score,
                },
                "accuracy": diagnosis.accuracy,
                "confidence": diagnosis.confidence,
            })

            if diagnosis.verdict == "decompose_step":
                steps_to_decompose.append(dag_step_id)
            elif diagnosis.verdict == "decompose_signature":
                sigs_to_decompose.append(node_id)
            elif diagnosis.verdict == "reroute":
                routing_misses.append((dag_step_id, node_id))

    logger.info(
        "[diagnostic] Post-mortem for %s: threshold=%.1f, pairs_analyzed=%d, "
        "steps_decompose=%d, sigs_decompose=%d, reroutes=%d",
        dag_id, failure_threshold, len(pair_failures),
        len(steps_to_decompose), len(sigs_to_decompose), len(routing_misses),
    )

    return {
        "dag_id": dag_id,
        "system_maturity": system_maturity,
        "failure_threshold": failure_threshold,
        "pairs_analyzed": len(pair_failures),
        "diagnoses": diagnoses,
        "steps_to_decompose": steps_to_decompose,
        "signatures_to_decompose": sigs_to_decompose,
        "routing_misses": routing_misses,
    }


def propagate_amplitude_to_signature_stats(dag_id: str, step_db) -> dict:
    """Propagate amplitude_post values to signature stats with step-level precision.

    Per beads mycelium-itkn + mycelium-7o8i: Close the loop from post-mortem to
    signature learning, with STEP-LEVEL success for precise blame attribution.

    Key insight: In a multi-step problem, only the failing step should be blamed.
    Steps 1-3 might succeed while step 4 fails - we should credit 1-3 and blame 4.

    Credit logic (priority order):
    1. step_success=1 (step itself succeeded) → FULL CREDIT (even if thread lost)
    2. step_success=0 (step itself failed) → BLAME (even if thread won - edge case)
    3. step_success=NULL + thread won → full credit (fallback)
    4. step_success=NULL + thread lost + high amplitude → partial credit
    5. step_success=NULL + thread lost + low amplitude → blame

    Credit propagates UP to parent routers with decay per CLAUDE.md.

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

    # Get per-node stats with STEP-LEVEL success (ss.step_success) as primary signal
    # Fall back to thread-level (t.success) + amplitude when step_success is NULL
    # NOTE: Uses mcts_step_summaries (lightweight) instead of mcts_thread_steps (detailed)
    cursor = conn.execute(
        """
        SELECT
            ss.node_id,
            COUNT(*) as total_steps,
            -- STEP-LEVEL: Steps that succeeded at execution (most precise signal)
            SUM(CASE WHEN ss.step_success = 1 THEN 1 ELSE 0 END) as step_succeeded,
            -- STEP-LEVEL: Steps that failed at execution (precise blame)
            SUM(CASE WHEN ss.step_success = 0 THEN 1 ELSE 0 END) as step_failed,
            -- FALLBACK: Steps in winning threads (when step_success is NULL)
            SUM(CASE WHEN ss.step_success IS NULL AND t.success = 1 THEN 1 ELSE 0 END) as thread_won_no_step,
            -- FALLBACK: High-confidence steps in losing threads (partial credit)
            SUM(CASE WHEN ss.step_success IS NULL AND t.success = 0 AND ss.amplitude >= ? THEN 1 ELSE 0 END) as high_conf_losing,
            -- FALLBACK: Low-confidence steps in losing threads (blame)
            SUM(CASE WHEN ss.step_success IS NULL AND t.success = 0 AND ss.amplitude < ? THEN 1 ELSE 0 END) as low_conf_losing,
            -- Average amplitude_post for reference
            AVG(ss.amplitude_post) as avg_amplitude_post
        FROM mcts_step_summaries ss
        JOIN mcts_threads t ON ss.thread_id = t.thread_id
        WHERE ss.dag_id = ? AND ss.amplitude_post IS NOT NULL
        GROUP BY ss.node_id
        """,
        (PARTIAL_CREDIT_HIGH_CONF_THRESHOLD, PARTIAL_CREDIT_HIGH_CONF_THRESHOLD, dag_id),
    )

    stats = {
        "nodes_processed": 0,
        "successes_credited": 0,
        "partial_credits": 0,
        "failures_credited": 0,
        "step_level_credit": 0,  # New: track how many used step-level success
        "step_level_blame": 0,   # New: track how many used step-level failure
    }

    for row in cursor.fetchall():
        (node_id, total_steps, step_succeeded, step_failed,
         thread_won_no_step, high_conf_losing, low_conf_losing, avg_amp) = row

        if total_steps == 0 or node_id is None:
            continue

        stats["nodes_processed"] += 1

        # Priority 1: STEP-LEVEL SUCCESS - step itself succeeded
        # This is the most precise signal - step executed without error
        if step_succeeded > 0:
            step_db.increment_signature_successes(node_id, count=1, propagate_to_parents=True)
            stats["successes_credited"] += 1
            stats["step_level_credit"] += 1
            logger.debug(
                "[mcts] Step-level credit to node %d (%d steps succeeded)",
                node_id, step_succeeded
            )

        # Priority 2: STEP-LEVEL FAILURE - step itself failed
        # This is precise blame - this specific step caused the problem
        elif step_failed > 0:
            step_db.increment_signature_failures(node_id, count=1, propagate_to_parents=True)
            stats["failures_credited"] += 1
            stats["step_level_blame"] += 1
            logger.debug(
                "[mcts] Step-level blame to node %d (%d steps failed)",
                node_id, step_failed
            )

        # Priority 3: FALLBACK - thread won but step_success not tracked
        elif thread_won_no_step > 0:
            step_db.increment_signature_successes(node_id, count=1, propagate_to_parents=True)
            stats["successes_credited"] += 1
            logger.debug(
                "[mcts] Thread-level credit to node %d (%d steps in winning thread)",
                node_id, thread_won_no_step
            )

        # Priority 4: FALLBACK - high-confidence in losing thread (partial credit)
        elif high_conf_losing > 0:
            step_db.increment_signature_partial_success(
                node_id, weight=PARTIAL_CREDIT_WEIGHT, propagate_to_parents=True
            )
            stats["partial_credits"] += 1
            logger.debug(
                "[mcts] Partial credit to node %d (%d high-conf losing steps, avg_amp=%.2f)",
                node_id, high_conf_losing, avg_amp or 0
            )

        # Priority 5: FALLBACK - low-confidence in losing thread (blame)
        elif low_conf_losing > 0:
            step_db.increment_signature_failures(node_id, count=1, propagate_to_parents=True)
            stats["failures_credited"] += 1
            logger.debug(
                "[mcts] Thread-level blame to node %d (%d low-conf losing steps, avg_amp=%.2f)",
                node_id, low_conf_losing, avg_amp or 0
            )

    if stats["nodes_processed"] > 0:
        logger.info(
            "[mcts] Credit propagation for DAG %s: %d nodes, +%d full, +%d partial, +%d failures "
            "(step-level: %d credit, %d blame)",
            dag_id, stats["nodes_processed"],
            stats["successes_credited"], stats["partial_credits"], stats["failures_credited"],
            stats["step_level_credit"], stats["step_level_blame"],
        )

    return stats


def propagate_success_similarity(dag_id: str, step_db) -> dict:
    """Propagate similarity scores from successful threads to leaf adaptive thresholds.

    Per mycelium-i601: When a thread succeeds, the similarity scores used during
    routing should be recorded on the leaf signatures. This feeds into the adaptive
    rejection threshold: threshold = mean - k * std.

    Only propagates similarity from thread steps where:
    1. The thread won (t.success = 1)
    2. A similarity_score was recorded (not NULL)

    Args:
        dag_id: The DAG to process
        step_db: StepSignatureDB instance for updating success_sim stats

    Returns:
        Dict with propagation statistics
    """
    from mycelium.config import CREDIT_PROPAGATION_ENABLED

    if not CREDIT_PROPAGATION_ENABLED:
        return {"updates": 0, "skipped": True}

    conn = get_db()

    # Get (node_id, similarity_score) pairs from winning threads
    # Use mcts_step_summaries (always populated) instead of mcts_thread_steps (failures only)
    cursor = conn.execute(
        """
        SELECT ss.node_id, ss.similarity_score
        FROM mcts_step_summaries ss
        JOIN mcts_threads t ON ss.thread_id = t.thread_id
        WHERE ss.dag_id = ?
          AND t.success = 1
          AND ss.similarity_score IS NOT NULL
          AND ss.node_id IS NOT NULL
        """,
        (dag_id,)
    )

    updates = []
    for row in cursor.fetchall():
        node_id = row["node_id"]
        similarity = row["similarity_score"]
        if node_id is not None and similarity is not None:
            updates.append((node_id, similarity))

    if updates:
        updated_count = step_db.update_success_similarity_batch(updates)
        logger.debug(
            "[mcts] Propagated %d success similarities for DAG %s",
            updated_count, dag_id
        )
        return {"updates": updated_count, "skipped": False}

    return {"updates": 0, "skipped": False}


# =============================================================================
# STEP-NODE STATS (Materialized (dag_step_type, node_id) performance)
# =============================================================================
# Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning"
# This closes the feedback loop: post-mortem → dag_step_node_stats → routing UCB1


def update_dag_step_node_stats(
    dag_step_type: str,
    node_id: int,
    won: bool,
    amplitude_post: float,
) -> None:
    """Upsert stats for a (dag_step_type, node_id) pair.

    Called during post-mortem to materialize (step, node) performance.
    Routing then queries these stats to make better decisions.

    Uses Welford's online algorithm for numerically stable variance tracking.
    High variance in amplitude_post = inconsistent performance = decomposition signal.

    Args:
        dag_step_type: The step type (e.g., "compute_sum", "compute_product")
        node_id: The signature ID that handled this step
        won: Whether the thread won (correct answer)
        amplitude_post: The post-observation amplitude for this step
    """
    from mycelium.config import (
        STEP_NODE_STATS_ENABLED,
        STEP_NODE_STATS_PRIOR_WINS,
        STEP_NODE_STATS_PRIOR_USES,
    )

    if not STEP_NODE_STATS_ENABLED:
        return

    # Validate dag_step_type
    if not dag_step_type or not isinstance(dag_step_type, str):
        logger.warning("[mcts] Invalid dag_step_type: %r, skipping stats update", dag_step_type)
        return
    # Truncate excessively long step types (sanity check)
    if len(dag_step_type) > 200:
        dag_step_type = dag_step_type[:200]

    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    win_inc = 1 if won else 0
    loss_inc = 0 if won else 1

    # Step 1: Check if row exists and get current Welford state
    cursor = conn.execute(
        """
        SELECT amp_post_count, amp_post_mean, amp_post_m2
        FROM dag_step_node_stats
        WHERE dag_step_type = ? AND node_id = ?
        """,
        (dag_step_type, node_id),
    )
    row = cursor.fetchone()

    if row is None:
        # New row: initialize with first observation
        # Welford's: n=1, mean=x, m2=0
        conn.execute(
            """
            INSERT INTO dag_step_node_stats (
                dag_step_type, node_id, uses, wins, losses,
                amplitude_post_sum, amp_post_count, amp_post_mean, amp_post_m2,
                last_updated
            )
            VALUES (?, ?, 1, ?, ?, ?, 1, ?, 0.0, ?)
            """,
            (
                dag_step_type,
                node_id,
                win_inc,
                loss_inc,
                amplitude_post,  # amplitude_post_sum (legacy)
                amplitude_post,  # amp_post_mean (Welford's)
                now,
            ),
        )
    else:
        # Existing row: update using Welford's algorithm
        old_count, old_mean, old_m2 = row
        old_count = old_count or 0
        old_mean = old_mean or 0.0
        old_m2 = old_m2 or 0.0

        # Welford's update formula:
        # n = n + 1
        # delta = x - mean
        # mean = mean + delta / n
        # delta2 = x - mean  (using NEW mean)
        # m2 = m2 + delta * delta2
        new_count = old_count + 1
        delta = amplitude_post - old_mean
        new_mean = old_mean + delta / new_count
        delta2 = amplitude_post - new_mean
        new_m2 = old_m2 + delta * delta2

        conn.execute(
            """
            UPDATE dag_step_node_stats
            SET uses = uses + 1,
                wins = wins + ?,
                losses = losses + ?,
                amplitude_post_sum = amplitude_post_sum + ?,
                amp_post_count = ?,
                amp_post_mean = ?,
                amp_post_m2 = ?,
                last_updated = ?
            WHERE dag_step_type = ? AND node_id = ?
            """,
            (
                win_inc,
                loss_inc,
                amplitude_post,
                new_count,
                new_mean,
                new_m2,
                now,
                dag_step_type,
                node_id,
            ),
        )

    # Step 2: Compute derived fields (win_rate, avg_amplitude_post) in Python
    # This is clearer than complex inline SQL and applies Bayesian priors correctly
    conn.execute(
        """
        UPDATE dag_step_node_stats
        SET win_rate = CAST(wins + ? AS REAL) / (uses + ?),
            avg_amplitude_post = amplitude_post_sum / uses
        WHERE dag_step_type = ? AND node_id = ?
        """,
        (
            STEP_NODE_STATS_PRIOR_WINS,
            STEP_NODE_STATS_PRIOR_USES,
            dag_step_type,
            node_id,
        ),
    )

    logger.debug(
        "[mcts] Updated step-node stats: %s/node_%d won=%s amp_post=%.2f",
        dag_step_type, node_id, won, amplitude_post
    )


def get_dag_step_node_stats_batch(
    dag_step_type: str,
    node_ids: list[int],
) -> dict[int, dict]:
    """Batch query stats for multiple (dag_step_type, node_id) pairs.

    Used by routing to efficiently fetch step-specific performance data.

    Args:
        dag_step_type: The step type to query
        node_ids: List of node IDs to fetch stats for

    Returns:
        Dict mapping node_id → stats dict with keys:
        - uses, wins, losses, win_rate, avg_amplitude_post
        - amp_post_variance, amp_post_std (computed from Welford's stats)
    """
    from mycelium.config import STEP_NODE_STATS_ENABLED

    if not STEP_NODE_STATS_ENABLED or not node_ids:
        return {}

    # Validate dag_step_type
    if not dag_step_type or not isinstance(dag_step_type, str):
        return {}

    conn = get_db()

    # Build placeholders for IN clause
    placeholders = ",".join("?" * len(node_ids))

    cursor = conn.execute(
        f"""
        SELECT node_id, uses, wins, losses, win_rate, avg_amplitude_post,
               amp_post_count, amp_post_mean, amp_post_m2
        FROM dag_step_node_stats
        WHERE dag_step_type = ? AND node_id IN ({placeholders})
        """,
        [dag_step_type] + list(node_ids),
    )

    result = {}
    for row in cursor.fetchall():
        # Compute variance from Welford's M2: variance = M2 / N
        amp_count = row["amp_post_count"] or 0
        amp_m2 = row["amp_post_m2"] or 0.0
        variance = amp_m2 / amp_count if amp_count > 0 else 0.0
        std = variance ** 0.5 if variance > 0 else 0.0

        result[row["node_id"]] = {
            "uses": row["uses"],
            "wins": row["wins"],
            "losses": row["losses"],
            "win_rate": row["win_rate"],
            "avg_amplitude_post": row["avg_amplitude_post"],
            # Welford's derived stats
            "amp_post_count": amp_count,
            "amp_post_mean": row["amp_post_mean"] or 0.0,
            "amp_post_variance": variance,
            "amp_post_std": std,
        }

    return result


def get_dag_step_node_stats_single(
    dag_step_type: str,
    node_id: int,
) -> Optional[dict]:
    """Get stats for a single (dag_step_type, node_id) pair.

    Args:
        dag_step_type: The step type
        node_id: The signature ID

    Returns:
        Stats dict or None if no data exists
    """
    batch = get_dag_step_node_stats_batch(dag_step_type, [node_id])
    return batch.get(node_id)


def get_high_variance_step_node_pairs(
    min_samples: int = 5,
    variance_threshold: float = 0.1,
    limit: int = 20,
) -> list[dict]:
    """Find (step_type, node_id) pairs with high amplitude_post variance.

    High variance indicates inconsistent performance - sometimes the node works
    well for this step type, sometimes it doesn't. This is a signal that the
    node is too generic and should be decomposed into specialized children.

    Per CLAUDE.md: "Destructive interference (mixed success/failure at same node)"
    triggers decomposition.

    Args:
        min_samples: Minimum observations before considering variance (cold start)
        variance_threshold: Minimum variance to flag as "high" (default 0.1)
        limit: Maximum number of pairs to return

    Returns:
        List of dicts with: dag_step_type, node_id, variance, std, uses, win_rate
        Sorted by variance descending (most inconsistent first)
    """
    from mycelium.config import STEP_NODE_STATS_ENABLED

    if not STEP_NODE_STATS_ENABLED:
        return []

    conn = get_db()

    # Query pairs with enough samples and compute variance
    cursor = conn.execute(
        """
        SELECT
            dag_step_type,
            node_id,
            amp_post_count,
            amp_post_mean,
            amp_post_m2,
            uses,
            win_rate
        FROM dag_step_node_stats
        WHERE amp_post_count >= ?
        ORDER BY amp_post_m2 / amp_post_count DESC
        LIMIT ?
        """,
        (min_samples, limit * 2),  # Fetch extra to filter by threshold
    )

    results = []
    for row in cursor.fetchall():
        count = row["amp_post_count"]
        m2 = row["amp_post_m2"] or 0.0
        variance = m2 / count if count > 0 else 0.0

        if variance >= variance_threshold:
            results.append({
                "dag_step_type": row["dag_step_type"],
                "node_id": row["node_id"],
                "amp_post_variance": variance,
                "amp_post_std": variance ** 0.5,
                "amp_post_mean": row["amp_post_mean"],
                "uses": row["uses"],
                "win_rate": row["win_rate"],
            })

        if len(results) >= limit:
            break

    return results


def propagate_step_node_stats(dag_id: str) -> dict:
    """Propagate post-mortem results to dag_step_node_stats table.

    Called after amplitude_post is computed to materialize (step, node) stats.
    These stats are then used by routing UCB1 to make better decisions.

    Args:
        dag_id: The DAG to process

    Returns:
        Dict with propagation statistics
    """
    from mycelium.config import STEP_NODE_STATS_ENABLED

    if not STEP_NODE_STATS_ENABLED:
        return {"pairs_updated": 0, "skipped": True}

    conn = get_db()

    # Get all step summaries with their thread outcomes and dag_step info
    # Use dsl_hint (operation type) when available, fall back to step_desc
    # Per mycelium-mgbs: dsl_hint provides better normalization than NL descriptions
    # NOTE: Uses mcts_step_summaries (lightweight) instead of mcts_thread_steps (detailed)
    cursor = conn.execute(
        """
        SELECT
            ss.node_id,
            ss.amplitude_post,
            t.success as thread_won,
            ds.dsl_hint,
            ds.step_desc
        FROM mcts_step_summaries ss
        JOIN mcts_threads t ON ss.thread_id = t.thread_id
        JOIN mcts_dag_steps ds ON ss.dag_step_id = ds.dag_step_id
        WHERE ss.dag_id = ? AND ss.amplitude_post IS NOT NULL AND ss.node_id IS NOT NULL
        """,
        (dag_id,),
    )

    stats = {"pairs_updated": 0}

    for row in cursor.fetchall():
        node_id, amplitude_post, thread_won, dsl_hint, step_desc = row

        # Use dsl_hint (e.g., "compute_sum") for better normalization
        # Fall back to step_desc if dsl_hint not available
        dag_step_type = dsl_hint or step_desc or "unknown"

        update_dag_step_node_stats(
            dag_step_type=dag_step_type,
            node_id=node_id,
            won=bool(thread_won),
            amplitude_post=amplitude_post,
        )
        stats["pairs_updated"] += 1

    if stats["pairs_updated"] > 0:
        logger.info(
            "[mcts] Step-node stats propagated for DAG %s: %d pairs updated",
            dag_id, stats["pairs_updated"]
        )

    return stats


# =============================================================================
# DAG_STEP EMBEDDINGS: Semantic similarity for decomposition decisions
def store_dag_step_embedding(
    dag_id: str,
    dag_step_id: str,
    step_desc: str,
    embedding: "np.ndarray",
    node_id: Optional[int] = None,
) -> int:
    """Store embedding for a dag_step.

    Called when a dag_step is created/executed to enable similarity lookups.

    Args:
        dag_id: The DAG this step belongs to
        dag_step_id: Unique step ID
        step_desc: The step description (task)
        embedding: The embedding vector
        node_id: Which leaf_node handled it (None if not yet executed)

    Returns:
        The ID of the inserted row
    """
    from datetime import datetime, timezone
    from mycelium.step_signatures.db import pack_embedding

    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    embedding_packed = pack_embedding(embedding)

    # Use the ConnectionManager's execute which auto-commits
    cursor = conn.execute(
        """INSERT INTO dag_step_embeddings
           (dag_id, dag_step_id, step_desc, embedding, node_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (dag_id, dag_step_id, step_desc, embedding_packed, node_id, now),
    )
    return cursor.lastrowid


def update_dag_step_embedding_outcome(
    dag_step_id: str,
    node_id: int,
    success: bool,
) -> None:
    """Update the outcome for a dag_step embedding record.

    Called after thread grading to record whether the step succeeded.

    Args:
        dag_step_id: The step ID
        node_id: The leaf_node that handled it
        success: Whether the step succeeded
    """
    conn = get_db()
    # ConnectionManager.execute() auto-commits
    conn.execute(
        """UPDATE dag_step_embeddings
           SET node_id = ?, success = ?
           WHERE dag_step_id = ?""",
        (node_id, 1 if success else 0, dag_step_id),
    )


def find_similar_dag_steps(
    embedding: "np.ndarray",
    limit: int = 20,
    min_similarity: float = 0.7,
) -> list[dict]:
    """Find dag_steps similar to the given embedding.

    Used to check: "Have we seen steps like this before? How did they do?"

    Args:
        embedding: The embedding to compare against
        limit: Maximum number of results
        min_similarity: Minimum cosine similarity threshold

    Returns:
        List of dicts with: dag_step_id, step_desc, node_id, success, similarity
    """
    import numpy as np
    from mycelium.step_signatures.db import unpack_embedding

    conn = get_db()
    cursor = conn.execute(
        """SELECT dag_step_id, step_desc, embedding, node_id, success
           FROM dag_step_embeddings
           WHERE embedding IS NOT NULL AND success IS NOT NULL"""
    )

    results = []
    embedding_norm = np.linalg.norm(embedding)
    if embedding_norm == 0:
        return results

    for row in cursor.fetchall():
        dag_step_id, step_desc, emb_packed, node_id, success = row
        stored_emb = unpack_embedding(emb_packed)
        if stored_emb is None:
            continue

        stored_norm = np.linalg.norm(stored_emb)
        if stored_norm == 0:
            continue

        similarity = float(np.dot(embedding, stored_emb) / (embedding_norm * stored_norm))

        if similarity >= min_similarity:
            results.append({
                "dag_step_id": dag_step_id,
                "step_desc": step_desc,
                "node_id": node_id,
                "success": success,
                "similarity": similarity,
            })

    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


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
# POSTMORTEM STATE MANAGEMENT (Database-backed for cross-process persistence)
# =============================================================================

import json


def _get_db_state_value(key: str, default: str = "0") -> str:
    """Get a value from db_metadata table."""
    db = get_db()
    with db.connection() as conn:
        row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def _set_db_state_value(key: str, value: str) -> None:
    """Set a value in db_metadata table (upsert)."""
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    with db.connection() as conn:
        conn.execute(
            """INSERT INTO db_metadata (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (key, value, now)
        )


# Keys for persistent state
_KEY_DSL_REGEN_COUNT = "postmortem_dsl_regen_count"
_KEY_HIGH_CONF_WRONG_NODES = "postmortem_high_conf_wrong_nodes"
_KEY_PROBLEM_COUNT = "postmortem_problem_count"
_KEY_NODES_FOR_SPLIT = "postmortem_nodes_for_split"
_KEY_POSTMORTEM_RUN_COUNT = "postmortem_run_count"
_KEY_BATCH_OPS_RUN_COUNT = "postmortem_batch_ops_count"


@dataclass
class PostmortemState:
    """Encapsulates mutable state for post-mortem batching.

    Now uses database for persistence across processes.
    In-memory fields are loaded from DB on access.
    """
    # In-memory cache (loaded from DB)
    _problem_count: int = field(default=None, repr=False)
    _nodes_for_split: list[int] = field(default=None, repr=False)
    _high_conf_wrong_nodes: list[int] = field(default=None, repr=False)
    _dsl_regen_problem_count: int = field(default=None, repr=False)
    _postmortem_run_count: int = field(default=None, repr=False)
    _batch_ops_run_count: int = field(default=None, repr=False)

    @property
    def problem_count(self) -> int:
        if self._problem_count is None:
            self._problem_count = int(_get_db_state_value(_KEY_PROBLEM_COUNT, "0"))
        return self._problem_count

    @property
    def nodes_for_split(self) -> list[int]:
        if self._nodes_for_split is None:
            raw = _get_db_state_value(_KEY_NODES_FOR_SPLIT, "[]")
            self._nodes_for_split = json.loads(raw)
        return self._nodes_for_split

    @property
    def high_conf_wrong_nodes(self) -> list[int]:
        if self._high_conf_wrong_nodes is None:
            raw = _get_db_state_value(_KEY_HIGH_CONF_WRONG_NODES, "[]")
            self._high_conf_wrong_nodes = json.loads(raw)
        return self._high_conf_wrong_nodes

    @property
    def dsl_regen_problem_count(self) -> int:
        if self._dsl_regen_problem_count is None:
            self._dsl_regen_problem_count = int(_get_db_state_value(_KEY_DSL_REGEN_COUNT, "0"))
        return self._dsl_regen_problem_count

    @property
    def postmortem_run_count(self) -> int:
        """Total number of times post-mortem analysis has run."""
        if self._postmortem_run_count is None:
            self._postmortem_run_count = int(_get_db_state_value(_KEY_POSTMORTEM_RUN_COUNT, "0"))
        return self._postmortem_run_count

    def increment_run_count(self) -> int:
        """Increment and persist the post-mortem run count. Returns new count."""
        current = self.postmortem_run_count
        self._postmortem_run_count = current + 1
        _set_db_state_value(_KEY_POSTMORTEM_RUN_COUNT, str(self._postmortem_run_count))
        return self._postmortem_run_count

    @property
    def batch_ops_run_count(self) -> int:
        """Total number of times batch operations (merge/split) have run."""
        if self._batch_ops_run_count is None:
            self._batch_ops_run_count = int(_get_db_state_value(_KEY_BATCH_OPS_RUN_COUNT, "0"))
        return self._batch_ops_run_count

    def increment_batch_ops_count(self) -> int:
        """Increment and persist the batch ops run count. Returns new count."""
        current = self.batch_ops_run_count
        self._batch_ops_run_count = current + 1
        _set_db_state_value(_KEY_BATCH_OPS_RUN_COUNT, str(self._batch_ops_run_count))
        return self._batch_ops_run_count

    def reset_merge_split(self) -> None:
        """Reset state after merge/split batch processing."""
        self._problem_count = 0
        self._nodes_for_split = []
        _set_db_state_value(_KEY_PROBLEM_COUNT, "0")
        _set_db_state_value(_KEY_NODES_FOR_SPLIT, "[]")

    def reset_dsl_regen(self) -> None:
        """Reset state after DSL regeneration batch."""
        self._high_conf_wrong_nodes = []
        self._dsl_regen_problem_count = 0
        _set_db_state_value(_KEY_HIGH_CONF_WRONG_NODES, "[]")
        _set_db_state_value(_KEY_DSL_REGEN_COUNT, "0")

    def accumulate_split_nodes(self, node_ids: list[int]) -> None:
        """Add nodes flagged for split."""
        # Load current state from DB
        current_nodes = list(self.nodes_for_split)
        current_count = self.problem_count

        # Update
        current_nodes.extend(node_ids)
        current_count += 1

        # Save back to DB
        self._nodes_for_split = current_nodes
        self._problem_count = current_count
        _set_db_state_value(_KEY_NODES_FOR_SPLIT, json.dumps(current_nodes))
        _set_db_state_value(_KEY_PROBLEM_COUNT, str(current_count))

    def accumulate_high_conf_wrong(self, nodes: list[dict]) -> None:
        """Add nodes with high-confidence wrong decisions."""
        # Load current state from DB (force refresh)
        self._high_conf_wrong_nodes = None
        self._dsl_regen_problem_count = None
        current_nodes = list(self.high_conf_wrong_nodes)
        current_count = self.dsl_regen_problem_count

        # Update
        for node in nodes:
            node_id = node.get("node_id")
            if node_id is not None and node_id not in current_nodes:
                current_nodes.append(node_id)
        current_count += 1

        # Save back to DB
        self._high_conf_wrong_nodes = current_nodes
        self._dsl_regen_problem_count = current_count
        _set_db_state_value(_KEY_HIGH_CONF_WRONG_NODES, json.dumps(current_nodes))
        _set_db_state_value(_KEY_DSL_REGEN_COUNT, str(current_count))

        logger.debug(
            "[postmortem] Accumulated high-conf-wrong: %d nodes, count=%d/%d",
            len(current_nodes), current_count, POSTMORTEM_DSL_REGEN_BATCH_SIZE
        )

    def invalidate_cache(self) -> None:
        """Force reload from DB on next access."""
        self._problem_count = None
        self._nodes_for_split = None
        self._high_conf_wrong_nodes = None
        self._dsl_regen_problem_count = None


# Singleton instance (thin wrapper, actual state in DB)
_postmortem_state: Optional[PostmortemState] = None


def get_postmortem_state() -> PostmortemState:
    """Get or create the singleton PostmortemState instance."""
    global _postmortem_state
    if _postmortem_state is None:
        _postmortem_state = PostmortemState()
    # Invalidate cache to ensure fresh read from DB
    _postmortem_state.invalidate_cache()
    return _postmortem_state


def reset_postmortem_state() -> None:
    """Reset the singleton state (useful for testing)."""
    global _postmortem_state
    _postmortem_state = PostmortemState()
    _postmortem_state.reset_merge_split()
    _postmortem_state.reset_dsl_regen()


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

    # Increment the run counter (tracks total post-mortem runs)
    run_count = state.increment_run_count()
    logger.debug("[mcts] Post-mortem run #%d", run_count)

    # First run amplitude_post computation - cheap, always run
    amplitude_stats = _compute_amplitude_post(dag_id)

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

    # Per mycelium-i601: Propagate success similarity for adaptive rejection thresholds
    # When threads win, record their similarity scores on leaves for adaptive thresholds
    success_sim_stats = propagate_success_similarity(dag_id, step_db)

    # Propagate to (dag_step_type, node_id) stats table
    # This enables routing UCB1 to use step-specific performance data
    step_node_stats = propagate_step_node_stats(dag_id)

    # Per beads mycelium-2rss: Divergence-point analysis for targeted blame
    # Compare winning vs losing thread paths to find exactly where failure occurred
    divergence_stats = assign_divergence_blame(dag_id, step_db)

    # Accumulate nodes flagged for split (using state object)
    state.accumulate_split_nodes(interference_result.nodes_flagged_split)

    # Variance-based decomposition: flag nodes with high amplitude_post variance
    # High variance = inconsistent performance = node too generic = should decompose
    # Per CLAUDE.md: "Destructive interference (mixed results)" triggers split
    from mycelium.config import (
        VARIANCE_DECOMPOSE_ENABLED, VARIANCE_MIN_SAMPLES,
        VARIANCE_THRESHOLD, VARIANCE_CHECK_LIMIT
    )
    variance_nodes_flagged = []
    if VARIANCE_DECOMPOSE_ENABLED:
        high_variance_pairs = get_high_variance_step_node_pairs(
            min_samples=VARIANCE_MIN_SAMPLES,
            variance_threshold=VARIANCE_THRESHOLD,
            limit=VARIANCE_CHECK_LIMIT,
        )
        if high_variance_pairs:
            variance_node_ids = [p["node_id"] for p in high_variance_pairs]
            state.accumulate_split_nodes(variance_node_ids)
            variance_nodes_flagged = variance_node_ids
            logger.info(
                "[mcts] Flagged %d high-variance nodes for decomposition (threshold=%.2f)",
                len(variance_node_ids), VARIANCE_THRESHOLD
            )
            for pair in high_variance_pairs[:3]:  # Log top 3 for debugging
                logger.debug(
                    "[mcts] High variance: node=%d step=%s var=%.3f std=%.3f uses=%d",
                    pair["node_id"], pair["dag_step_type"][:30],
                    pair["amp_post_variance"], pair["amp_post_std"], pair["uses"]
                )

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
        # Increment batch ops counter
        batch_ops_count = state.increment_batch_ops_count()
        logger.info("[mcts] Batch operations run #%d (every %d problems)", batch_ops_count, MERGE_SPLIT_BATCH_SIZE)

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

        # Collapse single-child routers to prevent chains
        # Per CLAUDE.md: "healthy tree would have ~5:1 ratio"
        collapse_result = collapse_single_child_routers()
        if collapse_result["collapsed"] > 0:
            logger.info(
                "[mcts] Collapsed %d single-child routers (chain prevention)",
                collapse_result["collapsed"],
            )

        # Flag high-rejection leaves for decomposition (MCTS-aware)
        # Per brainstorm: Only decompose if no alternative leaves could handle rejected steps
        rejection_flagged = flag_high_rejection_leaves_for_decomposition(step_db)
        if rejection_flagged:
            logger.info(
                "[mcts] Flagged %d high-rejection leaves for decomposition",
                len(rejection_flagged),
            )

        # Reset merge/split state
        state.reset_merge_split()

        logger.info(
            "[mcts] Batch operations complete: %d merges, %d splits flagged, %d rejection decomps",
            merge_split_result["merges_succeeded"],
            merge_split_result["splits_flagged"],
            len(rejection_flagged) if rejection_flagged else 0,
        )

    # Run decomposition analysis to find nodes/steps needing decomposition
    decomp_analysis = analyze_decomposition_needs(min_attempts=3, max_win_rate=0.5)

    if decomp_analysis["stats"]["nodes_failing"] > 0 or decomp_analysis["stats"]["steps_failing"] > 0:
        logger.info(
            "[mcts] Decomposition analysis: %d nodes, %d steps need decomposition",
            decomp_analysis["stats"]["nodes_failing"],
            decomp_analysis["stats"]["steps_failing"],
        )

    return {
        **amplitude_stats,
        "postmortem_run_count": run_count,
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
        "batch_ops_run_count": state.batch_ops_run_count,
        # DSL regeneration info (per beads mycelium-flbq)
        "dsl_regen_nodes_accumulated": len(state.high_conf_wrong_nodes),
        "dsl_regen_ready": should_trigger_dsl_regen(),
        # Amplitude credit propagation (per beads mycelium-itkn)
        "credit_nodes_processed": credit_stats.get("nodes_processed", 0),
        "credit_successes": credit_stats.get("successes_credited", 0),
        "credit_failures": credit_stats.get("failures_credited", 0),
        # Success similarity propagation for adaptive rejection (per mycelium-i601)
        "success_sim_updates": success_sim_stats.get("updates", 0),
        # Divergence-point analysis (per beads mycelium-2rss)
        "divergence_points_found": divergence_stats.get("divergence_points_found", 0),
        "divergence_blame_assigned": divergence_stats.get("divergence_blame_assigned", 0),
        "suffix_blame_assigned": divergence_stats.get("suffix_blame_assigned", 0),
        "shared_prefix_credit": divergence_stats.get("shared_prefix_credit", 0),
        # Decomposition analysis (nodes/steps that need decomposition)
        "nodes_needing_decomposition": [rec.target_id for rec in decomp_analysis["nodes_to_decompose"]],
        "steps_needing_decomposition": [(rec.target_id, rec.target_desc) for rec in decomp_analysis["steps_to_decompose"]],
        # Variance-based decomposition (Welford's algorithm)
        "variance_nodes_flagged": variance_nodes_flagged,
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
            sig = step_db.get_signature(node_id)
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

    Note: This is now handled internally by accumulate_high_conf_wrong.
    Kept for backward compatibility - increments via empty accumulate call.
    """
    state = get_postmortem_state()
    # Use accumulate with empty list to just increment counter
    state.accumulate_high_conf_wrong([])
    return state.dsl_regen_problem_count


# =============================================================================
# DECOMPOSITION ANALYSIS: Determine what needs decomposition
# =============================================================================
# Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning"
# Analyze failure patterns to decide whether to decompose the NODE or the STEP.


@dataclass
class DecompositionRecommendation:
    """Recommendation for what to decompose."""
    target_type: str  # "node" or "step" or "pair" (step_type, node_id)
    target_id: int  # node_id or dag_step_id
    target_desc: str  # Description for logging
    reason: str  # Why this needs decomposition
    win_rate: float  # Current success rate
    attempts: int  # Number of attempts
    variance: float = 0.0  # Amplitude variance (high = inconsistent performance)


def analyze_decomposition_needs(min_attempts: int = 3, max_win_rate: float = 0.5) -> dict:
    """Analyze (node, step) pairs to find what needs decomposition.

    Decision logic:
    1. NODE fails across ALL steps → Decompose the NODE (bad DSL or orphan)
    2. STEP fails across ALL nodes → Decompose the STEP (too complex)
    3. Specific (node, step) pair fails but node succeeds elsewhere → Create specialized child

    Args:
        min_attempts: Minimum attempts before considering for decomposition
        max_win_rate: Maximum win rate to be considered failing

    Returns:
        Dict with 'nodes_to_decompose', 'steps_to_decompose', 'stats'
    """
    db = get_db()

    # 1. NODE-CENTRIC: Find nodes failing across all steps
    node_stats = {}
    with db.connection() as conn:
        cursor = conn.execute("""
            SELECT
                t.node_id,
                COUNT(*) as total,
                SUM(CASE WHEN t.step_success = 1 THEN 1 ELSE 0 END) as wins
            FROM mcts_thread_steps t
            WHERE t.node_id IS NOT NULL
            GROUP BY t.node_id
            HAVING COUNT(*) >= ?
        """, (min_attempts,))

        for row in cursor.fetchall():
            node_id, total, wins = row
            win_rate = wins / total if total > 0 else 0
            node_stats[node_id] = {
                "total": total,
                "wins": wins,
                "win_rate": win_rate,
            }

    # 2. STEP-CENTRIC: Find steps failing across all nodes
    step_stats = {}
    with db.connection() as conn:
        cursor = conn.execute("""
            SELECT
                s.dag_step_id,
                s.step_desc,
                COUNT(DISTINCT t.node_id) as nodes_tried,
                COUNT(*) as total,
                SUM(CASE WHEN t.step_success = 1 THEN 1 ELSE 0 END) as wins
            FROM mcts_thread_steps t
            JOIN mcts_dag_steps s ON t.dag_step_id = s.dag_step_id
            WHERE t.node_id IS NOT NULL
            GROUP BY s.dag_step_id
            HAVING COUNT(*) >= ?
        """, (min_attempts,))

        for row in cursor.fetchall():
            step_id, step_desc, nodes_tried, total, wins = row
            win_rate = wins / total if total > 0 else 0
            step_stats[step_id] = {
                "step_desc": step_desc,
                "nodes_tried": nodes_tried,
                "total": total,
                "wins": wins,
                "win_rate": win_rate,
            }

    # 3. Analyze and generate recommendations
    nodes_to_decompose = []
    steps_to_decompose = []

    # Get signature info for failing nodes (batch query to avoid N+1)
    failing_node_ids = [nid for nid, stats in node_stats.items() if stats["win_rate"] <= max_win_rate]

    if failing_node_ids:
        with db.connection() as conn:
            # Batch query all failing nodes at once
            placeholders = ",".join("?" * len(failing_node_ids))
            cursor = conn.execute(
                f"SELECT id, step_type, dsl_type, dsl_script FROM step_signatures WHERE id IN ({placeholders})",
                failing_node_ids
            )
            node_details = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}

        for node_id in failing_node_ids:
            stats = node_stats[node_id]
            if node_id in node_details:
                step_type, dsl_type, dsl_script = node_details[node_id]
                # Check if it's an orphan router (no DSL)
                is_orphan = dsl_type == "router" and not dsl_script
                reason = "orphan router (no DSL)" if is_orphan else f"failing across all steps ({stats['win_rate']*100:.0f}% win rate)"

                nodes_to_decompose.append(DecompositionRecommendation(
                    target_type="node",
                    target_id=node_id,
                    target_desc=step_type,
                    reason=reason,
                    win_rate=stats["win_rate"],
                    attempts=stats["total"],
                ))

    # Analyze failing steps
    for step_id, stats in step_stats.items():
        if stats["win_rate"] <= max_win_rate:
            # Check if ALL nodes that tried this step failed
            # If yes, the step itself is too complex
            if stats["nodes_tried"] >= 1:  # At least one node tried
                steps_to_decompose.append(DecompositionRecommendation(
                    target_type="step",
                    target_id=step_id,
                    target_desc=stats["step_desc"],
                    reason=f"failing with {stats['nodes_tried']} node(s) tried ({stats['win_rate']*100:.0f}% win rate)",
                    win_rate=stats["win_rate"],
                    attempts=stats["total"],
                ))

    # 4. HIGH-VARIANCE PAIRS: Find (step_type, node) pairs with inconsistent performance
    # Per CLAUDE.md: "Destructive interference (mixed success/failure at same node)"
    # High variance = node is too generic for this step type = decompose into specialized children
    high_variance_pairs = get_high_variance_step_node_pairs(
        min_samples=min_attempts,
        variance_threshold=0.1,  # Flag pairs with >0.1 amplitude variance
        limit=10,
    )
    pairs_to_decompose = []
    for pair in high_variance_pairs:
        pairs_to_decompose.append(DecompositionRecommendation(
            target_type="pair",
            target_id=pair["node_id"],
            target_desc=f"{pair['dag_step_type']}/node_{pair['node_id']}",
            reason=f"high variance (std={pair['amp_post_std']:.3f}, {pair['win_rate']*100:.0f}% win rate)",
            win_rate=pair["win_rate"],
            attempts=pair["uses"],
            variance=pair["amp_post_variance"],
        ))

    return {
        "nodes_to_decompose": nodes_to_decompose,
        "steps_to_decompose": steps_to_decompose,
        "pairs_to_decompose": pairs_to_decompose,
        "stats": {
            "total_nodes_analyzed": len(node_stats),
            "total_steps_analyzed": len(step_stats),
            "nodes_failing": len(nodes_to_decompose),
            "steps_failing": len(steps_to_decompose),
            "pairs_high_variance": len(pairs_to_decompose),
        }
    }


def get_failing_nodes_for_decomposition(min_attempts: int = 3, max_win_rate: float = 0.5) -> list[int]:
    """Get list of node IDs that need decomposition based on failure analysis.

    Returns nodes that are failing across all steps they handle.
    """
    result = analyze_decomposition_needs(min_attempts, max_win_rate)
    return [rec.target_id for rec in result["nodes_to_decompose"]]


def get_failing_steps_for_decomposition(min_attempts: int = 3, max_win_rate: float = 0.5) -> list[tuple[str, str]]:
    """Get list of (dag_step_id, step_desc) tuples that need decomposition.

    Returns steps that are failing regardless of which node handles them.
    """
    result = analyze_decomposition_needs(min_attempts, max_win_rate)
    return [(rec.target_id, rec.target_desc) for rec in result["steps_to_decompose"]]


def get_mcts_win_rates(min_attempts: int = 1) -> dict[int, dict]:
    """Get MCTS win rates for all nodes with sufficient data.

    Returns actual win rates from mcts_thread_steps, which tracks
    step-level outcomes (success/failure). This is ground truth for
    operational correctness, as opposed to signature.success_rate which
    includes partial credit.

    Args:
        min_attempts: Minimum step attempts to include a node

    Returns:
        Dict mapping node_id to {"total": int, "wins": int, "win_rate": float}
    """
    db = get_db()
    node_stats = {}

    with db.connection() as conn:
        cursor = conn.execute("""
            SELECT
                t.node_id,
                COUNT(*) as total,
                SUM(CASE WHEN t.step_success = 1 THEN 1 ELSE 0 END) as wins
            FROM mcts_thread_steps t
            WHERE t.node_id IS NOT NULL
            GROUP BY t.node_id
            HAVING COUNT(*) >= ?
        """, (min_attempts,))

        for row in cursor.fetchall():
            node_id, total, wins = row
            win_rate = wins / total if total > 0 else 0
            node_stats[node_id] = {
                "total": total,
                "wins": wins,
                "win_rate": win_rate,
            }

    return node_stats


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


def collapse_single_child_routers() -> dict:
    """Collapse router signatures that have only one child.

    Single-child routers add indirection without value. This function:
    1. Finds routers with exactly one child
    2. Promotes the child to replace the router in relationships
    3. Removes the redundant router

    Returns:
        Dict with collapse statistics
    """
    db = get_db()
    collapsed = 0
    collapsed_ids = []

    try:
        with db.connection() as conn:
            # Find umbrella routers with exactly one child
            cursor = conn.execute("""
                SELECT sr.parent_id, sr.child_id, s.step_type
                FROM signature_relationships sr
                JOIN step_signatures s ON s.id = sr.parent_id
                WHERE s.is_semantic_umbrella = 1
                  AND s.is_archived = 0
                GROUP BY sr.parent_id
                HAVING COUNT(sr.child_id) = 1
            """)
            single_child_routers = cursor.fetchall()

            for row in single_child_routers:
                parent_id = row["parent_id"]
                child_id = row["child_id"]

                # Skip if parent is the root (id=1 or has no parent)
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM signature_relationships WHERE child_id = ?",
                    (parent_id,)
                )
                if cursor.fetchone()["cnt"] == 0:
                    # This router has no parent - it's at the root level, don't collapse
                    continue

                # Find grandparent relationships (who points to this router)
                cursor = conn.execute(
                    "SELECT parent_id FROM signature_relationships WHERE child_id = ?",
                    (parent_id,)
                )
                grandparents = [r["parent_id"] for r in cursor.fetchall()]

                # Skip if multiple grandparents - would violate UNIQUE(child_id) constraint
                if len(grandparents) > 1:
                    logger.debug(
                        "[mcts] Skipping collapse of router %d: multiple grandparents",
                        parent_id
                    )
                    continue

                # IMPORTANT: Delete router -> child relationship FIRST
                # This frees up the child_id for the grandparent relationship
                conn.execute(
                    "DELETE FROM signature_relationships WHERE parent_id = ? AND child_id = ?",
                    (parent_id, child_id)
                )

                # Now update grandparent -> router to grandparent -> child
                for grandparent_id in grandparents:
                    conn.execute("""
                        UPDATE signature_relationships
                        SET child_id = ?
                        WHERE parent_id = ? AND child_id = ?
                    """, (child_id, grandparent_id, parent_id))

                    # FIX: Update child's depth to reflect new parent
                    # Child's depth should be grandparent's depth + 1
                    grandparent_depth_row = conn.execute(
                        "SELECT depth FROM step_signatures WHERE id = ?",
                        (grandparent_id,)
                    ).fetchone()
                    new_depth = (grandparent_depth_row["depth"] + 1) if grandparent_depth_row else 1
                    conn.execute(
                        "UPDATE step_signatures SET depth = ? WHERE id = ?",
                        (new_depth, child_id)
                    )
                    logger.debug(
                        "[mcts] Updated child %d depth to %d (grandparent %d depth + 1)",
                        child_id, new_depth, grandparent_id
                    )

                # Mark router as archived (don't delete, preserve history)
                conn.execute(
                    "UPDATE step_signatures SET is_archived = 1 WHERE id = ?",
                    (parent_id,)
                )

                collapsed += 1
                collapsed_ids.append(parent_id)
                logger.debug(
                    "[mcts] Collapsed single-child router %d, promoted child %d",
                    parent_id, child_id
                )
            # Connection context manager handles commit

    except Exception as e:
        logger.warning("[mcts] Failed to collapse single-child routers: %s", e)
        # Connection context manager handles rollback on exception

    return {
        "collapsed": collapsed,
        "collapsed_ids": collapsed_ids,
    }


def repair_signature_depths() -> dict:
    """Repair signature depths that became inconsistent after collapse operations.

    This fixes the bug where collapse_single_child_routers() was reparenting
    signatures without updating their depth to match the new parent.

    For each signature, the correct depth is: parent's depth + 1 (or 0 if root).

    Returns:
        Dict with repair statistics
    """
    db = get_db()
    repaired = 0
    repairs = []

    try:
        with db.connection() as conn:
            # Get all non-root signatures with their parent depth
            cursor = conn.execute("""
                SELECT
                    s.id,
                    s.depth as current_depth,
                    s.is_root,
                    p.id as parent_id,
                    p.depth as parent_depth
                FROM step_signatures s
                LEFT JOIN signature_relationships r ON r.child_id = s.id
                LEFT JOIN step_signatures p ON p.id = r.parent_id
                WHERE s.is_archived = 0
            """)
            rows = cursor.fetchall()

            for row in rows:
                sig_id = row["id"]
                current_depth = row["current_depth"] or 0
                is_root = row["is_root"]
                parent_id = row["parent_id"]
                parent_depth = row["parent_depth"]

                # Root should always be depth 0
                if is_root:
                    if current_depth != 0:
                        conn.execute(
                            "UPDATE step_signatures SET depth = 0 WHERE id = ?",
                            (sig_id,)
                        )
                        repairs.append({
                            "id": sig_id,
                            "old_depth": current_depth,
                            "new_depth": 0,
                            "reason": "root must be depth 0"
                        })
                        repaired += 1
                    continue

                # Non-root with no parent (orphan) - shouldn't exist but handle it
                if parent_id is None:
                    expected_depth = 1  # Assume direct child of root
                else:
                    expected_depth = (parent_depth or 0) + 1

                if current_depth != expected_depth:
                    conn.execute(
                        "UPDATE step_signatures SET depth = ? WHERE id = ?",
                        (expected_depth, sig_id)
                    )
                    repairs.append({
                        "id": sig_id,
                        "old_depth": current_depth,
                        "new_depth": expected_depth,
                        "parent_id": parent_id,
                        "parent_depth": parent_depth
                    })
                    repaired += 1
                    logger.info(
                        "[mcts] Repaired sig %d depth: %d -> %d (parent %s at depth %s)",
                        sig_id, current_depth, expected_depth, parent_id, parent_depth
                    )

    except Exception as e:
        logger.warning("[mcts] Failed to repair signature depths: %s", e)

    if repaired > 0:
        logger.info("[mcts] Repaired %d signature depths", repaired)

    return {
        "repaired": repaired,
        "repairs": repairs,
    }


# =============================================================================
# DIAGNOSTIC POST-MORTEM (Accuracy-driven decomposition decisions)
# =============================================================================
# Per CLAUDE.md: "Failures are valuable data points" + "Failing signatures get decomposed"
#
# Uses smooth continuous functions based on:
# - Accuracy = successes / uses (percent, not absolute counts)
# - Confidence = smooth ramp with uses (more data → more trust)
# - Maturity = sigmoid over signature count
#
# Key insight: A signature with 60 successes + 4 failures (93.75%) should NOT
# be decomposed. We use accuracy (percent), not failure count.


@dataclass
class DiagnosticResult:
    """Result of diagnostic post-mortem analysis."""
    decompose_step_score: float    # 0-1, how much to decompose the dag_step
    decompose_sig_score: float     # 0-1, how much to decompose the signature
    reroute_score: float           # 0-1, how much this looks like a routing miss

    # Context for logging
    accuracy: float
    confidence: float
    step_distance: float

    @property
    def verdict(self) -> str:
        """Highest score wins, but only if above threshold."""
        from mycelium.config import DIAGNOSTIC_ACTION_THRESHOLD

        scores = {
            "decompose_step": self.decompose_step_score,
            "decompose_signature": self.decompose_sig_score,
            "reroute": self.reroute_score,
        }
        best = max(scores, key=scores.get)

        if scores[best] < DIAGNOSTIC_ACTION_THRESHOLD:
            return "wait"
        return best

    @property
    def max_score(self) -> float:
        """The winning score."""
        return max(self.decompose_step_score, self.decompose_sig_score, self.reroute_score)


def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Smooth S-curve from 0 to 1.

    Args:
        x: Input value
        midpoint: Value where sigmoid = 0.5
        steepness: Controls sharpness (higher = sharper transition)

    Returns:
        Value between 0 and 1
    """
    import math
    z = (x - midpoint) / steepness if steepness > 0 else 0
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def get_diagnostic_failure_threshold(system_maturity: float) -> float:
    """Compute failure threshold using smooth sigmoid.

    Cold start (low maturity): act fast, low threshold
    Mature (high maturity): be patient, higher threshold

    Args:
        system_maturity: Signature count (used as maturity proxy)

    Returns:
        Failure threshold (float, can be fractional)
    """
    from mycelium.config import (
        DIAGNOSTIC_THRESHOLD_MIN,
        DIAGNOSTIC_THRESHOLD_MAX,
        MATURITY_SIGMOID_MIDPOINT,
        MATURITY_SIGMOID_STEEPNESS,
    )

    maturity_factor = _sigmoid(
        system_maturity,
        midpoint=MATURITY_SIGMOID_MIDPOINT,
        steepness=MATURITY_SIGMOID_STEEPNESS,
    )

    # Interpolate between min and max threshold
    return DIAGNOSTIC_THRESHOLD_MIN + (
        (DIAGNOSTIC_THRESHOLD_MAX - DIAGNOSTIC_THRESHOLD_MIN) * maturity_factor
    )


def accuracy_confidence(uses: int) -> float:
    """Compute confidence in accuracy signal based on sample size.

    Uses exponential decay: confidence approaches 1.0 as uses increase.
    Low uses → don't trust accuracy yet
    High uses → accuracy is reliable

    Args:
        uses: Number of times signature was used

    Returns:
        Confidence between 0 and 1
    """
    import math
    from mycelium.config import DIAGNOSTIC_CONFIDENCE_HALFLIFE

    if uses <= 0:
        return 0.0

    # 1 - e^(-uses/halflife) approaches 1 as uses → ∞
    return 1.0 - math.exp(-uses / DIAGNOSTIC_CONFIDENCE_HALFLIFE)


def compute_decompose_score(
    accuracy: float,
    uses: int,
    step_centroid_distance: float = 0.0,
) -> float:
    """Compute continuous score indicating how much this signature should be decomposed.

    Components:
    - Low accuracy → high decompose score (weighted by confidence)
    - High step distance from centroid → suggests step doesn't belong here

    Args:
        accuracy: Success rate (0-1)
        uses: Number of times used
        step_centroid_distance: Distance from step embedding to signature centroid (0-1)

    Returns:
        Decompose score between 0 and 1 (higher = stronger signal to decompose)
    """
    from mycelium.config import (
        DIAGNOSTIC_ACCURACY_WEIGHT,
        DIAGNOSTIC_DISTANCE_WEIGHT,
    )

    confidence = accuracy_confidence(uses)

    # Accuracy component: invert accuracy, weight by confidence
    # accuracy=1.0 → 0 score, accuracy=0.0 → max score
    accuracy_score = (1.0 - accuracy) * confidence * DIAGNOSTIC_ACCURACY_WEIGHT

    # Distance component: far from centroid suggests wrong routing or complex step
    distance_score = step_centroid_distance * DIAGNOSTIC_DISTANCE_WEIGHT

    # Combine (weighted sum)
    raw_score = accuracy_score + distance_score

    # Normalize to 0-1 range
    return min(1.0, max(0.0, raw_score))


def diagnose_failure(
    signature_id: int,
    step_embedding,
    step_db,
) -> DiagnosticResult:
    """Diagnose a failure to determine decomposition target.

    Uses smooth continuous functions to compute scores for:
    - Decompose the step (step is too complex)
    - Decompose the signature (approach is wrong)
    - Reroute (wrong routing, similar steps succeeded elsewhere)

    Args:
        signature_id: ID of the signature that was used
        step_embedding: Embedding of the dag_step (numpy array)
        step_db: StepSignatureDB instance

    Returns:
        DiagnosticResult with scores and verdict
    """
    import numpy as np
    from mycelium.config import (
        DIAGNOSTIC_REROUTE_SIMILARITY_MIN,
        DIAGNOSTIC_REROUTE_LOOKBACK_DAYS,
        DIAGNOSTIC_GOOD_SIG_ACCURACY,
    )

    # Guard against None step_db
    if step_db is None:
        return DiagnosticResult(
            decompose_step_score=0.0,
            decompose_sig_score=0.0,
            reroute_score=0.0,
            accuracy=0.0,
            confidence=0.0,
            step_distance=0.0,
        )

    # Get signature stats
    sig = step_db.get_signature(signature_id)
    if sig is None:
        # Signature not found - can't diagnose
        return DiagnosticResult(
            decompose_step_score=0.0,
            decompose_sig_score=0.0,
            reroute_score=0.0,
            accuracy=0.0,
            confidence=0.0,
            step_distance=0.0,
        )

    # Core stats
    accuracy = sig.success_rate
    uses = sig.uses or 0
    confidence = accuracy_confidence(uses)

    # Compute step distance from signature centroid
    step_distance = 0.0
    if step_embedding is not None and sig.centroid is not None:
        try:
            # Cosine distance = 1 - cosine_similarity
            sig_centroid = np.array(sig.centroid, dtype=np.float32)
            step_emb = np.array(step_embedding, dtype=np.float32)

            norm_a = np.linalg.norm(sig_centroid)
            norm_b = np.linalg.norm(step_emb)
            if norm_a > 0 and norm_b > 0:
                cosine_sim = np.dot(sig_centroid, step_emb) / (norm_a * norm_b)
                step_distance = 1.0 - cosine_sim
        except Exception:
            step_distance = 0.0

    # === Decompose STEP score ===
    # High when: signature is accurate overall, but this step is distant
    # Good signature + outlier step → step needs decomposition
    sig_is_good = _sigmoid(accuracy, midpoint=DIAGNOSTIC_GOOD_SIG_ACCURACY, steepness=0.1)
    step_is_outlier = step_distance
    decompose_step_score = sig_is_good * step_is_outlier * confidence

    # === Decompose SIGNATURE score ===
    # High when: signature has poor accuracy with sufficient confidence
    decompose_sig_score = compute_decompose_score(
        accuracy=accuracy,
        uses=uses,
        step_centroid_distance=0.0,  # Don't double-count distance
    )

    # === Reroute score ===
    # High when: similar steps succeeded with OTHER signatures
    # Find similar successful steps from different signatures
    reroute_score = 0.0
    try:
        similar_successes = step_db.find_similar_successful_steps(
            embedding=step_embedding,
            exclude_signature_id=signature_id,
            min_similarity=DIAGNOSTIC_REROUTE_SIMILARITY_MIN,
            limit=3,
            lookback_days=DIAGNOSTIC_REROUTE_LOOKBACK_DAYS,
        )
        if similar_successes:
            # Best alternative similarity
            best_alt_similarity = similar_successes[0].get("similarity", 0.0)
            reroute_score = best_alt_similarity * confidence
    except AttributeError:
        # Method not implemented on step_db - expected during migration
        logger.debug("[diagnostic] find_similar_successful_steps not available")
    except Exception as e:
        # Unexpected error - log but don't crash diagnosis
        logger.warning("[diagnostic] Error finding similar successful steps: %s", e)

    result = DiagnosticResult(
        decompose_step_score=decompose_step_score,
        decompose_sig_score=decompose_sig_score,
        reroute_score=reroute_score,
        accuracy=accuracy,
        confidence=confidence,
        step_distance=step_distance,
    )

    logger.debug(
        "[diagnostic] sig=%d accuracy=%.1f%% conf=%.2f verdict=%s "
        "(step=%.2f sig=%.2f reroute=%.2f)",
        signature_id,
        accuracy * 100,
        confidence,
        result.verdict,
        decompose_step_score,
        decompose_sig_score,
        reroute_score,
    )

    return result


def run_diagnostic_postmortem(
    dag_id: str,
    step_db,
    step_embeddings: dict[str, Any] = None,
) -> dict:
    """Thin wrapper for backward compatibility. Use run_postmortem() instead.

    Calls _run_diagnostic_analysis() after checking if diagnostics are enabled.

    Args:
        dag_id: The DAG to analyze
        step_db: StepSignatureDB instance
        step_embeddings: Optional dict mapping dag_step_id to embeddings

    Returns:
        Dict with diagnostic results and recommendations
    """
    from mycelium.config import DIAGNOSTIC_POSTMORTEM_ENABLED

    if not DIAGNOSTIC_POSTMORTEM_ENABLED:
        return {"skipped": True, "reason": "DIAGNOSTIC_POSTMORTEM_ENABLED is False"}

    return _run_diagnostic_analysis(dag_id, step_db, step_embeddings)


