"""MCTS Data Layer - Minimal implementation for local decomposition.

Tracks DAGs and their outcomes for Welford learning.
Complex thread/amplitude tracking removed for simplicity.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from mycelium.data_layer.connection import get_db

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MCTSDag:
    """A problem DAG."""
    id: str
    problem_text: str
    created_at: str
    success: Optional[bool] = None

@dataclass
class MCTSDagStep:
    """A step in a DAG."""
    id: str
    dag_id: str
    step_text: str
    step_index: int

@dataclass
class RejectionDecision:
    """Result of rejection check."""
    rejected: bool
    rejection_count: int
    should_decompose: bool = False


# =============================================================================
# DAG MANAGEMENT
# =============================================================================

def create_dag(problem_text: str) -> str:
    """Create a new DAG for tracking."""
    dag_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        """
        INSERT INTO mcts_dags (id, problem_text, created_at)
        VALUES (?, ?, ?)
        """,
        (dag_id, problem_text, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return dag_id


def grade_dag(dag_id: str, success: bool) -> None:
    """Grade a DAG as success or failure."""
    conn = get_db()
    conn.execute(
        "UPDATE mcts_dags SET success = ? WHERE id = ?",
        (1 if success else 0, dag_id),
    )
    conn.commit()


def create_dag_steps(dag_id: str, step_texts: list[str]) -> list[str]:
    """Create steps for a DAG.

    Args:
        dag_id: The DAG ID
        step_texts: List of step descriptions

    Returns:
        List of step IDs
    """
    conn = get_db()
    step_ids = []

    for i, text in enumerate(step_texts):
        step_id = f"{dag_id}_step_{i}"
        conn.execute(
            """
            INSERT INTO mcts_dag_steps (id, dag_id, step_text, step_index)
            VALUES (?, ?, ?, ?)
            """,
            (step_id, dag_id, text, i),
        )
        step_ids.append(step_id)

    conn.commit()
    return step_ids


# =============================================================================
# POST-MORTEM (simplified)
# =============================================================================

def run_postmortem(dag_id: str, step_db=None) -> dict:
    """Run post-mortem analysis on a DAG.

    Simplified version - just propagates success/failure to signatures.
    """
    conn = get_db()

    # Get DAG outcome
    row = conn.execute(
        "SELECT success FROM mcts_dags WHERE id = ?",
        (dag_id,)
    ).fetchone()

    if row is None:
        return {"error": "DAG not found"}

    success = bool(row[0]) if row[0] is not None else None

    return {
        "dag_id": dag_id,
        "success": success,
        "analyzed": True,
    }


# =============================================================================
# REJECTION TRACKING (simplified)
# =============================================================================

def reject_dag_step(
    signature_id: int,
    similarity: float,
    step_text: str,
    dag_step_id: str = None,
    problem_context: str = None,
    reason: str = "below_threshold",
    conn=None,
) -> RejectionDecision:
    """Record a rejection and return decision.

    Simplified - just increments rejection count.
    """
    if conn is None:
        conn = get_db()

    # Increment rejection count
    conn.execute(
        """
        UPDATE step_signatures
        SET rejection_count = COALESCE(rejection_count, 0) + 1
        WHERE id = ?
        """,
        (signature_id,),
    )
    conn.commit()

    # Get current count
    row = conn.execute(
        "SELECT rejection_count FROM step_signatures WHERE id = ?",
        (signature_id,),
    ).fetchone()

    rejection_count = row[0] if row else 0

    return RejectionDecision(
        rejected=True,
        rejection_count=rejection_count,
        should_decompose=rejection_count >= 10,  # Simple threshold
    )


def get_rejection_count_threshold() -> int:
    """Get rejection threshold. Simplified to constant."""
    return 10


def record_leaf_rejection(signature_id: int, similarity: float, step_text: str) -> int:
    """Legacy interface - calls reject_dag_step."""
    decision = reject_dag_step(signature_id, similarity, step_text)
    return decision.rejection_count


def get_leaf_rejection_stats(signature_id: int) -> dict:
    """Get rejection stats for a signature."""
    conn = get_db()
    row = conn.execute(
        "SELECT rejection_count, uses FROM step_signatures WHERE id = ?",
        (signature_id,),
    ).fetchone()

    if row is None:
        return {"rejection_count": 0, "uses": 0, "rejection_rate": 0.0}

    rejection_count = row[0] or 0
    uses = row[1] or 0
    total = uses + rejection_count

    return {
        "rejection_count": rejection_count,
        "uses": uses,
        "rejection_rate": rejection_count / total if total > 0 else 0.0,
    }


# =============================================================================
# STUBS for backward compatibility
# =============================================================================

# Thread tracking - stubbed
class MCTSThread:
    pass

class MCTSThreadStep:
    pass

def create_thread(*args, **kwargs) -> str:
    return str(uuid.uuid4())

def complete_thread(*args, **kwargs) -> None:
    pass

def grade_thread(*args, **kwargs) -> None:
    pass

def log_thread_step(*args, **kwargs) -> str:
    return str(uuid.uuid4())

def update_amplitude_post(*args, **kwargs) -> None:
    pass

def update_summary_amplitude_post(*args, **kwargs) -> None:
    pass

def batch_update_amplitudes(*args, **kwargs) -> None:
    pass

def batch_update_summary_amplitudes(*args, **kwargs) -> None:
    pass

def get_thread_steps_for_dag(*args, **kwargs) -> list:
    return []

def get_node_step_stats(*args, **kwargs) -> dict:
    return {}

def get_dag_step_node_performance(*args, **kwargs) -> dict:
    return {}

# Stats - stubbed
def update_dag_step_node_stats(*args, **kwargs) -> None:
    pass

def get_dag_step_node_stats_batch(*args, **kwargs) -> dict:
    return {}

def get_dag_step_node_stats_single(*args, **kwargs) -> dict:
    return {}

def propagate_step_node_stats(*args, **kwargs) -> dict:
    return {}

def get_problem_nodes_needing_attention(*args, **kwargs) -> list:
    return []

# DSL regen - stubbed
def trigger_dsl_regeneration_for_nodes(*args, **kwargs) -> dict:
    return {}

def get_accumulated_failing_nodes(*args, **kwargs) -> list:
    return []

def clear_accumulated_failing_nodes(*args, **kwargs) -> None:
    pass

def should_trigger_dsl_regen(*args, **kwargs) -> bool:
    return False

# Divergence - stubbed
@dataclass
class ThreadPath:
    thread_id: str = ""
    success: bool = False
    steps: list = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []

@dataclass
class DivergencePoint:
    winning_thread_id: str = ""
    losing_thread_id: str = ""
    shared_prefix_len: int = 0
    divergence_step_idx: int = 0
    divergence_dag_step_id: Optional[str] = None
    winning_node_at_divergence: Optional[int] = None
    losing_node_at_divergence: Optional[int] = None
    losing_suffix: list = None

    def __post_init__(self):
        if self.losing_suffix is None:
            self.losing_suffix = []

def get_thread_paths(*args, **kwargs) -> list:
    return []

def find_divergence_points(*args, **kwargs) -> list:
    return []

def assign_divergence_blame(*args, **kwargs) -> dict:
    return {}

@dataclass
class DiagnosticResult:
    pass

# Embeddings - stubbed
def store_dag_step_embedding(*args, **kwargs) -> None:
    pass

def update_dag_step_embedding_outcome(*args, **kwargs) -> None:
    pass

def find_similar_dag_steps(*args, **kwargs) -> list:
    return []

# Plan stats - stubbed
def compute_plan_signature(*args, **kwargs) -> tuple:
    return ("", 0, "")

def record_plan_outcome(*args, **kwargs) -> None:
    pass

def get_plan_stats_summary(*args, **kwargs) -> dict:
    return {}

def get_top_plans(*args, **kwargs) -> list:
    return []

def get_worst_plans(*args, **kwargs) -> list:
    return []

# Decomposition queue - stubbed
def check_substeps_match_existing(*args, **kwargs) -> dict:
    return {"all_match": False}

def queue_for_decomposition(*args, **kwargs) -> int:
    return 0

def get_pending_decompositions(*args, **kwargs) -> list:
    return []

def get_decomposition_queue_size(*args, **kwargs) -> int:
    return 0

def get_oldest_pending_age_seconds(*args, **kwargs) -> float:
    return 0.0

def mark_decomposition_processed(*args, **kwargs) -> None:
    pass

def get_decomposition_queue_stats(*args, **kwargs) -> dict:
    return {}

def get_decomposition_results(*args, **kwargs) -> dict:
    return {}

def are_decompositions_ready(*args, **kwargs) -> bool:
    return False

def get_pending_queue_ids(*args, **kwargs) -> list:
    return []

# Rejection helpers - stubbed
def get_failing_step_descriptions(*args, **kwargs) -> list:
    return []

def get_leaves_needing_decomposition(*args, **kwargs) -> list:
    return []

def flag_high_rejection_leaves_for_decomposition(*args, **kwargs) -> list:
    return []

def get_adaptive_rejection_rate_threshold(*args, **kwargs) -> float:
    return 0.3

# Maturity - stubbed
def compute_db_maturity(*args, **kwargs) -> float:
    return 0.5

# Segmentation - stubbed
def get_segmentation_novelty_stats(*args, **kwargs) -> dict:
    return {}

def save_segmentation_novelty_stats(*args, **kwargs) -> None:
    pass
