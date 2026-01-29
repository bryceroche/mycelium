"""MCTS Data Layer - Minimal DAG tracking for local decomposition."""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from mycelium.data_layer.connection import get_db

logger = logging.getLogger(__name__)


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


def create_dag(problem_text: str) -> str:
    """Create a new DAG for tracking."""
    dag_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO mcts_dags (id, problem_text, created_at) VALUES (?, ?, ?)",
        (dag_id, problem_text, datetime.now(timezone.utc).isoformat()),
    )
    return dag_id


def grade_dag(dag_id: str, success: bool) -> None:
    """Grade a DAG as success or failure."""
    conn = get_db()
    conn.execute(
        "UPDATE mcts_dags SET success = ? WHERE id = ?",
        (1 if success else 0, dag_id),
    )


def create_dag_steps(dag_id: str, step_texts: list[str]) -> list[str]:
    """Create steps for a DAG."""
    conn = get_db()
    step_ids = []
    for i, text in enumerate(step_texts):
        step_id = f"{dag_id}_step_{i}"
        conn.execute(
            "INSERT INTO mcts_dag_steps (id, dag_id, step_text, step_index) VALUES (?, ?, ?, ?)",
            (step_id, dag_id, text, i),
        )
        step_ids.append(step_id)
    return step_ids


def run_postmortem(dag_id: str, step_db=None) -> dict:
    """Run post-mortem analysis on a DAG."""
    conn = get_db()
    row = conn.execute(
        "SELECT success FROM mcts_dags WHERE id = ?", (dag_id,)
    ).fetchone()
    if row is None:
        return {"error": "DAG not found"}
    return {"dag_id": dag_id, "success": bool(row[0]) if row[0] is not None else None, "analyzed": True}


def reject_dag_step(signature_id: int, similarity: float, step_text: str, **kwargs) -> dict:
    """Record a rejection. Returns rejection info."""
    conn = get_db()
    conn.execute(
        "UPDATE step_signatures SET rejection_count = COALESCE(rejection_count, 0) + 1 WHERE id = ?",
        (signature_id,),
    )
    row = conn.execute(
        "SELECT rejection_count FROM step_signatures WHERE id = ?", (signature_id,)
    ).fetchone()
    count = row[0] if row else 0
    return {"rejected": True, "rejection_count": count, "should_decompose": count >= 10}
