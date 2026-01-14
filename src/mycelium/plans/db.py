"""Database operations for recursive DAG of DAGs.

Provides CRUD operations for Plans and Steps with full recursive support.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import init_db
from ..planner import DAGPlan, Step

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "plans.db"


class PlansDB:
    """Database interface for recursive DAG of DAGs."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            init_db(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PlansDB":
        self._get_conn()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # =========================================================================
    # SAVE OPERATIONS
    # =========================================================================

    def save_plan(self, plan: DAGPlan, plan_id: Optional[str] = None) -> str:
        """Save a DAGPlan and all nested sub-plans recursively.

        Args:
            plan: The DAGPlan to save
            plan_id: Optional ID (generated if not provided)

        Returns:
            The plan ID
        """
        conn = self._get_conn()
        plan_id = plan_id or f"plan-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat()

        # Insert plan
        conn.execute("""
            INSERT INTO plans (id, problem, parent_step_id, depth, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (plan_id, plan.problem, plan.parent_step_id, plan.depth, now))

        # Insert steps
        for step in plan.steps:
            self._save_step(conn, step, plan_id, now)

        # Insert dependencies
        for step in plan.steps:
            for dep_local_id in step.depends_on:
                step_id = f"{plan_id}:{step.id}"
                dep_id = f"{plan_id}:{dep_local_id}"
                conn.execute("""
                    INSERT OR IGNORE INTO step_dependencies (step_id, depends_on_id)
                    VALUES (?, ?)
                """, (step_id, dep_id))

        conn.commit()
        logger.info(f"Saved plan {plan_id} with {len(plan.steps)} steps (depth={plan.depth})")
        return plan_id

    def _save_step(self, conn: sqlite3.Connection, step: Step, plan_id: str, now: str) -> str:
        """Save a single step (and recursively its sub-plan if composite)."""
        step_id = f"{plan_id}:{step.id}"

        # Serialize extracted_values
        values_json = json.dumps(step.extracted_values) if step.extracted_values else None

        conn.execute("""
            INSERT INTO steps (id, plan_id, local_id, task, status, result, success,
                             extracted_values, dsl_hint, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            step_id,
            plan_id,
            step.id,
            step.task,
            'completed' if step.success else 'pending',
            step.result,
            1 if step.success else 0,
            values_json,
            step.dsl_hint,
            now
        ))

        # Recursively save sub-plan if composite
        if step.sub_plan is not None:
            sub_plan = step.sub_plan
            sub_plan.parent_step_id = step_id
            sub_plan_id = f"{plan_id}:{step.id}:sub"
            self.save_plan(sub_plan, plan_id=sub_plan_id)

        return step_id

    # =========================================================================
    # LOAD OPERATIONS
    # =========================================================================

    def load_plan(self, plan_id: str) -> Optional[DAGPlan]:
        """Load a DAGPlan and all nested sub-plans recursively.

        Args:
            plan_id: The plan ID to load

        Returns:
            DAGPlan or None if not found
        """
        conn = self._get_conn()

        # Load plan metadata
        row = conn.execute("""
            SELECT * FROM plans WHERE id = ?
        """, (plan_id,)).fetchone()

        if not row:
            return None

        # Load steps
        step_rows = conn.execute("""
            SELECT * FROM steps WHERE plan_id = ? ORDER BY local_id
        """, (plan_id,)).fetchall()

        # Load dependencies
        deps_rows = conn.execute("""
            SELECT step_id, depends_on_id FROM step_dependencies
            WHERE step_id LIKE ?
        """, (f"{plan_id}:%",)).fetchall()

        # Build dependency map: step_id -> [depends_on_local_ids]
        deps_map: dict[str, list[str]] = {}
        for dep_row in deps_rows:
            step_id = dep_row["step_id"]
            # Extract local_id from depends_on_id (format: plan_id:local_id)
            dep_local = dep_row["depends_on_id"].split(":")[-1]
            if step_id not in deps_map:
                deps_map[step_id] = []
            deps_map[step_id].append(dep_local)

        # Convert rows to Step objects
        steps = []
        for step_row in step_rows:
            step_id = step_row["id"]
            step = Step(
                id=step_row["local_id"],
                task=step_row["task"],
                depends_on=deps_map.get(step_id, []),
                result=step_row["result"],
                success=bool(step_row["success"]),
                extracted_values=json.loads(step_row["extracted_values"] or "{}"),
                dsl_hint=step_row["dsl_hint"],
            )

            # Check for sub-plan
            sub_plan_id = f"{plan_id}:{step.id}:sub"
            sub_plan = self.load_plan(sub_plan_id)
            if sub_plan:
                step.sub_plan = sub_plan

            steps.append(step)

        plan = DAGPlan(
            steps=steps,
            problem=row["problem"],
            depth=row["depth"],
            parent_step_id=row["parent_step_id"],
        )

        return plan

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    def update_step_result(
        self,
        plan_id: str,
        local_id: str,
        result: str,
        success: bool
    ) -> None:
        """Update a step's result after execution."""
        conn = self._get_conn()
        step_id = f"{plan_id}:{local_id}"
        now = datetime.utcnow().isoformat()

        conn.execute("""
            UPDATE steps
            SET result = ?, success = ?, status = ?, completed_at = ?
            WHERE id = ?
        """, (result, 1 if success else 0, 'completed' if success else 'failed', now, step_id))
        conn.commit()

    def update_plan_status(self, plan_id: str, status: str, result: Optional[str] = None) -> None:
        """Update a plan's status."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()

        if status in ('completed', 'failed'):
            conn.execute("""
                UPDATE plans SET status = ?, result = ?, completed_at = ? WHERE id = ?
            """, (status, result, now, plan_id))
        else:
            conn.execute("""
                UPDATE plans SET status = ? WHERE id = ?
            """, (status, plan_id))
        conn.commit()

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def get_plan_stats(self, plan_id: str) -> dict:
        """Get statistics for a plan including nested sub-plans."""
        plan = self.load_plan(plan_id)
        if not plan:
            return {}

        return {
            "plan_id": plan_id,
            "problem": plan.problem[:50] + "..." if len(plan.problem) > 50 else plan.problem,
            "total_steps": plan.total_steps(),
            "max_depth": plan.max_depth(),
            "direct_steps": len(plan.steps),
            "composite_steps": sum(1 for s in plan.steps if s.is_composite),
            "atomic_steps": sum(1 for s in plan.steps if s.is_atomic),
        }

    def list_plans(self, limit: int = 100, status: Optional[str] = None) -> list[dict]:
        """List recent plans."""
        conn = self._get_conn()

        if status:
            rows = conn.execute("""
                SELECT id, problem, depth, status, created_at
                FROM plans
                WHERE parent_step_id IS NULL AND status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (status, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, problem, depth, status, created_at
                FROM plans
                WHERE parent_step_id IS NULL
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [dict(row) for row in rows]

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan and all nested sub-plans (cascades via FK)."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM plans WHERE id = ? OR id LIKE ?", (plan_id, f"{plan_id}:%"))
        conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # RECURSIVE QUERIES
    # =========================================================================

    def get_plan_hierarchy(self, plan_id: str) -> list[dict]:
        """Get the full hierarchy of a plan using the recursive view."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM plan_hierarchy WHERE root_plan_id = ?
        """, (plan_id,)).fetchall()
        return [dict(row) for row in rows]

    def find_composite_steps(self, plan_id: str) -> list[dict]:
        """Find all composite steps in a plan (steps with sub-plans)."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM steps_with_subplan
            WHERE plan_id = ? AND is_composite = 1
        """, (plan_id,)).fetchall()
        return [dict(row) for row in rows]


# Convenience function
def get_plans_db(db_path: Optional[Path] = None) -> PlansDB:
    """Get a PlansDB instance."""
    return PlansDB(db_path)
