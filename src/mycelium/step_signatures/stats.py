"""Rich Step-Level Statistics for Routing Decisions.

Per CLAUDE.md: 'db audit for signature step level stats'

This module tracks detailed step execution metrics that feed into MCTS routing:
- Per-step execution times
- Param extraction success rates
- DSL vs decompose ratios
- Signature health metrics

These stats help routing decisions by identifying:
- Fast, reliable signatures (prefer for similar problems)
- Slow or flaky signatures (consider alternatives)
- Decompose-heavy signatures (might need restructuring)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StepExecution:
    """A single step execution record.

    Captures what happened during one step execution for learning.
    """

    signature_id: int
    step_text: str
    execution_time_ms: float  # How long the step took
    step_completed: bool  # Whether step returned a result
    was_injected: bool  # Whether DSL was injected
    dsl_type: str  # 'math', 'decompose', 'router', etc.
    param_count: int  # Number of params extracted
    param_extraction_success: bool  # Whether all required params were found
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SignatureStats:
    """Aggregated statistics for a single signature.

    These metrics inform routing decisions:
    - High avg_execution_ms + low success_rate = slow and unreliable
    - High dsl_execution_ratio + high success_rate = efficient
    - High param_extraction_rate = good param specifications
    """

    signature_id: int
    total_executions: int = 0

    # Timing stats
    avg_execution_ms: float = 0.0
    min_execution_ms: float = float("inf")
    max_execution_ms: float = 0.0
    p50_execution_ms: float = 0.0  # Median
    p95_execution_ms: float = 0.0  # 95th percentile

    # Success metrics
    success_count: int = 0
    success_rate: float = 0.0

    # DSL vs decompose
    dsl_executions: int = 0  # Times DSL was executed (math, sympy, custom)
    decompose_executions: int = 0  # Times step required decomposition
    router_executions: int = 0  # Times signature routed to children
    dsl_execution_ratio: float = 0.0  # dsl / (dsl + decompose + router)

    # Param extraction
    total_param_extractions: int = 0
    successful_param_extractions: int = 0
    param_extraction_rate: float = 0.0

    # Injection stats
    injection_count: int = 0
    injection_rate: float = 0.0

    @property
    def is_reliable(self) -> bool:
        """Check if signature is reliable enough for zero-LLM routing."""
        return (
            self.total_executions >= 5
            and self.success_rate >= 0.7
            and self.param_extraction_rate >= 0.8
        )

    @property
    def efficiency_score(self) -> float:
        """Compute efficiency score for routing decisions.

        Higher = more efficient (fast + successful + good param extraction).
        Range: 0.0 to 1.0
        """
        if self.total_executions == 0:
            return 0.5  # Neutral for new signatures

        # Normalize execution time (assume 1000ms is "slow")
        time_score = max(0, 1.0 - (self.avg_execution_ms / 1000.0))

        # Weight factors
        return (
            0.4 * self.success_rate
            + 0.3 * time_score
            + 0.2 * self.param_extraction_rate
            + 0.1 * self.dsl_execution_ratio
        )


@dataclass
class RoutingContext:
    """Stats context for making routing decisions.

    Provides summary metrics across signatures to help MCTS routing.
    """

    total_signatures: int = 0
    total_executions: int = 0
    global_success_rate: float = 0.0
    global_avg_execution_ms: float = 0.0

    # Top performers for reference
    fastest_signatures: list[tuple[int, float]] = field(default_factory=list)
    most_reliable_signatures: list[tuple[int, float]] = field(default_factory=list)
    most_efficient_signatures: list[tuple[int, float]] = field(default_factory=list)


# =============================================================================
# STATS COLLECTOR
# =============================================================================


class StepStatsCollector:
    """Collects and queries step-level execution statistics.

    Extends the existing step_usage_log with richer metrics for routing.
    Thread-safe: uses its own connection per operation.
    """

    # Schema extension for step_usage_log (adds execution_time_ms)
    SCHEMA_EXTENSION = """
    -- Add execution_time_ms if not exists (migration)
    -- SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we check programmatically
    """

    def __init__(self, db_path: str):
        """Initialize stats collector.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _ensure_schema(self):
        """Ensure step_usage_log has execution_time_ms column."""
        conn = self._get_connection()
        try:
            # Check if column exists
            cursor = conn.execute("PRAGMA table_info(step_usage_log)")
            columns = {row["name"] for row in cursor.fetchall()}

            if "execution_time_ms" not in columns:
                conn.execute(
                    "ALTER TABLE step_usage_log ADD COLUMN execution_time_ms REAL DEFAULT 0"
                )
                logger.info("[stats] Added execution_time_ms column to step_usage_log")

            if "dsl_type" not in columns:
                conn.execute(
                    "ALTER TABLE step_usage_log ADD COLUMN dsl_type TEXT DEFAULT 'unknown'"
                )
                logger.info("[stats] Added dsl_type column to step_usage_log")

            if "param_extraction_success" not in columns:
                conn.execute(
                    "ALTER TABLE step_usage_log ADD COLUMN param_extraction_success INTEGER DEFAULT 1"
                )
                logger.info("[stats] Added param_extraction_success column to step_usage_log")

            conn.commit()
        finally:
            conn.close()

    def record_execution(
        self,
        signature_id: int,
        step_text: str,
        execution_time_ms: float,
        step_completed: bool,
        was_injected: bool,
        dsl_type: str,
        params_extracted: Optional[dict],
        param_extraction_success: bool,
    ) -> int:
        """Record a step execution with rich metrics.

        This extends the existing record_usage() with timing and DSL type info.

        Args:
            signature_id: ID of the signature used
            step_text: The step text that was executed
            execution_time_ms: Time taken to execute the step
            step_completed: Whether the step returned a result
            was_injected: Whether DSL was injected
            dsl_type: Type of DSL used ('math', 'decompose', 'router', etc.)
            params_extracted: Dict of extracted parameters
            param_extraction_success: Whether all required params were found

        Returns:
            ID of the inserted row
        """
        now = datetime.now(timezone.utc).isoformat()
        params_json = json.dumps(params_extracted) if params_extracted else None

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO step_usage_log
                   (signature_id, step_text, step_completed, was_injected,
                    params_extracted, execution_time_ms, dsl_type,
                    param_extraction_success, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signature_id,
                    step_text,
                    1 if step_completed else 0,
                    1 if was_injected else 0,
                    params_json,
                    execution_time_ms,
                    dsl_type,
                    1 if param_extraction_success else 0,
                    now,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            logger.debug(
                "[stats] Recorded execution: sig=%d time=%.1fms type=%s success=%s",
                signature_id,
                execution_time_ms,
                dsl_type,
                step_completed,
            )
            return row_id
        finally:
            conn.close()

    def get_signature_stats(self, signature_id: int) -> SignatureStats:
        """Get aggregated statistics for a signature.

        Args:
            signature_id: ID of the signature

        Returns:
            SignatureStats with aggregated metrics
        """
        conn = self._get_connection()
        try:
            # Get all execution records for this signature
            cursor = conn.execute(
                """SELECT execution_time_ms, step_completed, was_injected,
                          dsl_type, param_extraction_success
                   FROM step_usage_log
                   WHERE signature_id = ?
                   ORDER BY created_at DESC""",
                (signature_id,),
            )
            rows = cursor.fetchall()

            if not rows:
                return SignatureStats(signature_id=signature_id)

            stats = SignatureStats(signature_id=signature_id)
            stats.total_executions = len(rows)

            # Collect timing data
            times = []
            for row in rows:
                exec_time = row["execution_time_ms"] or 0
                times.append(exec_time)

                if row["step_completed"]:
                    stats.success_count += 1

                if row["was_injected"]:
                    stats.injection_count += 1

                dsl_type = row["dsl_type"] or "unknown"
                if dsl_type in ("math", "sympy", "custom"):
                    stats.dsl_executions += 1
                elif dsl_type == "decompose":
                    stats.decompose_executions += 1
                elif dsl_type == "router":
                    stats.router_executions += 1

                stats.total_param_extractions += 1
                if row["param_extraction_success"]:
                    stats.successful_param_extractions += 1

            # Calculate timing percentiles
            if times:
                times.sort()
                stats.avg_execution_ms = sum(times) / len(times)
                stats.min_execution_ms = times[0]
                stats.max_execution_ms = times[-1]
                stats.p50_execution_ms = times[len(times) // 2]
                p95_idx = int(len(times) * 0.95)
                stats.p95_execution_ms = times[min(p95_idx, len(times) - 1)]

            # Calculate rates
            n = stats.total_executions
            stats.success_rate = stats.success_count / n if n > 0 else 0.0
            stats.injection_rate = stats.injection_count / n if n > 0 else 0.0

            total_typed = (
                stats.dsl_executions + stats.decompose_executions + stats.router_executions
            )
            stats.dsl_execution_ratio = (
                stats.dsl_executions / total_typed if total_typed > 0 else 0.0
            )

            stats.param_extraction_rate = (
                stats.successful_param_extractions / stats.total_param_extractions
                if stats.total_param_extractions > 0
                else 0.0
            )

            return stats
        finally:
            conn.close()

    def get_routing_context(self, limit: int = 10) -> RoutingContext:
        """Get global routing context with top performers.

        Args:
            limit: Number of top signatures to include

        Returns:
            RoutingContext with global metrics and top performers
        """
        conn = self._get_connection()
        try:
            ctx = RoutingContext()

            # Get global counts
            row = conn.execute(
                "SELECT COUNT(DISTINCT signature_id), COUNT(*) FROM step_usage_log"
            ).fetchone()
            ctx.total_signatures = row[0] or 0
            ctx.total_executions = row[1] or 0

            if ctx.total_executions == 0:
                return ctx

            # Global success rate
            success_row = conn.execute(
                "SELECT SUM(step_completed), AVG(execution_time_ms) FROM step_usage_log"
            ).fetchone()
            total_success = success_row[0] or 0
            ctx.global_success_rate = total_success / ctx.total_executions
            ctx.global_avg_execution_ms = success_row[1] or 0.0

            # Top performers by different metrics
            # Fastest (avg execution time, min 5 executions)
            cursor = conn.execute(
                """SELECT signature_id, AVG(execution_time_ms) as avg_time
                   FROM step_usage_log
                   GROUP BY signature_id
                   HAVING COUNT(*) >= 5
                   ORDER BY avg_time ASC
                   LIMIT ?""",
                (limit,),
            )
            ctx.fastest_signatures = [(row[0], row[1]) for row in cursor.fetchall()]

            # Most reliable (highest success rate, min 5 executions)
            cursor = conn.execute(
                """SELECT signature_id,
                          CAST(SUM(step_completed) AS REAL) / COUNT(*) as success_rate
                   FROM step_usage_log
                   GROUP BY signature_id
                   HAVING COUNT(*) >= 5
                   ORDER BY success_rate DESC
                   LIMIT ?""",
                (limit,),
            )
            ctx.most_reliable_signatures = [(row[0], row[1]) for row in cursor.fetchall()]

            return ctx
        finally:
            conn.close()

    def get_dsl_type_breakdown(self, signature_id: Optional[int] = None) -> dict:
        """Get breakdown of DSL types used.

        Args:
            signature_id: Optional filter by signature

        Returns:
            Dict with counts per DSL type
        """
        conn = self._get_connection()
        try:
            if signature_id is not None:
                cursor = conn.execute(
                    """SELECT dsl_type, COUNT(*) as count
                       FROM step_usage_log
                       WHERE signature_id = ?
                       GROUP BY dsl_type""",
                    (signature_id,),
                )
            else:
                cursor = conn.execute(
                    """SELECT dsl_type, COUNT(*) as count
                       FROM step_usage_log
                       GROUP BY dsl_type"""
                )

            return {row["dsl_type"] or "unknown": row["count"] for row in cursor.fetchall()}
        finally:
            conn.close()

    def get_recent_executions(
        self, signature_id: int, limit: int = 20
    ) -> list[StepExecution]:
        """Get recent execution records for a signature.

        Args:
            signature_id: ID of the signature
            limit: Maximum number of records

        Returns:
            List of StepExecution records
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT signature_id, step_text, execution_time_ms, step_completed,
                          was_injected, dsl_type, params_extracted,
                          param_extraction_success, created_at
                   FROM step_usage_log
                   WHERE signature_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (signature_id, limit),
            )

            results = []
            for row in cursor.fetchall():
                params = row["params_extracted"]
                param_count = 0
                if params:
                    try:
                        param_count = len(json.loads(params))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.debug("[stats] Failed to parse params_extracted: %s", e)

                results.append(
                    StepExecution(
                        signature_id=row["signature_id"],
                        step_text=row["step_text"] or "",
                        execution_time_ms=row["execution_time_ms"] or 0,
                        step_completed=bool(row["step_completed"]),
                        was_injected=bool(row["was_injected"]),
                        dsl_type=row["dsl_type"] or "unknown",
                        param_count=param_count,
                        param_extraction_success=bool(row["param_extraction_success"]),
                        created_at=row["created_at"] or "",
                    )
                )

            return results
        finally:
            conn.close()


# =============================================================================
# ROUTING HELPERS
# =============================================================================


def compute_routing_bonus(stats: SignatureStats) -> float:
    """Compute a routing bonus based on signature stats.

    This bonus is added to the UCB1 score to prefer efficient signatures.

    Args:
        stats: SignatureStats for the signature

    Returns:
        Bonus value (0.0 to 0.2)
    """
    if stats.total_executions < 3:
        return 0.0  # Not enough data

    # Bonus components:
    # - Fast execution: up to +0.1
    # - High param extraction rate: up to +0.05
    # - High DSL execution ratio: up to +0.05

    time_bonus = 0.0
    if stats.avg_execution_ms < 100:
        time_bonus = 0.1
    elif stats.avg_execution_ms < 500:
        time_bonus = 0.05

    param_bonus = 0.05 * stats.param_extraction_rate
    dsl_bonus = 0.05 * stats.dsl_execution_ratio

    return time_bonus + param_bonus + dsl_bonus


def should_prefer_decomposition(stats: SignatureStats) -> bool:
    """Determine if a signature should prefer decomposition over DSL.

    Args:
        stats: SignatureStats for the signature

    Returns:
        True if decomposition historically works better for this signature
    """
    if stats.total_executions < 5:
        return False  # Not enough data

    # Prefer decomposition if:
    # 1. DSL success rate is low
    # 2. Decompose executions have higher success than DSL
    # 3. Param extraction is flaky

    dsl_heavy = stats.dsl_execution_ratio > 0.7
    low_success = stats.success_rate < 0.5
    flaky_params = stats.param_extraction_rate < 0.6

    return dsl_heavy and (low_success or flaky_params)


def get_signature_health(stats: SignatureStats) -> str:
    """Get a health assessment for a signature.

    Args:
        stats: SignatureStats for the signature

    Returns:
        Health status: 'healthy', 'degraded', 'unhealthy', or 'unknown'
    """
    if stats.total_executions < 3:
        return "unknown"

    score = stats.efficiency_score

    if score >= 0.7:
        return "healthy"
    elif score >= 0.4:
        return "degraded"
    else:
        return "unhealthy"
