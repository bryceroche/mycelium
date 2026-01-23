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

import random

from mycelium.config import STEP_STATS_ENABLED, STEP_STATS_SAMPLE_RATE
from mycelium.data_layer import configure_connection

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
        configure_connection(conn, enable_foreign_keys=False)
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



# =============================================================================
# STEP STATS TABLE (new analytics table)
# =============================================================================


def record_step_stats(
    db_path: str,
    step_text: str,
    latency_ms: float,
    signature_id: Optional[int] = None,
    embed_latency_ms: float = 0,
    route_latency_ms: float = 0,
    exec_latency_ms: float = 0,
    routing_depth: int = 0,
    was_routed: bool = False,
    route_path: Optional[list[int]] = None,
    embed_cache_hit: bool = False,
    success: bool = False,
    used_dsl: bool = False,
) -> Optional[int]:
    """Record step execution stats to step_stats table.

    Feature-flagged by STEP_STATS_ENABLED and sampled by STEP_STATS_SAMPLE_RATE.

    Args:
        db_path: Path to SQLite database
        step_text: The step being executed
        latency_ms: Total step execution time
        signature_id: ID of signature used (None if no match)
        embed_latency_ms: Time spent on embedding lookup/compute
        route_latency_ms: Time spent on routing decision
        exec_latency_ms: Time spent on DSL/LLM execution
        routing_depth: How deep in umbrella tree
        was_routed: Whether routed through umbrella
        route_path: List of signature IDs traversed
        embed_cache_hit: Whether embedding was cached
        success: Whether step completed successfully
        used_dsl: Whether DSL executed (vs LLM fallback)

    Returns:
        Row ID if recorded, None if skipped (disabled or sampled out)
    """
    if not STEP_STATS_ENABLED:
        return None

    # Sample rate check
    if STEP_STATS_SAMPLE_RATE < 1.0 and random.random() > STEP_STATS_SAMPLE_RATE:
        return None

    now = datetime.now(timezone.utc).isoformat()
    route_path_json = json.dumps(route_path) if route_path else None

    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        configure_connection(conn, enable_foreign_keys=False)

        cursor = conn.execute(
            """INSERT INTO step_stats
               (signature_id, step_text, latency_ms, embed_latency_ms,
                route_latency_ms, exec_latency_ms, routing_depth, was_routed,
                route_path, embed_cache_hit, success, used_dsl, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signature_id,
                step_text[:500],  # Truncate long step text
                latency_ms,
                embed_latency_ms,
                route_latency_ms,
                exec_latency_ms,
                routing_depth,
                1 if was_routed else 0,
                route_path_json,
                1 if embed_cache_hit else 0,
                1 if success else 0,
                1 if used_dsl else 0,
                now,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()

        logger.debug(
            "[step_stats] Recorded: sig=%s latency=%.1fms depth=%d cache_hit=%s",
            signature_id, latency_ms, routing_depth, embed_cache_hit,
        )
        return row_id

    except Exception as e:
        logger.warning("[step_stats] Failed to record: %s", e)
        return None




