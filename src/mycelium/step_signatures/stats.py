"""Step-Level Statistics Recording.

Per CLAUDE.md: 'db audit for signature step level stats'
Records step execution metrics for potential future analysis.
"""

import json
import logging
import random
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from mycelium.config import STEP_STATS_ENABLED, STEP_STATS_SAMPLE_RATE
from mycelium.data_layer import configure_connection

logger = logging.getLogger(__name__)


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
                step_text[:500],
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
        return row_id

    except Exception as e:
        logger.warning("[step_stats] Failed to record: %s", e)
        return None
