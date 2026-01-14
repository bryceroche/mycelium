"""Scoring and Normalization for Signature Routing.

Pure functions for computing routing scores and normalizing text.
All configurable values come from config.py.
"""

import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

from mycelium.config import (
    ROUTING_SIM_WEIGHT,
    ROUTING_SUCCESS_WEIGHT,
    ROUTING_PRIOR_SUCCESSES,
    ROUTING_PRIOR_USES,
    STALENESS_DECAY_ENABLED,
    STALENESS_DECAY_RATE,
    STALENESS_MAX_PENALTY,
    STALENESS_GRACE_DAYS,
    TRAFFIC_DECAY_ENABLED,
    TRAFFIC_MIN_SHARE,
    TRAFFIC_DECAY_RATE,
    TRAFFIC_CACHE_TTL,
    TRAFFIC_GRACE_PROBLEMS,
    DB_PATH,
)


# =============================================================================
# CACHED TOTAL PROBLEMS COUNTER
# =============================================================================
# Module-level cache to avoid DB hits on every routing decision

_total_problems_cache = {
    "value": 0,
    "expires_at": 0.0,
}


def get_total_problems_solved(db_path: str = DB_PATH) -> int:
    """Get total problems solved with TTL caching.

    Returns cached value if fresh, otherwise queries DB and updates cache.
    This is called frequently during routing, so we cache aggressively.
    """
    now = time.time()

    # Return cached value if still valid
    if now < _total_problems_cache["expires_at"]:
        return _total_problems_cache["value"]

    # Query DB for fresh value
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'total_problems_solved'"
        ).fetchone()
        conn.close()

        value = int(row["value"]) if row else 0
    except (sqlite3.Error, ValueError, TypeError):
        value = 0

    # Update cache
    _total_problems_cache["value"] = value
    _total_problems_cache["expires_at"] = now + TRAFFIC_CACHE_TTL

    return value


def increment_total_problems(db_path: str = DB_PATH) -> int:
    """Increment total problems counter and return new value.

    Called once per problem completion. Also updates the cache.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        now_iso = datetime.now(timezone.utc).isoformat()

        # Upsert the counter
        conn.execute("""
            INSERT INTO db_metadata (key, value, updated_at)
            VALUES ('total_problems_solved', '1', ?)
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                updated_at = ?
        """, (now_iso, now_iso))
        conn.commit()

        # Get new value
        row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'total_problems_solved'"
        ).fetchone()
        conn.close()

        new_value = int(row["value"]) if row else 1

        # Update cache immediately
        _total_problems_cache["value"] = new_value
        _total_problems_cache["expires_at"] = time.time() + TRAFFIC_CACHE_TTL

        return new_value
    except (sqlite3.Error, ValueError, TypeError):
        return 0


def invalidate_traffic_cache() -> None:
    """Force cache refresh on next access."""
    _total_problems_cache["expires_at"] = 0.0


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def normalize_step_text(text: str) -> str:
    """Normalize step text for embedding by replacing specific numbers with placeholders.

    This helps match similar operations regardless of specific values:
    - "Calculate 15 factorial" → "Calculate N factorial"
    - "Raise 5 to power 3" → "Raise N to power N"

    Args:
        text: Raw step text

    Returns:
        Normalized text with numbers replaced by N
    """
    # Replace standalone numbers (not part of words) with N
    # Keep decimal points for now
    normalized = re.sub(r'\b\d+\.?\d*\b', 'N', text)
    return normalized


def compute_staleness_penalty(last_used_at: Optional[str]) -> float:
    """Compute staleness penalty based on days since last use.

    Args:
        last_used_at: ISO timestamp of last use, or None if never used

    Returns:
        Penalty to subtract from routing score (0.0 to STALENESS_MAX_PENALTY)
    """
    if not STALENESS_DECAY_ENABLED or not last_used_at:
        return 0.0

    try:
        # Parse ISO timestamp
        last_used = datetime.fromisoformat(last_used_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        days_since_use = (now - last_used).total_seconds() / 86400.0

        # Apply grace period
        if days_since_use <= STALENESS_GRACE_DAYS:
            return 0.0

        # Calculate penalty: rate * days, capped at max
        penalty = (days_since_use - STALENESS_GRACE_DAYS) * STALENESS_DECAY_RATE
        return min(penalty, STALENESS_MAX_PENALTY)
    except (ValueError, TypeError):
        return 0.0


def compute_traffic_penalty(uses: int, total_problems: Optional[int] = None) -> float:
    """Compute traffic penalty for low-usage signatures.

    Signatures that get very little traffic relative to total problems
    are deprioritized. This helps prune shadowed/dead signatures.

    Args:
        uses: Number of times this signature was used
        total_problems: Total problems solved (uses cache if None)

    Returns:
        Penalty to subtract from routing score (0.0 to TRAFFIC_DECAY_RATE)
    """
    if not TRAFFIC_DECAY_ENABLED:
        return 0.0

    # Get total from cache if not provided
    if total_problems is None:
        total_problems = get_total_problems_solved()

    # Grace period: don't penalize during cold start
    if total_problems < TRAFFIC_GRACE_PROBLEMS:
        return 0.0

    # Calculate traffic share
    traffic_share = uses / total_problems if total_problems > 0 else 0.0

    # No penalty if above minimum share
    if traffic_share >= TRAFFIC_MIN_SHARE:
        return 0.0

    # Linear penalty based on how far below threshold
    # At 0 traffic: full penalty. At threshold: 0 penalty.
    deficit_ratio = 1.0 - (traffic_share / TRAFFIC_MIN_SHARE)
    return TRAFFIC_DECAY_RATE * deficit_ratio


def compute_routing_score(
    cosine_sim: float,
    uses: int,
    successes: int,
    last_used_at: Optional[str] = None,
    total_problems: Optional[int] = None,
) -> float:
    """Compute routing score blending cosine similarity with success rate.

    Uses Bayesian prior to handle cold start (configurable in config.py).
    Formula: base_score - staleness_penalty - traffic_penalty

    Args:
        cosine_sim: Cosine similarity between step and signature
        uses: Number of times signature was used
        successes: Number of successful uses
        last_used_at: ISO timestamp of last use (for staleness decay)
        total_problems: Total problems solved (for traffic decay, uses cache if None)

    Returns:
        Routing score (higher = better match)
    """
    effective_rate = (successes + ROUTING_PRIOR_SUCCESSES) / (uses + ROUTING_PRIOR_USES)
    base_score = ROUTING_SIM_WEIGHT * cosine_sim + ROUTING_SUCCESS_WEIGHT * effective_rate

    # Apply staleness penalty (time-based decay)
    staleness_penalty = compute_staleness_penalty(last_used_at)

    # Apply traffic penalty (usage-based decay)
    traffic_penalty = compute_traffic_penalty(uses, total_problems)

    return base_score - staleness_penalty - traffic_penalty
