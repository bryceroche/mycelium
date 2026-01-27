"""Scoring and Normalization for Signature Routing.

Pure functions for computing routing scores and normalizing text.
All configurable values come from config.py.

MCTS-style UCB1 scoring enables exploration/exploitation balance:
- Exploitation: prefer high-similarity, high-success-rate signatures
- Exploration: give bonus to under-visited signatures (may find better paths)
"""

import logging
import math
import re
import time

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import Optional

from mycelium.config import (
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
    MCTS_SIMILARITY_WEIGHT,
    MCTS_SUCCESS_WEIGHT,
    MCTS_MIN_VISITS_FOR_UCB,
)


# =============================================================================
# UTC TIMESTAMP HELPER
# =============================================================================

def utc_now_iso() -> str:
    """Generate ISO timestamp in UTC with 'Z' suffix.

    Use this instead of datetime.utcnow().isoformat() for consistent
    timezone-aware timestamps that work correctly with staleness calculations.

    Returns:
        ISO format timestamp with 'Z' suffix, e.g., '2024-01-15T12:30:45.123456Z'
    """
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


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

    Note: db_path param kept for API compatibility but ignored (uses data layer).
    """
    import sqlite3
    now = time.time()

    # Return cached value if still valid
    if now < _total_problems_cache["expires_at"]:
        return _total_problems_cache["value"]

    # Query DB for fresh value via data layer
    # Lazy import to avoid circular dependency
    from mycelium.data_layer import get_db
    try:
        db = get_db()
        row = db.fetchone(
            "SELECT value FROM db_metadata WHERE key = 'total_problems_solved'"
        )
        value = int(row["value"]) if row else 0
    except (sqlite3.Error, ValueError, TypeError) as e:
        logger.warning("[scoring] Failed to get total problems: %s", e)
        value = 0

    # Update cache
    _total_problems_cache["value"] = value
    _total_problems_cache["expires_at"] = now + TRAFFIC_CACHE_TTL

    return value


def increment_total_problems(db_path: str = DB_PATH) -> int:
    """Increment total problems counter and return new value.

    Called once per problem completion. Also updates the cache.

    Note: db_path param kept for API compatibility but ignored (uses data layer).
    """
    import sqlite3
    # Lazy import to avoid circular dependency
    from mycelium.data_layer import get_db
    try:
        db = get_db()
        now_iso = datetime.now(timezone.utc).isoformat()

        with db.connection() as conn:
            # Upsert the counter
            conn.execute("""
                INSERT INTO db_metadata (key, value, updated_at)
                VALUES ('total_problems_solved', '1', ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = CAST(CAST(value AS INTEGER) + 1 AS TEXT),
                    updated_at = ?
            """, (now_iso, now_iso))

            # Get new value
            row = conn.execute(
                "SELECT value FROM db_metadata WHERE key = 'total_problems_solved'"
            ).fetchone()

        new_value = int(row["value"]) if row else 1

        # Update cache immediately
        _total_problems_cache["value"] = new_value
        _total_problems_cache["expires_at"] = time.time() + TRAFFIC_CACHE_TTL

        return new_value
    except (sqlite3.Error, ValueError, TypeError) as e:
        logger.warning("[scoring] Failed to increment total problems: %s", e)
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
        last_used_at: ISO timestamp of last use, or None if never used.
                      Accepts UTC timestamps with 'Z' suffix, '+00:00' suffix,
                      or naive timestamps (assumed UTC).

    Returns:
        Penalty to subtract from routing score (0.0 to STALENESS_MAX_PENALTY)
    """
    if not STALENESS_DECAY_ENABLED or not last_used_at:
        return 0.0

    try:
        # Parse ISO timestamp - handle various UTC formats
        # Replace 'Z' with '+00:00' for fromisoformat compatibility
        ts = last_used_at.replace('Z', '+00:00')
        last_used = datetime.fromisoformat(ts)

        # If parsed timestamp is naive (no timezone), assume UTC
        if last_used.tzinfo is None:
            last_used = last_used.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        days_since_use = (now - last_used).total_seconds() / 86400.0

        # Apply grace period
        if days_since_use <= STALENESS_GRACE_DAYS:
            return 0.0

        # Calculate penalty: rate * days, capped at max
        penalty = (days_since_use - STALENESS_GRACE_DAYS) * STALENESS_DECAY_RATE
        return min(penalty, STALENESS_MAX_PENALTY)
    except (ValueError, TypeError, AttributeError) as e:
        # ValueError: invalid timestamp format
        # TypeError: wrong type passed to fromisoformat
        # AttributeError: non-string passed (no .replace method)
        logger.debug("[scoring] Failed to compute staleness penalty: %s", e)
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

    # Guard against invalid threshold config (would cause division issues)
    if TRAFFIC_MIN_SHARE <= 0:
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
    denominator = uses + ROUTING_PRIOR_USES
    effective_rate = (successes + ROUTING_PRIOR_SUCCESSES) / denominator if denominator > 0 else 0.5
    base_score = MCTS_SIMILARITY_WEIGHT * cosine_sim + MCTS_SUCCESS_WEIGHT * effective_rate

    # Apply staleness penalty (time-based decay)
    staleness_penalty = compute_staleness_penalty(last_used_at)

    # Apply traffic penalty (usage-based decay)
    traffic_penalty = compute_traffic_penalty(uses, total_problems)

    return base_score - staleness_penalty - traffic_penalty


def compute_ucb1_score(
    cosine_sim: float,
    uses: int,
    successes: int,
    parent_uses: int,
    last_used_at: Optional[str] = None,
    exploration_c: Optional[float] = None,
    step_node_stats: Optional[dict] = None,
) -> float:
    """Compute MCTS UCB1 score for signature routing.

    UCB1 (Upper Confidence Bound) balances exploitation vs exploration:
    - Exploitation: prefer signatures with high similarity and success rate
    - Exploration: give bonus to under-visited signatures

    Formula: exploit_score + C * sqrt(ln(N) / n)
    Where:
    - exploit_score = similarity * success_rate (weighted)
    - C = exploration constant (adaptive or fixed)
    - N = parent visits (total opportunities at this routing level)
    - n = child visits (this signature's uses)

    Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning."
    If step_node_stats is provided, we blend step-specific performance with
    signature-level stats for more accurate routing decisions.

    Args:
        cosine_sim: Cosine similarity between step and signature
        uses: Number of times this signature was used (n)
        successes: Number of successful uses
        parent_uses: Total uses at parent level (N)
        last_used_at: ISO timestamp of last use (for staleness decay)
        exploration_c: Override exploration constant (None = use adaptive)
        step_node_stats: Optional dict with step-specific stats:
            - uses, wins, losses, win_rate, avg_amplitude_post
            From dag_step_node_stats table (post-mortem feedback)

    Returns:
        UCB1 score (higher = better choice)
    """
    from mycelium.config import (
        STEP_NODE_STATS_ENABLED,
        STEP_NODE_STATS_MIN_USES,
        STEP_NODE_STATS_WEIGHT,
        AMPLITUDE_POST_PENALTY_THRESHOLD,
        AMPLITUDE_POST_PENALTY_MULT,
    )

    # Get exploration constant: use adaptive if not overridden
    if exploration_c is None:
        from mycelium.mcts.adaptive import AdaptiveExploration
        exploration_c = AdaptiveExploration.get_instance().exploration_weight

    # Compute signature-level success rate (baseline)
    denominator = uses + ROUTING_PRIOR_USES
    sig_success_rate = (successes + ROUTING_PRIOR_SUCCESSES) / denominator if denominator > 0 else 0.5

    # Blend with step-node stats if available and trusted
    # Per CLAUDE.md: Post-mortem amplitude_post feeds back into routing
    effective_rate = sig_success_rate
    amplitude_penalty = 1.0  # No penalty by default

    if (
        STEP_NODE_STATS_ENABLED
        and step_node_stats
        and step_node_stats.get("uses", 0) >= STEP_NODE_STATS_MIN_USES
    ):
        step_win_rate = step_node_stats.get("win_rate", 0.5)
        step_amp_post = step_node_stats.get("avg_amplitude_post", 1.0)
        step_uses = step_node_stats.get("uses", 0)

        # Blend step-level and signature-level success rates
        # STEP_NODE_STATS_WEIGHT controls the blend (0.6 = 60% step, 40% sig)
        effective_rate = (
            STEP_NODE_STATS_WEIGHT * step_win_rate +
            (1 - STEP_NODE_STATS_WEIGHT) * sig_success_rate
        )

        # Debug: Log step-node stats influence on routing
        logger.debug(
            "[scoring] Step-node stats applied: sig_rate=%.2f -> effective_rate=%.2f "
            "(step_win=%.2f, step_uses=%d, weight=%.1f)",
            sig_success_rate, effective_rate, step_win_rate, step_uses, STEP_NODE_STATS_WEIGHT
        )

        # Penalize if avg_amplitude_post is low (indicates high-confidence failures)
        # This surfaces destructive interference patterns from post-mortem
        if step_amp_post < AMPLITUDE_POST_PENALTY_THRESHOLD:
            amplitude_penalty = AMPLITUDE_POST_PENALTY_MULT
            logger.debug(
                "[scoring] Amplitude penalty applied: avg_amp_post=%.2f < %.2f, penalty=%.2f",
                step_amp_post, AMPLITUDE_POST_PENALTY_THRESHOLD, amplitude_penalty
            )
    elif step_node_stats and step_node_stats.get("uses", 0) < STEP_NODE_STATS_MIN_USES:
        # Log why stats weren't applied (insufficient data)
        logger.debug(
            "[scoring] Step-node stats skipped: uses=%d < min=%d required",
            step_node_stats.get("uses", 0), STEP_NODE_STATS_MIN_USES
        )

    # Exploitation term: similarity weighted by success rate
    exploit_score = MCTS_SIMILARITY_WEIGHT * cosine_sim + MCTS_SUCCESS_WEIGHT * effective_rate
    exploit_score *= amplitude_penalty  # Apply amplitude-based penalty

    # Exploration term: UCB1 bonus for under-visited signatures
    # sqrt(ln(N) / n) gives higher bonus to less-visited children
    if uses >= MCTS_MIN_VISITS_FOR_UCB and parent_uses > 0:
        # Standard UCB1 exploration bonus
        exploration_bonus = exploration_c * math.sqrt(math.log(parent_uses) / uses)
    elif uses == 0:
        # Unvisited signatures get maximum exploration bonus
        exploration_bonus = exploration_c * 2.0  # High bonus for unexplored
    else:
        # Very few visits: give moderate bonus
        exploration_bonus = exploration_c * 1.0

    # Apply staleness penalty (time-based decay)
    staleness_penalty = compute_staleness_penalty(last_used_at)

    return exploit_score + exploration_bonus - staleness_penalty
