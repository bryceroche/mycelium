"""Centralized db_metadata access per CLAUDE.md New Favorite Pattern.

All db_metadata keys and access patterns consolidated here.
Single source of truth for application state stored in database.
"""

import json
import logging
import math
import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional, NamedTuple

from mycelium.data_layer.connection import get_db

logger = logging.getLogger(__name__)


class WelfordStats(NamedTuple):
    """Welford algorithm stats for online variance computation."""
    count: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        """Sample variance."""
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        """Sample standard deviation."""
        return math.sqrt(self.variance)


class StateManager:
    """Centralized db_metadata access per CLAUDE.md New Favorite Pattern.

    Consolidates all db_metadata keys and access patterns into a single class.
    Provides type-safe getters/setters, caching, and atomic operations.

    Usage:
        from mycelium.data_layer import state_manager

        # Simple get/set
        count = state_manager.get_int(StateManager.KEY_TOTAL_PROBLEMS)
        state_manager.set(StateManager.KEY_TOTAL_PROBLEMS, count + 1)

        # Atomic increment
        new_count = state_manager.increment(StateManager.KEY_TOTAL_PROBLEMS)

        # Welford stats
        stats = state_manager.get_welford_stats("sim_stats_match")
        state_manager.update_welford_stats("sim_stats_match", 0.85)
    """

    # =========================================================================
    # KEY CONSTANTS - All db_metadata keys defined in one place
    # =========================================================================

    # Problem counter
    KEY_TOTAL_PROBLEMS = "total_problems_solved"

    # Tree restructuring
    KEY_LAST_RESTRUCTURE_COUNT = "last_restructure_count"

    # MCTS post-mortem state (per mcts.py)
    KEY_NODES_FOR_SPLIT = "nodes_for_split"
    KEY_HIGH_CONF_WRONG_NODES = "high_conf_wrong_nodes"
    KEY_DSL_REGEN_COUNT = "dsl_regen_count"

    # Similarity stats (Welford) - stored as prefixes with _count/_mean/_m2 suffixes
    PREFIX_SIM_STATS_MATCH = "sim_stats_match"
    PREFIX_SIM_STATS_CLUSTER = "sim_stats_cluster"

    # Segmentation novelty (TreeGuidedPlanner)
    KEY_SEGMENTATION_NOVELTY = "segmentation_novelty_stats"

    # =========================================================================
    # CACHE CONFIGURATION
    # =========================================================================

    # Cache TTL in seconds
    DEFAULT_CACHE_TTL = 5.0

    # Keys that should be cached (frequently accessed, rarely change)
    CACHED_KEYS = {
        KEY_TOTAL_PROBLEMS: 5.0,  # 5 second TTL
    }

    # =========================================================================
    # SINGLETON
    # =========================================================================

    _instance: Optional["StateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StateManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return
            self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)
            self._initialized = True

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def get(self, key: str, default: str = "") -> str:
        """Get a string value from db_metadata.

        Args:
            key: The metadata key
            default: Default value if key not found

        Returns:
            The string value
        """
        # Check cache first
        if key in self.CACHED_KEYS:
            cached = self._cache.get(key)
            if cached and cached[1] > time.time():
                return cached[0]

        db = get_db()
        with db.connection() as conn:
            row = conn.execute(
                "SELECT value FROM db_metadata WHERE key = ?", (key,)
            ).fetchone()
            value = row["value"] if row else default

        # Update cache if applicable
        if key in self.CACHED_KEYS:
            ttl = self.CACHED_KEYS[key]
            self._cache[key] = (value, time.time() + ttl)

        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer value from db_metadata."""
        try:
            return int(self.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float value from db_metadata."""
        try:
            return float(self.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def get_json(self, key: str, default: Any = None) -> Any:
        """Get a JSON value from db_metadata."""
        raw = self.get(key, "")
        if not raw:
            return default if default is not None else {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default if default is not None else {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in db_metadata (upsert).

        Args:
            key: The metadata key
            value: The value (will be converted to string)
        """
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()
        str_value = str(value)

        with db.connection() as conn:
            conn.execute(
                """INSERT INTO db_metadata (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (key, str_value, now)
            )

        # Update cache if applicable
        if key in self.CACHED_KEYS:
            ttl = self.CACHED_KEYS[key]
            self._cache[key] = (str_value, time.time() + ttl)

    def set_json(self, key: str, value: Any) -> None:
        """Set a JSON value in db_metadata."""
        self.set(key, json.dumps(value))

    def increment(self, key: str, delta: int = 1) -> int:
        """Atomically increment an integer value and return the new value.

        Args:
            key: The metadata key
            delta: Amount to increment (default 1)

        Returns:
            The new value after increment
        """
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        with db.connection() as conn:
            # Upsert with atomic increment
            conn.execute(
                """INSERT INTO db_metadata (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value = CAST(CAST(value AS INTEGER) + ? AS TEXT),
                       updated_at = ?""",
                (key, str(delta), now, delta, now)
            )

            # Get new value
            row = conn.execute(
                "SELECT value FROM db_metadata WHERE key = ?", (key,)
            ).fetchone()

        new_value = int(row["value"]) if row else delta

        # Update cache if applicable
        if key in self.CACHED_KEYS:
            ttl = self.CACHED_KEYS[key]
            self._cache[key] = (str(new_value), time.time() + ttl)

        return new_value

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cached values.

        Args:
            key: Specific key to invalidate, or None to clear all
        """
        if key is None:
            self._cache.clear()
        elif key in self._cache:
            del self._cache[key]

    # =========================================================================
    # WELFORD STATS METHODS
    # =========================================================================

    def get_welford_stats(self, prefix: str) -> WelfordStats:
        """Get Welford stats (count, mean, m2) for a given prefix.

        Args:
            prefix: The stats prefix (e.g., 'sim_stats_match')

        Returns:
            WelfordStats namedtuple with count, mean, m2
        """
        db = get_db()
        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT key, value FROM db_metadata WHERE key LIKE ?",
                (f'{prefix}_%',)
            )
            stats = {}
            for row in cursor:
                # Parse JSON value format: {"value": X}
                try:
                    parsed = json.loads(row["value"])
                    stats[row["key"]] = parsed.get("value", parsed) if isinstance(parsed, dict) else parsed
                except (json.JSONDecodeError, TypeError):
                    stats[row["key"]] = 0

        return WelfordStats(
            count=int(stats.get(f'{prefix}_count', 0)),
            mean=float(stats.get(f'{prefix}_mean', 0.0)),
            m2=float(stats.get(f'{prefix}_m2', 0.0)),
        )

    def update_welford_stats(self, prefix: str, value: float) -> WelfordStats:
        """Update Welford stats with a new observation.

        Args:
            prefix: The stats prefix (e.g., 'sim_stats_match')
            value: The new observation

        Returns:
            Updated WelfordStats
        """
        db = get_db()
        now = datetime.now(timezone.utc).isoformat()

        with db.connection() as conn:
            # Read current stats
            cursor = conn.execute(
                "SELECT key, value FROM db_metadata WHERE key LIKE ?",
                (f'{prefix}_%',)
            )
            stats = {}
            for row in cursor:
                try:
                    parsed = json.loads(row["value"])
                    stats[row["key"]] = parsed.get("value", parsed) if isinstance(parsed, dict) else parsed
                except (json.JSONDecodeError, TypeError):
                    stats[row["key"]] = 0

            count = int(stats.get(f'{prefix}_count', 0))
            mean = float(stats.get(f'{prefix}_mean', 0.0))
            m2 = float(stats.get(f'{prefix}_m2', 0.0))

            # Welford update
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2

            # Write back
            for key, val in [(f'{prefix}_count', count), (f'{prefix}_mean', mean), (f'{prefix}_m2', m2)]:
                json_val = json.dumps({'value': val})
                conn.execute(
                    """INSERT INTO db_metadata (key, value, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
                    (key, json_val, now, json_val, now)
                )

        result = WelfordStats(count=count, mean=mean, m2=m2)
        logger.debug(
            "[state_manager] Updated %s stats: count=%d, mean=%.3f, stddev=%.3f",
            prefix, count, mean, result.stddev
        )
        return result

    # =========================================================================
    # CONVENIENCE METHODS FOR COMMON KEYS
    # =========================================================================

    def get_total_problems_solved(self) -> int:
        """Get total problems solved counter."""
        return self.get_int(self.KEY_TOTAL_PROBLEMS)

    def increment_total_problems_solved(self) -> int:
        """Increment and return total problems solved."""
        return self.increment(self.KEY_TOTAL_PROBLEMS)

    def get_last_restructure_count(self) -> int:
        """Get last restructure problem count."""
        return self.get_int(self.KEY_LAST_RESTRUCTURE_COUNT)

    def set_last_restructure_count(self, count: int) -> None:
        """Set last restructure problem count."""
        self.set(self.KEY_LAST_RESTRUCTURE_COUNT, count)

    def get_similarity_thresholds(self) -> tuple[WelfordStats, WelfordStats]:
        """Get Welford stats for both match and cluster similarity.

        Returns:
            Tuple of (match_stats, cluster_stats)
        """
        match_stats = self.get_welford_stats(self.PREFIX_SIM_STATS_MATCH)
        cluster_stats = self.get_welford_stats(self.PREFIX_SIM_STATS_CLUSTER)
        return match_stats, cluster_stats


# Module-level singleton accessor
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the StateManager singleton."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


# Convenience functions for module-level access
def get(key: str, default: str = "") -> str:
    """Get a string value from db_metadata."""
    return get_state_manager().get(key, default)


def get_int(key: str, default: int = 0) -> int:
    """Get an integer value from db_metadata."""
    return get_state_manager().get_int(key, default)


def get_float(key: str, default: float = 0.0) -> float:
    """Get a float value from db_metadata."""
    return get_state_manager().get_float(key, default)


def get_json(key: str, default: Any = None) -> Any:
    """Get a JSON value from db_metadata."""
    return get_state_manager().get_json(key, default)


def set(key: str, value: Any) -> None:
    """Set a value in db_metadata."""
    get_state_manager().set(key, value)


def set_json(key: str, value: Any) -> None:
    """Set a JSON value in db_metadata."""
    get_state_manager().set_json(key, value)


def increment(key: str, delta: int = 1) -> int:
    """Atomically increment an integer value."""
    return get_state_manager().increment(key, delta)


def get_welford_stats(prefix: str) -> WelfordStats:
    """Get Welford stats for a prefix."""
    return get_state_manager().get_welford_stats(prefix)


def update_welford_stats(prefix: str, value: float) -> WelfordStats:
    """Update Welford stats with a new observation."""
    return get_state_manager().update_welford_stats(prefix, value)
