"""Slow Decay System: Lifecycle management for signature health.

Per CLAUDE.md: "slow decay: sig_uses / total_problems"

This module provides comprehensive decay tracking beyond basic penalties:
- DecayState: Persistent decay tracking per signature
- DecayManager: Orchestrate decay across all signatures
- Lifecycle hooks: Archive, demote, merge based on thresholds
- Recovery tracking: Monitor when decayed signatures revive

The goal is SMOOTH and CONTINUOUS learning - signatures that don't
pull their weight gradually fade, making room for better ones.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from mycelium.step_signatures.db import get_step_db
from mycelium.step_signatures.utils import invalidate_centroid_cache
from mycelium.config import (
    DB_PATH,
    TRAFFIC_MIN_SHARE,
    TRAFFIC_GRACE_PROBLEMS,
    DECAY_CHECK_INTERVAL_SEC,
    DECAY_ARCHIVE_THRESHOLD,
    DECAY_DEMOTE_THRESHOLD,
    DECAY_WARNING_THRESHOLD,
    DECAY_RECOVERY_THRESHOLD,
    DECAY_ARCHIVE_GRACE_DAYS,
    DECAY_MAX_ACTIONS_PER_RUN,
)

logger = logging.getLogger(__name__)


class DecayStatus(Enum):
    """Decay status levels for a signature."""
    HEALTHY = "healthy"           # Above min traffic share
    WARNING = "warning"           # Below warning threshold
    CRITICAL = "critical"         # Below demote threshold
    ARCHIVED = "archived"         # Soft-deleted due to decay
    RECOVERING = "recovering"     # Was critical, now improving


@dataclass
class DecayState:
    """Persistent decay state for a signature.

    Tracks not just current decay level but history and trends.
    This enables smooth decay rather than sudden cutoffs.
    """
    signature_id: int
    status: DecayStatus = DecayStatus.HEALTHY

    # Traffic metrics
    current_traffic_share: float = 0.0
    min_traffic_share_seen: float = 1.0
    max_traffic_share_seen: float = 0.0

    # Timing
    first_seen_at: Optional[str] = None
    last_healthy_at: Optional[str] = None
    entered_warning_at: Optional[str] = None
    entered_critical_at: Optional[str] = None
    archived_at: Optional[str] = None

    # Recovery tracking
    recovery_attempts: int = 0
    last_recovery_at: Optional[str] = None

    # Computed trend (positive = improving, negative = declining)
    trend_7d: float = 0.0
    trend_30d: float = 0.0

    def days_in_current_status(self) -> float:
        """How many days has this signature been in current status."""
        now = datetime.now(timezone.utc)

        def parse_ts(ts_str: str) -> float:
            """Parse timestamp and return days since then, or 0.0 on error."""
            if not ts_str or not ts_str.strip():
                return 0.0
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                # Ensure timezone-aware comparison
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return (now - ts).total_seconds() / 86400.0
            except (ValueError, TypeError):
                return 0.0

        if self.status == DecayStatus.ARCHIVED and self.archived_at:
            return parse_ts(self.archived_at)
        elif self.status == DecayStatus.CRITICAL and self.entered_critical_at:
            return parse_ts(self.entered_critical_at)
        elif self.status == DecayStatus.WARNING and self.entered_warning_at:
            return parse_ts(self.entered_warning_at)
        elif self.last_healthy_at:
            return parse_ts(self.last_healthy_at)

        return 0.0

    def is_improving(self) -> bool:
        """Is this signature's traffic trending upward."""
        return self.trend_7d > 0.01  # 1% improvement threshold

    def should_archive(self) -> bool:
        """Should this signature be archived due to decay."""
        if self.status == DecayStatus.ARCHIVED:
            return False  # Already archived

        # Must be critical for long enough
        if self.status != DecayStatus.CRITICAL:
            return False

        days_critical = self.days_in_current_status()
        if days_critical < DECAY_ARCHIVE_GRACE_DAYS:
            return False

        # Don't archive if recovering
        if self.is_improving():
            return False

        return True


@dataclass
class DecayAction:
    """An action to take based on decay analysis."""
    signature_id: int
    action: str  # "archive", "demote", "warn", "recover"
    reason: str
    old_status: DecayStatus
    new_status: DecayStatus
    traffic_share: float


@dataclass
class DecayReport:
    """Summary of decay analysis across all signatures."""
    total_signatures: int = 0
    healthy_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    archived_count: int = 0
    recovering_count: int = 0

    actions_taken: list[DecayAction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Aggregate metrics
    avg_traffic_share: float = 0.0
    median_traffic_share: float = 0.0
    total_uses: int = 0
    total_problems: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Decay Report: {self.total_signatures} signatures\n"
            f"  Healthy: {self.healthy_count}, Warning: {self.warning_count}, "
            f"Critical: {self.critical_count}, Archived: {self.archived_count}\n"
            f"  Recovering: {self.recovering_count}\n"
            f"  Actions: {len(self.actions_taken)}, Errors: {len(self.errors)}\n"
            f"  Avg traffic share: {self.avg_traffic_share:.2%}"
        )


class DecayManager:
    """Orchestrate slow decay across all signatures.

    This class:
    - Computes decay state for each signature
    - Tracks decay history in DB
    - Triggers lifecycle actions (archive, demote, warn)
    - Monitors recovery when decayed signatures get traffic

    Usage:
        manager = DecayManager()
        report = manager.run_decay_cycle()
        print(report.summary())
    """

    def __init__(self, db_path: str = DB_PATH):
        """Initialize DecayManager.

        Args:
            db_path: Kept for API compatibility but ignored (uses data layer).
        """
        self._db = get_step_db()
        self._last_run_at: float = 0
        self._decay_states: dict[int, DecayState] = {}

    def _connection(self):
        """Get DB connection via data layer.

        Per CLAUDE.md "New Favorite Pattern": Centralized connection management.
        """
        return self._db._connection()

    def _utc_now_iso(self) -> str:
        """Current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    def _ensure_decay_table(self) -> None:
        """Create decay tracking table if needed."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signature_decay (
                    signature_id INTEGER PRIMARY KEY,
                    status TEXT DEFAULT 'healthy',
                    current_traffic_share REAL DEFAULT 0,
                    min_traffic_share_seen REAL DEFAULT 1,
                    max_traffic_share_seen REAL DEFAULT 0,
                    first_seen_at TEXT,
                    last_healthy_at TEXT,
                    entered_warning_at TEXT,
                    entered_critical_at TEXT,
                    archived_at TEXT,
                    recovery_attempts INTEGER DEFAULT 0,
                    last_recovery_at TEXT,
                    trend_7d REAL DEFAULT 0,
                    trend_30d REAL DEFAULT 0,
                    updated_at TEXT,
                    FOREIGN KEY (signature_id) REFERENCES step_signatures(id)
                )
            """)

            # Also create history table for trend analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decay_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature_id INTEGER,
                    traffic_share REAL,
                    status TEXT,
                    recorded_at TEXT,
                    FOREIGN KEY (signature_id) REFERENCES step_signatures(id)
                )
            """)

            # Index for efficient trend queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decay_history_sig_time
                ON decay_history(signature_id, recorded_at)
            """)

            # Index for status aggregation queries (e.g., count by status)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decay_status
                ON signature_decay(status)
            """)
            conn.commit()

    def _get_total_problems(self) -> int:
        """Get total problems solved from metadata.

        Per CLAUDE.md New Favorite Pattern: Consolidated db_metadata access.
        """
        from mycelium.data_layer.state_manager import get_state_manager, StateManager
        return get_state_manager().get_int(StateManager.KEY_TOTAL_PROBLEMS)

    def _get_all_signature_stats(self) -> list[dict]:
        """Get uses for all non-archived signatures."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT id, uses, successes, is_semantic_umbrella, dsl_type,
                       created_at, last_used_at
                FROM step_signatures
                WHERE is_archived = 0 OR is_archived IS NULL
            """).fetchall()

            return [dict(row) for row in rows]

    def _load_decay_state(self, sig_id: int) -> Optional[DecayState]:
        """Load decay state from DB."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM signature_decay WHERE signature_id = ?",
                (sig_id,)
            ).fetchone()

            if not row:
                return None

            return DecayState(
                signature_id=row["signature_id"],
                status=DecayStatus(row["status"]),
                current_traffic_share=row["current_traffic_share"] or 0,
                min_traffic_share_seen=row["min_traffic_share_seen"] or 1,
                max_traffic_share_seen=row["max_traffic_share_seen"] or 0,
                first_seen_at=row["first_seen_at"],
                last_healthy_at=row["last_healthy_at"],
                entered_warning_at=row["entered_warning_at"],
                entered_critical_at=row["entered_critical_at"],
                archived_at=row["archived_at"],
                recovery_attempts=row["recovery_attempts"] or 0,
                last_recovery_at=row["last_recovery_at"],
                trend_7d=row["trend_7d"] or 0,
                trend_30d=row["trend_30d"] or 0,
            )

    def _save_decay_state(self, state: DecayState) -> None:
        """Persist decay state to DB."""
        now = self._utc_now_iso()

        with self._connection() as conn:
            conn.execute("""
                INSERT INTO signature_decay (
                    signature_id, status, current_traffic_share,
                    min_traffic_share_seen, max_traffic_share_seen,
                    first_seen_at, last_healthy_at, entered_warning_at,
                    entered_critical_at, archived_at, recovery_attempts,
                    last_recovery_at, trend_7d, trend_30d, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature_id) DO UPDATE SET
                    status = excluded.status,
                    current_traffic_share = excluded.current_traffic_share,
                    min_traffic_share_seen = excluded.min_traffic_share_seen,
                    max_traffic_share_seen = excluded.max_traffic_share_seen,
                    first_seen_at = COALESCE(signature_decay.first_seen_at, excluded.first_seen_at),
                    last_healthy_at = excluded.last_healthy_at,
                    entered_warning_at = excluded.entered_warning_at,
                    entered_critical_at = excluded.entered_critical_at,
                    archived_at = excluded.archived_at,
                    recovery_attempts = excluded.recovery_attempts,
                    last_recovery_at = excluded.last_recovery_at,
                    trend_7d = excluded.trend_7d,
                    trend_30d = excluded.trend_30d,
                    updated_at = excluded.updated_at
            """, (
                state.signature_id, state.status.value, state.current_traffic_share,
                state.min_traffic_share_seen, state.max_traffic_share_seen,
                state.first_seen_at, state.last_healthy_at, state.entered_warning_at,
                state.entered_critical_at, state.archived_at, state.recovery_attempts,
                state.last_recovery_at, state.trend_7d, state.trend_30d, now
            ))
            conn.commit()

    def _record_history(self, sig_id: int, traffic_share: float, status: DecayStatus) -> None:
        """Record a point in decay history for trend analysis."""
        now = self._utc_now_iso()

        with self._connection() as conn:
            conn.execute("""
                INSERT INTO decay_history (signature_id, traffic_share, status, recorded_at)
                VALUES (?, ?, ?, ?)
            """, (sig_id, traffic_share, status.value, now))
            conn.commit()

    def _compute_trend(self, sig_id: int, days: int) -> float:
        """Compute traffic trend over N days.

        Returns:
            Trend value: positive = improving, negative = declining
            0 if not enough data
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._connection() as conn:
            rows = conn.execute("""
                SELECT traffic_share, recorded_at
                FROM decay_history
                WHERE signature_id = ? AND recorded_at > ?
                ORDER BY recorded_at ASC
            """, (sig_id, cutoff)).fetchall()

        if len(rows) < 2:
            return 0.0

        # Simple linear trend: (last - first) / first
        first_share = rows[0]["traffic_share"]
        last_share = rows[-1]["traffic_share"]

        if first_share <= 0:
            return 1.0 if last_share > 0 else 0.0

        return (last_share - first_share) / first_share

    def _compute_decay_status(
        self,
        traffic_share: float,
        current_state: Optional[DecayState],
    ) -> DecayStatus:
        """Determine decay status from traffic share."""
        # Thresholds relative to TRAFFIC_MIN_SHARE
        archive_threshold = TRAFFIC_MIN_SHARE * DECAY_ARCHIVE_THRESHOLD
        demote_threshold = TRAFFIC_MIN_SHARE * DECAY_DEMOTE_THRESHOLD
        warning_threshold = TRAFFIC_MIN_SHARE * DECAY_WARNING_THRESHOLD
        recovery_threshold = TRAFFIC_MIN_SHARE * DECAY_RECOVERY_THRESHOLD

        # Check for recovery first
        if current_state and current_state.status in (DecayStatus.CRITICAL, DecayStatus.WARNING):
            if traffic_share >= recovery_threshold:
                return DecayStatus.RECOVERING

        # Determine status from thresholds
        if traffic_share >= TRAFFIC_MIN_SHARE:
            return DecayStatus.HEALTHY
        elif traffic_share >= warning_threshold:
            return DecayStatus.WARNING
        elif traffic_share >= demote_threshold:
            return DecayStatus.CRITICAL
        else:
            # Below archive threshold - but don't auto-archive, just mark critical
            # Actual archiving requires additional checks (time in critical, etc.)
            return DecayStatus.CRITICAL

    def analyze_signature(
        self,
        sig_id: int,
        uses: int,
        total_problems: int,
        created_at: Optional[str] = None,
    ) -> tuple[DecayState, Optional[DecayAction]]:
        """Analyze decay state for a single signature.

        Returns:
            (updated_state, action_if_any)
        """
        now = self._utc_now_iso()

        # Calculate traffic share
        traffic_share = uses / total_problems if total_problems > 0 else 0.0

        # Load existing state or create new
        state = self._load_decay_state(sig_id)
        if state is None:
            state = DecayState(
                signature_id=sig_id,
                first_seen_at=created_at or now,
                last_healthy_at=now,
            )

        # Update traffic metrics
        old_status = state.status
        state.current_traffic_share = traffic_share
        state.min_traffic_share_seen = min(state.min_traffic_share_seen, traffic_share)
        state.max_traffic_share_seen = max(state.max_traffic_share_seen, traffic_share)

        # Compute trends
        state.trend_7d = self._compute_trend(sig_id, 7)
        state.trend_30d = self._compute_trend(sig_id, 30)

        # Determine new status
        new_status = self._compute_decay_status(traffic_share, state)
        state.status = new_status

        # Update timestamps based on status transitions
        action = None

        if new_status == DecayStatus.HEALTHY:
            state.last_healthy_at = now
            state.entered_warning_at = None
            state.entered_critical_at = None

            if old_status in (DecayStatus.RECOVERING, DecayStatus.CRITICAL, DecayStatus.WARNING):
                action = DecayAction(
                    signature_id=sig_id,
                    action="recover",
                    reason=f"Traffic share recovered to {traffic_share:.2%}",
                    old_status=old_status,
                    new_status=new_status,
                    traffic_share=traffic_share,
                )
                state.last_recovery_at = now
                state.recovery_attempts += 1

        elif new_status == DecayStatus.WARNING:
            if old_status == DecayStatus.HEALTHY:
                state.entered_warning_at = now
                action = DecayAction(
                    signature_id=sig_id,
                    action="warn",
                    reason=f"Traffic share dropped to {traffic_share:.2%}",
                    old_status=old_status,
                    new_status=new_status,
                    traffic_share=traffic_share,
                )

        elif new_status == DecayStatus.CRITICAL:
            if old_status in (DecayStatus.HEALTHY, DecayStatus.WARNING):
                state.entered_critical_at = now
                action = DecayAction(
                    signature_id=sig_id,
                    action="demote",
                    reason=f"Traffic share critical at {traffic_share:.2%}",
                    old_status=old_status,
                    new_status=new_status,
                    traffic_share=traffic_share,
                )

        elif new_status == DecayStatus.RECOVERING:
            if old_status == DecayStatus.CRITICAL:
                action = DecayAction(
                    signature_id=sig_id,
                    action="recovering",
                    reason=f"Traffic share improving to {traffic_share:.2%}",
                    old_status=old_status,
                    new_status=new_status,
                    traffic_share=traffic_share,
                )

        # Check for archive condition
        if state.should_archive():
            state.status = DecayStatus.ARCHIVED
            state.archived_at = now
            action = DecayAction(
                signature_id=sig_id,
                action="archive",
                reason=f"Traffic share at {traffic_share:.2%} for {state.days_in_current_status():.0f} days",
                old_status=old_status,
                new_status=DecayStatus.ARCHIVED,
                traffic_share=traffic_share,
            )

        return state, action

    def _apply_action(self, action: DecayAction) -> bool:
        """Apply a decay action to the database.

        Returns:
            True if action was applied successfully
        """
        try:
            with self._connection() as conn:
                if action.action == "archive":
                    # Soft-delete the signature via consolidated db method
                    self._db.archive_signature(action.signature_id, reason=action.reason)

                elif action.action == "demote":
                    # If umbrella with no healthy children, demote to leaf
                    row = conn.execute(
                        "SELECT is_semantic_umbrella FROM step_signatures WHERE id = ?",
                        (action.signature_id,)
                    ).fetchone()

                    if row and row["is_semantic_umbrella"]:
                        # Check if has healthy children
                        child_count = conn.execute("""
                            SELECT COUNT(*) as cnt FROM signature_relationships sr
                            JOIN step_signatures s ON sr.child_id = s.id
                            WHERE sr.parent_id = ?
                            AND (s.is_archived = 0 OR s.is_archived IS NULL)
                        """, (action.signature_id,)).fetchone()["cnt"]

                        if child_count == 0:
                            # Per CLAUDE.md "New Favorite Pattern": use consolidated method
                            self._db.demote_umbrella_to_leaf(
                                signature_id=action.signature_id,
                                reason="no_healthy_children",
                                conn=conn,
                            )

                elif action.action == "warn":
                    logger.warning(
                        "[decay] Signature %d entering warning state: %s",
                        action.signature_id, action.reason
                    )

                elif action.action == "recover":
                    logger.info(
                        "[decay] Signature %d recovered: %s",
                        action.signature_id, action.reason
                    )

                conn.commit()
                return True

        except Exception as e:
            logger.error(
                "[decay] Failed to apply action %s for sig %d: %s",
                action.action, action.signature_id, e
            )
            return False

    def run_decay_cycle(self, force: bool = False) -> DecayReport:
        """Run a full decay analysis cycle.

        Args:
            force: Run even if not enough time has passed

        Returns:
            DecayReport with results
        """
        # Check if enough time has passed
        now = time.time()
        if not force and (now - self._last_run_at) < DECAY_CHECK_INTERVAL_SEC:
            return DecayReport()

        self._last_run_at = now
        report = DecayReport()

        try:
            # Ensure tables exist
            self._ensure_decay_table()

            # Get totals
            total_problems = self._get_total_problems()
            report.total_problems = total_problems

            # Grace period - don't decay during cold start
            if total_problems < TRAFFIC_GRACE_PROBLEMS:
                logger.debug(
                    "[decay] Skipping decay cycle - cold start (%d < %d problems)",
                    total_problems, TRAFFIC_GRACE_PROBLEMS
                )
                return report

            # Get all signatures
            signatures = self._get_all_signature_stats()
            report.total_signatures = len(signatures)

            if not signatures:
                return report

            # Calculate aggregate metrics
            total_uses = sum(s["uses"] for s in signatures)
            report.total_uses = total_uses

            traffic_shares = []
            actions_to_apply = []

            # Analyze each signature
            for sig in signatures:
                state, action = self.analyze_signature(
                    sig_id=sig["id"],
                    uses=sig["uses"],
                    total_problems=total_problems,
                    created_at=sig.get("created_at"),
                )

                # Save state and record history
                self._save_decay_state(state)
                self._record_history(sig["id"], state.current_traffic_share, state.status)

                # Count statuses
                traffic_shares.append(state.current_traffic_share)
                if state.status == DecayStatus.HEALTHY:
                    report.healthy_count += 1
                elif state.status == DecayStatus.WARNING:
                    report.warning_count += 1
                elif state.status == DecayStatus.CRITICAL:
                    report.critical_count += 1
                elif state.status == DecayStatus.ARCHIVED:
                    report.archived_count += 1
                elif state.status == DecayStatus.RECOVERING:
                    report.recovering_count += 1

                # Collect actions
                if action:
                    actions_to_apply.append(action)

            # Calculate aggregate metrics
            if traffic_shares:
                report.avg_traffic_share = sum(traffic_shares) / len(traffic_shares)
                sorted_shares = sorted(traffic_shares)
                mid = len(sorted_shares) // 2
                report.median_traffic_share = sorted_shares[mid]

            # Apply actions (with limit)
            for action in actions_to_apply[:DECAY_MAX_ACTIONS_PER_RUN]:
                if self._apply_action(action):
                    report.actions_taken.append(action)
                else:
                    report.errors.append(f"Failed to apply {action.action} for sig {action.signature_id}")

            logger.info("[decay] %s", report.summary())

        except Exception as e:
            logger.error("[decay] Decay cycle failed: %s", e)
            report.errors.append(str(e))

        return report

    def get_signature_health(self, sig_id: int) -> Optional[DecayState]:
        """Get decay health for a specific signature."""
        return self._load_decay_state(sig_id)

    def get_decay_summary(self) -> dict:
        """Get summary of all signature decay states."""
        try:
            self._ensure_decay_table()

            with self._connection() as conn:
                rows = conn.execute("""
                    SELECT status, COUNT(*) as cnt
                    FROM signature_decay
                    GROUP BY status
                """).fetchall()

            summary = {
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "archived": 0,
                "recovering": 0,
            }

            for row in rows:
                summary[row["status"]] = row["cnt"]

            return summary

        except Exception as e:
            logger.error("[decay] Failed to get decay summary: %s", e)
            return {"error": str(e)}

    def restore_signature(self, sig_id: int) -> bool:
        """Restore an archived signature.

        Use this when manually reviewing and deciding a signature
        should be given another chance.
        """
        try:
            # Un-archive via consolidated db method (handles cache invalidation)
            self._db.unarchive_signature(sig_id)

            # Reset decay state
            with self._connection() as conn:
                now = self._utc_now_iso()
                conn.execute("""
                    UPDATE signature_decay
                    SET status = 'warning',
                        archived_at = NULL,
                        recovery_attempts = recovery_attempts + 1,
                        last_recovery_at = ?,
                        updated_at = ?
                    WHERE signature_id = ?
                """, (now, now, sig_id))
                conn.commit()

            logger.info("[decay] Restored signature %d from archive", sig_id)
            return True

        except Exception as e:
            logger.error("[decay] Failed to restore signature %d: %s", sig_id, e)
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_decay_manager: Optional[DecayManager] = None


def get_decay_manager() -> DecayManager:
    """Get singleton DecayManager instance."""
    global _decay_manager
    if _decay_manager is None:
        _decay_manager = DecayManager()
    return _decay_manager


def run_decay_cycle(force: bool = False) -> DecayReport:
    """Run decay cycle using singleton manager."""
    return get_decay_manager().run_decay_cycle(force=force)


def get_signature_decay_state(sig_id: int) -> Optional[DecayState]:
    """Get decay state for a signature."""
    return get_decay_manager().get_signature_health(sig_id)


def get_decay_summary() -> dict:
    """Get summary of all decay states."""
    return get_decay_manager().get_decay_summary()
