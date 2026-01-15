"""Signature Health Metrics.

Provides at-a-glance view of signature performance:
- Success rates
- Traffic (uses count)
- Staleness (days since last use)
- Overall health status
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from mycelium.config import (
    DB_PATH,
    STALENESS_DECAY_RATE,
    STALENESS_MAX_PENALTY,
    STALENESS_GRACE_DAYS,
    RELIABILITY_MIN_USES,
    RELIABILITY_MIN_SUCCESS_RATE,
)


HealthStatus = Literal["healthy", "degraded", "unhealthy", "cold", "unknown"]


@dataclass
class SignatureHealth:
    """Health metrics for a single signature."""

    id: int
    signature_id: str
    step_type: str
    description: str
    depth: int

    # Core metrics
    uses: int
    successes: int
    success_rate: float

    # Staleness
    days_since_use: float
    staleness_penalty: float

    # Classification
    is_umbrella: bool
    is_root: bool
    dsl_type: str

    # Computed health
    health_status: HealthStatus
    efficiency_score: float

    # Child count (for umbrellas)
    child_count: int = 0


@dataclass
class HealthSummary:
    """Aggregated health summary across all signatures."""

    total_signatures: int = 0
    total_umbrellas: int = 0
    total_leaves: int = 0

    # Health distribution
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    cold_count: int = 0

    # Traffic distribution
    total_uses: int = 0
    total_successes: int = 0
    global_success_rate: float = 0.0

    # Top performers
    top_by_uses: list[SignatureHealth] = field(default_factory=list)
    top_by_success: list[SignatureHealth] = field(default_factory=list)
    most_stale: list[SignatureHealth] = field(default_factory=list)
    worst_performers: list[SignatureHealth] = field(default_factory=list)

    # DSL type breakdown
    dsl_type_counts: dict[str, int] = field(default_factory=dict)

    # Depth distribution
    depth_counts: dict[int, int] = field(default_factory=dict)


def _compute_health_status(
    uses: int,
    success_rate: float,
    days_since_use: float,
) -> HealthStatus:
    """Compute health status from metrics."""
    if uses < RELIABILITY_MIN_USES:
        return "cold"

    # Check for staleness first
    if days_since_use > 30:
        return "unhealthy"

    # Check success rate
    if success_rate >= RELIABILITY_MIN_SUCCESS_RATE:
        return "healthy"
    elif success_rate >= 0.4:
        return "degraded"
    else:
        return "unhealthy"


def _compute_efficiency_score(
    success_rate: float,
    uses: int,
    days_since_use: float,
) -> float:
    """Compute efficiency score (0.0 to 1.0).

    Higher = more efficient (successful + recent + proven).
    """
    if uses == 0:
        return 0.5  # Neutral for new signatures

    # Components:
    # - Success rate (40%)
    # - Usage confidence (30%) - more uses = more confident
    # - Freshness (30%) - recent usage = better

    usage_score = min(1.0, uses / 20.0)  # Cap at 20 uses
    freshness_score = max(0.0, 1.0 - (days_since_use / 30.0))  # 30 days = 0

    return (
        0.4 * success_rate
        + 0.3 * usage_score
        + 0.3 * freshness_score
    )


def _compute_staleness_penalty(days_since_use: float) -> float:
    """Compute staleness penalty based on days since last use."""
    if days_since_use <= STALENESS_GRACE_DAYS:
        return 0.0
    penalty = (days_since_use - STALENESS_GRACE_DAYS) * STALENESS_DECAY_RATE
    return min(penalty, STALENESS_MAX_PENALTY)


def get_signature_health_report(
    db_path: str = None,
    limit: int = 100,
    include_cold: bool = True,
) -> list[SignatureHealth]:
    """Get health report for all signatures.

    Args:
        db_path: Path to database (default from config)
        limit: Maximum signatures to return
        include_cold: Whether to include cold-start signatures

    Returns:
        List of SignatureHealth objects sorted by efficiency score
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    now = datetime.now(timezone.utc)
    results = []

    try:
        # Get all signatures with child counts
        cursor = conn.execute(
            """SELECT s.*,
                      (SELECT COUNT(*) FROM signature_relationships r
                       WHERE r.parent_id = s.id) as child_count
               FROM step_signatures s
               ORDER BY s.uses DESC
               LIMIT ?""",
            (limit * 2,)  # Fetch extra to filter
        )

        for row in cursor.fetchall():
            uses = row["uses"] or 0
            successes = row["successes"] or 0
            success_rate = successes / uses if uses > 0 else 0.0

            # Calculate staleness
            last_used = row["last_used_at"]
            if last_used:
                try:
                    last_dt = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    days_since = (now - last_dt).total_seconds() / 86400
                except (ValueError, TypeError):
                    days_since = 999.0
            else:
                # Never used - check created_at
                created = row["created_at"]
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                        days_since = (now - created_dt).total_seconds() / 86400
                    except (ValueError, TypeError):
                        days_since = 999.0
                else:
                    days_since = 999.0

            staleness_penalty = _compute_staleness_penalty(days_since)
            health_status = _compute_health_status(uses, success_rate, days_since)
            efficiency_score = _compute_efficiency_score(success_rate, uses, days_since)

            # Filter cold signatures if requested
            if not include_cold and health_status == "cold":
                continue

            health = SignatureHealth(
                id=row["id"],
                signature_id=row["signature_id"],
                step_type=row["step_type"] or "unknown",
                description=(row["description"] or "")[:80],
                depth=row["depth"] or 0,
                uses=uses,
                successes=successes,
                success_rate=success_rate,
                days_since_use=days_since,
                staleness_penalty=staleness_penalty,
                is_umbrella=bool(row["is_semantic_umbrella"]),
                is_root=bool(row["is_root"]),
                dsl_type=row["dsl_type"] or "unknown",
                health_status=health_status,
                efficiency_score=efficiency_score,
                child_count=row["child_count"] or 0,
            )
            results.append(health)

            if len(results) >= limit:
                break

        # Sort by efficiency score descending
        results.sort(key=lambda h: h.efficiency_score, reverse=True)

    finally:
        conn.close()

    return results


def get_health_summary(db_path: str = None) -> HealthSummary:
    """Get aggregated health summary.

    Args:
        db_path: Path to database (default from config)

    Returns:
        HealthSummary with aggregate metrics
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    summary = HealthSummary()
    now = datetime.now(timezone.utc)

    try:
        # Get basic counts
        row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
        summary.total_signatures = row[0] if row else 0

        row = conn.execute(
            "SELECT COUNT(*) FROM step_signatures WHERE is_semantic_umbrella = 1"
        ).fetchone()
        summary.total_umbrellas = row[0] if row else 0
        summary.total_leaves = summary.total_signatures - summary.total_umbrellas

        # Get totals
        row = conn.execute(
            "SELECT SUM(uses), SUM(successes) FROM step_signatures"
        ).fetchone()
        summary.total_uses = row[0] or 0
        summary.total_successes = row[1] or 0
        summary.global_success_rate = (
            summary.total_successes / summary.total_uses
            if summary.total_uses > 0 else 0.0
        )

        # DSL type breakdown
        cursor = conn.execute(
            "SELECT dsl_type, COUNT(*) FROM step_signatures GROUP BY dsl_type"
        )
        for row in cursor.fetchall():
            dsl_type = row[0] or "unknown"
            summary.dsl_type_counts[dsl_type] = row[1]

        # Depth distribution
        cursor = conn.execute(
            "SELECT depth, COUNT(*) FROM step_signatures GROUP BY depth ORDER BY depth"
        )
        for row in cursor.fetchall():
            depth = row[0] or 0
            summary.depth_counts[depth] = row[1]

        # Get all signatures for health classification
        all_health = get_signature_health_report(db_path, limit=9999, include_cold=True)

        for h in all_health:
            if h.health_status == "healthy":
                summary.healthy_count += 1
            elif h.health_status == "degraded":
                summary.degraded_count += 1
            elif h.health_status == "unhealthy":
                summary.unhealthy_count += 1
            elif h.health_status == "cold":
                summary.cold_count += 1

        # Top performers by uses
        by_uses = sorted(all_health, key=lambda h: h.uses, reverse=True)
        summary.top_by_uses = by_uses[:5]

        # Top by success rate (min 5 uses)
        proven = [h for h in all_health if h.uses >= 5]
        by_success = sorted(proven, key=lambda h: h.success_rate, reverse=True)
        summary.top_by_success = by_success[:5]

        # Most stale (min 3 uses, not cold)
        active = [h for h in all_health if h.uses >= 3 and h.health_status != "cold"]
        by_stale = sorted(active, key=lambda h: h.days_since_use, reverse=True)
        summary.most_stale = by_stale[:5]

        # Worst performers (min 5 uses, lowest success rate)
        worst = sorted(proven, key=lambda h: h.success_rate)
        summary.worst_performers = worst[:5]

    finally:
        conn.close()

    return summary


def format_health_report(
    report: list[SignatureHealth],
    show_description: bool = False,
) -> str:
    """Format health report as a text table.

    Args:
        report: List of SignatureHealth objects
        show_description: Whether to include description column

    Returns:
        Formatted string table
    """
    if not report:
        return "No signatures found."

    lines = []

    # Header
    if show_description:
        header = (
            f"{'ID':>4} {'Type':12} {'Status':9} {'Uses':>6} {'Succ%':>6} "
            f"{'Stale':>5} {'Eff':>5} {'DSL':8} {'Description':40}"
        )
    else:
        header = (
            f"{'ID':>4} {'Type':12} {'Status':9} {'Uses':>6} {'Succ%':>6} "
            f"{'Stale':>5} {'Eff':>5} {'DSL':8} {'Depth':>5} {'Kids':>4}"
        )
    lines.append(header)
    lines.append("-" * len(header))

    # Status symbols
    status_symbols = {
        "healthy": "[OK]",
        "degraded": "[--]",
        "unhealthy": "[!!]",
        "cold": "[  ]",
        "unknown": "[??]",
    }

    for h in report:
        status = status_symbols.get(h.health_status, "[??]")
        success_pct = f"{h.success_rate * 100:.0f}%"
        stale_days = f"{h.days_since_use:.0f}d" if h.days_since_use < 999 else "new"
        eff_score = f"{h.efficiency_score:.2f}"

        step_type = h.step_type[:12] if h.step_type else "unknown"
        dsl_type = h.dsl_type[:8] if h.dsl_type else "none"

        if show_description:
            desc = h.description[:40] if h.description else ""
            line = (
                f"{h.id:>4} {step_type:12} {status:9} {h.uses:>6} {success_pct:>6} "
                f"{stale_days:>5} {eff_score:>5} {dsl_type:8} {desc:40}"
            )
        else:
            line = (
                f"{h.id:>4} {step_type:12} {status:9} {h.uses:>6} {success_pct:>6} "
                f"{stale_days:>5} {eff_score:>5} {dsl_type:8} {h.depth:>5} {h.child_count:>4}"
            )
        lines.append(line)

    return "\n".join(lines)


def format_health_summary(summary: HealthSummary) -> str:
    """Format health summary as text.

    Args:
        summary: HealthSummary object

    Returns:
        Formatted string
    """
    lines = []

    lines.append("=== SIGNATURE HEALTH SUMMARY ===")
    lines.append("")

    # Overview
    lines.append(f"Total Signatures: {summary.total_signatures}")
    lines.append(f"  Umbrellas (routers): {summary.total_umbrellas}")
    lines.append(f"  Leaves (executors):  {summary.total_leaves}")
    lines.append("")

    # Traffic
    lines.append(f"Total Uses: {summary.total_uses}")
    lines.append(f"Total Successes: {summary.total_successes}")
    lines.append(f"Global Success Rate: {summary.global_success_rate * 100:.1f}%")
    lines.append("")

    # Health distribution
    lines.append("Health Distribution:")
    lines.append(f"  [OK] Healthy:   {summary.healthy_count:>4}")
    lines.append(f"  [--] Degraded:  {summary.degraded_count:>4}")
    lines.append(f"  [!!] Unhealthy: {summary.unhealthy_count:>4}")
    lines.append(f"  [  ] Cold:      {summary.cold_count:>4}")
    lines.append("")

    # DSL types
    if summary.dsl_type_counts:
        lines.append("DSL Types:")
        for dsl_type, count in sorted(summary.dsl_type_counts.items()):
            lines.append(f"  {dsl_type:12}: {count:>4}")
        lines.append("")

    # Depth distribution
    if summary.depth_counts:
        lines.append("Depth Distribution:")
        for depth, count in sorted(summary.depth_counts.items()):
            bar = "#" * min(count, 30)
            lines.append(f"  {depth:>2}: {bar} ({count})")
        lines.append("")

    # Top performers
    if summary.top_by_uses:
        lines.append("Top by Traffic:")
        for h in summary.top_by_uses:
            lines.append(f"  {h.id:>4}: {h.step_type:20} ({h.uses} uses)")
        lines.append("")

    if summary.top_by_success:
        lines.append("Top by Success Rate (min 5 uses):")
        for h in summary.top_by_success:
            lines.append(
                f"  {h.id:>4}: {h.step_type:20} ({h.success_rate*100:.0f}%)"
            )
        lines.append("")

    if summary.worst_performers:
        lines.append("Worst Performers (min 5 uses):")
        for h in summary.worst_performers:
            lines.append(
                f"  {h.id:>4}: {h.step_type:20} ({h.success_rate*100:.0f}%, {h.uses} uses)"
            )
        lines.append("")

    if summary.most_stale:
        lines.append("Most Stale:")
        for h in summary.most_stale:
            lines.append(
                f"  {h.id:>4}: {h.step_type:20} ({h.days_since_use:.0f} days)"
            )

    return "\n".join(lines)
