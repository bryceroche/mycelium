"""Cold-Start Progress Metrics.

Tracks the system's progress from cold-start (empty DB) to maturity:
- Signature count vs thresholds
- Big bang expansion progress
- Adaptive threshold status
"""

import sqlite3
from dataclasses import dataclass
from typing import Literal

from mycelium.config import (
    DB_PATH,
    MIN_MATCH_THRESHOLD,
    MIN_MATCH_THRESHOLD_COLD_START,
    MIN_MATCH_RAMP_SIGNATURES,
    AUTO_DEMOTE_RAMP_DIVISOR,
    AUTO_DEMOTE_MIN_USES_FLOOR,
    AUTO_DEMOTE_MIN_USES_CAP,
    DSL_OPERATION_INFERENCE_COLD_START,
    DSL_OPERATION_INFERENCE_MATURE,
    DSL_OPERATION_INFERENCE_RAMP_SIGS,
    BIG_BANG_EXPANSION_ENABLED,
    DEPTH_FORCE_DECOMPOSE_DEPTH,
    DEPTH_DECOMPOSE_DECAY_BASE,
)


ColdStartPhase = Literal["cold", "warming", "warm", "mature"]


@dataclass
class ColdStartProgress:
    """Cold-start progress metrics."""

    # Current state
    signature_count: int
    total_uses: int
    total_problems: int

    # Phase
    phase: ColdStartPhase
    phase_description: str

    # Thresholds
    current_match_threshold: float
    target_match_threshold: float
    match_threshold_progress: float  # 0.0 to 1.0

    current_dsl_inference_threshold: float
    target_dsl_inference_threshold: float

    current_demote_min_uses: int
    target_demote_min_uses: int

    # Big Bang status
    big_bang_enabled: bool
    force_decompose_depth: int
    decompose_decay_base: float

    # Tree growth
    max_depth: int
    umbrella_count: int
    leaf_count: int
    avg_children_per_umbrella: float

    # Milestones
    milestones_reached: list[str]
    next_milestone: str
    next_milestone_at: int


def _compute_adaptive_threshold(
    current_count: int,
    cold_value: float,
    mature_value: float,
    ramp_count: int,
) -> float:
    """Compute adaptive threshold based on signature count.

    Linear interpolation from cold_value to mature_value as count grows.
    """
    if current_count >= ramp_count:
        return mature_value

    progress = current_count / ramp_count
    return cold_value + (mature_value - cold_value) * progress


def _compute_phase(
    signature_count: int,
    total_uses: int,
    max_depth: int,
) -> tuple[ColdStartPhase, str]:
    """Determine cold-start phase and description."""
    if signature_count == 0:
        return "cold", "Empty database - ready for first problems"

    if signature_count < 10:
        return "cold", "Initial bootstrapping - building first signatures"

    if signature_count < MIN_MATCH_RAMP_SIGNATURES:
        return "warming", f"Warming up - {signature_count}/{MIN_MATCH_RAMP_SIGNATURES} to mature thresholds"

    if total_uses < 100:
        return "warm", "Warm - building usage history"

    if max_depth < DEPTH_FORCE_DECOMPOSE_DEPTH:
        return "warm", f"Warm - building tree depth ({max_depth}/{DEPTH_FORCE_DECOMPOSE_DEPTH})"

    return "mature", "Mature - system is stabilized"


def get_cold_start_progress(db_path: str = None) -> ColdStartProgress:
    """Get cold-start progress metrics.

    Args:
        db_path: Path to database (default from config)

    Returns:
        ColdStartProgress with all metrics
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        # Basic counts
        row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
        signature_count = row[0] if row else 0

        row = conn.execute("SELECT SUM(uses) FROM step_signatures").fetchone()
        total_uses = row[0] or 0

        # Get total problems from db_metadata
        row = conn.execute(
            "SELECT value FROM db_metadata WHERE key = 'total_problems'"
        ).fetchone()
        total_problems = int(row[0]) if row and row[0] else 0

        # Tree metrics
        row = conn.execute("SELECT MAX(depth) FROM step_signatures").fetchone()
        max_depth = row[0] or 0

        row = conn.execute(
            "SELECT COUNT(*) FROM step_signatures WHERE is_semantic_umbrella = 1"
        ).fetchone()
        umbrella_count = row[0] if row else 0

        leaf_count = signature_count - umbrella_count

        # Average children per umbrella
        row = conn.execute(
            """SELECT AVG(child_count) FROM (
                SELECT COUNT(*) as child_count
                FROM signature_relationships
                GROUP BY parent_id
            )"""
        ).fetchone()
        avg_children = row[0] or 0.0

        # Compute phase
        phase, phase_desc = _compute_phase(signature_count, total_uses, max_depth)

        # Compute current adaptive thresholds
        current_match = _compute_adaptive_threshold(
            signature_count,
            MIN_MATCH_THRESHOLD_COLD_START,
            MIN_MATCH_THRESHOLD,
            MIN_MATCH_RAMP_SIGNATURES,
        )

        current_dsl_inference = _compute_adaptive_threshold(
            signature_count,
            DSL_OPERATION_INFERENCE_COLD_START,
            DSL_OPERATION_INFERENCE_MATURE,
            DSL_OPERATION_INFERENCE_RAMP_SIGS,
        )

        # Demote min uses
        current_demote = min(
            AUTO_DEMOTE_MIN_USES_FLOOR + signature_count // AUTO_DEMOTE_RAMP_DIVISOR,
            AUTO_DEMOTE_MIN_USES_CAP,
        )

        # Threshold progress
        match_progress = min(1.0, signature_count / MIN_MATCH_RAMP_SIGNATURES)

        # Milestones
        milestones = []
        if signature_count >= 1:
            milestones.append("First signature created")
        if signature_count >= 10:
            milestones.append("10 signatures (initial cluster)")
        if signature_count >= MIN_MATCH_RAMP_SIGNATURES:
            milestones.append(f"{MIN_MATCH_RAMP_SIGNATURES} signatures (mature threshold)")
        if signature_count >= 100:
            milestones.append("100 signatures (solid foundation)")
        if signature_count >= 250:
            milestones.append("250 signatures (rich vocabulary)")
        if signature_count >= 500:
            milestones.append("500 signatures (comprehensive)")
        if total_uses >= 100:
            milestones.append("100 total uses (proven paths)")
        if total_uses >= 1000:
            milestones.append("1000 total uses (well-traveled)")
        if max_depth >= DEPTH_FORCE_DECOMPOSE_DEPTH:
            milestones.append(f"Depth {DEPTH_FORCE_DECOMPOSE_DEPTH}+ (deep decomposition)")
        if umbrella_count >= 10:
            milestones.append("10 umbrellas (routing hierarchy)")

        # Next milestone
        milestone_targets = [
            (1, "First signature"),
            (10, "10 signatures"),
            (MIN_MATCH_RAMP_SIGNATURES, "Mature thresholds"),
            (100, "100 signatures"),
            (250, "250 signatures"),
            (500, "500 signatures"),
            (1000, "1000 signatures"),
        ]

        next_milestone = "System fully mature"
        next_milestone_at = signature_count

        for target, name in milestone_targets:
            if signature_count < target:
                next_milestone = name
                next_milestone_at = target
                break

        return ColdStartProgress(
            signature_count=signature_count,
            total_uses=total_uses,
            total_problems=total_problems,
            phase=phase,
            phase_description=phase_desc,
            current_match_threshold=current_match,
            target_match_threshold=MIN_MATCH_THRESHOLD,
            match_threshold_progress=match_progress,
            current_dsl_inference_threshold=current_dsl_inference,
            target_dsl_inference_threshold=DSL_OPERATION_INFERENCE_MATURE,
            current_demote_min_uses=current_demote,
            target_demote_min_uses=AUTO_DEMOTE_MIN_USES_CAP,
            big_bang_enabled=BIG_BANG_EXPANSION_ENABLED,
            force_decompose_depth=DEPTH_FORCE_DECOMPOSE_DEPTH,
            decompose_decay_base=DEPTH_DECOMPOSE_DECAY_BASE,
            max_depth=max_depth,
            umbrella_count=umbrella_count,
            leaf_count=leaf_count,
            avg_children_per_umbrella=avg_children,
            milestones_reached=milestones,
            next_milestone=next_milestone,
            next_milestone_at=next_milestone_at,
        )

    finally:
        conn.close()


def format_cold_start_progress(progress: ColdStartProgress) -> str:
    """Format cold-start progress as text.

    Args:
        progress: ColdStartProgress object

    Returns:
        Formatted string
    """
    lines = []

    # Phase indicator
    phase_symbols = {
        "cold": "[____]",
        "warming": "[==__]",
        "warm": "[===_]",
        "mature": "[====]",
    }
    phase_sym = phase_symbols.get(progress.phase, "[????]")

    lines.append("=== COLD-START PROGRESS ===")
    lines.append("")
    lines.append(f"Phase: {phase_sym} {progress.phase.upper()}")
    lines.append(f"  {progress.phase_description}")
    lines.append("")

    # Progress bars
    lines.append("System Growth:")
    sig_bar = _progress_bar(progress.signature_count, 500)
    lines.append(f"  Signatures: {sig_bar} {progress.signature_count}/500")

    uses_bar = _progress_bar(progress.total_uses, 1000)
    lines.append(f"  Total Uses: {uses_bar} {progress.total_uses}/1000")

    problems_bar = _progress_bar(progress.total_problems, 500)
    lines.append(f"  Problems:   {problems_bar} {progress.total_problems}/500")

    lines.append("")

    # Threshold status
    lines.append("Adaptive Thresholds:")
    threshold_bar = _progress_bar(
        int(progress.match_threshold_progress * 100), 100
    )
    lines.append(
        f"  Match Threshold:  {threshold_bar} "
        f"{progress.current_match_threshold:.2f} -> {progress.target_match_threshold:.2f}"
    )
    lines.append(
        f"  DSL Inference:    {progress.current_dsl_inference_threshold:.2f} -> "
        f"{progress.target_dsl_inference_threshold:.2f}"
    )
    lines.append(
        f"  Demote Min Uses:  {progress.current_demote_min_uses} -> "
        f"{progress.target_demote_min_uses}"
    )
    lines.append("")

    # Tree structure
    lines.append("Tree Structure:")
    lines.append(f"  Max Depth:       {progress.max_depth}")
    lines.append(f"  Umbrellas:       {progress.umbrella_count}")
    lines.append(f"  Leaves:          {progress.leaf_count}")
    lines.append(f"  Avg Children:    {progress.avg_children_per_umbrella:.1f}")
    lines.append("")

    # Big Bang status
    bb_status = "ENABLED" if progress.big_bang_enabled else "DISABLED"
    lines.append(f"Big Bang Expansion: {bb_status}")
    if progress.big_bang_enabled:
        lines.append(f"  Force decompose at depth <= {progress.force_decompose_depth}")
        lines.append(f"  Decay base: {progress.decompose_decay_base}")
    lines.append("")

    # Milestones
    lines.append("Milestones Reached:")
    if progress.milestones_reached:
        for m in progress.milestones_reached:
            lines.append(f"  [x] {m}")
    else:
        lines.append("  (none yet)")
    lines.append("")

    # Next milestone
    if progress.signature_count < 1000:
        remaining = progress.next_milestone_at - progress.signature_count
        lines.append(f"Next Milestone: {progress.next_milestone}")
        lines.append(f"  Need {remaining} more signatures")

    return "\n".join(lines)


def _progress_bar(current: int, target: int, width: int = 20) -> str:
    """Create a simple progress bar."""
    if target <= 0:
        return "[" + " " * width + "]"

    progress = min(1.0, current / target)
    filled = int(progress * width)
    empty = width - filled

    return "[" + "#" * filled + "-" * empty + "]"


def get_growth_velocity(db_path: str = None, hours: int = 24) -> dict:
    """Get growth velocity metrics over recent time period.

    Args:
        db_path: Path to database (default from config)
        hours: Time window to measure

    Returns:
        Dict with velocity metrics
    """
    if db_path is None:
        db_path = DB_PATH

    from datetime import datetime, timedelta, timezone

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    try:
        # New signatures in period
        row = conn.execute(
            "SELECT COUNT(*) FROM step_signatures WHERE created_at >= ?",
            (cutoff,)
        ).fetchone()
        new_sigs = row[0] if row else 0

        # New uses in period (from usage log)
        row = conn.execute(
            "SELECT COUNT(*) FROM step_usage_log WHERE created_at >= ?",
            (cutoff,)
        ).fetchone()
        new_uses = row[0] if row else 0

        # New examples in period
        row = conn.execute(
            "SELECT COUNT(*) FROM step_examples WHERE created_at >= ?",
            (cutoff,)
        ).fetchone()
        new_examples = row[0] if row else 0

        return {
            "period_hours": hours,
            "new_signatures": new_sigs,
            "new_uses": new_uses,
            "new_examples": new_examples,
            "sigs_per_hour": new_sigs / hours if hours > 0 else 0.0,
            "uses_per_hour": new_uses / hours if hours > 0 else 0.0,
        }

    finally:
        conn.close()


def format_growth_velocity(velocity: dict) -> str:
    """Format growth velocity as text."""
    hours = velocity.get("period_hours", 24)
    lines = []
    lines.append(f"=== GROWTH VELOCITY (last {hours}h) ===")
    lines.append("")
    lines.append(f"New Signatures: {velocity.get('new_signatures', 0)}")
    lines.append(f"New Uses:       {velocity.get('new_uses', 0)}")
    lines.append(f"New Examples:   {velocity.get('new_examples', 0)}")
    lines.append("")
    lines.append(f"Signatures/hour: {velocity.get('sigs_per_hour', 0):.2f}")
    lines.append(f"Uses/hour:       {velocity.get('uses_per_hour', 0):.2f}")
    return "\n".join(lines)
