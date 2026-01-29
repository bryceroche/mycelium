"""Unified rejection checking for leaf signatures.

Per CLAUDE.md "New Favorite Pattern": Consolidate leaf_node rejection of dag_steps
to a single entry point. This module provides the canonical rejection check used
by both solver.py and db.py.

Database Statistics → Welford → Tree Structure (The Flow)
Adaptive thresholds derived from Welford statistics guide rejection decisions.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from mycelium.step_signatures.models import StepSignature

logger = logging.getLogger(__name__)


@dataclass
class RejectionResult:
    """Result of a rejection check.

    Per CLAUDE.md "New Favorite Pattern": Single result struct with all
    rejection-related info so callers don't need to coordinate multiple calls.

    Attributes:
        rejected: Whether the step was rejected
        signature: The signature (None if rejected)
        similarity: The similarity score
        threshold: The adaptive threshold used for comparison
        reason: Why rejection happened (None if not rejected)
        rejection_count: Current rejection count for signature (if rejected)
        should_decompose: Whether signature has hit decomposition threshold
    """
    rejected: bool
    signature: Optional["StepSignature"]
    similarity: float
    threshold: float
    reason: Optional[str] = None
    rejection_count: int = 0
    should_decompose: bool = False


def check_rejection(
    signature: "StepSignature",
    similarity: float,
    is_cold_start: bool,
    *,
    step_text: Optional[str] = None,
    problem_context: Optional[str] = None,
    dag_step_id: Optional[str] = None,
    conn=None,
    record: bool = True,
) -> RejectionResult:
    """Unified rejection check per CLAUDE.md New Favorite Pattern.

    Checks if a leaf signature should reject a dag_step based on adaptive
    Welford thresholds. Optionally records the rejection for learning.

    Args:
        signature: The leaf signature to check
        similarity: The similarity score between step and signature
        is_cold_start: Whether system is in cold start (skip rejection)
        step_text: Text of the step being matched (for recording)
        problem_context: Problem text for context (for recording)
        dag_step_id: Optional dag_step identifier (for recording)
        conn: Optional DB connection (reuse to avoid locks)
        record: Whether to record rejection (default True)

    Returns:
        RejectionResult with rejection status and details
    """
    from mycelium.config import (
        ADAPTIVE_REJECTION_K,
        ADAPTIVE_REJECTION_MIN_SAMPLES,
        ADAPTIVE_REJECTION_DEFAULT_THRESHOLD,
    )

    # Cold start: skip rejection while building vocabulary
    if is_cold_start:
        logger.debug(
            "[rejection] Cold start: skipping rejection for sig %d",
            signature.id
        )
        return RejectionResult(
            rejected=False,
            signature=signature,
            similarity=similarity,
            threshold=0.0,
            reason=None,
        )

    # Get adaptive threshold from signature's success similarity history
    threshold = signature.get_adaptive_rejection_threshold(
        k=ADAPTIVE_REJECTION_K,
        min_samples=ADAPTIVE_REJECTION_MIN_SAMPLES,
        default_threshold=ADAPTIVE_REJECTION_DEFAULT_THRESHOLD,
    )

    # Check if similarity is below threshold
    if similarity < threshold:
        rejection_count = 0
        should_decompose = False

        # Record rejection for learning if requested
        # Per CLAUDE.md "New Favorite Pattern": Use consolidated reject_dag_step()
        if record and step_text is not None:
            from mycelium.data_layer.mcts import reject_dag_step
            decision = reject_dag_step(
                signature_id=signature.id,
                similarity=similarity,
                step_text=step_text,
                dag_step_id=dag_step_id,
                problem_context=problem_context,
                reason="below_threshold",
                conn=conn,
            )
            rejection_count = decision.rejection_count
            should_decompose = decision.should_decompose

        logger.debug(
            "[rejection] Sig %d (%s) rejected: sim=%.3f < threshold=%.3f "
            "(n=%d, mean=%.3f, total_rejections=%d, should_decompose=%s)",
            signature.id, signature.step_type, similarity, threshold,
            signature.success_sim_count, signature.success_sim_mean,
            rejection_count, should_decompose,
        )

        return RejectionResult(
            rejected=True,
            signature=None,
            similarity=similarity,
            threshold=threshold,
            reason="below_threshold",
            rejection_count=rejection_count,
            should_decompose=should_decompose,
        )

    # Accepted
    return RejectionResult(
        rejected=False,
        signature=signature,
        similarity=similarity,
        threshold=threshold,
        reason=None,
    )
