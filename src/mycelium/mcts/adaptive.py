"""Adaptive MCTS exploration based on training accuracy.

Key insight: Exploration weight and split thresholds should adapt to learning progress.
- Cold start (low accuracy): Explore aggressively, tolerate failures
- Mature system (high accuracy): Exploit learned paths, split confidently

This implements CLAUDE.md principle:
"cold-start aware thresholds (adaptive branching more aggressive during cold start - big bang)"
"""

import logging
from collections import deque
from typing import Optional

from mycelium.config import (
    ADAPTIVE_EXPLORATION_C_MAX,
    ADAPTIVE_EXPLORATION_C_MIN,
    ADAPTIVE_SPLIT_THRESHOLD_LENIENT,
    ADAPTIVE_SPLIT_THRESHOLD_STRICT,
    ADAPTIVE_ACCURACY_WINDOW_SIZE,
    UCB1_ADJUSTMENT_ENABLED,
    UCB1_ADJUSTMENT_WINDOW,
    UCB1_ADJUSTMENT_MAX_DELTA,
    UCB1_ADJUSTMENT_SENSITIVITY,
)

logger = logging.getLogger(__name__)


def get_adaptive_exploration_weight(global_accuracy: float) -> float:
    """Compute exploration weight (C) based on global accuracy.

    Low accuracy → high exploration (discover what works)
    High accuracy → low exploration (exploit learned paths)

    Args:
        global_accuracy: Rolling accuracy (0.0 to 1.0)

    Returns:
        Exploration constant C for UCB1 formula

    Examples:
        accuracy=0%  → C=2.0 (maximum exploration)
        accuracy=50% → C=1.25
        accuracy=90% → C=0.6 (mostly exploit)
    """
    # Clamp accuracy to valid range
    accuracy = max(0.0, min(1.0, global_accuracy))

    # Linear interpolation from C_MAX (cold) to C_MIN (mature)
    return ADAPTIVE_EXPLORATION_C_MAX - (accuracy * (ADAPTIVE_EXPLORATION_C_MAX - ADAPTIVE_EXPLORATION_C_MIN))


def get_adaptive_split_threshold(global_accuracy: float) -> float:
    """Compute cluster split threshold based on global accuracy.

    Low accuracy → lenient (don't split too fast, still learning)
    High accuracy → strict (splits indicate real operational differences)

    The threshold is the FAILURE rate at which we trigger a split.

    Args:
        global_accuracy: Rolling accuracy (0.0 to 1.0)

    Returns:
        Failure rate threshold for splitting (0.0 to 1.0)

    Examples:
        accuracy=0%  → threshold=0.7 (tolerate 70% failure before split)
        accuracy=50% → threshold=0.55
        accuracy=90% → threshold=0.4 (split at 40% failure)
    """
    # Clamp accuracy to valid range
    accuracy = max(0.0, min(1.0, global_accuracy))

    # Linear interpolation from lenient (cold) to strict (mature)
    return ADAPTIVE_SPLIT_THRESHOLD_LENIENT - (
        accuracy * (ADAPTIVE_SPLIT_THRESHOLD_LENIENT - ADAPTIVE_SPLIT_THRESHOLD_STRICT)
    )


class AdaptiveExploration:
    """Track global accuracy and provide adaptive MCTS parameters.

    This class maintains a rolling window of recent results to compute
    global accuracy, which then drives exploration weight and split thresholds.

    Usage:
        adaptive = AdaptiveExploration.get_instance()
        adaptive.record_result(success=True)

        # In scoring.py:
        exploration_c = adaptive.exploration_weight

        # In umbrella_learner.py:
        split_threshold = adaptive.split_threshold
    """

    _instance: Optional["AdaptiveExploration"] = None

    def __init__(self, window_size: int = None):
        """Initialize with rolling window for accuracy tracking.

        Args:
            window_size: Number of recent results to track (default from config)
        """
        self.window_size = window_size or ADAPTIVE_ACCURACY_WINDOW_SIZE
        self.recent_results: deque[bool] = deque(maxlen=self.window_size)
        self._total_problems = 0
        self._total_successes = 0

        # Hit/miss tracking for UCB1 adjustment (per beads issue mycelium-nirq)
        # Tracks post-mortem patterns: high_conf_wrong, low_conf_right
        self._hitmiss_window = UCB1_ADJUSTMENT_WINDOW
        self._high_conf_wrong: deque[int] = deque(maxlen=self._hitmiss_window)
        self._low_conf_right: deque[int] = deque(maxlen=self._hitmiss_window)
        self._total_high_conf: deque[int] = deque(maxlen=self._hitmiss_window)
        self._total_low_conf: deque[int] = deque(maxlen=self._hitmiss_window)
        self._ucb1_adjustment = 0.0  # Current adjustment to C

    @classmethod
    def get_instance(cls, window_size: int = None) -> "AdaptiveExploration":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(window_size)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def record_result(self, success: bool):
        """Record a problem result.

        Args:
            success: Whether the problem was solved correctly
        """
        # Defensive: ensure success is a bool (sum() needs numeric values)
        self.recent_results.append(bool(success))
        self._total_problems += 1
        if success:
            self._total_successes += 1

        # Log periodically
        if self._total_problems % 100 == 0:
            logger.info(
                "[adaptive] Progress: %d problems, %.1f%% rolling accuracy, C=%.2f (adj=%.2f), split_threshold=%.2f",
                self._total_problems,
                self.global_accuracy * 100,
                self.exploration_weight,
                self._ucb1_adjustment,
                self.split_threshold,
            )

    def record_postmortem_stats(
        self,
        high_conf_wrong: int,
        low_conf_right: int,
        total_high_conf: int,
        total_low_conf: int,
    ):
        """Record post-mortem hit/miss stats for UCB1 adjustment.

        Called after run_postmortem to update hit/miss tracking.

        Args:
            high_conf_wrong: Count of high-confidence wrong decisions
            low_conf_right: Count of low-confidence right decisions
            total_high_conf: Total high-confidence decisions
            total_low_conf: Total low-confidence decisions
        """
        if not UCB1_ADJUSTMENT_ENABLED:
            return

        self._high_conf_wrong.append(high_conf_wrong)
        self._low_conf_right.append(low_conf_right)
        self._total_high_conf.append(total_high_conf)
        self._total_low_conf.append(total_low_conf)

        # Recompute UCB1 adjustment from rolling window
        self._update_ucb1_adjustment()

    def _update_ucb1_adjustment(self):
        """Update UCB1 adjustment based on hit/miss patterns.

        Logic:
        - high_conf_wrong_rate = sum(high_conf_wrong) / sum(total_high_conf)
        - low_conf_right_rate = sum(low_conf_right) / sum(total_low_conf)

        Adjustment signals:
        - High low_conf_right_rate → exploration is finding good paths → increase C
        - High high_conf_wrong_rate → confident picks are wrong → increase C
        - Both low → well-calibrated → no adjustment
        """
        total_hc = sum(self._total_high_conf)
        total_lc = sum(self._total_low_conf)

        if total_hc == 0 and total_lc == 0:
            self._ucb1_adjustment = 0.0
            return

        # Compute rates
        hc_wrong_rate = sum(self._high_conf_wrong) / total_hc if total_hc > 0 else 0.0
        lc_right_rate = sum(self._low_conf_right) / total_lc if total_lc > 0 else 0.0

        # Adjustment logic:
        # - hc_wrong_rate high → we're over-confident → explore more (+)
        # - lc_right_rate high → exploration finds good stuff → explore more (+)
        # Combined signal, scaled by sensitivity
        raw_signal = (hc_wrong_rate + lc_right_rate) / 2.0
        adjustment = raw_signal * UCB1_ADJUSTMENT_SENSITIVITY * UCB1_ADJUSTMENT_MAX_DELTA * 2

        # Clamp to max delta
        self._ucb1_adjustment = max(-UCB1_ADJUSTMENT_MAX_DELTA, min(UCB1_ADJUSTMENT_MAX_DELTA, adjustment))

        logger.debug(
            "[adaptive] UCB1 adjustment: hc_wrong=%.2f, lc_right=%.2f, adj=%.3f",
            hc_wrong_rate, lc_right_rate, self._ucb1_adjustment
        )

    @property
    def global_accuracy(self) -> float:
        """Get rolling window accuracy (0.0 to 1.0)."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def lifetime_accuracy(self) -> float:
        """Get lifetime accuracy (0.0 to 1.0)."""
        if self._total_problems == 0:
            return 0.0
        return self._total_successes / self._total_problems

    @property
    def exploration_weight(self) -> float:
        """Get adaptive exploration weight (C) for UCB1.

        Combines accuracy-based weight with post-mortem hit/miss adjustment.
        """
        base_c = get_adaptive_exploration_weight(self.global_accuracy)
        adjusted_c = base_c + self._ucb1_adjustment

        # Clamp to valid range
        return max(ADAPTIVE_EXPLORATION_C_MIN, min(ADAPTIVE_EXPLORATION_C_MAX, adjusted_c))

    @property
    def split_threshold(self) -> float:
        """Get adaptive split threshold (failure rate)."""
        return get_adaptive_split_threshold(self.global_accuracy)

    @property
    def total_problems(self) -> int:
        """Get total problems processed."""
        return self._total_problems

    def get_stats(self) -> dict:
        """Get current adaptive exploration stats."""
        total_hc = sum(self._total_high_conf)
        total_lc = sum(self._total_low_conf)

        return {
            "total_problems": self._total_problems,
            "total_successes": self._total_successes,
            "window_size": self.window_size,
            "window_count": len(self.recent_results),
            "global_accuracy": self.global_accuracy,
            "lifetime_accuracy": self.lifetime_accuracy,
            "exploration_weight": self.exploration_weight,
            "split_threshold": self.split_threshold,
            # Hit/miss stats for UCB1 adjustment
            "ucb1_adjustment": self._ucb1_adjustment,
            "hitmiss_window": len(self._high_conf_wrong),
            "high_conf_wrong_rate": sum(self._high_conf_wrong) / total_hc if total_hc > 0 else 0.0,
            "low_conf_right_rate": sum(self._low_conf_right) / total_lc if total_lc > 0 else 0.0,
        }
