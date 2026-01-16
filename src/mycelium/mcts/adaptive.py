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
        self.recent_results.append(success)
        self._total_problems += 1
        if success:
            self._total_successes += 1

        # Log periodically
        if self._total_problems % 100 == 0:
            logger.info(
                "[adaptive] Progress: %d problems, %.1f%% rolling accuracy, C=%.2f, split_threshold=%.2f",
                self._total_problems,
                self.global_accuracy * 100,
                self.exploration_weight,
                self.split_threshold,
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
        """Get adaptive exploration weight (C) for UCB1."""
        return get_adaptive_exploration_weight(self.global_accuracy)

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
        return {
            "total_problems": self._total_problems,
            "total_successes": self._total_successes,
            "window_size": self.window_size,
            "window_count": len(self.recent_results),
            "global_accuracy": self.global_accuracy,
            "lifetime_accuracy": self.lifetime_accuracy,
            "exploration_weight": self.exploration_weight,
            "split_threshold": self.split_threshold,
        }
