"""
Consolidated Welford Statistics Module for Mycelium.

This module provides the single source of truth for Welford's online algorithm
implementation, following the consolidation pattern described in CLAUDE.md.

Welford's algorithm enables incremental computation of mean and variance
without storing all data points - critical for the variance-based decomposition
signals used in MCTS post-mortem analysis.

Key applications:
- Embedding variance tracking (similarity signals)
- Outcome variance tracking (success/failure signals)
- The two-signal interpretation: outcome variance vs embedding variance
- Closed loop: "ONE node high variance -> decompose node,
                MULTIPLE nodes high variance -> refine type"
"""

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class WelfordStats:
    """
    Running statistics using Welford's online algorithm.

    Tracks mean and variance incrementally without storing all values.
    This is critical for the variance-based decomposition signals
    described in CLAUDE.md.

    Attributes:
        count: Number of observations
        mean: Running mean of values
        m2: Sum of squared differences from mean (for variance calculation)

    Example:
        >>> stats = WelfordStats()
        >>> for value in [10, 20, 30]:
        ...     stats.update(value)
        >>> print(f"Mean: {stats.mean}, Variance: {stats.variance}")
        Mean: 20.0, Variance: 100.0
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean

    def update(self, value: float) -> None:
        """
        Update statistics with new value using Welford's algorithm.

        This is the core incremental update that maintains numerical stability
        even with large datasets or values far from the mean.

        Args:
            value: New observation to incorporate
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """
        Return sample variance (or 0 if insufficient data).

        Uses Bessel's correction (n-1) for unbiased sample variance.
        Returns 0.0 if count < 2 to avoid division by zero.
        """
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Return sample standard deviation."""
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for storage/JSON.

        Returns:
            Dictionary with count, mean, and m2 fields
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WelfordStats":
        """
        Deserialize from dictionary.

        Args:
            d: Dictionary with count, mean, and m2 fields

        Returns:
            WelfordStats instance with restored state
        """
        stats = cls()
        stats.count = int(d.get("count", 0))
        stats.mean = float(d.get("mean", 0.0))
        stats.m2 = float(d.get("m2", 0.0))
        return stats

    @classmethod
    def from_db_row(cls, count: int, mean: float, m2: float) -> "WelfordStats":
        """
        Create from database row values.

        Convenience method for constructing from database query results.

        Args:
            count: Number of observations
            mean: Running mean
            m2: Sum of squared differences

        Returns:
            WelfordStats instance with given values
        """
        stats = cls()
        stats.count = count
        stats.mean = mean
        stats.m2 = m2
        return stats

    def merge(self, other: "WelfordStats") -> "WelfordStats":
        """
        Merge two WelfordStats instances using parallel algorithm.

        Useful for combining statistics from different sources or
        parallel computations.

        Args:
            other: Another WelfordStats instance to merge

        Returns:
            New WelfordStats with combined statistics
        """
        if other.count == 0:
            return WelfordStats(self.count, self.mean, self.m2)
        if self.count == 0:
            return WelfordStats(other.count, other.mean, other.m2)

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        combined_mean = self.mean + delta * other.count / combined_count
        combined_m2 = (
            self.m2 + other.m2 +
            delta * delta * self.count * other.count / combined_count
        )

        return WelfordStats(combined_count, combined_mean, combined_m2)

    def __repr__(self) -> str:
        return f"WelfordStats(count={self.count}, mean={self.mean:.4f}, std={self.std:.4f})"
