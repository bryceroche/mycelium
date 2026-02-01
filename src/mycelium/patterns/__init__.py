"""Pattern-based math problem solving with embedding matching."""
from .registry import PATTERNS, Pattern, get_pattern
from .matcher import match_pattern
from .executor import execute_pattern
from .welford import (
    record_similarity,
    get_adaptive_threshold,
    get_pattern_stats,
    get_global_stats,
    WelfordState,
)
from .coverage import propose_example, check_coverage

__all__ = [
    "PATTERNS",
    "Pattern",
    "get_pattern",
    "match_pattern",
    "execute_pattern",
    # Welford stats
    "record_similarity",
    "get_adaptive_threshold",
    "get_pattern_stats",
    "get_global_stats",
    "WelfordState",
    # Coverage
    "propose_example",
    "check_coverage",
]
