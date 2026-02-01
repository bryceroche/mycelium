"""Pattern-based math problem solving with embedding matching."""
from .registry import PATTERNS, Pattern, get_pattern
from .matcher import match_pattern
from .executor import execute_pattern

__all__ = [
    "PATTERNS",
    "Pattern",
    "get_pattern",
    "match_pattern",
    "execute_pattern",
]
