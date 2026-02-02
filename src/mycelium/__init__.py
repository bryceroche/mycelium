"""Mycelium: Pattern-based math problem solver."""

__version__ = "3.0.0"

from .engine import PatternEngine, solve
from .patterns import PATTERNS, Pattern, match_pattern, execute_pattern

__all__ = [
    "PatternEngine",
    "Pattern",
    "PATTERNS",
    "solve",
    "match_pattern",
    "execute_pattern",
]
