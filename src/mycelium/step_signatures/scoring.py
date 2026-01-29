"""Scoring stubs - minimal implementations for cleanup."""

import re
from typing import Optional

def normalize_step_text(text: str) -> str:
    """Normalize step text by replacing numbers with N."""
    return re.sub(r'\b\d+\.?\d*\b', 'N', text)

def compute_routing_score(similarity: float, **kwargs) -> float:
    """Simplified routing score = similarity."""
    return similarity

def compute_ucb1_score(wins: int, total: int, parent_total: int, c: float = 1.414) -> float:
    """UCB1 exploration score."""
    import math
    if total == 0:
        return float('inf')
    exploitation = wins / total
    exploration = c * math.sqrt(math.log(parent_total + 1) / total)
    return exploitation + exploration

def increment_total_problems() -> int:
    """Stub - return 1."""
    return 1

def invalidate_traffic_cache() -> None:
    """Stub - no-op."""
    pass
