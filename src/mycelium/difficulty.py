"""Problem difficulty estimation for adaptive tree behavior.

Estimates problem complexity to enable:
- Difficulty-aware decomposition depth (harder → deeper)
- Difficulty-weighted credit (harder problems worth more signal)
- Difficulty-aware routing (prefer sigs proven at similar difficulty)

Difficulty scale: 0.0 (trivial) to 1.0 (competition math)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# DIFFICULTY SIGNALS
# =============================================================================

# Keywords indicating harder problems (proofs, abstract concepts)
HARD_KEYWORDS = [
    r"\bprove\b", r"\bshow that\b", r"\bdemonstrate\b",
    r"\bfor all\b", r"\bfor every\b", r"\bthere exists\b",
    r"\binduction\b", r"\bcontradiction\b", r"\blemma\b", r"\btheorem\b",
    r"\biff\b", r"\bif and only if\b",
    r"\blim\b", r"\blimit\b", r"\bderivative\b", r"\bintegral\b",
    r"\bconverges?\b", r"\bdiverges?\b",
    r"\bmodulo\b", r"\bmod\b", r"\bgcd\b", r"\blcm\b",
    r"\bprime\b", r"\bfactorization\b",
    r"\bcombinatorics\b", r"\bpermutation\b", r"\bcombination\b",
    r"\bprobability\b", r"\bexpected value\b",
]

# Keywords indicating easier problems (basic arithmetic)
EASY_KEYWORDS = [
    r"\bhow many\b", r"\bhow much\b",
    r"\btotal\b", r"\bsum\b", r"\bdifference\b",
    r"\bcost\b", r"\bprice\b", r"\bmoney\b", r"\bdollars?\b",
    r"\bminutes?\b", r"\bhours?\b", r"\bdays?\b",
    r"\bapples?\b", r"\boranges?\b", r"\bcookies?\b",  # Word problem objects
]

# Math notation patterns (LaTeX, symbols)
MATH_NOTATION = [
    r"\$.*?\$",  # Inline LaTeX
    r"\\frac", r"\\sqrt", r"\\sum", r"\\prod", r"\\int",
    r"\\geq?", r"\\leq?", r"\\neq",
    r"\^{?\d+}?",  # Exponents like x^2 or x^{10}
    r"_{?\d+}?",  # Subscripts
    r"[∑∏∫∂∇]",  # Unicode math symbols
    r"≥|≤|≠|∈|∀|∃",  # Unicode comparison/logic
]

# Dataset difficulty mappings
DATASET_DIFFICULTY = {
    "gsm8k": 0.2,
    "math_l1": 0.4,
    "math_l2": 0.5,
    "math_l3": 0.6,
    "math_l4": 0.8,
    "math_l5": 1.0,
    "aime": 0.95,
    "imo": 1.0,
}


def estimate_difficulty(
    problem: str,
    dataset: Optional[str] = None,
    level: Optional[int] = None,
) -> float:
    """Estimate problem difficulty from text and metadata.

    Args:
        problem: The problem text
        dataset: Optional dataset name (gsm8k, math, aime, etc.)
        level: Optional explicit difficulty level (1-5 for MATH)

    Returns:
        Difficulty score from 0.0 (trivial) to 1.0 (competition math)
    """
    # If explicit level provided, use it directly
    if level is not None:
        return min(1.0, level / 5.0)

    # If dataset provided, use as baseline
    if dataset:
        dataset_lower = dataset.lower()
        # Check for exact match
        if dataset_lower in DATASET_DIFFICULTY:
            return DATASET_DIFFICULTY[dataset_lower]
        # Check for partial match (e.g., "math" in "math_l3")
        for key, value in DATASET_DIFFICULTY.items():
            if key in dataset_lower:
                return value

    # Otherwise, estimate from problem text
    return _estimate_from_text(problem)


def _estimate_from_text(problem: str) -> float:
    """Estimate difficulty purely from problem text."""
    problem_lower = problem.lower()

    # Start with baseline
    difficulty = 0.3

    # Length signal (longer problems tend to be harder)
    word_count = len(problem.split())
    if word_count > 200:
        difficulty += 0.15
    elif word_count > 100:
        difficulty += 0.1
    elif word_count < 30:
        difficulty -= 0.1

    # Hard keyword signals
    hard_count = sum(1 for pattern in HARD_KEYWORDS if re.search(pattern, problem_lower))
    difficulty += min(0.3, hard_count * 0.08)

    # Easy keyword signals
    easy_count = sum(1 for pattern in EASY_KEYWORDS if re.search(pattern, problem_lower))
    difficulty -= min(0.15, easy_count * 0.03)

    # Math notation signals (LaTeX, symbols)
    notation_count = sum(1 for pattern in MATH_NOTATION if re.search(pattern, problem))
    difficulty += min(0.2, notation_count * 0.05)

    # Multi-part problems (a), (b), (c)
    if re.search(r"\([a-c]\)", problem_lower):
        difficulty += 0.1

    # Clamp to valid range
    return max(0.0, min(1.0, difficulty))


def get_difficulty_label(difficulty: float) -> str:
    """Get human-readable label for difficulty score."""
    if difficulty < 0.25:
        return "elementary"
    elif difficulty < 0.45:
        return "grade_school"  # GSM8K level
    elif difficulty < 0.65:
        return "intermediate"  # MATH L1-L3
    elif difficulty < 0.85:
        return "advanced"  # MATH L4
    else:
        return "competition"  # MATH L5, AIME, IMO


def get_recommended_depth(difficulty: float) -> int:
    """Get recommended decomposition depth for difficulty level.

    Harder problems need deeper decomposition to handle complex reasoning.
    """
    if difficulty < 0.3:
        return 3  # Simple problems: shallow
    elif difficulty < 0.5:
        return 4  # GSM8K level
    elif difficulty < 0.7:
        return 5  # MATH L1-L3
    elif difficulty < 0.9:
        return 7  # MATH L4-L5
    else:
        return 10  # Competition math


def get_credit_multiplier(difficulty: float) -> float:
    """Get credit multiplier for difficulty level.

    Harder problems provide more valuable signal when solved.
    Range: 1.0 (trivial) to 5.0 (competition math)
    """
    return 1.0 + (difficulty * 4.0)


def get_exploration_budget(difficulty: float, base_budget: float = 3.0) -> float:
    """Get MCTS exploration budget scaled by difficulty.

    Harder problems get more exploration budget.
    """
    return base_budget * (1.0 + difficulty)
