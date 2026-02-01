"""
Pattern-based math problem solver.

Flow: match_pattern (embedding) -> specialized prompt -> SymPy/eval -> answer

Uses specialized patterns for each problem type:
1. Match problem to best pattern via embedding similarity
2. Use pattern's specialized prompt to guide LLM
3. LLM outputs structured decomposition
4. Execute with SymPy (symbolic) or eval (arithmetic)
5. Return the answer
"""
import logging
from typing import Any, Optional

from mycelium.patterns import match_pattern, execute_pattern
from mycelium.patterns.coverage import propose_example
from mycelium.patterns.welford import record_similarity, get_adaptive_threshold

logger = logging.getLogger(__name__)

# Default proposal threshold - overridden by Welford adaptive thresholds
PROPOSAL_THRESHOLD = 0.85  # Fallback only


class PatternEngine:
    """
    Solve math problems using pattern matching with specialized prompts.

    The flow:
    1. Match problem to best pattern via embedding similarity
    2. Use pattern's specialized prompt
    3. LLM outputs structured decomposition
    4. Execute with SymPy or eval
    5. If correct + low similarity, propose as new example
    """

    def __init__(self, proposal_threshold: float = PROPOSAL_THRESHOLD):
        self.proposal_threshold = proposal_threshold

    def solve(self, problem: str, expected_answer: Any = None) -> Any:
        """
        Solve a problem using pattern matching.

        Args:
            problem: The problem text
            expected_answer: If provided, used to verify and propose examples

        Returns:
            The computed answer
        """
        # Step 1: Match pattern via embedding similarity
        pattern, similarity = match_pattern(problem)

        if pattern is None:
            logger.warning("[engine] No pattern matched")
            return None

        logger.info(f"[engine] Matched pattern '{pattern.name}' (sim={similarity:.3f})")

        # Step 2: Execute pattern
        result = execute_pattern(problem, pattern)

        logger.info(f"[engine] Result: {result}")

        # Step 3: Record observation and propose example if coverage gap
        is_correct = None
        if expected_answer is not None and result is not None:
            try:
                is_correct = (result == expected_answer or
                            abs(float(result) - float(expected_answer)) < 0.01)
            except (TypeError, ValueError):
                pass  # Can't compare

        # Always record for Welford stats (even without expected_answer)
        problem_hash = str(hash(problem))
        record_similarity(pattern.name, similarity, is_correct, problem_hash)

        # Propose if correct but low similarity (adaptive threshold)
        if is_correct:
            threshold = get_adaptive_threshold(pattern.name)
            if similarity < threshold:
                propose_example(problem, pattern.name, similarity, threshold, was_correct=True)

        return result

    def solve_batch(self, problems: list) -> list:
        """Solve multiple problems."""
        return [self.solve(problem) for problem in problems]


# Convenience function
def solve(problem: str) -> Any:
    """Solve a single problem."""
    engine = PatternEngine()
    return engine.solve(problem)


# Legacy alias for backwards compatibility
TemplateEngine = PatternEngine
