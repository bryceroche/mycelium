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

logger = logging.getLogger(__name__)


class PatternEngine:
    """
    Solve math problems using pattern matching with specialized prompts.

    The flow:
    1. Match problem to best pattern via embedding similarity
    2. Use pattern's specialized prompt
    3. LLM outputs structured decomposition
    4. Execute with SymPy or eval
    5. Return answer
    """

    def solve(self, problem: str) -> Any:
        """
        Solve a problem using pattern matching.

        Args:
            problem: The problem text

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
