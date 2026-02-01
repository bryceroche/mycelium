"""
Template-based math problem solver.

Flow: match_template -> map_slots -> execute_graph

This replaces the old decomposition approach with a simpler,
more transformer-friendly approach:
1. Find the nearest template via embedding similarity
2. Ask LLM to map problem text to template slots (pure attention)
3. Execute the pre-built computation graph

Supports chained execution for multi-step problems:
- chunk_problem breaks complex problems into sub-problems
- Each chunk can reference [RESULT] from the previous step
- Results chain through until final answer
"""
import logging
from typing import Any, List, Optional, Tuple

from .templates.models import Template, Example, ExampleProposal
from .templates.db import (
    get_template, save_proposal, find_nearest_example,
    get_all_templates, save_example
)
from .templates.matcher import match_template, map_slots
from .templates.graphs import execute_graph
from .embedding_cache import cached_embed
from .templates.chunker import chunk_problem

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Solve math problems using template matching.

    The key insight: transformers are great at alignment (mapping tokens to slots).
    We leverage this by:
    1. Curating templates with pre-built computation graphs
    2. Matching problems to templates via embedding similarity
    3. Asking the LLM only to map slots (its strength)
    4. Executing graphs deterministically (no LLM variance)
    """

    def __init__(self, propose_threshold: float = 0.90):
        """
        Args:
            propose_threshold: Similarity below this triggers a proposal
        """
        self.propose_threshold = propose_threshold

    def solve(self, problem: str, expected_answer: Any = None) -> Any:
        """
        Solve a problem using template matching.

        Supports chained execution for multi-step problems:
        - Chunks the problem into sub-problems
        - Each chunk can reference [RESULT] from previous step
        - Results chain through until final answer

        Args:
            problem: The problem text
            expected_answer: If provided, will propose novel examples

        Returns:
            The computed answer
        """
        # Chunk the problem into sub-problems
        chunks = chunk_problem(problem)
        n = len(chunks)

        if n == 1:
            # Single chunk: use standard flow
            return self._solve_single(problem, expected_answer)

        # Multiple chunks: chain execution
        logger.info(f"[engine] Problem split into {n} chunks")
        result = None

        for i, chunk in enumerate(chunks):
            logger.info(f"[engine] Chunk {i+1}/{n}: {chunk[:50]}...")
            result = self._solve_chunk(chunk, result)

            if result is None:
                logger.warning(f"[engine] Chunk {i+1}/{n} failed")
                return None

        # Propose as example if novel and correct (use original problem)
        if expected_answer is not None and result == expected_answer:
            template, similarity = match_template(problem)
            if template and similarity < self.propose_threshold:
                slots = map_slots(problem, template)
                if slots:
                    self._propose_example(problem, template, similarity, slots, result, expected_answer)

        return result

    def _solve_chunk(self, chunk: str, previous_result: Any = None) -> Any:
        """
        Solve a single chunk, substituting [RESULT] with previous result.

        Args:
            chunk: The chunk text (may contain [RESULT] placeholder)
            previous_result: Result from previous chunk (if any)

        Returns:
            The computed answer for this chunk
        """
        # Substitute [RESULT] with previous result if present
        if previous_result is not None:
            chunk = chunk.replace("[RESULT]", str(previous_result))

        # Match template
        template, similarity = match_template(chunk)

        if template is None:
            logger.warning("[engine] No template matched for chunk")
            return None

        logger.info(f"[engine] Matched template '{template.name}' (sim={similarity:.3f})")

        # Map slots
        slots = map_slots(chunk, template)

        if not slots:
            logger.warning("[engine] Failed to map slots for chunk")
            return None

        logger.info(f"[engine] Slots: {slots}")

        # Execute graph
        try:
            answer = execute_graph(template.graph.to_dict(), slots)
            logger.info(f"[engine] Chunk answer: {answer}")
            return answer
        except Exception as e:
            logger.error(f"[engine] Graph execution failed: {e}")
            return None

    def _solve_single(self, problem: str, expected_answer: Any = None) -> Any:
        """
        Solve a single problem (no chunking).

        This is the original solve logic for single-chunk problems.

        Args:
            problem: The problem text
            expected_answer: If provided, will propose novel examples

        Returns:
            The computed answer
        """
        # Step 1: Match template
        template, similarity = match_template(problem)

        if template is None:
            logger.warning("[engine] No template matched")
            return None

        logger.info(f"[engine] Matched template '{template.name}' (sim={similarity:.3f})")

        # Step 2: Map slots
        slots = map_slots(problem, template)

        if not slots:
            logger.warning("[engine] Failed to map slots")
            return None

        logger.info(f"[engine] Slots: {slots}")

        # Step 3: Execute graph
        try:
            answer = execute_graph(template.graph.to_dict(), slots)
            logger.info(f"[engine] Answer: {answer}")
        except Exception as e:
            logger.error(f"[engine] Graph execution failed: {e}")
            return None

        # Step 4: Propose as example if novel and correct
        if expected_answer is not None and answer == expected_answer:
            if similarity < self.propose_threshold:
                self._propose_example(problem, template, similarity, slots, answer, expected_answer)

        return answer

    def _propose_example(
        self,
        problem: str,
        template: Template,
        similarity: float,
        slots: dict,
        computed: Any,
        expected: Any
    ):
        """Propose a novel example for human review."""
        embedding = cached_embed(problem)

        proposal = ExampleProposal(
            problem_text=problem,
            embedding=embedding,
            template_id=template.id,
            similarity_to_nearest=similarity,
            slots_mapped=slots,
            computed_answer=computed,
            expected_answer=expected,
            status="pending"
        )

        proposal_id = save_proposal(proposal)
        logger.info(f"[engine] Proposed example #{proposal_id} (sim={similarity:.3f})")

    def solve_batch(self, problems: list, expected_answers: list = None) -> list:
        """Solve multiple problems."""
        results = []
        expected = expected_answers or [None] * len(problems)

        for problem, exp in zip(problems, expected):
            result = self.solve(problem, exp)
            results.append(result)

        return results


# Convenience function
def solve(problem: str) -> Any:
    """Solve a single problem."""
    engine = TemplateEngine()
    return engine.solve(problem)
