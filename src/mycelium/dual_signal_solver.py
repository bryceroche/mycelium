"""Dual-Signal Solver Integration for Mycelium.

This module integrates the dual-signal template system with the solver flow.
It provides a solver that uses BOTH embedding similarity AND attention patterns
for operation classification, addressing the lexical vs operational similarity
problem described in CLAUDE.md.

Key principles (from CLAUDE.md):
- Always route to best match, let failures drive learning
- Record outcomes for Welford stats
- Failures are valuable data points
- Dual-signal approach routes by what operations DO, not what they SOUND LIKE

The solver:
1. Loads operation templates from JSON (pre-computed from training data)
2. Uses attention patterns + embeddings for span classification
3. Records outcomes to update Welford statistics
4. Provides decomposition signals when variance is high
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from mycelium.dual_signal_templates import (
    TemplateStore,
    DualSignalTemplate,
)
from mycelium.dual_signal_pipeline import (
    DualSignalPipeline,
    MatchedOperation,
    PipelineOutput,
)
from mycelium.types import Operation as SolverOperation


# Paths — template loading handled by DualSignalPipeline
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "minilm_contrastive.pt"


@dataclass
class SolverResult:
    """Result from solving a problem."""
    answer: float
    operations: List[SolverOperation]
    state: Dict[str, float]
    spans_detected: int
    success: Optional[bool] = None  # Set after validation


class DualSignalSolver:
    """Solver using dual-signal (embedding + attention) routing.

    This solver integrates the dual-signal template system with problem solving:
    1. Segments problem into spans using attention-based detection
    2. Matches each span to operation templates using BOTH signals
    3. Executes operations in sequence (linear chain)
    4. Records outcomes for continuous learning

    Key insight from CLAUDE.md: The combination of (dag_step_id, dag_step_type / node_id)
    is what we're learning. A node might be great for step 2 but terrible for step 5.
    """

    # Entity detection handled by attention_graph.py (attention_received signal)

    def __init__(
        self,
        templates_path: Optional[str] = None,
        model_path: Optional[str] = None,
        embedding_weight: float = 0.9,
        attention_weight: float = 0.1,
        use_db: bool = False,
        mock_model: bool = False,
    ):
        """Initialize the dual-signal solver.

        Args:
            templates_path: Path to templates JSON (pipeline auto-discovers if None)
            model_path: Path to fine-tuned MiniLM model
            embedding_weight: Weight for embedding similarity [0-1]
            attention_weight: Weight for attention similarity [0-1]
            use_db: Whether to persist to database
            mock_model: If True, use mock embeddings for testing without GPU
        """
        self.templates_path = templates_path
        self.model_path = model_path or str(MODEL_PATH)
        self.embedding_weight = embedding_weight
        self.attention_weight = attention_weight
        self.use_db = use_db
        self.mock_model = mock_model

        # Initialize components
        self._pipeline: Optional[DualSignalPipeline] = None
        self._templates_loaded = False

        # Track outcomes for reporting
        self._outcomes: List[Tuple[str, bool]] = []  # (template_id, success)

    def _ensure_pipeline(self) -> DualSignalPipeline:
        """Lazy-load the pipeline."""
        if self._pipeline is None:
            if self.mock_model:
                # Mock mode: no actual model loading
                self._pipeline = _create_mock_pipeline(
                    self.embedding_weight,
                    self.attention_weight
                )
            else:
                # Check if model exists
                model_to_use = self.model_path
                if not os.path.exists(model_to_use):
                    print(f"Note: Fine-tuned model not found at {model_to_use}")
                    print("Using base MiniLM weights")
                    model_to_use = None

                self._pipeline = DualSignalPipeline(
                    model_path=model_to_use,
                    embedding_weight=self.embedding_weight,
                    attention_weight=self.attention_weight,
                )

            # Pipeline loads templates in __init__ (qwen_templates.json or deduplicated_templates.json)
            # If explicit path was given, REPLACE with that instead
            if not self.mock_model and self.templates_path:
                self._pipeline.load_templates(self.templates_path, replace=True)

            self._templates_loaded = True

        return self._pipeline

    def solve(self, problem: str) -> SolverResult:
        """Solve a math word problem using dual-signal routing.

        Uses attention-based graph execution for span composition:
        1. Build span graph using attention signals (not hardcoded lists)
        2. Match spans to templates
        3. Execute graph to compute answer

        Args:
            problem: The problem text

        Returns:
            SolverResult with answer, operations, and state
        """
        pipeline = self._ensure_pipeline()

        # Process problem through dual-signal pipeline
        output = pipeline.process_problem(problem)

        # Graph execution is the ONLY execution path
        # Entity detection comes from attention_received signals in attention_graph.py
        # No hardcoded entity extraction or pronoun lists
        if output.execution_result and output.execution_result.answer is not None:
            # Build operations list for reporting
            operations = []
            for matched_op in output.matched_operations:
                # Entity comes from attention-detected entities in the graph
                # (populated by attention_graph.py using attention_received signal)
                entity = "X"
                if output.execution_result.entity_values:
                    entity = next(iter(output.execution_result.entity_values), "X")

                op = SolverOperation(
                    subgraph=matched_op.subgraph,
                    value=self._extract_number(matched_op.span_text),
                    entity=entity,
                    confidence=matched_op.confidence,
                    embedding_sim=matched_op.embedding_similarity,
                    attention_sim=matched_op.attention_similarity,
                    span_text=matched_op.span_text,
                    template_id=matched_op.template_id,
                )
                operations.append(op)

            return SolverResult(
                answer=output.execution_result.answer,
                operations=operations,
                state=output.execution_result.entity_values,
                spans_detected=output.spans_detected,
            )

        # No graph execution result - return zero (no fallback to hardcoded heuristics)
        return SolverResult(
            answer=0.0,
            operations=[],
            state={},
            spans_detected=output.spans_detected,
        )

    def _extract_number(self, text: str) -> float:
        """Extract the first number from text.

        Handles:
        - Plain numbers: 42, 3.14
        - Comma-separated: 80,000 -> 80000
        - Dollar amounts: $80,000 -> 80000
        - Percentages: 15% -> 15
        - Word numbers: one, two, three...
        """
        import re

        # First try to find numbers with commas (e.g., 80,000 or $80,000)
        # Pattern: optional $, digits with optional commas, optional decimal
        comma_matches = re.findall(r'\$?([\d,]+\.?\d*)', text)
        for match in comma_matches:
            # Remove commas and convert
            clean = match.replace(',', '')
            if clean and clean != '.':
                try:
                    return float(clean)
                except ValueError:
                    continue

        # Fall back to simple number extraction
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            return float(matches[0])

        # Check for word numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
            'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000,
        }
        text_lower = text.lower()
        for word, value in word_numbers.items():
            if word in text_lower:
                return float(value)

        return 0.0

    # ================================================================
    # REMOVED: Hardcoded NOISE_WORDS and _extract_entity with 3-word window
    # Entity detection is now done by attention_graph.py using attention_received
    # signals. No hardcoded word lists needed.
    # ================================================================

    def record_outcome(self, result: SolverResult, correct: bool) -> None:
        """Record outcome for learning.

        Per CLAUDE.md: Failures are valuable data points.
        Record every failure - it feeds the post-mortem analysis.
        """
        pipeline = self._ensure_pipeline()
        result.success = correct

        for op in result.operations:
            # Record outcome for template
            pipeline.record_outcome(
                op.template_id,
                success=correct,
                embedding_sim=op.embedding_sim,
                attention_sim=op.attention_sim,
            )

            # Track locally
            self._outcomes.append((op.template_id, correct))

            # Also record to database if enabled
            if self.use_db:
                try:
                    from mycelium.db import update_welford_stats
                    stat_type = f"dual_signal_{op.template_id}_outcome"
                    update_welford_stats(stat_type, 1.0 if correct else 0.0)
                except Exception:
                    pass

    def get_decomposition_candidates(self) -> List[str]:
        """Get templates that should be considered for decomposition.

        Per CLAUDE.md: High variance signals need for decomposition.
        "ONE node high variance -> decompose node"
        """
        pipeline = self._ensure_pipeline()
        candidates = pipeline.get_decomposition_candidates()
        return [t.template_id for t in candidates]

    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        pipeline = self._ensure_pipeline()

        # Count outcomes
        total_outcomes = len(self._outcomes)
        successes = sum(1 for _, success in self._outcomes if success)

        return {
            "templates_loaded": len(pipeline.store.templates),
            "embedding_weight": pipeline.store.embedding_weight,
            "attention_weight": pipeline.store.attention_weight,
            "total_outcomes": total_outcomes,
            "success_rate": successes / total_outcomes if total_outcomes > 0 else 0.0,
            "high_variance_templates": len(pipeline.get_decomposition_candidates()),
        }

    def print_stats(self) -> None:
        """Print solver statistics."""
        stats = self.get_stats()
        print("\n=== Dual-Signal Solver Stats ===")
        print(f"Templates loaded: {stats['templates_loaded']}")
        print(f"Signal weights: embedding={stats['embedding_weight']:.2f}, "
              f"attention={stats['attention_weight']:.2f}")
        print(f"Outcomes recorded: {stats['total_outcomes']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"High variance templates: {stats['high_variance_templates']}")
        print("=" * 35 + "\n")


def _create_mock_pipeline(
    embedding_weight: float,
    attention_weight: float
) -> DualSignalPipeline:
    """Create a mock pipeline for testing without GPU.

    Mock mode is minimal - no hardcoded verb lists.
    Returns empty results; real testing requires the model.
    """
    class MockPipeline:
        def __init__(self, emb_w, att_w):
            self.store = TemplateStore(emb_w, att_w)

        def process_problem(self, text: str) -> PipelineOutput:
            return PipelineOutput(
                problem_text=text,
                matched_operations=[],
                spans_detected=0,
                templates_available=0,
            )

        def record_outcome(self, template_id, success, embedding_sim=None, attention_sim=None):
            pass

        def get_decomposition_candidates(self):
            return []

        def bootstrap_from_examples(self):
            return 0

    return MockPipeline(embedding_weight, attention_weight)


# ============================================================
# Main / CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Signal Solver")
    parser.add_argument(
        "--templates",
        default=None,
        help="Path to operation templates JSON",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model for testing without GPU",
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Single problem to solve",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dual-Signal Solver")
    print("=" * 60)

    # Initialize solver
    solver = DualSignalSolver(
        templates_path=args.templates,
        mock_model=args.mock,
    )

    # Test problems
    if args.problem:
        problems = [args.problem]
    else:
        problems = [
            "John has 5 apples. He gives 2 apples to Mary. How many apples does John have now?",
            "Lisa has 12 cookies. She ate 3 cookies and gave 4 to her brother.",
            "Tom had 8 dollars. He earned 5 more dollars mowing lawns.",
        ]

    print("\nSolving problems...")
    for i, problem in enumerate(problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"Text: {problem}")

        result = solver.solve(problem)

        print(f"Answer: {result.answer}")
        print(f"State: {result.state}")
        print(f"Spans detected: {result.spans_detected}")
        print("Operations:")
        for op in result.operations:
            # Extract op name from subgraph for display
            op_name = "SET"
            if op.subgraph and op.subgraph.get("steps"):
                op_name = op.subgraph["steps"][-1].get("op", "SET")
            print(f"  - {op.entity} [{op_name}] {op.value}")
            print(f"    Confidence: {op.confidence:.3f}")
            print(f"    Similarity: emb={op.embedding_sim:.3f}, att={op.attention_sim:.3f}")

    # Print stats
    solver.print_stats()
