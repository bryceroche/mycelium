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
    SpanDetector,
    TemplateStore,
    DualSignalTemplate,
    OperationType,
    WelfordStats,
)
from mycelium.dual_signal_pipeline import (
    DualSignalPipeline,
    MatchedOperation,
    PipelineOutput,
)
# Import Operation from types.py (canonical definition)
# Alias as SolverOperation for backward compatibility
from mycelium.types import Operation as SolverOperation
# Import centralized DSL framework for operation execution
from mycelium.span_templates import get_dsl, infer_dsl_expr, execute_dsl_expr
# Import database functions for loading templates
from mycelium.db import get_all_templates, get_template_count


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATES_JSON = PROJECT_ROOT / "operation_templates.json"
TEMPLATES_EXPORT = PROJECT_ROOT / "span_templates_export.json"  # Fine-grained templates
MODEL_PATH = PROJECT_ROOT / "models" / "minilm_attention_finetuned.pt"


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

    # Import from span_normalizer (single source of truth)
    from mycelium.span_normalizer import PRONOUNS

    def __init__(
        self,
        templates_path: Optional[str] = None,
        model_path: Optional[str] = None,
        embedding_weight: float = 0.5,
        attention_weight: float = 0.5,
        use_db: bool = False,
        mock_model: bool = False,
    ):
        """Initialize the dual-signal solver.

        Args:
            templates_path: Path to operation_templates.json
            model_path: Path to fine-tuned MiniLM model
            embedding_weight: Weight for embedding similarity [0-1]
            attention_weight: Weight for attention similarity [0-1]
            use_db: Whether to persist to database
            mock_model: If True, use mock embeddings for testing without GPU
        """
        self.templates_path = templates_path or str(TEMPLATES_JSON)
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

            # Load templates: prefer database > export JSON > coarse JSON > bootstrap
            if not self.mock_model:
                templates_loaded = False

                # 1. Try database first (has fine-grained templates)
                try:
                    db_count = get_template_count()
                    if db_count > 0:
                        self._load_templates_from_db()
                        templates_loaded = True
                except Exception as e:
                    print(f"Database not available ({e})")

                # 2. Try exported fine-grained templates (span_templates_export.json)
                if not templates_loaded and os.path.exists(TEMPLATES_EXPORT):
                    self.templates_path = TEMPLATES_EXPORT
                    self._load_templates_from_json()
                    templates_loaded = True

                # 3. Try coarse operation templates (operation_templates.json)
                if not templates_loaded and os.path.exists(self.templates_path):
                    self._load_templates_from_json()
                    templates_loaded = True

                # 4. Bootstrap with examples as last resort
                if not templates_loaded:
                    print("No templates found, bootstrapping with examples...")
                    self._pipeline.bootstrap_from_examples()

            self._templates_loaded = True

        return self._pipeline

    def _load_templates_from_json(self) -> int:
        """Load operation templates from JSON file.

        Supports two formats:
        1. Coarse format (operation_templates.json): One centroid per operation
        2. Fine-grained format (span_templates_export.json): One template per pattern
        """
        try:
            with open(self.templates_path, 'r') as f:
                data = json.load(f)

            count = 0
            # Detect format: coarse has operation names as keys (SUB, ADD, etc.)
            # Fine-grained has template IDs as keys (add_n_has_v_i_0, etc.)
            operation_names = {"SET", "ADD", "SUB", "MUL", "DIV"}
            is_coarse_format = all(k.upper() in operation_names for k in data.keys())

            for template_id, template_data in data.items():
                # Get pattern (for DSL inference)
                pattern = template_data.get("pattern", "")

                # Get operation type from template data
                if is_coarse_format:
                    op_str = template_id.upper()
                else:
                    op_str = (template_data.get("operation_type") or template_data.get("operation", "SET")).upper()

                # Infer DSL expression from pattern (overrides mislabeled operations)
                dsl_expr = infer_dsl_expr(pattern, op_str) if pattern else None

                # Correct operation type based on inferred DSL
                if dsl_expr:
                    if "entity - " in dsl_expr or "ref - " in dsl_expr:
                        op_str = "SUB"
                    elif "entity + " in dsl_expr or "ref + " in dsl_expr:
                        op_str = "ADD"
                    elif "entity * " in dsl_expr or "ref * " in dsl_expr:
                        op_str = "MUL"
                    elif "entity / " in dsl_expr or "ref / " in dsl_expr:
                        op_str = "DIV"
                    elif dsl_expr == "value":
                        op_str = "SET"

                try:
                    op_type = OperationType(op_str)
                except ValueError:
                    op_type = OperationType.SET

                # Get centroid (required)
                centroid_data = template_data.get("embedding_centroid")
                if centroid_data is None:
                    continue
                centroid = np.array(centroid_data, dtype=np.float32)

                # Get attention signature (optional)
                if "attention_signature" in template_data:
                    attention = np.array(template_data["attention_signature"], dtype=np.float32)
                else:
                    attention = np.zeros(100, dtype=np.float32)

                # Create template with pattern and custom DSL
                tid = f"{template_id}_centroid" if is_coarse_format else template_id
                template = DualSignalTemplate(
                    template_id=tid,
                    operation_type=op_type,
                    embedding_centroid=centroid,
                    attention_signature=attention,
                    pattern=pattern,
                    dsl_expr=dsl_expr or "value",
                    span_examples=(template_data.get("span_examples") or template_data.get("examples", []))[:10],
                    match_count=template_data.get("count", 0),
                )

                # Load Welford stats if present
                if "welford_count" in template_data:
                    template.embedding_welford.count = template_data["welford_count"]
                    template.embedding_welford.mean = template_data.get("welford_mean", 0.0)
                    template.embedding_welford.M2 = template_data.get("welford_m2", 0.0)

                self._pipeline.store.add_template(template)
                count += 1

            print(f"Loaded {count} operation templates from {self.templates_path}")
            return count

        except Exception as e:
            print(f"Error loading templates: {e}")
            return 0

    def _load_templates_from_db(self) -> int:
        """Load fine-grained span templates from PostgreSQL database.

        The database contains specialized templates like:
        - "[NAME] sold [N] [ITEM]" → SUB
        - "[NAME] has [N] more than [REF]" → COMPARE_MORE

        These are more fine-grained than the JSON centroids.
        """
        try:
            templates = get_all_templates()
            if not templates:
                return 0

            count = 0
            for t in templates:
                if t.centroid is None:
                    continue

                # Map operation string to OperationType enum
                op_str = t.operation.upper() if t.operation else "SET"
                try:
                    op_type = OperationType(op_str)
                except ValueError:
                    op_type = OperationType.SET

                template = DualSignalTemplate(
                    template_id=t.template_id,
                    operation_type=op_type,
                    embedding_centroid=t.centroid,
                    attention_signature=np.zeros(100, dtype=np.float32),  # DB doesn't store attention
                    span_examples=t.examples[:10] if t.examples else [],
                    match_count=t.count,
                )

                # Load Welford stats from DB
                if t.welford_count > 0:
                    template.embedding_welford.count = t.welford_count
                    template.embedding_welford.mean = t.welford_mean
                    template.embedding_welford.M2 = t.welford_m2

                self._pipeline.store.add_template(template)
                count += 1

            print(f"Loaded {count} fine-grained templates from database")
            return count

        except Exception as e:
            print(f"Error loading templates from DB: {e}")
            return 0

    def solve(self, problem: str) -> SolverResult:
        """Solve a math word problem using dual-signal routing.

        Args:
            problem: The problem text

        Returns:
            SolverResult with answer, operations, and state
        """
        pipeline = self._ensure_pipeline()

        # Process problem through dual-signal pipeline
        output = pipeline.process_problem(problem)

        # Convert matched operations to solver operations
        operations = []
        state: Dict[str, float] = {}
        main_entity = None

        # Extract numbers and entities from spans
        for matched_op in output.matched_operations:
            # Extract numeric value from span
            value = self._extract_number(matched_op.span_text)

            # Extract entity from span
            entity = self._extract_entity(matched_op.span_text)

            # Handle pronouns
            if entity and entity.lower() in self.PRONOUNS and main_entity:
                entity = main_entity
            elif entity and entity.lower() not in self.PRONOUNS:
                main_entity = entity

            if entity is None:
                entity = main_entity or "X"
                if main_entity is None:
                    main_entity = entity  # Track first entity seen

            op = SolverOperation(
                op_type=matched_op.operation_type.value,
                value=value,
                entity=entity,
                confidence=matched_op.confidence,
                embedding_sim=matched_op.embedding_similarity,
                attention_sim=matched_op.attention_similarity,
                span_text=matched_op.span_text,
                template_id=matched_op.template_id,
            )
            operations.append(op)

            # Execute using custom DSL expression from matched template
            # Each template has its own dsl_expr like "entity - value" or "ref * 2"
            if entity not in state:
                state[entity] = 0

            # Use custom DSL expression (preferred) or fall back to operation-based
            dsl_expr = getattr(matched_op, 'dsl_expr', None)
            if dsl_expr and dsl_expr != "value":
                # Execute custom DSL expression
                state[entity] = execute_dsl_expr(dsl_expr, state, entity, op.value, None)
            else:
                # Fall back to operation-based DSL
                dsl_fn = get_dsl(op.op_type, "simple")
                state[entity] = dsl_fn(state, entity, op.value, None)

        # Get answer (main entity's final value)
        answer = state.get(main_entity, 0) if main_entity else 0

        return SolverResult(
            answer=answer,
            operations=operations,
            state=state,
            spans_detected=output.spans_detected,
        )

    def _extract_number(self, text: str) -> float:
        """Extract the first number from text."""
        import re

        # Find numbers (including decimals)
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            return float(matches[0])

        # Check for word numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        }
        text_lower = text.lower()
        for word, value in word_numbers.items():
            if word in text_lower:
                return float(value)

        return 0.0

    # Words that look like entities but aren't (sentence starters, articles, etc.)
    NOISE_WORDS = frozenset([
        'then', 'the', 'each', 'every', 'some', 'all', 'and', 'but', 'after',
        'before', 'when', 'if', 'so', 'a', 'an', 'now', 'later', 'first',
    ])

    def _extract_entity(self, text: str) -> Optional[str]:
        """Extract the SUBJECT entity from text.

        Key insight: The subject is typically in the first 2-3 words.
        "She gave 8 to Bob" - subject is "She", Bob is the object (recipient).

        Priority:
        1. Look at first 3 words for subject
        2. If subject is pronoun, return None (caller resolves to main_entity)
        3. If subject is proper noun, return it

        This INTENTIONALLY returns None for pronouns so the solver
        can resolve them to the main_entity from previous sentences.
        """
        words = text.split()

        # Only look at first 3 words for subject (subject-verb-object structure)
        subject_window = words[:3]

        for word in subject_window:
            clean_word = word.strip('.,!?\'\"')
            if not clean_word:
                continue

            lower = clean_word.lower()

            # Skip noise words (Then, The, etc.)
            if lower in self.NOISE_WORDS:
                continue

            # If it's a pronoun, return None to signal pronoun resolution
            if lower in self.PRONOUNS:
                return None

            # If it's capitalized (proper noun), return it
            if clean_word[0].isupper():
                return clean_word

        # No subject found in first 3 words - return None
        return None

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
                    stat_type = f"dual_signal_{op.op_type}_outcome"
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

    Uses random embeddings and predefined operation patterns.
    """
    # Create a custom mock pipeline
    class MockPipeline:
        def __init__(self, emb_w, att_w):
            self.store = TemplateStore(emb_w, att_w)
            self._setup_mock_templates()

        def _setup_mock_templates(self):
            """Create mock templates for each operation type."""
            # Create deterministic "embeddings" for each operation
            np.random.seed(42)  # Reproducible

            ops = [
                (OperationType.SET, ["has", "have", "had", "starts with", "begins with", "started with"]),
                (OperationType.ADD, ["found", "received", "got", "earned", "bought", "finds", "gets", "earns", "buys", "more"]),
                (OperationType.SUB, ["gave", "gives", "sold", "sells", "lost", "loses", "spent", "spends", "ate", "eats"]),
                (OperationType.MUL, ["doubled", "tripled", "times", "multiplied", "each"]),
                (OperationType.DIV, ["divided", "split", "shared", "half", "equally"]),
            ]

            for op_type, keywords in ops:
                embedding = np.random.randn(384).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)

                template = DualSignalTemplate(
                    template_id=f"{op_type.value}_mock",
                    operation_type=op_type,
                    embedding_centroid=embedding,
                    attention_signature=np.zeros(100, dtype=np.float32),
                    span_examples=keywords,
                )
                self.store.add_template(template)

        def process_problem(self, text: str) -> PipelineOutput:
            """Process problem using keyword matching (mock)."""
            import re

            # Simple clause splitting
            clauses = re.split(r'[.!?]|\band\b|\bthen\b', text)
            clauses = [c.strip() for c in clauses if c.strip()]

            matched_ops = []
            for clause in clauses:
                # Skip question clauses
                if '?' in clause or 'how many' in clause.lower():
                    continue

                # Find matching operation by keywords
                best_match = None
                best_score = 0.0

                clause_lower = clause.lower()
                for template in self.store.templates.values():
                    score = 0.0
                    for example in template.span_examples:
                        if example.lower() in clause_lower:
                            score += 1.0

                    if score > best_score:
                        best_score = score
                        best_match = template

                # Default to SET if no match
                if best_match is None:
                    best_match = self.store.get_template("SET_mock")
                    best_score = 0.3

                if best_match:
                    matched_ops.append(MatchedOperation(
                        span_text=clause,
                        operation_type=best_match.operation_type,
                        template_id=best_match.template_id,
                        combined_score=best_score / 5.0,
                        embedding_similarity=0.7,
                        attention_similarity=0.6,
                        confidence=best_score / 5.0 + 0.5,
                    ))

            return PipelineOutput(
                problem_text=text,
                matched_operations=matched_ops,
                spans_detected=len(clauses),
                templates_available=len(self.store.templates),
            )

        def record_outcome(self, template_id, success, embedding_sim=None, attention_sim=None):
            """Record outcome for learning."""
            template = self.store.get_template(template_id)
            if template:
                template.record_outcome(success)

        def get_decomposition_candidates(self):
            """Get high variance templates."""
            return self.store.get_high_variance_templates()

        def bootstrap_from_examples(self):
            """Already bootstrapped in init."""
            return len(self.store.templates)

    return MockPipeline(embedding_weight, attention_weight)


# ============================================================
# Main / CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Signal Solver")
    parser.add_argument(
        "--templates",
        default=str(TEMPLATES_JSON),
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
            print(f"  - {op.entity} {op.op_type} {op.value}")
            print(f"    Confidence: {op.confidence:.3f}")
            print(f"    Similarity: emb={op.embedding_sim:.3f}, att={op.attention_sim:.3f}")

    # Print stats
    solver.print_stats()
