"""Dual-Signal Pipeline Integration for Mycelium.

This module integrates the dual-signal template system (attention + embeddings)
with the mycelium pipeline for math word problem solving.

Key features:
1. Uses fine-tuned MiniLM for attention-based span detection
2. Dual-signal matching: embedding similarity + attention pattern correlation
3. Welford statistics for confidence tracking and decomposition signals
4. Bootstrap templates from existing labeled data

The dual-signal approach routes by what operations DO, not what they SOUND LIKE,
addressing the lexical vs operational similarity problem.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from mycelium.dual_signal_templates import (
    SpanDetector,
    TemplateStore,
    DualSignalTemplate,
    OperationType,
    WelfordStats,
    create_template_from_span,
)


# Default model path - relative to this file's location in src/mycelium/
# Goes up to mycelium/ project root, then into models/
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "minilm_attention_finetuned.pt"


@dataclass
class MatchedOperation:
    """Result of matching a span to a template."""
    span_text: str
    operation_type: OperationType
    template_id: str
    combined_score: float
    embedding_similarity: float
    attention_similarity: float
    confidence: float  # Welford-adjusted confidence


@dataclass
class PipelineOutput:
    """Complete output from processing a problem."""
    problem_text: str
    matched_operations: List[MatchedOperation]
    spans_detected: int
    templates_available: int


class DualSignalPipeline:
    """Pipeline that uses dual-signal (attention + embedding) for operation matching.

    This integrates the fine-tuned MiniLM model with the template store
    to provide attention-aware classification of math problem spans.

    The dual-signal approach achieves better operational similarity matching
    by using BOTH embedding content and attention patterns.
    """

    # Operation type mapping from string labels
    OP_TYPE_MAP = {
        "SET": OperationType.SET,
        "ADD": OperationType.ADD,
        "SUB": OperationType.SUB,
        "MUL": OperationType.MUL,
        "DIV": OperationType.DIV,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_weight: float = 0.5,
        attention_weight: float = 0.5,
        device: str = "auto",
        templates_path: Optional[str] = None,
    ):
        """Initialize the dual-signal pipeline.

        Args:
            model_path: Path to fine-tuned MiniLM model checkpoint.
                       Uses default path if not provided.
            embedding_weight: Weight for embedding similarity in matching [0-1]
            attention_weight: Weight for attention similarity in matching [0-1]
            device: Device for inference ("auto", "cuda", "cpu")
            templates_path: Optional path to load/save templates JSON
        """
        # Resolve model path
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            print("Initializing with base MiniLM weights (no fine-tuning)")
            model_path = None

        # Initialize span detector with fine-tuned model
        self.detector = SpanDetector(model_path=model_path, device=device)

        # Initialize template store
        self.store = TemplateStore(
            embedding_weight=embedding_weight,
            attention_weight=attention_weight,
        )

        # Track templates file path for persistence
        self.templates_path = templates_path

        # Load templates if path provided and exists
        if templates_path and os.path.exists(templates_path):
            self.load_templates(templates_path)

    def process_problem(self, text: str) -> PipelineOutput:
        """Process a math problem and match spans to templates.

        This is the main entry point for the pipeline:
        1. Detects operational spans using attention patterns
        2. Matches each span to templates using dual-signal
        3. Returns matched operations with confidence scores

        Args:
            text: The math problem text to process

        Returns:
            PipelineOutput with matched operations for each span
        """
        # Step 1: Detect spans using attention-based community detection
        spans = self.detector.extract_span_features(text, method="community")

        matched_operations = []

        # Step 2: Match each span to templates
        for span in spans:
            match_result = self._match_span_to_template(span)
            if match_result:
                matched_operations.append(match_result)

        return PipelineOutput(
            problem_text=text,
            matched_operations=matched_operations,
            spans_detected=len(spans),
            templates_available=len(self.store.templates),
        )

    def _match_span_to_template(
        self,
        span: Dict[str, Any]
    ) -> Optional[MatchedOperation]:
        """Match a single span to the best template.

        Uses dual-signal scoring: embedding similarity + attention correlation.
        Following CLAUDE.md: Always route to best match, let failures drive learning.

        Args:
            span: Span dictionary from extract_span_features()

        Returns:
            MatchedOperation if templates exist, None otherwise
        """
        if not self.store.templates:
            # No templates yet - return None (caller should bootstrap)
            return None

        # Get span signals
        embedding = span["embedding"]
        attention = span["attention_pattern"]

        # Find best matching template (always route to best per CLAUDE.md)
        result = self.store.find_best_match(embedding, attention)

        if result is None:
            return None

        template, combined_score, emb_sim, att_sim = result

        # Update Welford stats for this template
        template.embedding_welford.update(emb_sim)
        template.attention_welford.update(att_sim)

        # Compute confidence using Welford z-score
        confidence = self._compute_confidence(template, combined_score)

        return MatchedOperation(
            span_text=span["text"],
            operation_type=template.operation_type,
            template_id=template.template_id,
            combined_score=combined_score,
            embedding_similarity=emb_sim,
            attention_similarity=att_sim,
            confidence=confidence,
        )

    def _compute_confidence(
        self,
        template: DualSignalTemplate,
        score: float
    ) -> float:
        """Compute Welford-based confidence for a match.

        Uses the template's running statistics to determine
        how this score compares to typical matches.
        """
        stats = template.embedding_welford

        if stats.count < 2 or stats.std < 1e-8:
            # Not enough data, use raw score
            return min(1.0, max(0.0, score))

        # Z-score: how many stds above/below mean
        z = (score - stats.mean) / stats.std

        # Sigmoid to [0, 1] range
        import math
        confidence = 1.0 / (1.0 + math.exp(-z))

        return confidence

    def record_outcome(
        self,
        template_id: str,
        success: bool,
        embedding_sim: Optional[float] = None,
        attention_sim: Optional[float] = None,
    ) -> None:
        """Record execution outcome for learning.

        This feeds the variance-based decomposition system
        and weight learning in the template store.

        Args:
            template_id: ID of the template that was used
            success: Whether execution was successful
            embedding_sim: Embedding similarity from the match
            attention_sim: Attention similarity from the match
        """
        template = self.store.get_template(template_id)
        if template:
            template.record_outcome(success)

        # Update signal weight learning
        if embedding_sim is not None and attention_sim is not None:
            self.store.update_weights_from_outcome(
                embedding_sim, attention_sim, success
            )

    def get_decomposition_candidates(self) -> List[DualSignalTemplate]:
        """Get templates that should be considered for decomposition.

        Per CLAUDE.md: High variance signals need for decomposition.
        "ONE node high variance -> decompose node"

        Returns:
            List of templates with high outcome variance
        """
        return self.store.get_high_variance_templates()

    # ================================================================
    # Bootstrap Methods - Initialize templates from existing data
    # ================================================================

    def bootstrap_from_labeled_spans(
        self,
        spans: List[Tuple[str, str]],  # (text, operation_label)
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
    ) -> int:
        """Bootstrap templates from labeled span data.

        Creates initial templates from existing labeled spans.
        This is the primary way to initialize the template store.

        Args:
            spans: List of (span_text, operation_label) tuples
            deduplicate: Skip spans too similar to existing templates
            similarity_threshold: Threshold for deduplication

        Returns:
            Number of templates created
        """
        created = 0

        for span_text, op_label in spans:
            # Map label to OperationType
            op_type = self.OP_TYPE_MAP.get(op_label.upper(), OperationType.UNKNOWN)

            # Extract features for this span
            embedding, attention, tokens = self.detector.extract_features(span_text)
            attention_flat = attention.flatten()

            # Check for duplicates
            if deduplicate and self.store.templates:
                best_match = self.store.find_best_match(embedding, attention_flat)
                if best_match and best_match[1] > similarity_threshold:
                    # Too similar to existing template, skip
                    continue

            # Create template
            import uuid
            template_id = f"{op_type.value}_{uuid.uuid4().hex[:8]}"

            template = DualSignalTemplate(
                template_id=template_id,
                operation_type=op_type,
                embedding_centroid=embedding.copy(),
                attention_signature=attention_flat.copy(),
                span_examples=[span_text],
            )

            self.store.add_template(template)
            created += 1

        return created

    def bootstrap_from_database(
        self,
        limit: int = 500,
        deduplicate: bool = True,
    ) -> int:
        """Bootstrap templates from the mycelium database.

        Loads labeled spans from the database and creates templates.

        Args:
            limit: Maximum number of spans to load
            deduplicate: Skip spans too similar to existing

        Returns:
            Number of templates created
        """
        try:
            from mycelium.db import get_labeled_spans

            spans = get_labeled_spans(limit=limit)
            labeled_data = [
                (span.span_text, span.operation)
                for span in spans
                if span.operation
            ]

            return self.bootstrap_from_labeled_spans(
                labeled_data,
                deduplicate=deduplicate
            )
        except ImportError:
            print("Warning: Could not import mycelium.db")
            return 0
        except Exception as e:
            print(f"Warning: Database bootstrap failed: {e}")
            return 0

    def bootstrap_from_examples(self) -> int:
        """Bootstrap with built-in example spans.

        Provides minimal templates for each operation type
        when no database is available.

        Returns:
            Number of templates created
        """
        example_spans = [
            # SET operations
            ("John has 5 apples", "SET"),
            ("Mary has 12 cookies", "SET"),
            ("Tom has 8 dollars", "SET"),

            # ADD operations
            ("He found 3 more coins", "ADD"),
            ("She received 10 dollars", "ADD"),
            ("They bought 6 oranges", "ADD"),

            # SUB operations
            ("She gave 4 to Jane", "SUB"),
            ("He sold 5 apples", "SUB"),
            ("She spent 8 dollars", "SUB"),

            # MUL operations
            ("She tripled her money", "MUL"),
            ("He doubled his score", "MUL"),
            ("Each bag has 5 apples", "MUL"),

            # DIV operations
            ("She split it into 3 parts", "DIV"),
            ("He divided by 2", "DIV"),
            ("They shared equally among 4", "DIV"),
        ]

        return self.bootstrap_from_labeled_spans(
            example_spans,
            deduplicate=False  # Keep all examples
        )

    # ================================================================
    # Persistence Methods
    # ================================================================

    def save_templates(self, path: Optional[str] = None) -> None:
        """Save templates to JSON file.

        Args:
            path: File path. Uses self.templates_path if not provided.
        """
        path = path or self.templates_path
        if not path:
            raise ValueError("No templates path specified")

        data = self.store.to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.store.templates)} templates to {path}")

    def load_templates(self, path: Optional[str] = None) -> None:
        """Load templates from JSON file.

        Args:
            path: File path. Uses self.templates_path if not provided.
        """
        path = path or self.templates_path
        if not path:
            raise ValueError("No templates path specified")

        with open(path, 'r') as f:
            data = json.load(f)

        self.store = TemplateStore.from_dict(data)
        print(f"Loaded {len(self.store.templates)} templates from {path}")

    # ================================================================
    # Diagnostic Methods
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with template counts, weights, etc.
        """
        templates_by_op = {}
        for template in self.store.templates.values():
            op = template.operation_type.value
            templates_by_op[op] = templates_by_op.get(op, 0) + 1

        return {
            "total_templates": len(self.store.templates),
            "templates_by_operation": templates_by_op,
            "embedding_weight": self.store.embedding_weight,
            "attention_weight": self.store.attention_weight,
            "high_variance_templates": len(self.get_decomposition_candidates()),
        }

    def print_stats(self) -> None:
        """Print pipeline statistics to console."""
        stats = self.get_stats()
        print("\n=== Dual-Signal Pipeline Stats ===")
        print(f"Total templates: {stats['total_templates']}")
        print(f"Templates by operation: {stats['templates_by_operation']}")
        print(f"Signal weights: embedding={stats['embedding_weight']:.2f}, "
              f"attention={stats['attention_weight']:.2f}")
        print(f"High variance templates: {stats['high_variance_templates']}")
        print("=" * 40 + "\n")


# ================================================================
# Main Block - Demonstration
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Signal Pipeline Demo")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to fine-tuned MiniLM model",
    )
    parser.add_argument(
        "--bootstrap",
        choices=["examples", "database", "none"],
        default="examples",
        help="How to bootstrap templates",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dual-Signal Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = DualSignalPipeline(
        model_path=args.model_path,
        embedding_weight=0.5,
        attention_weight=0.5,
    )

    # Bootstrap templates
    print(f"\n2. Bootstrapping templates ({args.bootstrap})...")
    if args.bootstrap == "examples":
        n_templates = pipeline.bootstrap_from_examples()
    elif args.bootstrap == "database":
        n_templates = pipeline.bootstrap_from_database()
    else:
        n_templates = 0
    print(f"   Created {n_templates} templates")

    # Print stats
    pipeline.print_stats()

    # Test on sample problems
    print("\n3. Processing sample problems...")

    test_problems = [
        "John has 5 apples. He gives 2 apples to Mary. How many apples does John have now?",
        "Lisa has 12 cookies. She ate 3 cookies and gave 4 to her brother.",
        "Tom had 8 dollars. He earned 5 more dollars mowing lawns.",
    ]

    for i, problem in enumerate(test_problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"Text: {problem}")

        output = pipeline.process_problem(problem)

        print(f"Spans detected: {output.spans_detected}")
        print(f"Matched operations:")

        for op in output.matched_operations:
            span_preview = f"{op.span_text[:40]}..." if len(op.span_text) > 40 else op.span_text
            print(f"  - '{span_preview}'")
            print(f"    Operation: {op.operation_type.value}")
            print(f"    Template: {op.template_id}")
            print(f"    Confidence: {op.confidence:.3f}")
            print(f"    Scores: combined={op.combined_score:.3f}, "
                  f"emb={op.embedding_similarity:.3f}, "
                  f"att={op.attention_similarity:.3f}")

    # Demonstrate outcome recording
    print("\n4. Demonstrating outcome recording...")
    if test_problems and pipeline.store.templates:
        output = pipeline.process_problem(test_problems[0])
        if output.matched_operations:
            op = output.matched_operations[0]
            pipeline.record_outcome(
                op.template_id,
                success=True,
                embedding_sim=op.embedding_similarity,
                attention_sim=op.attention_similarity,
            )
            print(f"   Recorded success for template: {op.template_id}")

    # Check for decomposition candidates
    print("\n5. Checking decomposition candidates...")
    candidates = pipeline.get_decomposition_candidates()
    if candidates:
        print(f"   Found {len(candidates)} high-variance templates:")
        for t in candidates:
            print(f"   - {t.template_id} (variance={t.outcome_welford.variance:.3f})")
    else:
        print("   No high-variance templates (need more data)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
