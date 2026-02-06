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
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from mycelium.dual_signal_templates import (
    SpanDetector,
    TemplateStore,
    DualSignalTemplate,
)
from mycelium.attention_graph import AttentionGraphBuilder, SpanGraph, Span, GraphExecutor, ExecutionResult
from mycelium.subgraph_dsl import SubGraphDSL, load_subgraph_dsls
from mycelium.graph_embedder import infer_span_graph_embedding_from_text


# Default model path - relative to this file's location in src/mycelium/
# Goes up to mycelium/ project root, then into models/
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "minilm_attention_finetuned.pt"

# Default templates path - prefer Qwen-generated, fall back to deduplicated
_PROJECT_ROOT = Path(__file__).parent.parent.parent
QWEN_TEMPLATES_PATH = _PROJECT_ROOT / "qwen_templates.json"
DEDUP_TEMPLATES_PATH = _PROJECT_ROOT / "deduplicated_templates.json"
DEFAULT_TEMPLATES_PATH = QWEN_TEMPLATES_PATH if QWEN_TEMPLATES_PATH.exists() else DEDUP_TEMPLATES_PATH


# ================================================================
# Inference Strategy
# ================================================================
# At inference, raw spans are embedded directly by MiniLM and matched
# to templates by cosine similarity. NO generalization needed at inference.
#
# Templates were generalized at training time using Qwen-7B (one-time batch).
# Template centroids are computed from RAW span examples (not generalized
# patterns), so raw inference spans naturally match.
#
# See scripts/generalize_with_qwen.py for training-time generalization.


@dataclass
class MatchedOperation:
    """Result of matching a span to a template."""
    span_text: str
    template_id: str
    combined_score: float
    embedding_similarity: float
    attention_similarity: float
    confidence: float  # Welford-adjusted confidence
    subgraph: Optional[Dict[str, Any]] = None  # SubGraphDSL dict for execution


@dataclass
class PipelineOutput:
    """Complete output from processing a problem."""
    problem_text: str
    matched_operations: List[MatchedOperation]
    spans_detected: int
    templates_available: int
    execution_result: Optional[ExecutionResult] = None  # Result of graph execution
    answer: Optional[float] = None  # Computed numeric answer


class DualSignalPipeline:
    """Pipeline that uses dual-signal (attention + embedding) for operation matching.

    This integrates the fine-tuned MiniLM model with the template store
    to provide attention-aware classification of math problem spans.

    The dual-signal approach achieves better operational similarity matching
    by using BOTH embedding content and attention patterns.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_weight: float = 0.5,
        attention_weight: float = 0.3,
        graph_weight: float = 0.2,
        device: str = "auto",
        templates_path: Optional[str] = None,
    ):
        """Initialize the triple-signal pipeline.

        Args:
            model_path: Path to fine-tuned MiniLM model checkpoint.
                       Uses default path if not provided.
            embedding_weight: Weight for embedding similarity in matching [0-1].
            attention_weight: Weight for attention similarity in matching [0-1].
            graph_weight: Weight for computation graph similarity [0-1].
            device: Device for inference ("auto", "cuda", "cpu")
            templates_path: Optional path to load/save templates JSON

        Triple-signal approach:
        - Embedding: Captures lexical/semantic similarity
        - Attention: Captures structural processing patterns
        - Graph: Captures operational computation structure
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

        # Initialize template store with triple-signal weights
        self.store = TemplateStore(
            embedding_weight=embedding_weight,
            attention_weight=attention_weight,
            graph_weight=graph_weight,
        )

        # Initialize attention graph builder for span detection
        # Uses attention signals (not hardcoded lists) per CLAUDE.md
        self.graph_builder = AttentionGraphBuilder(
            entity_threshold=0.08,  # Tuned: "jordan" has 0.112 attention_received
            connectivity_threshold=0.1,
            boundary_drop_threshold=0.5,
        )

        # Initialize graph executor for computing answers
        self.graph_executor = GraphExecutor()

        # Sub-graph DSLs (keyed by template_id)
        self.subgraph_dsls: Dict[str, SubGraphDSL] = {}

        # Track templates file path for persistence
        self.templates_path = templates_path

        # Load templates: prefer explicit path, then default path
        if templates_path and os.path.exists(templates_path):
            self.load_templates(templates_path)
        elif os.path.exists(DEFAULT_TEMPLATES_PATH):
            self.load_templates(str(DEFAULT_TEMPLATES_PATH))

    def process_problem(self, text: str) -> PipelineOutput:
        """Process a math problem and match spans to templates.

        Uses attention signals (not hardcoded lists) per CLAUDE.md:
        1. Extract attention matrix from MiniLM
        2. Build span graph using attention connectivity
        3. Detect entities via attention_received signal
        4. Compose sub-graphs using cross-span attention
        5. Match each span to template and return in execution order

        Args:
            text: The math problem text to process

        Returns:
            PipelineOutput with matched operations for each span
        """
        return self._process_with_attention_graph(text)

    def _process_with_attention_graph(self, text: str) -> PipelineOutput:
        """Process using attention-based span detection and graph composition.

        No hardcoded entity lists - uses attention signals:
        - attention_received → detect entities
        - span_connectivity → find boundaries
        - cross_attention → compose sub-graphs

        Returns PipelineOutput with computed answer from graph execution.
        """
        # Extract attention matrix from MiniLM
        embedding, attention_matrix, tokens = self.detector.extract_features(text)

        # Average attention across heads if needed
        if attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)

        # Build span graph using attention signals
        graph = self.graph_builder.build_graph(attention_matrix, tokens, text)

        # Match each span to templates and collect SubGraphDSL for execution
        matched_operations = []
        span_subgraphs = []  # (span_idx, subgraph_dict)

        num_spans = len(graph.spans)
        for span_idx, span in enumerate(graph.spans):
            # Compute backward attention: how much this span attends to earlier spans
            # High backward attention = span depends on upstream context
            # Edge (src, dst, weight) means span[src] attends to span[dst]
            # So span_idx looking back = edges where src == span_idx and dst < span_idx
            backward_attention = sum(
                weight for src, dst, weight in graph.edges
                if src == span_idx and dst < span_idx
            )

            # Span position (0=first, 1=last) for graph embedding inference
            span_position = span_idx / max(1, num_spans - 1) if num_spans > 1 else 0.0

            match_result = self._match_span_to_template(
                span, attention_matrix, tokens,
                incoming_cross_attention=backward_attention,
                span_position=span_position,
            )
            if match_result:
                matched_operations.append(match_result)
                # Every template now has a subgraph (converted from legacy if needed)
                if match_result.subgraph:
                    span_subgraphs.append((span_idx, match_result.subgraph))

        # Execute using SubGraphDSL
        execution_result = self.graph_executor.execute_graph_with_subgraphs(
            graph, span_subgraphs, attention_matrix
        )

        return PipelineOutput(
            problem_text=text,
            matched_operations=matched_operations,
            spans_detected=len(graph.spans),
            templates_available=len(self.store.templates),
            execution_result=execution_result,
            answer=execution_result.answer if execution_result else None,
        )

    def _match_span_to_template(
        self,
        span: Span,
        full_attention_matrix: np.ndarray,
        full_tokens: List[str],
        incoming_cross_attention: float = 0.0,
        span_position: float = 0.5,
    ) -> Optional[MatchedOperation]:
        """Match a detected span to the best template using triple signals.

        Uses three signals for matching:
        1. Embedding similarity (text semantic)
        2. Attention pattern correlation (structural)
        3. Graph embedding similarity (operational)

        The graph embedding is inferred from span features (number count,
        backward attention, position) to match templates by operation type.

        Args:
            span: Detected span with token_indices into full attention matrix
            full_attention_matrix: Full attention matrix for the WHOLE problem
            full_tokens: All tokens from the full problem
            incoming_cross_attention: Sum of attention weights from previous spans
            span_position: Position in sequence (0=first, 1=last)

        Returns:
            MatchedOperation if match found, None otherwise
        """
        if not self.store.templates:
            return None

        # Embed the raw span directly - no generalization needed at inference
        # Template centroids were computed from raw span examples at training time
        span_embedding, span_attention, _ = self.detector.extract_features(span.text)

        # Average attention if needed
        if span_attention.ndim > 2:
            span_attention = span_attention.mean(axis=0)

        # Flatten attention for matching
        attention_flat = span_attention.flatten()

        # Infer span graph embedding from features
        # This encodes expected operation type based on:
        # - Number count (arity hint)
        # - Backward attention (needs upstream?)
        # - Position (early = SET, late = compute)
        span_graph_embedding = infer_span_graph_embedding_from_text(
            span_text=span.text,
            backward_attention=incoming_cross_attention,
            span_position=span_position,
            extract_numbers_fn=self.graph_executor.extract_numbers,
        )

        # Find best template match using triple signals
        result = self.store.find_best_match(
            span_embedding, attention_flat,
            graph_embedding=span_graph_embedding,
            needs_upstream=incoming_cross_attention > 0.05
        )

        if result:
            template, combined_score, emb_sim, att_sim, graph_sim = result
            confidence = self._compute_confidence(template, combined_score)

            return MatchedOperation(
                span_text=span.text,
                template_id=template.template_id,
                combined_score=combined_score,
                embedding_similarity=emb_sim,
                attention_similarity=att_sim,
                confidence=confidence,
                subgraph=template.subgraph,
            )

        return None

    # ================================================================
    # REMOVED: All hardcoded verb lists and pattern-based classification
    # Per CLAUDE.md: "AVOID Verb Classification Like The Plague"
    #
    # Operation type comes ONLY from template matching:
    # - Templates were classified by Qwen at training time
    # - At inference, raw span embedding → cosine match → template → operation + DSL
    # - No verb lists, no pattern matching, no heuristics
    # ================================================================

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
        graph_sim: Optional[float] = None,
    ) -> None:
        """Record execution outcome for learning.

        This feeds the variance-based decomposition system
        and weight learning in the template store.

        Args:
            template_id: ID of the template that was used
            success: Whether execution was successful
            embedding_sim: Embedding similarity from the match
            attention_sim: Attention similarity from the match
            graph_sim: Graph structure similarity from the match
        """
        template = self.store.get_template(template_id)
        if template:
            template.record_outcome(success)

        # Update signal weight learning (all 3 signals)
        if embedding_sim is not None and attention_sim is not None:
            self.store.update_weights_from_outcome(
                embedding_sim, attention_sim, success,
                graph_sim=graph_sim if graph_sim is not None else 0.5
            )

    def get_decomposition_candidates(self) -> List[DualSignalTemplate]:
        """Get templates that should be considered for decomposition.

        Per CLAUDE.md: High variance signals need for decomposition.
        "ONE node high variance -> decompose node"

        Returns:
            List of templates with high outcome variance
        """
        return self.store.get_high_variance_templates()

    def load_subgraph_dsls(self, path: str) -> int:
        """Load sub-graph DSLs from a JSON file.

        Each template gets a 1:1 sub-graph DSL for composable execution.

        Args:
            path: Path to JSON file with sub-graph DSLs

        Returns:
            Number of DSLs loaded
        """
        self.subgraph_dsls = load_subgraph_dsls(path)
        print(f"Loaded {len(self.subgraph_dsls)} sub-graph DSLs from {path}")
        return len(self.subgraph_dsls)

    # ================================================================
    # Bootstrap Methods - Initialize templates from existing data
    # ================================================================

    def bootstrap_from_labeled_spans(
        self,
        spans: List[Tuple[str, Any]],  # (text, subgraph_dict or legacy dsl_expr string)
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
    ) -> int:
        """Bootstrap templates from labeled span data.

        Creates initial templates from existing labeled spans.

        Args:
            spans: List of (span_text, subgraph_or_dsl_expr) tuples
                   - subgraph: Dict with SubGraphDSL format
                   - dsl_expr: Legacy string like "value", "entity + value" (converted)
            deduplicate: Skip spans too similar to existing templates
            similarity_threshold: Threshold for deduplication

        Returns:
            Number of templates created
        """
        created = 0

        for span_text, subgraph_or_dsl in spans:
            # Extract features for this span
            embedding, attention, tokens = self.detector.extract_features(span_text)
            attention_flat = attention.flatten()

            # Check for duplicates
            if deduplicate and self.store.templates:
                best_match = self.store.find_best_match(embedding, attention_flat)
                if best_match and best_match[1] > similarity_threshold:
                    continue

            # Create template
            import uuid
            template_id = f"tpl_{uuid.uuid4().hex[:8]}"

            # Convert legacy dsl_expr string to subgraph if needed
            if isinstance(subgraph_or_dsl, str):
                subgraph = self._dsl_expr_to_subgraph(subgraph_or_dsl, template_id)
            else:
                subgraph = subgraph_or_dsl

            template = DualSignalTemplate(
                template_id=template_id,
                embedding_centroid=embedding.copy(),
                attention_signature=attention_flat.copy(),
                subgraph=subgraph,
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
                (span.span_text, getattr(span, 'dsl_expr', 'value'))
                for span in spans
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

        Provides minimal templates when no data is available.

        Returns:
            Number of templates created
        """
        example_spans = [
            ("John has 5 apples", "value"),
            ("Mary has 12 cookies", "value"),
            ("He found 3 more coins", "entity + value"),
            ("She received 10 dollars", "entity + value"),
            ("She gave 4 to Jane", "entity - value"),
            ("He sold 5 apples", "entity - value"),
            ("Each bag has 5 apples", "entity * value"),
            ("She split it into 3 parts", "entity / value"),
        ]

        return self.bootstrap_from_labeled_spans(
            example_spans,
            deduplicate=False
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

    def load_templates(self, path: Optional[str] = None, replace: bool = False) -> None:
        """Load templates from JSON file.

        Supports two formats:
        1. TemplateStore format: {"embedding_weight": ..., "templates": {...}}
        2. Flat list format: [{template_id, operation, ...}, ...]

        Args:
            path: File path. Uses self.templates_path if not provided.
            replace: If True, clear existing templates before loading.
        """
        path = path or self.templates_path
        if not path:
            raise ValueError("No templates path specified")

        with open(path, 'r') as f:
            data = json.load(f)

        # Clear existing templates if replacing
        if replace:
            self.store = TemplateStore()

        # Detect format
        if isinstance(data, list):
            # Flat list format (deduplicated_templates.json style)
            for tpl_dict in data:
                template = self._convert_simple_template(tpl_dict)
                if template:
                    self.store.add_template(template)
        elif isinstance(data, dict) and "templates" in data:
            # TemplateStore format
            self.store = TemplateStore.from_dict(data)
        else:
            raise ValueError(f"Unknown template format in {path}")

        print(f"Loaded {len(self.store.templates)} templates from {path}")

    def _convert_simple_template(self, tpl_dict: Dict[str, Any]) -> Optional[DualSignalTemplate]:
        """Convert a simple template dict to DualSignalTemplate.

        Supports three template formats:
        1. Atomic pipeline: centroid + subgraph (SubGraphDSL steps) — preferred
        2. Legacy Qwen: embedding_centroid + base_dsl/dsl string → converted to subgraph
        3. Legacy format: re-embeds from pattern_examples text

        Args:
            tpl_dict: Dict with template_id, subgraph, base_dsl/dsl, etc.

        Returns:
            DualSignalTemplate or None if invalid
        """
        try:
            # Get SubGraphDSL — prefer explicit subgraph, convert legacy dsl_expr strings
            subgraph = tpl_dict.get("subgraph")

            if subgraph is None:
                # Convert legacy dsl_expr string to SubGraphDSL format
                dsl_expr = tpl_dict.get("base_dsl", tpl_dict.get("dsl", tpl_dict.get("dsl_expr", "value")))
                subgraph = self._dsl_expr_to_subgraph(dsl_expr, tpl_dict.get("template_id", "unknown"))

            # Pattern examples: check both field names
            patterns = tpl_dict.get("pattern_examples", tpl_dict.get("span_examples", []))

            # Check for pre-computed centroid (supports both field names)
            centroid = tpl_dict.get("embedding_centroid", tpl_dict.get("centroid", None))
            if centroid is not None:
                embedding = np.array(centroid, dtype=np.float32)

                # Load pre-computed attention signature if available
                if "attention_signature" in tpl_dict:
                    attention_flat = np.array(tpl_dict["attention_signature"], dtype=np.float32)
                else:
                    # Default to zeros if not pre-computed
                    attention_flat = np.zeros(100, dtype=np.float32)
            else:
                # Legacy: re-embed from pattern examples or description
                text_for_embedding = tpl_dict.get("description", tpl_dict.get("pattern", ""))
                if patterns:
                    text_for_embedding = " ".join(patterns[:3])

                embedding, attention, _ = self.detector.extract_features(text_for_embedding)
                if attention.ndim > 2:
                    attention = attention.mean(axis=0)
                attention_flat = attention.flatten()

            # Load graph embedding if present
            graph_emb = None
            if "graph_embedding" in tpl_dict:
                graph_emb = np.array(tpl_dict["graph_embedding"], dtype=np.float32)

            return DualSignalTemplate(
                template_id=tpl_dict["template_id"],
                embedding_centroid=embedding,
                attention_signature=attention_flat,
                pattern=tpl_dict.get("pattern", ""),
                subgraph=subgraph,
                graph_embedding=graph_emb,
                span_examples=patterns,
            )
        except Exception as e:
            print(f"Warning: Failed to convert template {tpl_dict.get('template_id')}: {e}")
            return None

    @staticmethod
    def _dsl_expr_to_subgraph(dsl_expr: str, template_id: str) -> Dict[str, Any]:
        """Convert legacy dsl_expr string to SubGraphDSL format.

        This is for backwards compatibility with old templates that used
        simple strings like "value", "entity + value", etc.
        """
        dsl_expr = dsl_expr.strip().lower() if dsl_expr else "value"

        # Default subgraph structure
        def make_subgraph(params, inputs, steps):
            return {
                "template_id": template_id,
                "pattern": "",
                "params": params,
                "inputs": inputs,
                "steps": steps,
                "output": "out",
            }

        if dsl_expr == "value":
            return make_subgraph({"n1": "value"}, {}, [{"var": "out", "op": "SET", "args": ["n1"]}])
        elif "+" in dsl_expr:  # "entity + value"
            return make_subgraph({"n1": "value"}, {"upstream": "entity"},
                                 [{"var": "out", "op": "ADD", "args": ["upstream", "n1"]}])
        elif "-" in dsl_expr:  # "entity - value"
            return make_subgraph({"n1": "value"}, {"upstream": "entity"},
                                 [{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}])
        elif "*" in dsl_expr:  # "entity * value"
            return make_subgraph({"n1": "value"}, {"upstream": "entity"},
                                 [{"var": "out", "op": "MUL", "args": ["upstream", "n1"]}])
        elif "/" in dsl_expr:  # "entity / value"
            return make_subgraph({"n1": "value"}, {"upstream": "entity"},
                                 [{"var": "out", "op": "DIV", "args": ["upstream", "n1"]}])
        else:
            return make_subgraph({"n1": "value"}, {}, [{"var": "out", "op": "SET", "args": ["n1"]}])

    def _compute_attention_centroid(self, pattern_examples: List[str]) -> np.ndarray:
        """Compute attention centroid from pattern examples via MiniLM.

        Runs extract_features on each example, flattens the attention matrices,
        and averages them (zero-padded to max length). This gives templates a
        REAL attention signature for dual-signal matching instead of dummy zeros.

        Args:
            pattern_examples: Raw span examples from the template

        Returns:
            Averaged flattened attention vector, or zeros if no examples
        """
        if not pattern_examples:
            return np.zeros(100, dtype=np.float32)

        attention_vectors = []
        for example in pattern_examples[:5]:  # Cap at 5 to limit startup cost
            try:
                _, attention, _ = self.detector.extract_features(example)
                if attention.ndim > 2:
                    attention = attention.mean(axis=0)
                attention_vectors.append(attention.flatten())
            except Exception:
                continue

        if not attention_vectors:
            return np.zeros(100, dtype=np.float32)

        # Pad all to max length, then average
        max_len = max(len(v) for v in attention_vectors)
        padded = np.zeros((len(attention_vectors), max_len), dtype=np.float32)
        for i, v in enumerate(attention_vectors):
            padded[i, :len(v)] = v

        return padded.mean(axis=0)

    # ================================================================
    # Diagnostic Methods
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Note: NO operation type categorization - per CLAUDE.md, we avoid
        classifying templates by operation type like the plague.

        Returns:
            Dictionary with template counts, weights, etc.
        """
        # Count templates with/without subgraphs
        with_subgraph = sum(1 for t in self.store.templates.values() if t.subgraph)

        return {
            "total_templates": len(self.store.templates),
            "templates_with_subgraph": with_subgraph,
            "embedding_weight": self.store.embedding_weight,
            "attention_weight": self.store.attention_weight,
            "high_variance_templates": len(self.get_decomposition_candidates()),
        }

    def print_stats(self) -> None:
        """Print pipeline statistics to console."""
        stats = self.get_stats()
        print("\n=== Dual-Signal Pipeline Stats ===")
        print(f"Total templates: {stats['total_templates']}")
        print(f"Templates with subgraph: {stats['templates_with_subgraph']}")
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
            # Extract operation from subgraph for display
            if op.subgraph and op.subgraph.get("steps"):
                dsl_op = op.subgraph["steps"][-1].get("op", "SET")
            else:
                dsl_op = "SET"
            print(f"    Op: {dsl_op}")
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
