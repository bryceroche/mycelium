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

# Primary template sources (17k spans → 207 specialized templates)
SPECIALIZED_TEMPLATES = PROJECT_ROOT / "specialized_templates.json"  # 17k raw templates (source of truth)
DEDUPLICATED_TEMPLATES = PROJECT_ROOT / "deduplicated_templates.json"  # 207 specialized (PREFERRED)

# Legacy/alternative template files (kept for backward compatibility)
OPERATION_SEPARATED = PROJECT_ROOT / "operation_separated_templates.json"  # 458 cluster+operation pairs
ENHANCED_DSL_LIBRARY = PROJECT_ROOT / "enhanced_dsl_library.json"  # 207 with Qwen signals
DSL_LIBRARY = PROJECT_ROOT / "dsl_library.json"  # LLM-generated
DUAL_SIGNAL_TEMPLATES = PROJECT_ROOT / "dual_signal_templates.json"  # Legacy embeddings + attention
TEMPLATES_EXPORT = PROJECT_ROOT / "span_templates_export.json"  # Fine-grained templates
TEMPLATES_JSON = PROJECT_ROOT / "operation_templates.json"  # Coarse templates

# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "minilm_contrastive.pt"  # Contrastive-trained encoder


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

            # Load templates: prefer deduplicated (207 specialized) > operation-separated > enhanced > specialized
            if not self.mock_model:
                templates_loaded = False

                # 1. FIRST: Try deduplicated_templates (207 specialized - balanced operations)
                if not templates_loaded and os.path.exists(DEDUPLICATED_TEMPLATES):
                    self.templates_path = str(DEDUPLICATED_TEMPLATES)
                    count = self._load_enhanced_templates()  # Compatible format
                    if count > 0:
                        templates_loaded = True
                        print(f"Using deduplicated templates (207 specialized from 17k spans)")

                # 2. Fallback to operation_separated (458 pure operation clusters)
                if not templates_loaded and os.path.exists(OPERATION_SEPARATED):
                    self.templates_path = str(OPERATION_SEPARATED)
                    count = self._load_enhanced_templates()
                    if count > 0:
                        templates_loaded = True
                        print(f"Using operation-separated templates (458 pure clusters)")

                # 3. Fallback to enhanced_dsl_library (207 mixed operation clusters)
                if not templates_loaded and os.path.exists(ENHANCED_DSL_LIBRARY):
                    self.templates_path = str(ENHANCED_DSL_LIBRARY)
                    count = self._load_enhanced_templates()
                    if count > 0:
                        templates_loaded = True
                        print(f"Using enhanced templates (207 deduped with aggregated signals)")

                # 4. Fallback to specialized_templates (17k raw)
                if not templates_loaded and os.path.exists(SPECIALIZED_TEMPLATES):
                    self.templates_path = str(SPECIALIZED_TEMPLATES)
                    count = self._load_specialized_templates()
                    if count > 0:
                        templates_loaded = True
                        print(f"Using specialized templates (17k raw spans)")

                # 5. Try database (has fine-grained templates)
                if not templates_loaded:
                    try:
                        db_count = get_template_count()
                        if db_count > 0:
                            self._load_templates_from_db()
                            templates_loaded = True
                    except Exception as e:
                        print(f"Database not available ({e})")

                # 6. Bootstrap with examples as last resort
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

    def _load_specialized_templates(self) -> int:
        """Load templates from specialized_templates.json (signal mapper training data).

        This is the SAME data the signal mapper was trained on, ensuring consistency
        between predicted attention signals and template matching.

        Format:
        - template_id, pattern, operation_type, dsl_expr
        - embedding_centroid (384-dim MiniLM)
        - attention_entropy, attention_received, attention_connection (Qwen-derived)
        """
        try:
            with open(self.templates_path, 'r') as f:
                data = json.load(f)

            count = 0
            for template_id, template_data in data.items():
                # Get operation type (specialized uses 'operation_type' not 'operation')
                op_str = template_data.get("operation_type", "SET").upper()
                try:
                    op_type = OperationType(op_str)
                except ValueError:
                    op_type = OperationType.SET

                # Get centroid (required)
                centroid_data = template_data.get("embedding_centroid")
                if centroid_data is None or len(centroid_data) == 0:
                    continue
                centroid = np.array(centroid_data, dtype=np.float32)

                # Build attention signature from Qwen signals
                attention_entropy = template_data.get("attention_entropy", 0.0)
                attention_received = template_data.get("attention_received", 0.0)
                attention_connection = template_data.get("attention_connection", 0.0)

                # Create attention signature matching signal mapper output format
                attention = np.array([
                    attention_entropy,
                    attention_received,
                    attention_connection,
                ] + [0.0] * 97, dtype=np.float32)  # Pad to 100 dims

                # Get DSL expression (specialized uses 'dsl_expr' not 'custom_dsl')
                dsl_expr = template_data.get("dsl_expr", "value")

                # Get pattern and examples
                # Use pattern_examples[0] if pattern is not set
                pattern = template_data.get("pattern", "")
                pattern_examples = template_data.get("pattern_examples", [])
                if not pattern and pattern_examples:
                    pattern = pattern_examples[0]
                spans = template_data.get("span_examples", [])

                # Create template
                template = DualSignalTemplate(
                    template_id=template_id,
                    operation_type=op_type,
                    embedding_centroid=centroid,
                    attention_signature=attention,
                    pattern=pattern,
                    dsl_expr=dsl_expr,
                    span_examples=spans[:10] if spans else [],
                    match_count=template_data.get("count", 0),
                )

                # Store Qwen attention signals as extra attributes
                template.attention_entropy = attention_entropy
                template.attention_received = attention_received
                template.attention_connection = attention_connection

                # Set cross-entity attention based on operation type
                # Derived from empirical analysis: SET=0, ADD=0.04, SUB/MUL=0.06
                CROSS_ENTITY_BY_OP = {
                    OperationType.SET: 0.0,
                    OperationType.ADD: 0.04,
                    OperationType.SUB: 0.06,
                    OperationType.MUL: 0.06,
                    OperationType.DIV: 0.04,
                    OperationType.UNKNOWN: 0.03,
                }
                template.cross_entity_attention = CROSS_ENTITY_BY_OP.get(op_type, 0.03)

                self._pipeline.store.add_template(template)
                count += 1

            print(f"Loaded {count} specialized templates (signal mapper training data)")
            return count

        except Exception as e:
            print(f"Error loading specialized templates: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _load_enhanced_templates(self) -> int:
        """Load enhanced templates with aggregated Qwen signals + variance.

        Enhanced templates are 207 deduped templates where each has:
        - aggregated_signals: mean/variance of Qwen signals from 17k source templates
        - source_count: how many source templates contributed
        - avg_similarity: average cosine similarity to sources

        The variance serves as a routing confidence metric:
        - Low variance = reliable signal (consistent across similar spans)
        - High variance = unreliable signal (may need decomposition)
        """
        try:
            with open(self.templates_path, 'r') as f:
                data = json.load(f)

            # Handle both dict and {templates: [...]} formats
            if isinstance(data, dict) and 'templates' in data:
                templates_list = data['templates']
            elif isinstance(data, list):
                templates_list = data
            else:
                templates_list = list(data.values())

            count = 0
            for template_data in templates_list:
                # Get template ID
                template_id = template_data.get('template_id', f'enhanced_{count}')

                # Get operation type
                op_str = template_data.get("operation", template_data.get("operation_type", "SET")).upper()
                try:
                    op_type = OperationType(op_str)
                except ValueError:
                    op_type = OperationType.SET

                # Get centroid (required)
                centroid_data = template_data.get("embedding_centroid")
                if centroid_data is None or len(centroid_data) == 0:
                    continue
                centroid = np.array(centroid_data, dtype=np.float32)

                # Get attention signals (supports both aggregated and flat formats)
                agg = template_data.get("aggregated_signals", {})
                entropy_mean = agg.get("entropy_mean", template_data.get("attention_entropy", 0.0))
                received_mean = agg.get("received_mean", template_data.get("attention_received", 0.0))
                connection_mean = agg.get("connection_mean", template_data.get("attention_connection", 0.0))

                # Create attention signature from aggregated signals
                attention = np.array([
                    entropy_mean,
                    received_mean,
                    connection_mean,
                ] + [0.0] * 97, dtype=np.float32)

                # Get DSL expression (supports multiple field names)
                dsl_expr = template_data.get("custom_dsl") or template_data.get("base_dsl") or template_data.get("dsl_expr", "value")

                # Get pattern and examples
                # Use pattern_examples[0] if pattern is not set
                pattern = template_data.get("pattern", "")
                pattern_examples = template_data.get("pattern_examples", [])
                if not pattern and pattern_examples:
                    pattern = pattern_examples[0]
                spans = template_data.get("span_examples", [])

                # Create template
                template = DualSignalTemplate(
                    template_id=template_id,
                    operation_type=op_type,
                    embedding_centroid=centroid,
                    attention_signature=attention,
                    pattern=pattern,
                    dsl_expr=dsl_expr,
                    span_examples=spans[:10] if spans else [],
                    match_count=template_data.get("count", 0),
                )

                # Store aggregated signals + variance as extra attributes
                template.entropy_mean = entropy_mean
                template.entropy_var = agg.get("entropy_var", 0.0)
                template.received_mean = received_mean
                template.received_var = agg.get("received_var", 0.0)
                template.connection_mean = connection_mean
                template.connection_var = agg.get("connection_var", 0.0)
                template.source_count = agg.get("source_count", 0)
                template.avg_similarity = agg.get("avg_similarity", 0.0)

                # Set cross-entity attention based on operation type
                # Derived from empirical analysis: SET=0, ADD=0.04, SUB/MUL=0.06
                CROSS_ENTITY_BY_OP = {
                    OperationType.SET: 0.0,
                    OperationType.ADD: 0.04,
                    OperationType.SUB: 0.06,
                    OperationType.MUL: 0.06,
                    OperationType.DIV: 0.04,
                    OperationType.UNKNOWN: 0.03,
                }
                template.cross_entity_attention = CROSS_ENTITY_BY_OP.get(op_type, 0.03)

                self._pipeline.store.add_template(template)
                count += 1

            print(f"Loaded {count} enhanced templates (207 deduped with aggregated signals)")
            return count

        except Exception as e:
            print(f"Error loading enhanced templates: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _load_dsl_library(self) -> int:
        """Load templates from DSL library with Qwen attention signals.

        The DSL library has the new format with:
        - embedding_centroid (384-dim MiniLM)
        - attention_entropy (Qwen-derived)
        - attention_received (Qwen-derived)
        - attention_connection (Qwen-derived)
        - custom_dsl (specialized DSL expression)
        """
        try:
            with open(self.templates_path, 'r') as f:
                data = json.load(f)

            count = 0
            for template_id, template_data in data.items():
                # Get operation type
                op_str = template_data.get("operation", "SET").upper()
                try:
                    op_type = OperationType(op_str)
                except ValueError:
                    op_type = OperationType.SET

                # Get centroid (required)
                centroid_data = template_data.get("embedding_centroid")
                if centroid_data is None or len(centroid_data) == 0:
                    continue
                centroid = np.array(centroid_data, dtype=np.float32)

                # Build attention signature from Qwen signals
                # Combine entropy, received, connection into a feature vector
                attention_entropy = template_data.get("attention_entropy", 0.0)
                attention_received = template_data.get("attention_received", 0.0)
                attention_connection = template_data.get("attention_connection", 0.0)

                # Create compact attention signature (can be expanded later)
                attention = np.array([
                    attention_entropy,
                    attention_received,
                    attention_connection,
                ] + [0.0] * 97, dtype=np.float32)  # Pad to 100 dims

                # Get pattern examples
                patterns = template_data.get("pattern_examples", [])
                spans = template_data.get("span_examples", [])

                # Get custom DSL expression
                custom_dsl = template_data.get("custom_dsl", template_data.get("base_dsl", "value"))

                # Create template
                template = DualSignalTemplate(
                    template_id=template_id,
                    operation_type=op_type,
                    embedding_centroid=centroid,
                    attention_signature=attention,
                    pattern=patterns[0] if patterns else "",
                    dsl_expr=custom_dsl,
                    span_examples=spans[:10],
                    match_count=template_data.get("count", 0),
                )

                # Store Qwen attention signals as extra attributes for dual-signal matching
                template.attention_entropy = attention_entropy
                template.attention_received = attention_received
                template.attention_connection = attention_connection

                self._pipeline.store.add_template(template)
                count += 1

            print(f"Loaded {count} DSL templates with Qwen attention signals")
            return count

        except Exception as e:
            print(f"Error loading DSL library: {e}")
            import traceback
            traceback.print_exc()
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
                    # Use the entity that was tracked during graph execution
                    entity = next(iter(output.execution_result.entity_values), "X")

                op = SolverOperation(
                    op_type=matched_op.operation_type.value,
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
