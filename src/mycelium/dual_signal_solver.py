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
                pattern = template_data.get("pattern", "")
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
                pattern = template_data.get("pattern", "")
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
            """Process problem using pattern-based template matching (mock mode).

            Uses pattern matching for operation classification,
            skipping GPU-based embedding extraction.
            """
            import re

            # Pattern-based operation inference (no verb classifier)
            def infer_operation(clause):
                """Infer operation from clause patterns."""
                clause_lower = clause.lower()

                # Check patterns in order of specificity
                sub_verbs = ['sold', 'sells', 'gave', 'gives', 'spent', 'spends',
                             'lost', 'loses', 'ate', 'eats', 'used', 'uses',
                             'took', 'takes', 'baked', 'bakes', 'threw', 'throws',
                             'lent', 'lends', 'traded', 'trades', 'donated', 'donates',
                             'paid', 'pays', 'drank', 'drinks']
                add_verbs = ['found', 'finds', 'received', 'receives', 'earned', 'earns',
                             'won', 'wins', 'bought', 'buys', 'got', 'gets',
                             'collected', 'collects', 'picked', 'picks',
                             'gathered', 'gathers', 'gained', 'gains']
                set_verbs = ['has', 'have', 'had', 'starts', 'started', 'owns',
                             'contains', 'there are', 'there were']

                # 1. Check VERY specific patterns first (price calculations)
                # "sells for $N each/per" = MUL (revenue = quantity * price)
                if re.search(r'(sells?|sold)\s+.*for\s+\$?\d+', clause_lower):
                    return (OperationType.MUL, "entity * value", 0.9)
                if re.search(r'for\s+\$?\d+.*\b(each|per)\b', clause_lower):
                    return (OperationType.MUL, "entity * value", 0.85)

                # 2. Check subtraction
                for verb in sub_verbs:
                    if verb in clause_lower:
                        return (OperationType.SUB, "entity - value", 0.85)

                # 3. Check addition
                for verb in add_verbs:
                    if verb in clause_lower:
                        return (OperationType.ADD, "entity + value", 0.85)

                # 4. Check other multiplication patterns
                # "each X has N" = MUL (total = count * per_item)
                # "N times" = MUL
                if re.search(r'each\s+\w+\s+has\s+\d+', clause_lower):
                    return (OperationType.MUL, "entity * value", 0.85)
                if 'times' in clause_lower and re.search(r'\d+\s+times', clause_lower):
                    return (OperationType.MUL, "entity * value", 0.85)
                if 'doubled' in clause_lower:
                    return (OperationType.MUL, "entity * 2", 0.9)
                if 'tripled' in clause_lower:
                    return (OperationType.MUL, "entity * 3", 0.9)

                # 4. Check division patterns
                if 'shared' in clause_lower and 'equally' in clause_lower:
                    return (OperationType.DIV, "entity / value", 0.85)
                if 'split' in clause_lower:
                    return (OperationType.DIV, "entity / value", 0.8)
                if 'divided' in clause_lower:
                    return (OperationType.DIV, "entity / value", 0.8)
                if 'half of' in clause_lower:
                    return (OperationType.DIV, "entity / 2", 0.85)
                if re.search(r'among\s+\d+', clause_lower):
                    return (OperationType.DIV, "entity / value", 0.75)

                # 5. Check set/initial values (last, as fallback)
                for verb in set_verbs:
                    if verb in clause_lower:
                        return (OperationType.SET, "value", 0.7)

                # Default: SET if has number
                if re.search(r'\d+', clause):
                    return (OperationType.SET, "value", 0.5)

                return None

            # Simple clause splitting
            clauses = re.split(r'[.!?]|\band\b|\bthen\b', text)
            clauses = [c.strip() for c in clauses if c.strip()]

            matched_ops = []
            for clause in clauses:
                # Skip question clauses
                clause_lower = clause.lower()
                if '?' in clause or any(q in clause_lower for q in
                        ['how many', 'how much', 'what is', 'what are']):
                    continue

                # Infer operation from patterns
                result = infer_operation(clause)

                if result:
                    op_type, dsl_expr, confidence = result
                    matched_ops.append(MatchedOperation(
                        span_text=clause,
                        operation_type=op_type,
                        template_id=f"{op_type.value}_pattern",
                        combined_score=confidence,
                        embedding_similarity=0.0,
                        attention_similarity=0.0,
                        confidence=confidence,
                        dsl_expr=dsl_expr,
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
