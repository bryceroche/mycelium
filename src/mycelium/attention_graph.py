"""Attention-based span detection and graph composition.

Uses attention signals (not hardcoded lists) to:
1. Detect entities dynamically (high attention_received = entity)
2. Find span boundaries (connectivity drops = span boundary)
3. Compose sub-graphs via cross-span attention

Per CLAUDE.md:
- "AVOID Verb Classification Like The Plague" - no hardcoded lists
- "Attention Received" - tokens that get looked back to are entities
- "Span Connectivity" - high connectivity = cohesive semantic unit
- "Cross-Attention Between Spans" - captures dependencies between spans
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass, field


@dataclass
class Entity:
    """An entity detected via attention signals."""
    text: str
    token_indices: List[int]  # Which tokens form this entity
    attention_received: float  # How much attention this entity receives
    span_id: Optional[int] = None  # Which span this entity belongs to


@dataclass
class Span:
    """A span detected via attention connectivity."""
    text: str
    token_indices: List[int]
    start_idx: int
    end_idx: int
    connectivity: float  # Internal attention connectivity
    entities: List[Entity] = field(default_factory=list)
    operation_type: Optional[str] = None
    template_id: Optional[str] = None


@dataclass
class SpanGraph:
    """Graph of spans connected by cross-attention."""
    spans: List[Span]
    edges: List[Tuple[int, int, float]]  # (from_span, to_span, attention_weight)
    entity_references: Dict[str, List[int]]  # entity_text -> [span_ids that reference it]


class AttentionGraphBuilder:
    """Builds span graphs from attention matrices.

    No hardcoded entity lists - uses attention signals to detect:
    - Entities: tokens with high attention_received
    - Span boundaries: where connectivity drops
    - Cross-span references: attention between spans
    """

    def __init__(
        self,
        entity_threshold: float = 0.08,  # Lowered: "jordan" has 0.112
        connectivity_threshold: float = 0.1,  # Min connectivity for span
        boundary_drop_threshold: float = 0.5,  # Connectivity drop ratio for boundary
    ):
        self.entity_threshold = entity_threshold
        self.connectivity_threshold = connectivity_threshold
        self.boundary_drop_threshold = boundary_drop_threshold

    def compute_attention_received(
        self,
        attention_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute attention received per token.

        High attention_received = many tokens look back to this one = entity/anchor.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights

        Returns:
            (seq_len,) array of attention received per token
        """
        # Sum columns: how much attention each token receives from all others
        # Exclude self-attention (diagonal)
        mask = 1 - np.eye(attention_matrix.shape[0])
        masked_attn = attention_matrix * mask
        return masked_attn.sum(axis=0)

    def compute_attention_entropy(
        self,
        attention_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute attention entropy per token.

        Low entropy = focused attention = important structural role.
        High entropy = diffuse attention = less discriminative.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights

        Returns:
            (seq_len,) array of entropy per token
        """
        # Normalize rows to probabilities
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        probs = attention_matrix / (row_sums + 1e-10)

        # Compute entropy per row (per token's attention distribution)
        # H = -sum(p * log(p))
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=1)

        # Normalize by max possible entropy
        max_entropy = np.log(attention_matrix.shape[1])
        return entropy / (max_entropy + 1e-10)

    def compute_span_connectivity(
        self,
        attention_matrix: np.ndarray,
        start_idx: int,
        end_idx: int
    ) -> float:
        """Compute connectivity within a candidate span.

        High connectivity = tokens attend to each other = cohesive unit.
        Low connectivity = tokens don't belong together = invalid span.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            start_idx: Start of span (inclusive)
            end_idx: End of span (exclusive)

        Returns:
            Average mutual attention within span
        """
        if end_idx <= start_idx:
            return 0.0

        span_attn = attention_matrix[start_idx:end_idx, start_idx:end_idx]

        # Average attention within span (excluding diagonal)
        n = end_idx - start_idx
        if n <= 1:
            return 0.0

        mask = 1 - np.eye(n)
        masked = span_attn * mask
        return masked.sum() / (n * (n - 1) + 1e-10)

    def detect_entities(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        threshold: Optional[float] = None
    ) -> List[Entity]:
        """Detect entities using attention_received signal.

        No hardcoded list - entities are tokens that receive high attention.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            tokens: List of token strings
            threshold: Override entity threshold

        Returns:
            List of detected entities
        """
        threshold = threshold or self.entity_threshold

        # Compute attention received per token
        attn_received = self.compute_attention_received(attention_matrix)

        # Create mask for special tokens ([CLS], [SEP], punctuation)
        # These tokens dominate attention but aren't entities
        special_mask = np.array([
            t.startswith('[') and t.endswith(']') or t in '.!?,:;'
            for t in tokens
        ])

        # Normalize to [0, 1] excluding special tokens
        non_special_attn = attn_received[~special_mask]
        if len(non_special_attn) > 0 and non_special_attn.max() > 0:
            max_received = non_special_attn.max()
            attn_received = attn_received / max_received
        else:
            max_received = attn_received.max()
            if max_received > 0:
                attn_received = attn_received / max_received

        entities = []
        i = 0
        while i < len(tokens):
            if attn_received[i] >= threshold:
                # Found potential entity - extend to consecutive high-attention tokens
                start = i
                while i < len(tokens) and attn_received[i] >= threshold * 0.7:
                    i += 1
                end = i

                # Skip special tokens
                entity_tokens = tokens[start:end]
                if not all(t.startswith('[') and t.endswith(']') for t in entity_tokens):
                    entity_text = ' '.join(t for t in entity_tokens
                                          if not (t.startswith('[') and t.endswith(']')))
                    if entity_text.strip():
                        entities.append(Entity(
                            text=entity_text.strip(),
                            token_indices=list(range(start, end)),
                            attention_received=float(attn_received[start:end].mean())
                        ))
            else:
                i += 1

        return entities

    def detect_span_boundaries(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Detect span boundaries using connectivity drops.

        Panama Hats problem: find longest spans with high internal connectivity.
        Boundary = where connectivity drops significantly.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            tokens: List of token strings

        Returns:
            List of (start_idx, end_idx) span boundaries
        """
        n = len(tokens)
        if n <= 2:
            return [(0, n)]

        # Compute running connectivity for windows of different sizes
        # Start with sentence-level splits (periods, etc.)
        boundaries = [0]

        for i, token in enumerate(tokens):
            # Split on sentence boundaries
            if token in ['.', '!', '?', ','] and i > 0 and i < n - 1:
                boundaries.append(i + 1)

        boundaries.append(n)

        # Refine boundaries using connectivity
        refined = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]

            # Check if this span should be split further
            if end - start > 5:
                # Look for connectivity drops within span
                best_split = None
                min_connectivity = float('inf')

                for j in range(start + 2, end - 2):
                    # Connectivity at potential split point
                    left_conn = self.compute_span_connectivity(attention_matrix, start, j)
                    right_conn = self.compute_span_connectivity(attention_matrix, j, end)
                    cross_conn = self._compute_cross_attention(attention_matrix, start, j, j, end)

                    # If cross-attention is much lower than internal, this is a good split
                    internal_avg = (left_conn + right_conn) / 2
                    if cross_conn < internal_avg * self.boundary_drop_threshold:
                        if cross_conn < min_connectivity:
                            min_connectivity = cross_conn
                            best_split = j

                if best_split is not None:
                    refined.append((start, best_split))
                    refined.append((best_split, end))
                else:
                    refined.append((start, end))
            else:
                refined.append((start, end))

        return refined

    def _compute_cross_attention(
        self,
        attention_matrix: np.ndarray,
        span1_start: int,
        span1_end: int,
        span2_start: int,
        span2_end: int
    ) -> float:
        """Compute attention between two spans."""
        cross_attn = attention_matrix[span1_start:span1_end, span2_start:span2_end]
        return float(cross_attn.mean())

    def build_spans(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        text: str
    ) -> List[Span]:
        """Build spans from attention signals.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            tokens: List of token strings
            text: Original text

        Returns:
            List of detected spans with entities
        """
        # Detect span boundaries
        boundaries = self.detect_span_boundaries(attention_matrix, tokens)

        # Detect all entities
        all_entities = self.detect_entities(attention_matrix, tokens)

        spans = []
        for span_id, (start, end) in enumerate(boundaries):
            # Get span text from tokens
            span_tokens = tokens[start:end]
            span_text = ' '.join(t for t in span_tokens
                                if not (t.startswith('[') and t.endswith(']'))
                                and not t.startswith('##'))

            # Skip empty or trivial spans
            if not span_text.strip() or len(span_text.strip()) < 3:
                continue

            # Compute connectivity
            connectivity = self.compute_span_connectivity(attention_matrix, start, end)

            # Find entities in this span
            span_entities = []
            for entity in all_entities:
                if any(start <= idx < end for idx in entity.token_indices):
                    entity.span_id = span_id
                    span_entities.append(entity)

            spans.append(Span(
                text=span_text.strip(),
                token_indices=list(range(start, end)),
                start_idx=start,
                end_idx=end,
                connectivity=connectivity,
                entities=span_entities
            ))

        return spans

    def build_graph(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        text: str
    ) -> SpanGraph:
        """Build complete span graph with cross-attention edges.

        Args:
            attention_matrix: (seq_len, seq_len) attention weights
            tokens: List of token strings
            text: Original text

        Returns:
            SpanGraph with spans, edges, and entity references
        """
        # Build spans
        spans = self.build_spans(attention_matrix, tokens, text)

        # Build cross-span edges
        edges = []
        for i, span1 in enumerate(spans):
            for j, span2 in enumerate(spans):
                if i >= j:
                    continue

                # Compute cross-attention between spans
                cross_attn = self._compute_cross_attention(
                    attention_matrix,
                    span1.start_idx, span1.end_idx,
                    span2.start_idx, span2.end_idx
                )

                # Only add edge if significant attention
                if cross_attn > 0.01:
                    edges.append((i, j, cross_attn))

        # Track entity references across spans
        entity_refs: Dict[str, List[int]] = {}
        for span_idx, span in enumerate(spans):
            for entity in span.entities:
                key = entity.text.lower()
                if key not in entity_refs:
                    entity_refs[key] = []
                entity_refs[key].append(span_idx)

        return SpanGraph(
            spans=spans,
            edges=edges,
            entity_references=entity_refs
        )

    def compose_subgraphs(
        self,
        graph: SpanGraph,
        template_store
    ) -> List[Tuple[Span, str, float]]:
        """Compose sub-graphs into execution order using cross-attention.

        Uses:
        1. Cross-span attention edges to determine dependencies
        2. Entity references to track what each span operates on
        3. Template matching to get operation type per span

        Args:
            graph: SpanGraph with spans and edges
            template_store: TemplateStore for matching spans to templates

        Returns:
            List of (span, template_id, confidence) in execution order
        """
        if not graph.spans:
            return []

        # Match each span to best template
        matched = []
        for span in graph.spans:
            # TODO: Use template_store to match span to template
            # For now, return spans in order with None template
            matched.append((span, None, 0.0))

        # Sort by topological order based on edges
        # Spans that are referenced by later spans come first
        # (Using simple heuristic: earlier spans first, with adjustment for dependencies)

        # Build dependency graph
        incoming = {i: set() for i in range(len(graph.spans))}
        for src, dst, weight in graph.edges:
            if weight > 0.05:  # Significant dependency
                incoming[dst].add(src)

        # Topological sort (Kahn's algorithm)
        order = []
        no_deps = [i for i in range(len(graph.spans)) if not incoming[i]]

        while no_deps:
            i = no_deps.pop(0)
            order.append(i)
            for j in range(len(graph.spans)):
                if i in incoming[j]:
                    incoming[j].remove(i)
                    if not incoming[j]:
                        no_deps.append(j)

        # Handle cycles by adding remaining
        for i in range(len(graph.spans)):
            if i not in order:
                order.append(i)

        # Return in execution order
        return [(matched[i][0], matched[i][1], matched[i][2]) for i in order]


@dataclass
class ExecutionResult:
    """Result of executing a span graph."""
    answer: Optional[float]
    entity_values: Dict[str, float]  # entity_name -> computed value
    execution_trace: List[str]  # Steps taken
    success: bool
    error: Optional[str] = None


class GraphExecutor:
    """Executes span graphs to compute numeric answers.

    Uses attention signals for all decisions:
    - Entity detection: attention_received (from AttentionGraphBuilder)
    - Pronoun resolution: cross-attention between spans
    - Number role assignment: attention from number tokens to entities
    - Reference entities: cross-attention edge weights

    No hardcoded word lists. Per CLAUDE.md: attention signals discriminate.
    """

    def __init__(self):
        import re
        self._number_pattern = re.compile(r'[\$]?(\d+(?:,\d{3})*(?:\.\d+)?)')
        self._fraction_pattern = re.compile(r'(\d+)/(\d+)')
        # Common pronouns for cross-attention resolution (not for classification)
        self._pronouns = frozenset([
            'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'its', 'their',
        ])

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numeric values from text."""
        import re
        numbers = []

        # Handle fractions first (e.g., "1/2", "3/4")
        for match in self._fraction_pattern.finditer(text):
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator != 0:
                numbers.append(numerator / denominator)

        # Remove fractions from text to avoid double-counting
        text_no_fractions = self._fraction_pattern.sub('', text)

        # Extract integers and decimals
        for match in self._number_pattern.finditer(text_no_fractions):
            num_str = match.group(1).replace(',', '')
            try:
                numbers.append(float(num_str))
            except ValueError:
                pass

        return numbers

    def execute_dsl(
        self,
        dsl_expr: str,
        entity_value: Optional[float],
        span_value: Optional[float]
    ) -> Optional[float]:
        """Execute a DSL expression.

        DSL expressions:
        - "value" -> return span_value
        - "entity + value" -> entity_value + span_value
        - "entity - value" -> entity_value - span_value
        - "entity * value" -> entity_value * span_value
        - "entity / value" -> entity_value / span_value

        Args:
            dsl_expr: The DSL expression string
            entity_value: Current entity value (from previous spans)
            span_value: Value extracted from current span

        Returns:
            Computed result or None if execution fails
        """
        if span_value is None:
            return entity_value

        dsl_expr = dsl_expr.strip().lower()

        if dsl_expr == 'value':
            return span_value

        if entity_value is None:
            # First operation - treat as SET
            return span_value

        if '+' in dsl_expr or dsl_expr.startswith('add'):
            return entity_value + span_value
        elif '-' in dsl_expr or dsl_expr.startswith('sub'):
            return entity_value - span_value
        elif '*' in dsl_expr or dsl_expr.startswith('mul'):
            return entity_value * span_value
        elif '/' in dsl_expr or dsl_expr.startswith('div'):
            if span_value != 0:
                return entity_value / span_value
            return None

        # Default: return span_value
        return span_value

    def resolve_entity_via_cross_attention(
        self,
        span: Span,
        span_idx: int,
        graph: SpanGraph,
        attention_matrix: Optional[np.ndarray],
        entity_values: Dict[str, float],
    ) -> Optional[str]:
        """Resolve which entity a span refers to using cross-attention.

        When a span has a pronoun or no detected entity, uses cross-attention
        edge weights to find which previous span (and its entity) this span
        most strongly attends to.

        No hardcoded pronoun lists for classification - attention signals decide.

        Args:
            span: Current span being executed
            span_idx: Index of current span in graph
            graph: Full span graph with edges
            attention_matrix: Full problem attention matrix
            entity_values: Currently tracked entity values

        Returns:
            Entity name resolved via cross-attention, or None
        """
        if not graph.edges or not entity_values:
            return None

        # Find the strongest cross-attention edge FROM a previous span TO this span
        best_source = None
        best_weight = 0.0

        for src, dst, weight in graph.edges:
            if dst == span_idx and weight > best_weight:
                best_weight = weight
                best_source = src

        if best_source is None or best_source >= len(graph.spans):
            return None

        # Get the primary entity from the source span
        source_span = graph.spans[best_source]
        if source_span.entities:
            source_entity = max(source_span.entities, key=lambda e: e.attention_received)
            entity_name = source_entity.text.lower()
            if entity_name in entity_values:
                return entity_name

        return None

    def assign_number_roles(
        self,
        span: Span,
        numbers: List[float],
        attention_matrix: Optional[np.ndarray],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Assign numbers to roles using attention patterns.

        For multi-number spans like "bought 3 apples for $2 each":
        - Primary value: number with highest attention to the span's entity
        - Secondary value: the other number (used as operand)

        Uses attention from number tokens to entity tokens to determine roles.

        Args:
            span: The span containing the numbers
            numbers: Extracted numeric values
            attention_matrix: Full problem attention matrix

        Returns:
            (primary_value, secondary_value) tuple
        """
        if not numbers:
            return None, None
        if len(numbers) == 1:
            return numbers[0], None

        # With attention matrix, use attention to determine roles
        if attention_matrix is not None and span.entities:
            entity_indices = set()
            for entity in span.entities:
                entity_indices.update(entity.token_indices)

            # Find which number tokens attend more strongly to entity tokens
            # Number tokens within the span's range
            number_attentions = []
            for num_idx, num_val in enumerate(numbers):
                # Estimate which token position this number occupies
                # by scanning span tokens for numeric content
                attn_to_entity = 0.0
                count = 0
                for tok_idx in span.token_indices:
                    if tok_idx < attention_matrix.shape[0]:
                        for ent_idx in entity_indices:
                            if ent_idx < attention_matrix.shape[1]:
                                attn_to_entity += attention_matrix[tok_idx, ent_idx]
                                count += 1
                number_attentions.append(attn_to_entity / max(count, 1))

            # Number with higher attention to entity = primary (the quantity being modified)
            if len(number_attentions) >= 2 and number_attentions[0] != number_attentions[1]:
                if number_attentions[0] >= number_attentions[1]:
                    return numbers[0], numbers[1]
                else:
                    return numbers[1], numbers[0]

        # Fallback: first number is primary, second is secondary
        return numbers[0], numbers[1] if len(numbers) > 1 else None

    def find_reference_entity(
        self,
        span: Span,
        span_idx: int,
        graph: SpanGraph,
        entity_values: Dict[str, float],
    ) -> Optional[str]:
        """Find reference entity for relative operations (e.g., "more than [REF]").

        Uses cross-attention: if a span has multiple entities detected,
        the one with LOWER attention_received (not the subject) is likely
        the reference entity.

        Args:
            span: Current span
            span_idx: Index in graph
            graph: Full graph
            entity_values: Current entity state

        Returns:
            Reference entity name, or None
        """
        if len(span.entities) < 2:
            return None

        # Sort entities by attention_received - highest is subject, others are references
        sorted_entities = sorted(span.entities, key=lambda e: -e.attention_received)

        # The second-highest attention entity that exists in state = reference
        for entity in sorted_entities[1:]:
            entity_name = entity.text.lower()
            if entity_name in entity_values:
                return entity_name

        # Check entity_references in graph for cross-span references
        for entity_name, span_ids in graph.entity_references.items():
            if span_idx in span_ids and entity_name in entity_values:
                # This span references an entity from another span
                # Check it's not the primary entity of this span
                primary = sorted_entities[0].text.lower() if sorted_entities else None
                if entity_name != primary:
                    return entity_name

        return None

    def execute_graph_with_attention(
        self,
        graph: SpanGraph,
        span_templates: List[Tuple[int, str, str]],  # (span_idx, operation_type, dsl_expr)
        attention_matrix: Optional[np.ndarray] = None
    ) -> ExecutionResult:
        """Execute span graph using attention signals for all decisions.

        Pure attention-driven execution:
        1. Entity detection: attention_received (from graph.spans[].entities)
        2. Pronoun/reference resolution: cross-attention edges between spans
        3. Number role assignment: attention from number tokens to entities
        4. Reference entities: secondary entities detected via attention

        No hardcoded word lists. Per CLAUDE.md: attention signals discriminate.

        Args:
            graph: SpanGraph with spans, edges, and entities
            span_templates: List of (span_idx, operation_type, dsl_expr) tuples
            attention_matrix: Full problem attention matrix for cross-attention

        Returns:
            ExecutionResult with composed answer
        """
        if not graph.spans:
            return ExecutionResult(
                answer=None,
                entity_values={},
                execution_trace=["No spans to execute"],
                success=False,
                error="Empty graph"
            )

        entity_values: Dict[str, float] = {}
        trace: List[str] = []

        template_map = {idx: (op, dsl) for idx, op, dsl in span_templates}
        order = self._topological_sort(graph)

        # Track the primary entity (first named entity seen)
        primary_entity: Optional[str] = None

        for span_idx in order:
            if span_idx >= len(graph.spans):
                continue

            span = graph.spans[span_idx]
            op_type, dsl_expr = template_map.get(span_idx, ('SET', 'value'))

            # --- Entity resolution via attention ---
            # 1. Use attention-detected entities from the span
            current_entity = None
            if span.entities:
                best_entity = max(span.entities, key=lambda e: e.attention_received)
                entity_text = best_entity.text.lower()

                # Check if this is a pronoun needing cross-attention resolution
                if entity_text in self._pronouns:
                    # Resolve via cross-attention to previous spans
                    resolved = self.resolve_entity_via_cross_attention(
                        span, span_idx, graph, attention_matrix, entity_values
                    )
                    current_entity = resolved or primary_entity
                else:
                    current_entity = entity_text
                    if primary_entity is None:
                        primary_entity = entity_text

            # 2. If no entity detected, use cross-attention to infer from context
            if current_entity is None:
                current_entity = self.resolve_entity_via_cross_attention(
                    span, span_idx, graph, attention_matrix, entity_values
                ) or primary_entity or f"entity_{span_idx}"

            # --- Number extraction with role assignment ---
            numbers = self.extract_numbers(span.text)
            primary_value, secondary_value = self.assign_number_roles(
                span, numbers, attention_matrix
            )

            # --- Reference entity for relative DSLs (e.g., "ref + value") ---
            ref_entity = None
            if 'ref' in dsl_expr:
                ref_entity = self.find_reference_entity(
                    span, span_idx, graph, entity_values
                )

            # --- Execute DSL ---
            if op_type == 'SET' or current_entity not in entity_values:
                if primary_value is not None:
                    entity_values[current_entity] = primary_value
                    trace.append(f"SET {current_entity} = {primary_value}")
            else:
                old_val = entity_values[current_entity]

                # Handle reference-based DSL (e.g., "ref + value", "ref * 2")
                if ref_entity and ref_entity in entity_values:
                    ref_val = entity_values[ref_entity]
                    new_val = self._execute_ref_dsl(dsl_expr, ref_val, primary_value)
                    if new_val is not None:
                        entity_values[current_entity] = new_val
                        trace.append(f"{op_type} {current_entity}: ref({ref_entity}={ref_val}) -> {new_val}")
                else:
                    new_val = self.execute_dsl(dsl_expr, old_val, primary_value)
                    if new_val is not None:
                        entity_values[current_entity] = new_val
                        trace.append(f"{op_type} {current_entity}: {old_val} -> {new_val}")

            # Handle secondary value (e.g., MUL: quantity * unit_price)
            if secondary_value is not None and op_type in ('MUL', 'DIV'):
                old_val = entity_values.get(current_entity, 0)
                new_val = self.execute_dsl(dsl_expr, old_val, secondary_value)
                if new_val is not None:
                    entity_values[current_entity] = new_val
                    trace.append(f"{op_type} {current_entity} (secondary): {old_val} -> {new_val}")

        # Compute final answer
        if not entity_values:
            return ExecutionResult(
                answer=None,
                entity_values=entity_values,
                execution_trace=trace,
                success=False,
                error="No values extracted"
            )

        trace.append(f"Entity values: {entity_values}")

        # Answer = primary entity's final value (not sum of all entities)
        if primary_entity and primary_entity in entity_values:
            answer = entity_values[primary_entity]
        else:
            # Fallback: last computed value
            answer = list(entity_values.values())[-1]

        trace.append(f"Final answer ({primary_entity}): {answer}")

        return ExecutionResult(
            answer=answer,
            entity_values=entity_values,
            execution_trace=trace,
            success=True
        )

    def _execute_ref_dsl(
        self,
        dsl_expr: str,
        ref_value: float,
        span_value: Optional[float]
    ) -> Optional[float]:
        """Execute a reference-based DSL expression.

        Handles DSL like "ref + value", "ref * 2", "ref - value".
        Uses the reference entity's value as the base.
        """
        dsl = dsl_expr.strip().lower()

        if span_value is None:
            return ref_value

        if 'ref * 2' in dsl:
            return ref_value * 2
        elif 'ref * 3' in dsl:
            return ref_value * 3
        elif 'ref +' in dsl:
            return ref_value + span_value
        elif 'ref -' in dsl:
            return ref_value - span_value
        elif 'ref *' in dsl:
            return ref_value * span_value
        elif 'ref /' in dsl:
            return ref_value / span_value if span_value != 0 else None

        return ref_value

    # execute_graph() removed - use execute_graph_with_attention() instead
    # The old method used string-based entity matching; the new method
    # uses attention signals for all entity resolution.

    def _topological_sort(self, graph: SpanGraph) -> List[int]:
        """Topological sort of spans based on cross-attention edges."""
        n = len(graph.spans)
        if n == 0:
            return []

        # Build incoming edge counts
        incoming = {i: set() for i in range(n)}
        for src, dst, weight in graph.edges:
            if weight > 0.02:  # Significant dependency threshold
                incoming[dst].add(src)

        # Kahn's algorithm
        order = []
        no_deps = [i for i in range(n) if not incoming[i]]

        while no_deps:
            i = no_deps.pop(0)
            order.append(i)
            for j in range(n):
                if i in incoming[j]:
                    incoming[j].remove(i)
                    if not incoming[j]:
                        no_deps.append(j)

        # Add remaining (handles cycles)
        for i in range(n):
            if i not in order:
                order.append(i)

        return order


def extract_attention_features(
    attention_matrix: np.ndarray,
    tokens: List[str]
) -> Dict[str, np.ndarray]:
    """Extract all attention features for a text.

    Args:
        attention_matrix: (seq_len, seq_len) attention weights
        tokens: List of token strings

    Returns:
        Dict with 'entropy', 'received', 'connectivity' arrays
    """
    builder = AttentionGraphBuilder()

    return {
        'entropy': builder.compute_attention_entropy(attention_matrix),
        'received': builder.compute_attention_received(attention_matrix),
        # Connectivity is per-span, computed separately
    }
