"""
Span Graph: Panama Hats-inspired compositional graph building.

Key insight from CLAUDE.md:
- "panama" = country, "panama hats" = different meaning
- "half" = 0.5, "half the eggs" = ONE operation

Attention connectivity defines span boundaries. Spans become subgraph nodes.
Cross-attention between spans defines edges (composition).

The Big 5:
1. Panama Hats Problem (guides span creation)
2. Attention Signals (Entropy, Received, Connectivity)
3. MiniLM distillation (trained with MSE attention loss)
4. Trained Signal Mapping (17k spans)
5. Cross-Attention Between Spans
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class Track(Enum):
    """Tracks for multi-track accounting (cost vs value problems)."""
    DEFAULT = "default"   # Single-track (most problems)
    COST = "cost"         # Money spent: buying, repairs, investment
    VALUE = "value"       # Asset value: appreciation, increases
    BOTH = "both"         # Affects both tracks (initial purchase)


@dataclass
class SpanNode:
    """A span that forms one node in the computation graph.

    Each span is a cohesive semantic unit (high attention connectivity).
    Examples:
    - "half the price of the cheese" (one MUL operation)
    - "twice as many apples as oranges" (one MUL with reference)
    """
    id: str
    text: str

    # Extracted semantics
    dsl_expr: str = "value"                # DSL expression: "value", "entity + value", etc.
    value: Optional[float] = None          # Numeric value if present
    entity: Optional[str] = None           # Subject entity ("Janet", "eggs")
    reference: Optional[str] = None        # Referenced entity for REF/comparisons

    # Multi-track accounting
    track: Track = Track.DEFAULT           # Which track this affects

    # Attention signals (from MiniLM → Qwen mapping)
    entropy: float = 0.0                   # Low = focused attention
    received: float = 0.0                  # High = structural importance
    connectivity: float = 0.0             # High = cohesive span

    # Position in problem
    position: int = 0                      # Span index in sequence

    # Embedding for template matching
    embedding: Optional[np.ndarray] = None


@dataclass
class SpanEdge:
    """Edge connecting two spans in the computation graph.

    Edges represent data flow / dependencies:
    - Entity reference: "she" → "Janet"
    - Value reference: "half that" → previous value
    - Comparison: "twice as many as X" → X's value
    """
    source_id: str
    target_id: str
    edge_type: str  # "entity_ref", "value_ref", "comparison", "sequence"
    weight: float = 1.0  # Cross-attention strength


@dataclass
class SpanGraph:
    """Computation graph built from connected spans.

    Nodes are spans (cohesive semantic units).
    Edges are connections (entity refs, value refs, sequence).

    The graph can be pattern-matched against templates.
    """
    nodes: Dict[str, SpanNode] = field(default_factory=dict)
    edges: List[SpanEdge] = field(default_factory=list)

    # Entity tracking across spans
    entity_mentions: Dict[str, List[str]] = field(default_factory=dict)  # entity -> [span_ids]

    def add_node(self, node: SpanNode):
        """Add a span node to the graph."""
        self.nodes[node.id] = node
        if node.entity:
            if node.entity not in self.entity_mentions:
                self.entity_mentions[node.entity] = []
            self.entity_mentions[node.entity].append(node.id)

    def add_edge(self, edge: SpanEdge):
        """Add an edge connecting two spans."""
        self.edges.append(edge)

    def get_node_edges(self, node_id: str) -> List[SpanEdge]:
        """Get all edges connected to a node."""
        return [e for e in self.edges if e.source_id == node_id or e.target_id == node_id]

    def topological_order(self) -> List[str]:
        """Get nodes in execution order (dependencies first)."""
        # Build adjacency for incoming edges
        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for edge in self.edges:
                if edge.source_id == node_id:
                    in_degree[edge.target_id] -= 1
                    if in_degree[edge.target_id] == 0:
                        queue.append(edge.target_id)

        # Add any remaining (cycles or disconnected)
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)

        return order


class SpanGraphBuilder:
    """Builds a SpanGraph from problem text using attention signals.

    Pipeline:
    1. Segment into candidate spans (sentence boundaries + attention connectivity)
    2. Extract semantics from each span (operation, value, entity)
    3. Link spans via entity tracking and cross-attention
    4. Build the computation graph
    """

    # Pronouns that reference previous entities
    PRONOUNS = {'she', 'he', 'it', 'they', 'her', 'his', 'their', 'its'}

    # Reference phrases that point to previous values
    VALUE_REFS = {
        'that much': 1.0,
        'the rest': 1.0,
        'the remainder': 1.0,
        'half that': 0.5,
        'twice that': 2.0,
        'half as much': 0.5,
        'twice as much': 2.0,
    }

    # Comparison patterns (entity references)
    COMPARISON_PATTERNS = [
        r'(\w+) times as many (?:\w+ )?as (\w+)',      # "4 times as many as Seattle"
        r'twice as many (?:\w+ )?as (\w+)',            # "twice as many as Charleston"
        r'half as many (?:\w+ )?as (\w+)',             # "half as many as X"
        r'(\d+) more than (\w+)',                       # "5 more than Janet"
        r'(\d+) fewer than (\w+)',                      # "3 fewer than Bob"
    ]

    def __init__(self, detector=None):
        """Initialize with optional span detector for attention signals."""
        self.detector = detector

    def build_graph(self, problem_text: str) -> SpanGraph:
        """Build a SpanGraph from problem text.

        This is the main entry point.
        """
        graph = SpanGraph()

        # 1. Segment into spans (using attention connectivity)
        spans = self._segment_spans(problem_text)

        # 2. Create nodes with extracted semantics
        entity_tracker = {}  # Most recent entity by name
        last_entity = None

        for i, span_text in enumerate(spans):
            node = self._create_span_node(span_text, i, last_entity, entity_tracker)
            if node:
                graph.add_node(node)
                if node.entity:
                    entity_tracker[node.entity.lower()] = node.id
                    last_entity = node.entity

        # 3. Link spans via entity references and cross-attention
        self._link_spans(graph, entity_tracker)

        return graph

    def _segment_spans(self, text: str) -> List[str]:
        """Segment text into spans using attention connectivity.

        Panama Hats insight: tokens with high mutual attention form a span.
        For now, use heuristics. TODO: use actual attention signals.
        """
        # Split on sentence boundaries
        parts = re.split(r'[.!?]', text)

        spans = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_lower = part.lower()

            # Extract values from questions before skipping
            # "if Seattle has 20 sheep" contains important data
            if any(q in part_lower for q in ['how many', 'how much', 'what is']):
                # Check for "if X has/is Y" patterns with values
                if_match = re.search(r'if (\w+) (?:has|is|have|are|had|was|were) (\d+)', part_lower)
                if if_match:
                    # Extract this as a SET span
                    entity = if_match.group(1).capitalize()
                    value = if_match.group(2)
                    spans.append(f"{entity} has {value}")
                continue

            # Split on "and" / "then" for compound sentences
            # But keep together: "twice as many apples as oranges"
            if ' and ' in part and 'as many' not in part_lower:
                # Split on "and" - but "half that much" should be its own span
                subparts = part.split(' and ')
                spans.extend([s.strip() for s in subparts if s.strip()])
            else:
                spans.append(part)

        return spans

    def _create_span_node(
        self,
        text: str,
        position: int,
        last_entity: Optional[str],
        entity_tracker: Dict[str, str]
    ) -> Optional[SpanNode]:
        """Create a SpanNode from span text with extracted semantics."""

        text_lower = text.lower()

        # Extract entity (subject of the span)
        entity = self._extract_entity(text)

        # Handle pronouns and demonstratives ("this", "that")
        if entity and entity.lower() in self.PRONOUNS:
            entity = last_entity
        elif entity and entity.lower() in ['this', 'that', 'it']:
            entity = last_entity

        # If no entity found but we have a value reference, carry forward from last entity
        # "half that much" continues the previous entity's accounting
        if entity is None and last_entity:
            for phrase in self.VALUE_REFS:
                if phrase in text_lower:
                    entity = last_entity
                    break

        # Extract numeric value
        value = self._extract_value(text)

        # Detect DSL expression and any references
        dsl_expr, reference, ref_multiplier = self._detect_operation(text, entity_tracker)

        # Detect which track this span affects (for multi-track accounting)
        track = self._detect_track(text, dsl_expr, reference)

        # If we have a reference multiplier (e.g., "twice that", "150%"), use it
        # For percentage patterns, always use the computed multiplier
        if ref_multiplier and ref_multiplier != 1.0:
            if reference == "_pct_increase" or value is None:
                value = ref_multiplier

        node = SpanNode(
            id=f"span_{position}",
            text=text,
            dsl_expr=dsl_expr,
            value=value,
            entity=entity,
            reference=reference,
            track=track,
            position=position,
        )

        return node

    def _detect_track(
        self,
        text: str,
        dsl_expr: str,
        reference: Optional[str]
    ) -> Track:
        """Detect which track this span affects for multi-track accounting.

        Track types:
        - COST: Money spent (buying, repairs, investment)
        - VALUE: Asset value (appreciation, increases, worth)
        - BOTH: Initial purchase (sets both cost basis and initial value)
        - DEFAULT: Single-track problems

        Returns Track.DEFAULT for simple single-track problems.
        """
        text_lower = text.lower()

        # Keywords that indicate we need multi-track accounting
        multi_track_signals = [
            'profit', 'loss', 'gain', 'margin', 'difference',
            'buys', 'bought', 'purchase', 'repair', 'renovate',
            'value', 'worth', 'appreciate', 'increase'
        ]

        # Check if this problem needs multi-track accounting
        # (Only activate if we see profit/value language)
        has_value_language = any(kw in text_lower for kw in ['value', 'worth', 'appreciate', 'increase'])
        has_cost_language = any(kw in text_lower for kw in ['repair', 'renovate', 'invest', 'put', 'spent'])

        # Initial purchase: "buys X for $Y" - sets BOTH tracks
        if re.search(r'buys?|bought|purchase', text_lower) and re.search(r'\$[\d,]+', text_lower):
            return Track.BOTH

        # Cost-only: repairs, renovations, investments INTO something
        cost_patterns = [
            r'puts?.*\$',           # "puts $50k into"
            r'spent.*\$',           # "spent $X on"
            r'invest',              # "invested"
            r'repair',              # "repairs"
            r'renovate',            # "renovation"
            r'maintenance',         # "maintenance costs"
        ]
        for pattern in cost_patterns:
            if re.search(pattern, text_lower):
                return Track.COST

        # Value-only: appreciation, increases, "the value"
        value_patterns = [
            r'value.*increase',     # "value increased"
            r'increase.*value',     # "increased the value"
            r'appreciate',          # "appreciated by"
            r'worth.*now',          # "worth X now"
            r'increased?.*%',       # "increased by X%" (typically value)
        ]
        for pattern in value_patterns:
            if re.search(pattern, text_lower):
                return Track.VALUE

        # If we have a percentage increase with _pct_increase reference, it's VALUE
        if reference == "_pct_increase":
            return Track.VALUE

        return Track.DEFAULT

    def _extract_entity(self, text: str) -> Optional[str]:
        """Extract the main entity from a span.

        Uses attention-received signal: high received = entity/anchor.
        For now, heuristic: first capitalized word or noun phrase.
        """
        # Look for capitalized words (proper nouns)
        caps = re.findall(r'\b([A-Z][a-z]+)\b', text)
        if caps:
            # Filter out sentence starters
            if caps[0] in ['A', 'The', 'She', 'He', 'It', 'They']:
                if len(caps) > 1:
                    return caps[1]
                # Check for pronoun
                for word in ['she', 'he', 'it', 'they']:
                    if word in text.lower().split()[:3]:
                        return word.capitalize()
            else:
                return caps[0]

        # Look for pronouns
        words = text.lower().split()
        for word in words[:5]:
            if word in self.PRONOUNS:
                return word.capitalize()

        return None

    def _extract_value(self, text: str) -> Optional[float]:
        """Extract numeric value from span.

        For multi-value spans like "3 sprints 3 times", returns the product.
        """
        text_lower = text.lower()

        # Handle dollar amounts with commas
        money = re.findall(r'\$([0-9,]+(?:\.\d+)?)', text)
        if money:
            return float(money[0].replace(',', ''))

        # Check for multi-value patterns FIRST
        # "X things Y times" → X * Y
        times_match = re.search(r'(\d+)\s+\w+\s+(\d+)\s+times', text_lower)
        if times_match:
            return float(times_match.group(1)) * float(times_match.group(2))

        # "X per Y" where both are numbers → X * Y (e.g., "3 per day for 5 days")
        per_match = re.search(r'(\d+)\s+per\s+\w+.*?(\d+)\s+\w+', text_lower)
        if per_match:
            return float(per_match.group(1)) * float(per_match.group(2))

        # Handle regular numbers - extract all and check for compound patterns
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            # If "times" or "each" present with multiple numbers, multiply them
            if len(numbers) >= 2 and ('times' in text_lower or 'x' in text_lower):
                result = 1.0
                for n in numbers:
                    result *= float(n)
                return result
            return float(numbers[0])

        # Word numbers
        word_nums = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3,
        }
        for word, val in word_nums.items():
            if word in text_lower:
                return float(val)

        return None

    def _detect_operation(
        self,
        text: str,
        entity_tracker: Dict[str, str]
    ) -> Tuple[str, Optional[str], Optional[float]]:
        """Detect DSL expression and any entity/value references.

        Returns: (dsl_expr, reference_entity, reference_multiplier)
        DSL expressions: "value", "entity + value", "entity - value",
                        "entity * value", "entity / value", "ref"
        """
        text_lower = text.lower()

        # Check for comparison patterns first (these have entity references)
        for pattern in self.COMPARISON_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                # Find the referenced entity
                ref_entity = groups[-1] if groups else None

                # Determine multiplier
                if 'twice' in text_lower:
                    return "entity * value", ref_entity, 2.0
                elif 'half' in text_lower:
                    return "entity * value", ref_entity, 0.5
                elif 'times' in text_lower and len(groups) > 1:
                    try:
                        mult = float(groups[0])
                        return "entity * value", ref_entity, mult
                    except:
                        pass
                elif 'more than' in text_lower:
                    return "entity + value", ref_entity, None
                elif 'fewer than' in text_lower:
                    return "entity - value", ref_entity, None

                return "ref", ref_entity, None

        # Check for percentage increase pattern FIRST
        # "increased by X%" or "increased the value by X%" → MUL with factor (1 + X/100)
        pct_match = re.search(r'increased?.*?(\d+)%', text_lower)
        if pct_match:
            pct = float(pct_match.group(1))
            return "entity * value", "_pct_increase", 1 + pct / 100

        # "X% more" → MUL with factor (1 + X/100)
        pct_more = re.search(r'(\d+)%\s+more', text_lower)
        if pct_more:
            pct = float(pct_more.group(1))
            return "entity * value", "_pct_increase", 1 + pct / 100

        # "X% of" → MUL with factor X/100
        pct_of = re.search(r'(\d+)%\s+of', text_lower)
        if pct_of:
            pct = float(pct_of.group(1))
            return "entity * value", None, pct / 100

        # Check for price/rate patterns - takes priority over value references
        # "$X per/each" or "for $X per" → MUL
        if re.search(r'\$\d+', text_lower) and ('per' in text_lower or 'each' in text_lower):
            return "entity * value", None, None

        # "X each Y" or "X per Y" without $ → MUL (unit rate)
        if re.search(r'\d+.*\b(each|per)\b', text_lower):
            return "entity * value", None, None

        # "for $X per" pattern
        if re.search(r'for \$\d+.*per', text_lower):
            return "entity * value", None, None

        # "sells/sold ... for $X" could be MUL if it's unit pricing
        if re.search(r'sells?.*for.*\$\d+', text_lower):
            return "entity * value", None, None

        # "remainder/rest" + price → multiply remaining by price
        if ('remainder' in text_lower or 'rest' in text_lower) and '$' in text_lower:
            return "entity * value", None, None

        # Check for value reference phrases - these ADD a fraction of previous value
        # But NOT if there's a price pattern (already handled above)
        for phrase, mult in self.VALUE_REFS.items():
            if phrase in text_lower and '$' not in text_lower:
                return "entity + value", "_prev", mult

        # Check for multiplication keywords
        if any(kw in text_lower for kw in ['times', 'multiplied', 'doubled', 'tripled']):
            return "entity * value", None, None

        # Check for division keywords
        if any(kw in text_lower for kw in ['divided', 'split', 'shared equally', 'half of']):
            return "entity / value", None, None

        # Subtraction verbs (consumption) - "bakes with X eggs" = uses eggs
        # Use word boundaries to avoid "seattle" matching "eat"
        # NOTE: "gave him/her" = receiving (not subtraction), so check context
        if re.search(r'\bgave\s+(him|her|them|me)\b', text_lower):
            return "value", None, None

        sub_verbs = ['ate', 'eats', 'eat', 'gave', 'gives', 'spent', 'lost',
                     'used', 'took', 'baked', 'bakes', 'bake', 'drank', 'threw']
        for verb in sub_verbs:
            if re.search(rf'\b{verb}\b', text_lower):
                return "entity - value", None, None

        # Selling without price pattern is also consumption (giving away inventory)
        # But only if not a "has/have" sentence (initial state)
        if any(v in text_lower for v in ['sold', 'sells', 'sell']):
            if ' has ' not in text_lower and ' have ' not in text_lower:
                return "entity - value", None, None

        # SET verbs (initial state) - check before ADD
        set_verbs = ['has', 'have', 'had', 'starts', 'started', 'owns',
                     'contains', 'is', 'was', 'are', 'were', 'lay', 'lays', 'takes']
        for verb in set_verbs:
            if re.search(rf'\b{verb}\b', text_lower):
                return "value", None, None

        # Addition verbs - includes spending/putting in (cost accumulation)
        add_verbs = ['found', 'received', 'earned', 'won', 'bought', 'got',
                     'collected', 'picked', 'gathered', 'gained',
                     'puts', 'put', 'spent', 'invested', 'added']
        for verb in add_verbs:
            if re.search(rf'\b{verb}\b', text_lower):
                return "entity + value", None, None

        # Default to value assignment
        return "value", None, None

    def _link_spans(self, graph: SpanGraph, entity_tracker: Dict[str, str]):
        """Link spans via entity references and sequence.

        This creates the edges that enable composition.
        """
        node_ids = list(graph.nodes.keys())

        for i, node_id in enumerate(node_ids):
            node = graph.nodes[node_id]

            # Link to previous span (sequence)
            if i > 0:
                prev_id = node_ids[i - 1]
                graph.add_edge(SpanEdge(
                    source_id=prev_id,
                    target_id=node_id,
                    edge_type="sequence",
                    weight=0.5
                ))

            # Link via entity reference
            if node.reference:
                ref_lower = node.reference.lower()
                if ref_lower in entity_tracker:
                    ref_node_id = entity_tracker[ref_lower]
                    graph.add_edge(SpanEdge(
                        source_id=ref_node_id,
                        target_id=node_id,
                        edge_type="comparison",
                        weight=1.0
                    ))

            # Link via pronoun resolution
            if node.entity and node.entity.lower() in self.PRONOUNS:
                # Find most recent non-pronoun entity
                for prev_idx in range(i - 1, -1, -1):
                    prev_node = graph.nodes[node_ids[prev_idx]]
                    if prev_node.entity and prev_node.entity.lower() not in self.PRONOUNS:
                        graph.add_edge(SpanEdge(
                            source_id=node_ids[prev_idx],
                            target_id=node_id,
                            edge_type="entity_ref",
                            weight=1.0
                        ))
                        break


class SpanGraphExecutor:
    """Execute a SpanGraph to compute the answer.

    Follows dependency order, resolving references as we go.
    Supports multi-track accounting for cost vs value problems.
    """

    def execute(self, graph: SpanGraph) -> Tuple[float, Dict[str, float]]:
        """Execute the graph and return (answer, state).

        State maps entity names to their current values.
        For multi-track problems, internally tracks cost and value separately.
        """
        # Check if this is a multi-track problem
        is_multi_track = any(
            node.track in (Track.COST, Track.VALUE, Track.BOTH)
            for node in graph.nodes.values()
        )

        if is_multi_track:
            return self._execute_multi_track(graph)
        else:
            return self._execute_single_track(graph)

    def _execute_single_track(self, graph: SpanGraph) -> Tuple[float, Dict[str, float]]:
        """Execute single-track problems (most common case)."""
        state: Dict[str, float] = {}

        # Build dependency-aware execution order
        order = self._dependency_order(graph)

        main_entity = None
        accumulator_entity = None

        for node_id in order:
            node = graph.nodes[node_id]

            # Track first entity as main
            if main_entity is None and node.entity and node.entity != "X":
                main_entity = node.entity
                accumulator_entity = node.entity

            entity = node.entity or accumulator_entity or "X"

            # Initialize entity if needed
            if entity not in state:
                state[entity] = 0.0

            # Get value
            value = node.value if node.value is not None else 0.0

            # Handle reference operations
            ref_entity = node.reference.capitalize() if node.reference else None
            if ref_entity and ref_entity.lower() != entity.lower():
                ref_key = None
                for key in state:
                    if key.lower() == ref_entity.lower():
                        ref_key = key
                        break

                if ref_key and ref_key in state:
                    ref_value = state[ref_key]
                    if '*' in node.dsl_expr:
                        multiplier = value if value else 2.0
                        state[entity] = ref_value * multiplier
                        continue
                    elif '+' in node.dsl_expr:
                        state[entity] = ref_value + value
                        continue
                    elif '-' in node.dsl_expr:
                        state[entity] = ref_value - value
                        continue
                    elif node.dsl_expr == "ref":
                        if value and value < 1.0:
                            state[entity] = state[entity] + (ref_value * value)
                        else:
                            state[entity] = ref_value
                        continue

            # Handle "_prev" reference
            if node.reference == "_prev" and '+' in node.dsl_expr:
                prev_value = state[entity]
                if value and value != 0:
                    state[entity] = prev_value + (prev_value * value)
                continue

            # Handle "_pct_increase"
            if node.reference == "_pct_increase" and '*' in node.dsl_expr:
                if value and value != 0:
                    state[entity] = state[entity] * value
                continue

            # Standard operations based on DSL expression
            if node.dsl_expr == "value":
                state[entity] = value
            elif '+' in node.dsl_expr:
                state[entity] = state[entity] + value
            elif '-' in node.dsl_expr:
                state[entity] = state[entity] - value
            elif '*' in node.dsl_expr:
                if value and state[entity]:
                    state[entity] = state[entity] * value
                elif value:
                    state[entity] = value
            elif '/' in node.dsl_expr:
                if value and value != 0:
                    state[entity] = state[entity] / value

        # Determine answer
        if len(state) == 1:
            answer = list(state.values())[0]
        elif main_entity and main_entity in state:
            unique_values = len(set(state.values()))
            if unique_values == len(state) and len(state) > 1:
                answer = sum(state.values())
            else:
                answer = state[main_entity]
        else:
            answer = sum(state.values()) if state else 0.0

        return answer, state

    def _execute_multi_track(self, graph: SpanGraph) -> Tuple[float, Dict[str, float]]:
        """Execute multi-track problems (cost vs value accounting).

        Maintains separate cost and value tracks:
        - COST: Money spent (buying, repairs)
        - VALUE: Asset value (initial + appreciation)
        - profit = value - cost
        """
        # Multi-track state: {entity: {"cost": X, "value": Y}}
        state: Dict[str, Dict[str, float]] = {}

        order = self._dependency_order(graph)

        main_entity = None

        for node_id in order:
            node = graph.nodes[node_id]

            if main_entity is None and node.entity and node.entity != "X":
                main_entity = node.entity

            entity = node.entity or main_entity or "X"

            # Initialize entity with both tracks
            if entity not in state:
                state[entity] = {"cost": 0.0, "value": 0.0}

            value = node.value if node.value is not None else 0.0
            track = node.track

            # Handle "_pct_increase" - applies to VALUE track only
            if node.reference == "_pct_increase" and '*' in node.dsl_expr:
                if value and value != 0:
                    # Apply percentage increase to VALUE (not cost)
                    state[entity]["value"] = state[entity]["value"] * value
                continue

            # Determine which track(s) to modify
            if track == Track.BOTH:
                # Initial purchase: sets both cost basis and initial value
                if node.dsl_expr == "value":
                    state[entity]["cost"] = value
                    state[entity]["value"] = value
                elif '+' in node.dsl_expr:
                    state[entity]["cost"] += value
                    state[entity]["value"] += value

            elif track == Track.COST:
                # Cost-only: repairs, investments
                if node.dsl_expr == "value":
                    state[entity]["cost"] = value
                elif '+' in node.dsl_expr:
                    state[entity]["cost"] += value
                elif '-' in node.dsl_expr:
                    state[entity]["cost"] -= value

            elif track == Track.VALUE:
                # Value-only: appreciation
                if node.dsl_expr == "value":
                    state[entity]["value"] = value
                elif '+' in node.dsl_expr:
                    state[entity]["value"] += value
                elif '*' in node.dsl_expr:
                    if value and state[entity]["value"]:
                        state[entity]["value"] *= value
                    elif value:
                        state[entity]["value"] = value

            else:
                # DEFAULT track - apply to both (single-track behavior)
                if node.dsl_expr == "value":
                    state[entity]["cost"] = value
                    state[entity]["value"] = value
                elif '+' in node.dsl_expr:
                    state[entity]["cost"] += value
                    state[entity]["value"] += value
                elif '-' in node.dsl_expr:
                    state[entity]["cost"] -= value
                    state[entity]["value"] -= value
                elif '*' in node.dsl_expr:
                    if value:
                        state[entity]["cost"] *= value
                        state[entity]["value"] *= value

        # Compute derived value: profit = value - cost
        if main_entity and main_entity in state:
            profit = state[main_entity]["value"] - state[main_entity]["cost"]
            answer = profit
        else:
            # Sum profits across all entities
            answer = sum(s["value"] - s["cost"] for s in state.values())

        # Flatten state for return (show both tracks + profit)
        flat_state = {}
        for entity, tracks in state.items():
            flat_state[f"{entity}_cost"] = tracks["cost"]
            flat_state[f"{entity}_value"] = tracks["value"]
            flat_state[f"{entity}_profit"] = tracks["value"] - tracks["cost"]

        return answer, flat_state

    def _dependency_order(self, graph: SpanGraph) -> List[str]:
        """Order nodes so references are resolved before dependents."""
        # Find all referenced entities and their defining spans
        entity_to_span = {}
        for node_id, node in graph.nodes.items():
            if node.entity:
                entity_to_span[node.entity.lower()] = node_id

        # Build dependency graph
        deps = {nid: set() for nid in graph.nodes}
        for node_id, node in graph.nodes.items():
            if node.reference:
                ref_lower = node.reference.lower()
                if ref_lower in entity_to_span:
                    deps[node_id].add(entity_to_span[ref_lower])

        # Topological sort with dependencies
        visited = set()
        order = []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for dep in deps[nid]:
                visit(dep)
            order.append(nid)

        # Start with nodes that have no dependencies
        for nid in graph.nodes:
            if not deps[nid]:
                visit(nid)
        # Then visit remaining
        for nid in graph.nodes:
            visit(nid)

        return order


def solve_with_graph(problem: str) -> Tuple[float, SpanGraph, Dict[str, float]]:
    """Solve a problem using span graph composition.

    Returns: (answer, graph, final_state)
    """
    builder = SpanGraphBuilder()
    graph = builder.build_graph(problem)

    executor = SpanGraphExecutor()
    answer, state = executor.execute(graph)

    return answer, graph, state


def debug_graph(problem: str):
    """Debug helper to visualize graph construction."""
    print("=" * 60)
    print("PROBLEM:", problem[:80], "..." if len(problem) > 80 else "")
    print("=" * 60)

    answer, graph, state = solve_with_graph(problem)

    # Check if multi-track
    is_multi_track = any(
        node.track in (Track.COST, Track.VALUE, Track.BOTH)
        for node in graph.nodes.values()
    )

    print("\nSPAN NODES:")
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        ref_str = f" (ref: {node.reference})" if node.reference else ""
        track_str = f" [{node.track.value}]" if node.track != Track.DEFAULT else ""
        print(f"  [{node_id}] {node.dsl_expr:20s} | {node.entity or 'X':12s} | "
              f"val={node.value}{ref_str}{track_str}")
        print(f"           \"{node.text[:50]}...\"" if len(node.text) > 50 else
              f"           \"{node.text}\"")

    print("\nEDGES:")
    for edge in graph.edges:
        print(f"  {edge.source_id} --[{edge.edge_type}]--> {edge.target_id}")

    if is_multi_track:
        print("\nMULTI-TRACK MODE ENABLED")

    print(f"\nFINAL STATE: {state}")
    print(f"ANSWER: {answer}")

    return answer, graph, state
