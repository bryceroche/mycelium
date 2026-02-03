"""Span Graph: Lightweight graph structure for span relationships.

Represents how spans in a math word problem reference and modify each other.
Works with sentence-transformers embeddings for CPU-friendly inference.

Key insight: The RELATIONSHIP between spans contains classification signal.
- "She sold 5" → modifies(She, Lisa) + operation → likely SUB
- "John has 3 more than Lisa" → references(John, Lisa) + comparative → likely ADD

This helps disambiguate ADD vs SUB by looking at the relationship context.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class RelationType(Enum):
    """Types of relationships between spans/entities."""
    DEFINES = "DEFINES"          # "Lisa has 12 apples" → defines Lisa's state
    MODIFIES = "MODIFIES"        # "She sold 5" → modifies (reduces) Lisa's state
    REFERENCES = "REFERENCES"    # "John has 3 more than Lisa" → references Lisa's value
    COMPARES = "COMPARES"        # "twice as many as" → comparative relationship
    TEMPORAL = "TEMPORAL"        # "then", "after" → temporal ordering


class OpHint(Enum):
    """Hints for operation classification derived from relationships."""
    SET = "SET"           # First mention, no prior state
    INCREASE = "INCREASE" # More, adds, gains, etc.
    DECREASE = "DECREASE" # Less, loses, sells, etc.
    MULTIPLY = "MULTIPLY" # Times, twice, etc.
    DIVIDE = "DIVIDE"     # Split, each, per
    UNKNOWN = "UNKNOWN"


@dataclass
class SpanNode:
    """A semantic span in the problem text.

    Captures:
    - The text and its embedding (for similarity matching)
    - Position in problem (for temporal ordering)
    - Entity this span relates to
    - What entities this span references
    - Operation hint from relationship analysis
    """
    text: str
    position: int                    # Sentence/clause index in problem
    embedding: Optional[np.ndarray] = None  # sentence-transformers embedding

    # Entity information
    entity: Optional[str] = None     # Primary entity this span is about
    value: Optional[float] = None    # Numeric value if present

    # Relationship information
    references: List[str] = field(default_factory=list)  # Entities this span references
    relation_type: RelationType = RelationType.DEFINES

    # Classification hints
    op_hint: OpHint = OpHint.UNKNOWN
    op_confidence: float = 0.0

    def __post_init__(self):
        if self.references is None:
            self.references = []


@dataclass
class SpanEdge:
    """An edge connecting two spans with a relationship."""
    source: str           # Source span text (key)
    target: str           # Target span text (key)
    relation: RelationType
    weight: float = 1.0   # Relationship strength (embedding similarity)


@dataclass
class EntityState:
    """Tracks an entity's state through the problem."""
    name: str
    initial_value: Optional[float] = None
    current_value: Optional[float] = None
    defining_span: Optional[str] = None  # Span text that first defined this entity
    modifying_spans: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.modifying_spans is None:
            self.modifying_spans = []


class SpanGraph:
    """Graph of span relationships for a math word problem.

    Lightweight design:
    - Uses sentence-transformers for embeddings (CPU-friendly)
    - Stores spans as nodes, relationships as edges
    - Tracks entity states for dependency resolution
    - Provides operation hints based on relationship context
    """

    # Import canonical patterns from verb_classifier (single source of truth)
    from mycelium.verb_classifier import (
        ADD_PATTERNS as INCREASE_PATTERNS,
        SUB_PATTERNS as DECREASE_PATTERNS,
        MUL_PATTERNS as MULTIPLY_PATTERNS,
        DIV_PATTERNS as DIVIDE_PATTERNS,
        REFERENCE_PATTERNS,
    )

    def __init__(self):
        self.nodes: Dict[str, SpanNode] = {}  # text -> node
        self.edges: List[SpanEdge] = []
        self.entities: Dict[str, EntityState] = {}  # name -> state
        self._embedding_model = None

    def _ensure_embedding_model(self):
        """Lazy load sentence-transformers model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2: ~80MB, runs on CPU, good quality
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get normalized embedding for text."""
        model = self._ensure_embedding_model()
        emb = model.encode(text, convert_to_numpy=True)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        import re
        numbers = []
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            numbers.append(float(match.group(1)))
        return numbers

    # Common pronouns for entity resolution
    PRONOUNS = frozenset([
        "she", "he", "they", "it", "her", "him", "them", "his", "hers", "its", "their"
    ])

    # Common non-entity words that might be capitalized at sentence start
    NON_ENTITY_WORDS = frozenset([
        "the", "a", "an", "this", "that", "these", "those", "there", "here",
        "what", "how", "when", "where", "why", "who", "which",
        "if", "then", "so", "but", "and", "or", "for", "to", "from"
    ])

    def _extract_entity(self, text: str) -> Optional[str]:
        """Extract primary entity from span text.

        Handles:
        - Proper nouns: "Lisa has 12 apples" → Lisa
        - Pronouns: "She sold 5" → She (will be resolved later)
        - First-word capitalization: Skip common words
        """
        import re
        words = text.split()
        if not words:
            return None

        # First, check if first word is a pronoun
        first_word_lower = words[0].lower()
        if first_word_lower in self.PRONOUNS:
            return words[0]  # Return pronoun for later resolution

        # Look for proper nouns (capitalized words)
        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word)
            if not clean:
                continue

            # Check if capitalized
            if clean[0].isupper():
                # Skip common words that might be capitalized at start
                if clean.lower() in self.NON_ENTITY_WORDS:
                    continue
                # Skip common verbs at sentence start
                if clean.lower() in {"has", "had", "is", "was", "are", "were", "do", "does", "did"}:
                    continue
                return clean

        return None

    def _detect_references(self, text: str) -> List[str]:
        """Detect entity references in span text.

        Looks for:
        - "than Lisa" → references Lisa
        - "as John" → references John
        - Pronouns that need resolution
        """
        import re
        references = []
        text_lower = text.lower()

        # Pattern: "than/as [Entity]"
        ref_pattern = r'\b(?:than|as)\s+([A-Z][a-z]+)'
        for match in re.finditer(ref_pattern, text):
            references.append(match.group(1))

        # Pattern: "[Entity]'s"
        poss_pattern = r"\b([A-Z][a-z]+)'s\b"
        for match in re.finditer(poss_pattern, text):
            references.append(match.group(1))

        return references

    def _infer_op_hint(self, text: str, has_reference: bool) -> Tuple[OpHint, float]:
        """Infer operation hint from text patterns.

        Key insight: "more than" vs "less than" + reference = ADD vs SUB
        But standalone "sold 5" with no reference = SUB from current state

        Returns (hint, confidence) where confidence is [0, 1].
        """
        text_lower = text.lower()
        words = set(text_lower.split())

        # Check for pattern matches
        increase_score = len(words & self.INCREASE_PATTERNS)
        decrease_score = len(words & self.DECREASE_PATTERNS)
        multiply_score = len(words & self.MULTIPLY_PATTERNS)
        divide_score = len(words & self.DIVIDE_PATTERNS)

        total = increase_score + decrease_score + multiply_score + divide_score

        if total == 0:
            # No strong signal - likely SET if no reference
            if not has_reference and self._extract_numbers(text):
                return (OpHint.SET, 0.6)
            return (OpHint.UNKNOWN, 0.0)

        # Determine strongest signal
        scores = [
            (OpHint.INCREASE, increase_score),
            (OpHint.DECREASE, decrease_score),
            (OpHint.MULTIPLY, multiply_score),
            (OpHint.DIVIDE, divide_score),
        ]
        best = max(scores, key=lambda x: x[1])
        confidence = best[1] / total if total > 0 else 0.0

        return (best[0], confidence)

    def _determine_relation_type(self, text: str, references: List[str],
                                  entity: Optional[str]) -> RelationType:
        """Determine the relationship type of this span."""
        has_reference = len(references) > 0

        # If references another entity → REFERENCES or COMPARES
        if has_reference:
            text_lower = text.lower()
            if "than" in text_lower or "as" in text_lower:
                return RelationType.COMPARES
            return RelationType.REFERENCES

        # If entity already exists → MODIFIES
        if entity and entity in self.entities:
            return RelationType.MODIFIES

        # First mention → DEFINES
        return RelationType.DEFINES

    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to the most recent entity.

        Simple heuristic: Most recent named entity that's not a pronoun.
        """
        if not self.entities:
            return None

        # Get entities sorted by their first mention position
        entity_positions = []
        for name, state in self.entities.items():
            if state.defining_span and state.defining_span in self.nodes:
                pos = self.nodes[state.defining_span].position
                # Skip if the entity itself is a pronoun
                if name.lower() not in self.PRONOUNS:
                    entity_positions.append((pos, name))

        if not entity_positions:
            return None

        # Return most recent (highest position)
        entity_positions.sort(key=lambda x: x[0], reverse=True)
        return entity_positions[0][1]

    def add_span(self, text: str, position: int) -> SpanNode:
        """Add a span to the graph.

        Automatically:
        - Computes embedding
        - Extracts entity and references
        - Resolves pronouns to prior entities
        - Infers operation hint
        - Creates edges to referenced entities
        """
        if text in self.nodes:
            return self.nodes[text]

        # Extract information
        numbers = self._extract_numbers(text)
        entity = self._extract_entity(text)
        references = self._detect_references(text)

        # Resolve pronouns
        resolved_entity = entity
        if entity and entity.lower() in self.PRONOUNS:
            resolved = self._resolve_pronoun(entity)
            if resolved:
                resolved_entity = resolved
                # Pronoun reference creates an implicit edge
                if resolved not in references:
                    references = references + [resolved]

        # Determine relationship type (use resolved entity for checking)
        relation_type = self._determine_relation_type(text, references, resolved_entity)

        # Infer operation hint
        op_hint, op_confidence = self._infer_op_hint(text, len(references) > 0)

        # Compute embedding
        embedding = self._get_embedding(text)

        # Create node (store resolved entity)
        node = SpanNode(
            text=text,
            position=position,
            embedding=embedding,
            entity=resolved_entity,
            value=numbers[0] if numbers else None,
            references=references,
            relation_type=relation_type,
            op_hint=op_hint,
            op_confidence=op_confidence,
        )
        self.nodes[text] = node

        # Update entity state (use resolved entity)
        if resolved_entity:
            if resolved_entity not in self.entities:
                self.entities[resolved_entity] = EntityState(
                    name=resolved_entity,
                    initial_value=node.value,
                    current_value=node.value,
                    defining_span=text,
                )
            else:
                self.entities[resolved_entity].modifying_spans.append(text)

        # Create edges to referenced entities
        for ref in references:
            if ref in self.entities:
                ref_span = self.entities[ref].defining_span
                if ref_span:
                    # Compute edge weight as embedding similarity
                    ref_emb = self.nodes[ref_span].embedding
                    weight = float(np.dot(embedding, ref_emb)) if ref_emb is not None else 0.5

                    self.edges.append(SpanEdge(
                        source=text,
                        target=ref_span,
                        relation=relation_type,
                        weight=weight,
                    ))

        return node

    def get_classification_context(self, span_text: str) -> Dict:
        """Get rich context for classifying a span's operation.

        Returns contextual features that help disambiguate ADD vs SUB:
        - Entity state (is this first mention or modification?)
        - References (does it depend on another entity?)
        - Relationship type
        - Operation hint from patterns
        - Embedding similarity to referenced entities
        """
        if span_text not in self.nodes:
            return {}

        node = self.nodes[span_text]

        context = {
            "entity": node.entity,
            "value": node.value,
            "position": node.position,
            "relation_type": node.relation_type.value,
            "op_hint": node.op_hint.value,
            "op_confidence": node.op_confidence,
            "references": node.references,
            "is_first_mention": (
                node.entity is not None and
                node.entity in self.entities and
                self.entities[node.entity].defining_span == span_text
            ),
        }

        # Add reference similarity if available
        if node.references and node.embedding is not None:
            ref_similarities = {}
            for ref in node.references:
                if ref in self.entities:
                    ref_span = self.entities[ref].defining_span
                    if ref_span and ref_span in self.nodes:
                        ref_emb = self.nodes[ref_span].embedding
                        if ref_emb is not None:
                            sim = float(np.dot(node.embedding, ref_emb))
                            ref_similarities[ref] = sim
            context["reference_similarities"] = ref_similarities

        return context

    def classify_operation(self, span_text: str) -> Tuple[str, float]:
        """Classify a span's operation using graph context.

        NOTE: Hard-coded rules REMOVED - dual-signal approach (attention + embeddings)
        will learn classification patterns. This method now just returns the op_hint
        as a weak baseline signal. Real classification happens in the dual-signal pipeline.

        Returns (operation, confidence) based on simple op_hint mapping.
        """
        context = self.get_classification_context(span_text)
        if not context:
            return ("SET", 0.3)

        op_hint = context.get("op_hint", "UNKNOWN")
        hint_confidence = context.get("op_confidence", 0.0)

        # Simple mapping from op_hint to operation (weak signal only)
        hint_to_op = {
            "INCREASE": "ADD",
            "DECREASE": "SUB",
            "MULTIPLY": "MUL",
            "DIVIDE": "DIV",
            "SET": "SET",
        }

        if op_hint in hint_to_op:
            return (hint_to_op[op_hint], 0.5 + hint_confidence * 0.3)

        # Default: SET with low confidence
        return ("SET", 0.3)

    def embed_relationship(self, source_text: str, target_text: str,
                           relation: RelationType) -> np.ndarray:
        """Embed a relationship between spans.

        Creates a composite embedding from:
        - Source span embedding
        - Target span embedding
        - Relationship type encoding

        This allows learning from relationship patterns:
        - "modifies(She, Lisa)" patterns cluster differently than
        - "references(John, Lisa)" patterns
        """
        if source_text not in self.nodes or target_text not in self.nodes:
            return np.zeros(384 * 2 + 5)  # all-MiniLM-L6-v2 dim + relation

        source_emb = self.nodes[source_text].embedding
        target_emb = self.nodes[target_text].embedding

        # Relation type one-hot encoding
        relation_encoding = np.zeros(5)
        relation_idx = list(RelationType).index(relation)
        relation_encoding[relation_idx] = 1.0

        # Concatenate: [source_emb, target_emb, relation_encoding]
        if source_emb is not None and target_emb is not None:
            return np.concatenate([source_emb, target_emb, relation_encoding])

        return np.zeros(384 * 2 + 5)

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary (without embeddings)."""
        return {
            "nodes": [
                {
                    "text": n.text,
                    "position": n.position,
                    "entity": n.entity,
                    "value": n.value,
                    "references": n.references,
                    "relation_type": n.relation_type.value,
                    "op_hint": n.op_hint.value,
                    "op_confidence": n.op_confidence,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation.value,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
            "entities": {
                name: {
                    "initial_value": state.initial_value,
                    "current_value": state.current_value,
                    "defining_span": state.defining_span,
                    "modifying_spans": state.modifying_spans,
                }
                for name, state in self.entities.items()
            },
        }


def build_span_graph(problem_text: str) -> SpanGraph:
    """Build a span graph from a math word problem.

    Steps:
    1. Split into sentences/clauses
    2. Add each as a span node
    3. Automatically detect relationships

    Example:
        problem = "Lisa has 12 apples. She sold 5. John has 3 more than Lisa."
        graph = build_span_graph(problem)

        # Now we can query relationships:
        context = graph.get_classification_context("She sold 5")
        # → {'entity': 'She', 'references': [], 'relation_type': 'MODIFIES', ...}

        context = graph.get_classification_context("John has 3 more than Lisa")
        # → {'entity': 'John', 'references': ['Lisa'], 'relation_type': 'COMPARES', ...}
    """
    import re

    graph = SpanGraph()

    # Split into sentences/clauses
    # Split on periods, exclamation, question marks, and "then"/"and" conjunctions
    clauses = re.split(r'[.!?]|\bthen\b|\band\b', problem_text)
    clauses = [c.strip() for c in clauses if c.strip()]

    # Add each clause as a span
    for i, clause in enumerate(clauses):
        graph.add_span(clause, position=i)

    return graph


# =============================================================================
# How relationships help with ADD vs SUB confusion
# =============================================================================
#
# The key insight: isolated spans like "sold 5" are ambiguous.
# But in graph context, the relationship provides signal:
#
# Case 1: "Lisa has 12 apples. She sold 5."
#   - "Lisa has 12 apples" → DEFINES Lisa, SET operation
#   - "She sold 5" → MODIFIES Lisa (pronoun resolution), "sold" → DECREASE → SUB
#
# Case 2: "John has 3 more than Lisa"
#   - REFERENCES Lisa + "more than" → INCREASE relative to Lisa → ADD
#
# Case 3: "Mary has 5 fewer books than Tom"
#   - REFERENCES Tom + "fewer than" → DECREASE relative to Tom → SUB
#
# The graph structure captures:
# 1. WHO the operation applies to (entity tracking)
# 2. WHAT prior state it depends on (references)
# 3. HOW it relates to that state (increase/decrease patterns)
#
# This context, combined with span embeddings, gives much stronger signal
# than trying to classify "sold 5" in isolation.


def demo():
    """Demonstrate span graph construction and classification."""
    print("=== Span Graph Demo ===\n")

    problems = [
        "Lisa has 12 apples. She sold 5.",
        "Tom has 8 coins. He found 3 more coins.",
        "Mary has 10 books. John has 3 more than Mary.",
        "Sarah had 15 stickers. She gave 4 to Mike. Then she lost 2.",
    ]

    for problem in problems:
        print(f"Problem: {problem}")
        print("-" * 50)

        graph = build_span_graph(problem)

        print("Nodes:")
        for text, node in graph.nodes.items():
            print(f"  '{text}'")
            print(f"    entity={node.entity}, value={node.value}")
            print(f"    relation={node.relation_type.value}, refs={node.references}")
            print(f"    op_hint={node.op_hint.value} (conf={node.op_confidence:.2f})")

            # Classify
            op, conf = graph.classify_operation(text)
            print(f"    → classified as {op} (conf={conf:.2f})")

        print("\nEdges:")
        for edge in graph.edges:
            print(f"  {edge.source[:30]}... → {edge.target[:30]}...")
            print(f"    relation={edge.relation.value}, weight={edge.weight:.3f}")

        print("\nEntities:")
        for name, state in graph.entities.items():
            print(f"  {name}: initial={state.initial_value}, modifiers={len(state.modifying_spans)}")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    demo()
