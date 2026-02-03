"""Simple KNN + Linear Chain Pipeline for math word problems.

NO TREE. NO MCTS. NO GPU REQUIRED. Just:
1. Segment span → extract reference, numbers
2. Get span embedding (sentence-transformers, CPU-friendly)
3. KNN lookup → find nearest labeled spans
4. Welford z-score → confidence
5. If confident: return operation. If not: try top-3
6. Update Welford stats with result

Uses all-MiniLM-L6-v2 for embeddings (~80MB, runs on CPU).
"""

import os
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Database URL from environment
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://mycelium:MyceliumDB2024!@mycelium-db.co1sisksw74e.us-east-1.rds.amazonaws.com:5432/mycelium"
)


@dataclass
class Operation:
    """A detected operation from a span."""
    op_type: str  # SET, ADD, SUB, MUL, DIV
    value: float
    entity: Optional[str]  # Who this applies to
    confidence: float
    span_text: str


@dataclass
class PipelineResult:
    """Result of running the pipeline on a problem."""
    answer: float
    steps: List[Operation]
    state: Dict[str, float]  # Entity states after execution


def classify_by_verb(text: str) -> Optional[str]:
    """Classify operation by verb taxonomy as a backup classifier.

    Returns ADD, SUB, MUL, DIV, or None if no clear verb signal.
    Used when KNN is uncertain between ADD/SUB.
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    # Verb patterns from span_graph.py
    INCREASE_VERBS = frozenset([
        "more", "added", "gained", "found", "received", "bought", "collected",
        "earned", "got", "picked", "additional", "extra", "plus"
    ])
    DECREASE_VERBS = frozenset([
        "less", "fewer", "sold", "gave", "lost", "spent", "used", "ate",
        "removed", "took", "subtracted", "minus"
    ])
    MULTIPLY_VERBS = frozenset([
        "times", "twice", "double", "triple"
    ])
    DIVIDE_VERBS = frozenset([
        "split", "divided", "shared"
    ])

    # Check for verb matches
    increase_count = len(words & INCREASE_VERBS)
    decrease_count = len(words & DECREASE_VERBS)
    multiply_count = len(words & MULTIPLY_VERBS)
    divide_count = len(words & DIVIDE_VERBS)

    # Return operation if there's a clear winner
    counts = [
        (increase_count, "ADD"),
        (decrease_count, "SUB"),
        (multiply_count, "MUL"),
        (divide_count, "DIV"),
    ]
    counts.sort(reverse=True)

    # Only return if there's a clear signal (at least 1 match and no tie)
    if counts[0][0] > 0 and counts[0][0] > counts[1][0]:
        return counts[0][1]

    return None


class SimplePipeline:
    """Simple KNN + Linear Chain pipeline with two-tier template matching.

    Two-tier KNN architecture:
    - Tier 1: Template centroids (clustered patterns) with 2x weight
    - Tier 2: Raw labeled spans with 1x weight

    Templates are "gold standard" anchors computed from clustering similar spans.
    """

    # Common pronouns that should trigger entity resolution
    PRONOUNS = {"she", "he", "they", "it", "her", "him", "them"}
    ENTITY_RESOLUTION_THRESHOLD = 0.7  # Cosine similarity threshold for resolution
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # CPU-friendly, ~80MB

    # Two-tier KNN weights
    TEMPLATE_WEIGHT = 2.0  # Templates count 2x vs individual spans
    TEMPLATE_CONFIDENCE_BONUS = 0.10  # Confidence boost for template matches

    # Position-based priors (position 0/1 = initial state, almost always SET)
    POSITION_PRIORS = {
        1: {"SET": 0.85, "ADD": 0.05, "SUB": 0.05, "MUL": 0.025, "DIV": 0.025},
        2: {"SET": 0.10, "ADD": 0.35, "SUB": 0.35, "MUL": 0.10, "DIV": 0.10},
        3: {"SET": 0.05, "ADD": 0.35, "SUB": 0.35, "MUL": 0.125, "DIV": 0.125},
    }

    # Positional prefixes for embedding context
    POSITION_PREFIXES = {
        1: "Initially:",
        2: "Then:",
        3: "After that:",
    }

    def __init__(self, use_db: bool = True):
        self.use_db = use_db
        self._segmenter = None
        self._embedding_model = None  # Lazy-loaded sentence-transformers model
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._labeled_spans: List[Tuple[str, str, np.ndarray]] = []  # (text, op, embedding)

        # Template centroids for two-tier KNN
        self._template_centroids: List[Tuple[str, str, np.ndarray]] = []  # (template_id, op, centroid)

        # Welford stats (loaded from DB or defaults)
        self._op_stats: Dict[str, Dict] = {}  # op -> {count, mean, m2}

        if use_db:
            self._load_from_db()

    def _load_from_db(self):
        """Load templates, labeled spans, and Welford stats from database."""
        try:
            from mycelium.db import (
                get_labeled_spans, get_all_welford_stats, get_embedding,
                get_template_centroids
            )

            # Load Welford stats
            for stat in get_all_welford_stats():
                if stat.stat_type.startswith("op_") and "_confidence" in stat.stat_type:
                    op = stat.stat_type.replace("op_", "").replace("_confidence", "")
                    self._op_stats[op] = {
                        "count": stat.count,
                        "mean": stat.mean,
                        "m2": stat.m2
                    }

            # Load template centroids (Tier 1 - gold standard anchors)
            self._template_centroids = get_template_centroids()

            # Load labeled spans with embeddings (Tier 2 - raw data)
            spans = get_labeled_spans(limit=2000)
            for span in spans:
                if span.operation:
                    # Try to get cached embedding (using sentence-transformers model)
                    emb = get_embedding(span.span_text, self.EMBEDDING_MODEL)
                    if emb is not None:
                        # Embeddings are already normalized
                        self._labeled_spans.append((span.span_text, span.operation, emb))

            print(f"Loaded {len(self._template_centroids)} templates, {len(self._labeled_spans)} spans, {len(self._op_stats)} op stats")
        except Exception as e:
            print(f"Could not load from DB: {e}")

    def _ensure_segmenter(self):
        """Lazy load the segmenter (CPU-only, no GPU required)."""
        if self._segmenter is None:
            from mycelium.simple_segmenter import SimpleSegmenter
            self._segmenter = SimpleSegmenter()
        return self._segmenter

    def _ensure_embedding_model(self):
        """Lazy load the sentence-transformers model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
        return self._embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using sentence-transformers (L2-normalized)."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        model = self._ensure_embedding_model()

        # Get embedding (sentence-transformers returns normalized by default)
        embedding = model.encode(text, convert_to_numpy=True)

        # Ensure normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding = embedding.astype(np.float32)
        self._embeddings_cache[text] = embedding

        # Store in DB if available
        if self.use_db:
            try:
                from mycelium.db import store_embedding
                store_embedding(text, embedding, self.EMBEDDING_MODEL)
            except:
                pass

        return embedding

    def _get_positional_embedding(self, text: str, position: int) -> np.ndarray:
        """Get embedding with positional context.

        Adds positional prefix to disambiguate operations by their position
        in the problem (e.g., "Initially: has 12" vs "Then: sold 5").
        """
        prefix = self.POSITION_PREFIXES.get(min(position, 3), "Then:")
        positioned_text = f"{prefix} {text}"
        return self._get_embedding(positioned_text)

    def _resolve_entity(self, text: str, known_entities: Dict[str, np.ndarray]) -> Optional[str]:
        """Resolve an entity (especially pronouns) to a known entity using embedding similarity.

        Returns the matching entity name if similarity > threshold, else None.
        """
        if not known_entities:
            return None

        # Get embedding for the candidate entity
        entity_emb = self._get_embedding(text)

        # Compare to known entities
        best_match = None
        best_sim = 0.0
        for entity_name, entity_emb_stored in known_entities.items():
            sim = float(np.dot(entity_emb, entity_emb_stored))
            if sim > best_sim:
                best_sim = sim
                best_match = entity_name

        # Return match if above threshold
        if best_sim >= self.ENTITY_RESOLUTION_THRESHOLD:
            return best_match
        return None

    def _knn_lookup(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[str, str, float]]:
        """Find k nearest labeled spans by cosine similarity.

        Note: Both embedding and labeled span embeddings are L2-normalized,
        so cosine similarity = dot product.
        """
        if not self._labeled_spans:
            return []

        # Compute similarities (dot product since normalized)
        similarities = []
        for text, op, emb in self._labeled_spans:
            sim = float(np.dot(embedding, emb))
            similarities.append((text, op, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:k]

    def _two_tier_knn_lookup(
        self,
        embedding: np.ndarray,
        k: int = 7
    ) -> List[Tuple[str, str, float, bool]]:
        """Two-tier KNN: templates first (2x weight), then raw spans.

        Like multi-object detection: templates are "gold standard" anchors,
        raw spans provide additional signal.

        Returns: List of (source_id, operation, weighted_similarity, is_template)
        """
        results = []

        # Tier 1: Template centroids (gold standard, 2x weight)
        for template_id, op, centroid in self._template_centroids:
            sim = float(np.dot(embedding, centroid))
            # Templates get weighted similarity boost
            weighted_sim = sim * self.TEMPLATE_WEIGHT
            results.append((template_id, op, weighted_sim, True))

        # Tier 2: Raw labeled spans (1x weight)
        for text, op, emb in self._labeled_spans:
            sim = float(np.dot(embedding, emb))
            results.append((text, op, sim, False))

        # Sort by weighted similarity descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def _get_position_prior(self, position: int) -> Dict[str, float]:
        """Get position-based prior probability for operations.

        Position 1 (first clause) is almost always SET (establishing initial state).
        Later positions are more likely to be modifications (ADD/SUB).
        """
        if position in self.POSITION_PRIORS:
            return self.POSITION_PRIORS[position].copy()
        # Default for position > 3
        return {"SET": 0.05, "ADD": 0.35, "SUB": 0.35, "MUL": 0.125, "DIV": 0.125}

    def _welford_zscore(self, op: str, similarity: float) -> float:
        """Get z-score for this similarity given operation's distribution."""
        if op not in self._op_stats or self._op_stats[op]["count"] < 2:
            return 0.0  # No data, neutral confidence

        stats = self._op_stats[op]
        variance = stats["m2"] / (stats["count"] - 1) if stats["count"] > 1 else 0
        std = math.sqrt(variance) if variance > 0 else 1e-8

        return (similarity - stats["mean"]) / std

    def _update_welford(self, op: str, similarity: float):
        """Update Welford stats for operation."""
        if op not in self._op_stats:
            self._op_stats[op] = {"count": 0, "mean": 0.0, "m2": 0.0}

        stats = self._op_stats[op]
        stats["count"] += 1
        delta = similarity - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = similarity - stats["mean"]
        stats["m2"] += delta * delta2

        # Update DB if available
        if self.use_db:
            try:
                from mycelium.db import update_welford_stats
                update_welford_stats(f"op_{op}_confidence", similarity)
            except:
                pass

    def classify_span(self, span_text: str, position: int = 1) -> Tuple[str, float]:
        """Classify a span into an operation type using two-tier KNN.

        Uses templates (gold standard anchors) + raw spans + position priors
        to classify operations. Like multi-object detection where each span
        is classified independently.

        Args:
            span_text: The text of the span to classify
            position: Position in the problem (1=first, 2=second, etc.)
                     Position 1 has strong prior for SET.

        Returns (operation, confidence).
        """
        # Use positional embedding for better disambiguation
        embedding = self._get_positional_embedding(span_text, position)

        # Two-tier lookup: templates (2x) + raw spans (1x)
        neighbors = self._two_tier_knn_lookup(embedding, k=7)

        if not neighbors:
            # No neighbors - use position prior only
            priors = self._get_position_prior(position)
            best_op = max(priors, key=priors.get)
            return (best_op, priors[best_op])

        # Score each operation: combine KNN votes + position prior
        priors = self._get_position_prior(position)
        op_scores: Dict[str, float] = {op: prior * 0.3 for op, prior in priors.items()}  # Prior weight
        template_matches: Dict[str, int] = {}

        for source_id, op, weighted_sim, is_template in neighbors:
            if op not in op_scores:
                op_scores[op] = 0.0
            # Add weighted similarity (templates already have 2x weight)
            op_scores[op] += weighted_sim

            # Track template matches
            if is_template:
                template_matches[op] = template_matches.get(op, 0) + 1

        # Get best operation
        best_op = max(op_scores, key=op_scores.get)
        best_score = op_scores[best_op]

        # Verb backup for ADD/SUB confusion
        sorted_ops = sorted(op_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_ops) >= 2:
            second_op, second_score = sorted_ops[1]
            # If top-2 are ADD/SUB and scores are close, use verb backup
            if {best_op, second_op} == {"ADD", "SUB"} and (best_score - second_score) < 0.1:
                verb_result = classify_by_verb(span_text)
                if verb_result:
                    best_op = verb_result
                    best_score = max(best_score, second_score)

        # Compute raw similarity for Welford (unweighted)
        raw_sim = best_score / self.TEMPLATE_WEIGHT if best_op in template_matches else best_score

        # Confidence from Welford z-score
        zscore = self._welford_zscore(best_op, raw_sim)
        confidence = 1 / (1 + math.exp(-zscore))  # Sigmoid to [0, 1]

        # Bonus confidence if matched a template
        if best_op in template_matches:
            confidence = min(1.0, confidence + self.TEMPLATE_CONFIDENCE_BONUS)

        return (best_op, confidence)

    def segment_problem(self, problem: str) -> List[Dict]:
        """Segment a problem into spans with extracted info."""
        segmenter = self._ensure_segmenter()

        # For now, split on periods/conjunctions as simple heuristic
        # TODO: Use attention-based clause detection
        import re
        clauses = re.split(r'[.!?]|\band\b|\bthen\b', problem)
        clauses = [c.strip() for c in clauses if c.strip()]

        results = []
        for clause in clauses:
            result = segmenter.segment(clause)
            results.append({
                "text": clause,
                "numbers": result.numbers,
                "reference": result.reference_entity,
                "segments": result.segments,
            })

        return results

    def solve(self, problem: str) -> PipelineResult:
        """Solve a math word problem using simple linear chain."""
        # 1. Segment into clauses
        clauses = self.segment_problem(problem)

        # 2. Process each clause
        steps = []
        state: Dict[str, float] = {}
        main_entity = None  # Most recent named entity (for pronoun resolution)
        position = 0  # Track position for positional embedding

        for clause in clauses:
            text = clause["text"]
            numbers = clause["numbers"]
            reference = clause["reference"]

            # Skip if no numbers (probably question clause)
            if not numbers:
                continue

            # Increment position counter (1-indexed for operations with numbers)
            position += 1

            # 3. Classify operation with positional context
            op_type, confidence = self.classify_span(text, position=position)

            # 4. Determine entity
            # First clause usually establishes main entity
            # Extract subject from first segment
            if clause["segments"]:
                first_seg = clause["segments"][0]
                if first_seg.segment_type == "operation":
                    entity = first_seg.text.split()[0] if first_seg.text else "X"
                else:
                    entity = first_seg.text
            else:
                entity = "X"

            # 5. Entity resolution: pronouns resolve to most recent named entity
            # This is simpler and more reliable than embedding-based resolution
            if entity.lower() in self.PRONOUNS and main_entity:
                entity = main_entity

            # Track most recent named entity (for pronoun resolution)
            if entity.lower() not in self.PRONOUNS:
                main_entity = entity

            # Use reference if this is a relative operation
            if reference:
                # "Lisa has 5 more than John" → Lisa's value depends on John
                if reference in state:
                    # Relative to reference
                    pass  # Will handle in execution

            # 6. Create operation
            op = Operation(
                op_type=op_type,
                value=numbers[0] if numbers else 0,
                entity=entity,
                confidence=confidence,
                span_text=text,
            )
            steps.append(op)

            # 7. Execute operation (linear chain)
            if entity not in state:
                state[entity] = 0

            if op_type == "SET":
                state[entity] = op.value
            elif op_type == "ADD":
                if reference and reference in state:
                    state[entity] = state[reference] + op.value
                else:
                    state[entity] += op.value
            elif op_type == "SUB":
                if reference and reference in state:
                    state[entity] = state[reference] - op.value
                else:
                    state[entity] -= op.value
            elif op_type == "MUL":
                if reference and reference in state:
                    state[entity] = state[reference] * op.value
                else:
                    state[entity] *= op.value
            elif op_type == "DIV":
                if reference and reference in state:
                    state[entity] = state[reference] / op.value if op.value != 0 else 0
                else:
                    state[entity] /= op.value if op.value != 0 else 1

        # Return answer (main entity's final value)
        answer = state.get(main_entity, 0) if main_entity else 0

        return PipelineResult(
            answer=answer,
            steps=steps,
            state=state,
        )

    def update_from_result(self, problem: str, correct: bool, steps: List[Operation]):
        """Update Welford stats based on result."""
        for step in steps:
            # Update confidence stats
            if correct:
                # Reinforce this classification
                self._update_welford(step.op_type, step.confidence + 0.1)
            else:
                # Weaken this classification
                self._update_welford(step.op_type, step.confidence - 0.1)

        # Log to DB
        if self.use_db:
            try:
                from mycelium.db import add_problem_result
                add_problem_result(
                    problem_id=str(hash(problem))[:16],
                    correct=correct,
                    predicted_answer=str(steps[-1].value if steps else 0),
                    actual_answer="",
                    problem_text=problem,
                    dag_steps=[{"op": s.op_type, "value": s.value, "entity": s.entity} for s in steps]
                )
            except:
                pass


def test_pipeline():
    """Test the simple pipeline."""
    print("=== Simple Pipeline Test ===\n")

    pipeline = SimplePipeline(use_db=True)

    test_problems = [
        "Lisa has 12 apples. She sold 5 apples.",
        "Tom had 8 coins. He found 3 more coins.",
        "Mary has 10 books. She gave 4 to John.",
    ]

    for problem in test_problems:
        print(f"Problem: {problem}")
        result = pipeline.solve(problem)
        print(f"  Answer: {result.answer}")
        print(f"  State: {result.state}")
        print(f"  Steps:")
        for step in result.steps:
            print(f"    {step.entity} {step.op_type} {step.value} (conf={step.confidence:.2f})")
        print()


def test_verb_classifier():
    """Test the verb taxonomy classifier."""
    print("=== Verb Classifier Test ===\n")

    test_cases = [
        # (text, expected_op)
        ("She sold 5 apples", "SUB"),
        ("He found 3 more coins", "ADD"),
        ("She gave 4 to John", "SUB"),
        ("Tom bought 6 oranges", "ADD"),
        ("Mary lost 2 pencils", "SUB"),
        ("He received 10 dollars", "ADD"),
        ("She spent 8 dollars", "SUB"),
        ("They collected 15 stamps", "ADD"),
        ("He ate 3 cookies", "SUB"),
        ("She earned 50 dollars", "ADD"),
    ]

    passed = 0
    for text, expected in test_cases:
        result = classify_by_verb(text)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  [{status}] '{text}' -> {result} (expected {expected})")

    print(f"\nVerb classifier: {passed}/{len(test_cases)} passed\n")


def test_positional_embedding():
    """Test that positional prefixes are applied correctly."""
    print("=== Positional Embedding Test ===\n")

    pipeline = SimplePipeline(use_db=False)

    # Test that different positions produce different embeddings
    text = "sold 5 apples"
    emb1 = pipeline._get_positional_embedding(text, position=1)
    emb2 = pipeline._get_positional_embedding(text, position=2)
    emb3 = pipeline._get_positional_embedding(text, position=3)

    # Embeddings should be different due to prefixes
    sim_1_2 = float(np.dot(emb1, emb2))
    sim_1_3 = float(np.dot(emb1, emb3))
    sim_2_3 = float(np.dot(emb2, emb3))

    print(f"  Position 1 vs 2 similarity: {sim_1_2:.4f}")
    print(f"  Position 1 vs 3 similarity: {sim_1_3:.4f}")
    print(f"  Position 2 vs 3 similarity: {sim_2_3:.4f}")

    # All should be similar but not identical (prefixes add context)
    assert sim_1_2 < 1.0, "Position 1 and 2 embeddings should differ"
    assert sim_1_3 < 1.0, "Position 1 and 3 embeddings should differ"
    print("\n  [PASS] Positional embeddings are distinct\n")


def test_add_sub_disambiguation():
    """Test that ADD/SUB confusion is resolved by verb backup."""
    print("=== ADD/SUB Disambiguation Test ===\n")

    # Problems that commonly confuse ADD/SUB
    test_cases = [
        # (problem, step_index, expected_op)
        ("Lisa has 12 apples. She sold 5 apples.", 1, "SUB"),  # sold -> SUB
        ("Tom had 8 coins. He found 3 more coins.", 1, "ADD"),  # found -> ADD
        ("Mary has 10 books. She gave 4 to John.", 1, "SUB"),  # gave -> SUB
        ("John had 20 dollars. He spent 8 on lunch.", 1, "SUB"),  # spent -> SUB
        ("Sarah has 5 stickers. She received 7 more.", 1, "ADD"),  # received -> ADD
    ]

    pipeline = SimplePipeline(use_db=False)

    passed = 0
    for problem, step_idx, expected in test_cases:
        result = pipeline.solve(problem)
        if len(result.steps) > step_idx:
            actual = result.steps[step_idx].op_type
            status = "PASS" if actual == expected else "FAIL"
            if actual == expected:
                passed += 1
            print(f"  [{status}] '{problem}'")
            print(f"          Step {step_idx}: {actual} (expected {expected})")
        else:
            print(f"  [SKIP] '{problem}' - not enough steps")

    print(f"\nADD/SUB disambiguation: {passed}/{len(test_cases)} passed\n")


if __name__ == "__main__":
    test_verb_classifier()
    test_positional_embedding()
    test_pipeline()
    test_add_sub_disambiguation()
