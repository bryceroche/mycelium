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


class SimplePipeline:
    """Simple KNN + Linear Chain pipeline."""

    # Common pronouns that should trigger entity resolution
    PRONOUNS = {"she", "he", "they", "it", "her", "him", "them"}
    ENTITY_RESOLUTION_THRESHOLD = 0.7  # Cosine similarity threshold for resolution
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # CPU-friendly, ~80MB

    def __init__(self, use_db: bool = True):
        self.use_db = use_db
        self._segmenter = None
        self._embedding_model = None  # Lazy-loaded sentence-transformers model
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._labeled_spans: List[Tuple[str, str, np.ndarray]] = []  # (text, op, embedding)

        # Welford stats (loaded from DB or defaults)
        self._op_stats: Dict[str, Dict] = {}  # op -> {count, mean, m2}

        if use_db:
            self._load_from_db()

    def _load_from_db(self):
        """Load labeled spans and Welford stats from database."""
        try:
            from mycelium.db import get_labeled_spans, get_all_welford_stats, get_embedding

            # Load Welford stats
            for stat in get_all_welford_stats():
                if stat.stat_type.startswith("op_") and "_confidence" in stat.stat_type:
                    op = stat.stat_type.replace("op_", "").replace("_confidence", "")
                    self._op_stats[op] = {
                        "count": stat.count,
                        "mean": stat.mean,
                        "m2": stat.m2
                    }

            # Load labeled spans with embeddings
            spans = get_labeled_spans(limit=2000)
            for span in spans:
                if span.operation:
                    # Try to get cached embedding (using sentence-transformers model)
                    emb = get_embedding(span.span_text, self.EMBEDDING_MODEL)
                    if emb is not None:
                        # Embeddings are already normalized
                        self._labeled_spans.append((span.span_text, span.operation, emb))

            print(f"Loaded {len(self._labeled_spans)} labeled spans, {len(self._op_stats)} op stats")
        except Exception as e:
            print(f"Could not load from DB: {e}")

    def _ensure_segmenter(self):
        """Lazy load the segmenter."""
        if self._segmenter is None:
            from mycelium.attention_segmenter import AttentionSegmenter
            self._segmenter = AttentionSegmenter()
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

    def classify_span(self, span_text: str) -> Tuple[str, float]:
        """Classify a span into an operation type.

        Returns (operation, confidence).
        """
        embedding = self._get_embedding(span_text)
        neighbors = self._knn_lookup(embedding, k=5)

        if not neighbors:
            return ("SET", 0.0)  # Default fallback

        # Vote among neighbors
        op_scores: Dict[str, List[float]] = {}
        for text, op, sim in neighbors:
            if op not in op_scores:
                op_scores[op] = []
            op_scores[op].append(sim)

        # Best operation by average similarity
        best_op = max(op_scores.keys(), key=lambda o: sum(op_scores[o]) / len(op_scores[o]))
        avg_sim = sum(op_scores[best_op]) / len(op_scores[best_op])

        # Confidence from Welford z-score (positive = better than average)
        zscore = self._welford_zscore(best_op, avg_sim)
        confidence = 1 / (1 + math.exp(-zscore))  # Sigmoid to [0, 1]

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

        for clause in clauses:
            text = clause["text"]
            numbers = clause["numbers"]
            reference = clause["reference"]

            # Skip if no numbers (probably question clause)
            if not numbers:
                continue

            # 3. Classify operation
            op_type, confidence = self.classify_span(text)

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


if __name__ == "__main__":
    test_pipeline()
