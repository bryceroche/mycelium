"""Pipeline for span extraction and graph construction.

Two-phase approach:
1. COLLECTION: Extract spans from problems using attention patterns
2. TAGGING: Label spans with operations (manual or LLM-assisted)
3. TRAINING: Use tagged spans to train tiny classifier

This module handles phase 1 - collecting spans for later tagging.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSpan:
    """A span extracted from problem text, pending operation tagging."""
    text: str                           # the span text (e.g., "eats three")
    start_idx: int                      # character start position
    end_idx: int                        # character end position
    tokens: List[str]                   # tokenized form
    token_indices: List[int]            # indices in full token list
    attention_score: float              # how strongly tokens attend to each other
    contains_number: bool               # whether span contains a numeric value
    number_value: Optional[float] = None  # extracted number if present

    # To be filled during tagging phase
    operation: Optional[str] = None     # SET, ADD, SUB, MUL, DIV, or None
    tagged_by: Optional[str] = None     # "manual", "llm", "regex"
    confidence: Optional[float] = None  # tagging confidence


@dataclass
class ProblemSpans:
    """All extracted spans from a single problem."""
    problem_id: str
    problem_text: str
    subject: str                        # detected subject/entity
    subject_token_idx: int              # token index of subject
    spans: List[ExtractedSpan] = field(default_factory=list)

    # Metadata
    ground_truth_answer: Optional[float] = None
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SpanCollector:
    """Collects spans from problems for later tagging."""

    def __init__(self, output_path: str = "collected_spans.jsonl"):
        self.output_path = Path(output_path)
        self.collected: List[ProblemSpans] = []

        # Number patterns for extraction
        self.number_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\b')
        self.word_numbers = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'half': 0.5,
            'twice': 2, 'double': 2, 'triple': 3,
        }

    def extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        # Try digit pattern first
        match = self.number_pattern.search(text)
        if match:
            return float(match.group(1))

        # Try word numbers
        text_lower = text.lower()
        for word, value in self.word_numbers.items():
            if word in text_lower:
                return value

        return None

    def extract_spans_simple(
        self,
        problem_id: str,
        problem_text: str,
        tokens: List[str],
        attention: 'np.ndarray',
        ground_truth: Optional[float] = None,
    ) -> ProblemSpans:
        """Extract spans using simple heuristics (no model needed for collection).

        Strategy: Find sentence segments containing numbers, these are likely operations.
        """
        import numpy as np

        # Find subject (highest attention sink)
        incoming_attn = attention.sum(axis=0)
        subject_idx = int(np.argmax(incoming_attn))
        subject = tokens[subject_idx].replace('Ġ', '').replace('▁', '')

        result = ProblemSpans(
            problem_id=problem_id,
            problem_text=problem_text,
            subject=subject,
            subject_token_idx=subject_idx,
            ground_truth_answer=ground_truth,
        )

        # Split into sentences/clauses
        clauses = re.split(r'[.!?;]|\band\b|\bthen\b', problem_text)

        char_pos = 0
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Find position in original text
            start_idx = problem_text.find(clause, char_pos)
            if start_idx == -1:
                start_idx = char_pos
            end_idx = start_idx + len(clause)
            char_pos = end_idx

            # Check if clause contains a number
            number = self.extract_number(clause)

            # Calculate attention score for this span
            # (simplified: use mean attention of tokens in span)
            span_tokens = []
            span_indices = []

            # Map clause back to tokens (approximate)
            clause_lower = clause.lower()
            for i, tok in enumerate(tokens):
                tok_clean = tok.replace('Ġ', '').replace('▁', '').lower()
                if tok_clean and tok_clean in clause_lower:
                    span_tokens.append(tok)
                    span_indices.append(i)

            if span_indices:
                # Attention score: mean pairwise attention within span
                span_attn = attention[np.ix_(span_indices, span_indices)]
                attn_score = float(np.mean(span_attn))
            else:
                attn_score = 0.0

            span = ExtractedSpan(
                text=clause,
                start_idx=start_idx,
                end_idx=end_idx,
                tokens=span_tokens,
                token_indices=span_indices,
                attention_score=attn_score,
                contains_number=number is not None,
                number_value=number,
            )
            result.spans.append(span)

        self.collected.append(result)
        return result

    def save(self):
        """Save collected spans to JSONL file."""
        with open(self.output_path, 'w') as f:
            for ps in self.collected:
                # Convert to dict, handling dataclasses
                data = {
                    "problem_id": ps.problem_id,
                    "problem_text": ps.problem_text,
                    "subject": ps.subject,
                    "subject_token_idx": ps.subject_token_idx,
                    "ground_truth_answer": ps.ground_truth_answer,
                    "extracted_at": ps.extracted_at,
                    "spans": [asdict(s) for s in ps.spans],
                }
                f.write(json.dumps(data) + '\n')

        logger.info(f"Saved {len(self.collected)} problems to {self.output_path}")

    def load(self) -> List[ProblemSpans]:
        """Load previously collected spans."""
        if not self.output_path.exists():
            return []

        results = []
        with open(self.output_path) as f:
            for line in f:
                data = json.loads(line)
                ps = ProblemSpans(
                    problem_id=data["problem_id"],
                    problem_text=data["problem_text"],
                    subject=data["subject"],
                    subject_token_idx=data["subject_token_idx"],
                    ground_truth_answer=data.get("ground_truth_answer"),
                    extracted_at=data.get("extracted_at", ""),
                    spans=[ExtractedSpan(**s) for s in data["spans"]],
                )
                results.append(ps)

        return results

    def stats(self) -> Dict:
        """Get statistics on collected spans."""
        total_spans = sum(len(ps.spans) for ps in self.collected)
        spans_with_numbers = sum(
            sum(1 for s in ps.spans if s.contains_number)
            for ps in self.collected
        )
        tagged_spans = sum(
            sum(1 for s in ps.spans if s.operation is not None)
            for ps in self.collected
        )

        return {
            "total_problems": len(self.collected),
            "total_spans": total_spans,
            "spans_with_numbers": spans_with_numbers,
            "tagged_spans": tagged_spans,
            "untagged_spans": total_spans - tagged_spans,
        }


class SpanClusterer:
    """Cluster spans by embedding similarity for efficient tagging.

    Instead of tagging 10,000 spans individually:
    1. Embed all spans
    2. Cluster into ~50-100 groups
    3. Tag ONE span per cluster
    4. Propagate labels to all spans in cluster

    This IS the tiny classifier - clusters become operation classes.
    """

    def __init__(self, n_clusters: int = 50):
        self.n_clusters = n_clusters
        self.embeddings: Dict[str, 'np.ndarray'] = {}  # span_text -> embedding
        self.cluster_labels: Dict[str, int] = {}       # span_text -> cluster_id
        self.cluster_operations: Dict[int, str] = {}   # cluster_id -> operation

    def embed_spans(self, spans: List[ExtractedSpan], embedder=None):
        """Embed all spans using the provided embedder."""
        import numpy as np

        if embedder is None:
            # Use default embedder
            from mycelium.embedder import get_embedding
            embedder = lambda x: get_embedding(x)

        for span in spans:
            if span.text not in self.embeddings:
                self.embeddings[span.text] = embedder(span.text)

        logger.info(f"Embedded {len(self.embeddings)} unique spans")

    def cluster(self):
        """Cluster embedded spans using k-means."""
        import numpy as np
        from sklearn.cluster import KMeans

        if not self.embeddings:
            raise ValueError("No embeddings to cluster. Call embed_spans first.")

        texts = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[t] for t in texts])

        # Adjust n_clusters if we have fewer spans
        n_clusters = min(self.n_clusters, len(texts))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        self.cluster_labels = {text: int(label) for text, label in zip(texts, labels)}
        logger.info(f"Clustered into {n_clusters} groups")

        return self._cluster_summary()

    def _cluster_summary(self) -> Dict[int, List[str]]:
        """Get summary of clusters (cluster_id -> list of span texts)."""
        clusters: Dict[int, List[str]] = {}
        for text, cluster_id in self.cluster_labels.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(text)
        return clusters

    def tag_cluster(self, cluster_id: int, operation: str):
        """Tag all spans in a cluster with an operation."""
        self.cluster_operations[cluster_id] = operation

    def propagate_tags(self, spans: List[ExtractedSpan]) -> List[ExtractedSpan]:
        """Propagate cluster tags to individual spans."""
        for span in spans:
            if span.text in self.cluster_labels:
                cluster_id = self.cluster_labels[span.text]
                if cluster_id in self.cluster_operations:
                    span.operation = self.cluster_operations[cluster_id]
                    span.tagged_by = "cluster"
                    span.confidence = 0.85
        return spans

    def get_representative_spans(self, n_per_cluster: int = 3) -> Dict[int, List[str]]:
        """Get representative spans from each cluster for manual review."""
        import numpy as np

        clusters = self._cluster_summary()
        representatives = {}

        for cluster_id, texts in clusters.items():
            # Get spans closest to cluster centroid
            if len(texts) <= n_per_cluster:
                representatives[cluster_id] = texts
            else:
                # Simple: just take first n (could do centroid distance)
                representatives[cluster_id] = texts[:n_per_cluster]

        return representatives

    def save(self, path: str):
        """Save cluster state."""
        import numpy as np

        data = {
            "n_clusters": self.n_clusters,
            "cluster_labels": self.cluster_labels,
            "cluster_operations": {str(k): v for k, v in self.cluster_operations.items()},
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load cluster state."""
        import numpy as np

        with open(path) as f:
            data = json.load(f)

        self.n_clusters = data["n_clusters"]
        self.cluster_labels = data["cluster_labels"]
        self.cluster_operations = {int(k): v for k, v in data["cluster_operations"].items()}
        self.embeddings = {k: np.array(v) for k, v in data["embeddings"].items()}


def tag_spans_with_regex(spans: List[ExtractedSpan]) -> List[ExtractedSpan]:
    """Apply regex-based tagging to spans (bootstrap tagging)."""

    patterns = [
        # SET operations (initialization)
        (r'\b(has|have|had|is|are|was|were|starts?|begins?|contains?)\b', 'SET'),
        (r'\b(lay|lays|laid|produces?|makes?|earns?|gets?|receives?|collects?)\b', 'SET'),

        # SUB operations (decrease)
        (r'\b(eats?|ate|uses?|used|spends?|spent|loses?|lost)\b', 'SUB'),
        (r'\b(gives?|gave|sold|sells?|removes?|takes?|took)\b', 'SUB'),

        # ADD operations (increase)
        (r'\b(buys?|bought|adds?|added|gains?|gained|finds?|found)\b', 'ADD'),
        (r'\b(receives?|received|gets?|got|wins?|won)\b', 'ADD'),

        # MUL operations
        (r'\b(times|multiply|multiplied)\b', 'MUL'),
        (r'\b(\d+)\s*%', 'MUL'),
        (r'\b(twice|double|triple)\b', 'MUL'),

        # DIV operations
        (r'\b(divides?|divided|split|half|halves)\b', 'DIV'),
    ]

    for span in spans:
        if span.operation is not None:
            continue  # already tagged

        text_lower = span.text.lower()
        for pattern, op in patterns:
            if re.search(pattern, text_lower):
                span.operation = op
                span.tagged_by = "regex"
                span.confidence = 0.7  # regex confidence
                break

    return spans
