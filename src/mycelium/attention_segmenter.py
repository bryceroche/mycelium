"""Hidden-state based clause segmentation.

NO KEYWORDS. Pure embedding similarity for:
1. Detecting variable references (entities that cluster separately)
2. Identifying operation boundaries (verbs that separate)
3. Extracting numbers as operands

Uses hidden states from middle layers (layer 14) instead of attention.
This bypasses the "attention sink" problem in causal models where all
tokens attend heavily to the first token.

Uses Qwen2.5-Math-7B-Instruct for superior math reasoning (91.6% GSM8K).
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class Segment:
    """A segmented clause with embedding-derived metadata."""
    text: str
    tokens: List[str]
    start_idx: int
    end_idx: int
    segment_type: str  # "operation", "reference", "number"
    similarity_score: float  # How similar tokens in this segment are


@dataclass
class SegmentedSpan:
    """A span broken into semantic segments via attention."""
    original_text: str
    segments: List[Segment]
    reference_entity: Optional[str]  # Detected variable reference (e.g., "John")
    numbers: List[float]  # Extracted numeric values


class AttentionSegmenter:
    """Segment spans into clauses using hidden state similarity.

    NO KEYWORDS. Uses hierarchical clustering on hidden state embeddings to find:
    - Reference entities (cluster separately with 2 clusters)
    - Operation verbs (cluster separately with more clusters)
    - Number tokens (cluster at boundaries)

    Uses middle-layer hidden states (layer 14) instead of attention.
    This bypasses the attention sink problem in causal models.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._hidden_layer = 14  # Middle layer for semantic info

    def _ensure_model(self):
        """Lazy load the model."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                output_hidden_states=True,  # Hidden states instead of attention
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            # Determine middle layer
            n_layers = self._model.config.num_hidden_layers
            self._hidden_layer = n_layers // 2
            print(f"Model loaded. Using hidden layer {self._hidden_layer}/{n_layers}")

    def _get_similarity_matrix(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Get token similarity matrix using hidden states.

        Uses middle-layer hidden states (layer 14) and computes cosine
        similarity between token embeddings. This bypasses the attention
        sink problem in causal models.
        """
        self._ensure_model()

        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # Get hidden states from middle layer
        hidden = outputs.hidden_states[self._hidden_layer][0].cpu().numpy()

        # Compute cosine similarity between all token pairs
        norms = np.linalg.norm(hidden, axis=1, keepdims=True)
        hidden_normed = hidden / (norms + 1e-10)
        similarity = hidden_normed @ hidden_normed.T

        # Ensure valid range [0, 1]
        similarity = np.clip(similarity, 0, 1)

        return similarity, tokens

    def _cluster_tokens(
        self,
        similarity: np.ndarray,
        tokens: List[str],
        n_clusters: int
    ) -> Dict[int, List[Tuple[int, str]]]:
        """Cluster tokens based on embedding similarity."""
        # Convert similarity to distance
        dist = 1 - similarity
        np.fill_diagonal(dist, 0)

        # Handle NaN/Inf values
        dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
        dist = np.clip(dist, 0, 1)

        # Hierarchical clustering
        dist_condensed = squareform(dist, checks=False)
        Z = linkage(dist_condensed, method="average")
        cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Group tokens by cluster
        clusters = {}
        for i, (tok, c_id) in enumerate(zip(tokens, cluster_ids)):
            if c_id not in clusters:
                clusters[c_id] = []
            clusters[c_id].append((i, tok))

        return clusters

    def _tokens_to_text(self, tokens: List[str]) -> str:
        """Convert tokens back to readable text."""
        # Handle different tokenizer formats (Ġ for GPT-style, ## for BERT-style)
        text = "".join(tokens)
        text = text.replace("Ġ", " ").replace("▁", " ").replace("##", "")
        return text.strip()

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text. No keywords - just pattern matching."""
        import re
        numbers = []

        # Find digit sequences
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            numbers.append(float(match.group(1)))

        return numbers

    def _compute_cluster_similarity_score(
        self,
        similarity: np.ndarray,
        indices: List[int]
    ) -> float:
        """Compute how similar tokens in a cluster are to each other."""
        if len(indices) < 2:
            return 1.0  # Single token is perfectly self-similar

        # Average similarity within cluster
        total = 0.0
        count = 0
        for i in indices:
            for j in indices:
                if i != j:
                    total += similarity[i, j]
                    count += 1

        return total / max(count, 1)

    def segment(self, text: str) -> SegmentedSpan:
        """Segment a span using hidden state similarity.

        Strategy:
        1. Use 4-5 clusters to detect reference entities (single tokens at end)
        2. Use 3-4 clusters to identify verb boundaries
        3. Extract numbers from the segments
        """
        similarity, tokens = self._get_similarity_matrix(text)
        n_tokens = len(tokens)

        segments = []
        reference_entity = None

        # Step 1: Check for reference entity by examining the LAST token directly
        # Reference entities like "John", "Tom" appear as the final token
        # and have LOW similarity to the rest of the clause
        if n_tokens >= 4:
            last_idx = n_tokens - 1
            last_tok = tokens[last_idx]
            last_text = self._tokens_to_text([last_tok])

            # Skip if last token is numeric
            if not last_text.strip().isdigit():
                # Check cross-similarity of last token to other tokens
                other_indices = [i for i in range(n_tokens) if i != last_idx]
                cross_sim = np.mean([similarity[last_idx, j] for j in other_indices])

                # Low cross-similarity (<0.34) suggests reference entity
                # Reference entities (John, Tom) have ~0.27-0.33 similarity
                # Direct objects (apples, coins) have ~0.34-0.40 similarity
                if cross_sim < 0.34:
                    reference_entity = last_text

        # Step 2: Use 3 clusters for main segmentation
        clusters_3 = self._cluster_tokens(similarity, tokens, n_clusters=3)

        # Sort clusters by their starting position
        sorted_clusters = []
        for c_id, toks in clusters_3.items():
            indices = [t[0] for t in toks]
            start = min(indices)
            end = max(indices)
            token_list = [t[1] for t in sorted(toks, key=lambda x: x[0])]
            sim_score = self._compute_cluster_similarity_score(similarity, indices)
            sorted_clusters.append((start, end, token_list, sim_score))

        sorted_clusters.sort(key=lambda x: x[0])

        # Convert to Segment objects
        for start, end, toks, sim_score in sorted_clusters:
            seg_text = self._tokens_to_text(toks)

            # Determine segment type based on position and content
            numbers = self._extract_numbers(seg_text)

            if seg_text == reference_entity:
                seg_type = "reference"
            elif numbers and len(seg_text.split()) <= 2:
                seg_type = "number"
            else:
                seg_type = "operation"

            segments.append(Segment(
                text=seg_text,
                tokens=toks,
                start_idx=start,
                end_idx=end,
                segment_type=seg_type,
                similarity_score=sim_score,
            ))

        # Extract all numbers from original text
        all_numbers = self._extract_numbers(text)

        return SegmentedSpan(
            original_text=text,
            segments=segments,
            reference_entity=reference_entity,
            numbers=all_numbers,
        )

    def segment_multi_clause(self, text: str) -> List[SegmentedSpan]:
        """Segment a potentially multi-clause span into separate operations.

        Uses hidden state similarity to find natural breakpoints.
        """
        similarity, tokens = self._get_similarity_matrix(text)
        n_tokens = len(tokens)

        # Count numbers to estimate clause count
        numbers = self._extract_numbers(text)
        estimated_clauses = max(1, len(numbers))

        if estimated_clauses <= 1:
            return [self.segment(text)]

        # Use more clusters for multi-clause
        n_clusters = min(estimated_clauses + 2, n_tokens // 2)
        clusters = self._cluster_tokens(attn, tokens, n_clusters=n_clusters)

        # Group adjacent clusters that form clauses
        # For now, return as single segmented span
        # TODO: Split into separate clauses based on verb clusters
        return [self.segment(text)]


def test_segmenter():
    """Test the attention segmenter."""
    segmenter = AttentionSegmenter()

    test_cases = [
        "Lisa has 5 more apples than John",
        "Mary has twice as many books as Tom",
        "She sold 5 apples",
        "He found 8 coins then spent 3",
    ]

    print("=== Attention Segmenter Test ===\n")

    for text in test_cases:
        print(f"Input: \"{text}\"")
        result = segmenter.segment(text)

        print(f"  Reference: {result.reference_entity}")
        print(f"  Numbers: {result.numbers}")
        print(f"  Segments:")
        for seg in result.segments:
            print(f"    [{seg.segment_type}] \"{seg.text}\" (attn={seg.attention_score:.3f})")
        print()


if __name__ == "__main__":
    test_segmenter()
