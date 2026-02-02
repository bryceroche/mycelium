"""Detect semantic spans from attention patterns.

The "Panama Hats" problem: we need the LONGEST contiguous span
that forms a coherent semantic unit.

"half the price of the cheese" = ONE span (operation)
not "half" + "the" + "price" + "of" + "the" + "cheese"
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A semantic span in the text."""
    start: int          # Start token index
    end: int            # End token index (exclusive)
    tokens: List[str]   # The tokens in this span
    text: str           # Joined text
    confidence: float   # How strongly tokens attend to each other


def build_adjacency(
    attention: np.ndarray,
    threshold: float = 0.15,
) -> np.ndarray:
    """Build token adjacency matrix from attention.

    Tokens are "connected" if they attend strongly to each other
    (bidirectional mutual attention).

    Args:
        attention: Attention matrix (seq_len, seq_len)
        threshold: Minimum mutual attention to be considered connected

    Returns:
        Boolean adjacency matrix (seq_len, seq_len)
    """
    # Mutual attention: average of forward and backward
    mutual = (attention + attention.T) / 2
    return mutual > threshold


def find_contiguous_spans(
    adjacency: np.ndarray,
    tokens: List[str],
    min_span_size: int = 2,
) -> List[Span]:
    """Find maximal contiguous spans of connected tokens.

    Key insight: we want CONTIGUOUS spans (adjacent tokens).
    Non-contiguous connections suggest different semantic units.

    Args:
        adjacency: Boolean adjacency matrix
        tokens: List of tokens
        min_span_size: Minimum tokens to form a span

    Returns:
        List of Span objects
    """
    n = len(tokens)
    spans = []
    visited = set()

    for start in range(n):
        if start in visited:
            continue

        # Greedy extend: keep adding if connected to ALL in span
        span_indices = [start]

        for next_idx in range(start + 1, n):
            # Check if next token connects to all tokens in current span
            all_connected = all(
                adjacency[next_idx, idx] or adjacency[idx, next_idx]
                for idx in span_indices
            )

            if all_connected:
                span_indices.append(next_idx)
            else:
                break  # Gap found, span ends

        if len(span_indices) >= min_span_size:
            # Calculate confidence as mean mutual attention within span
            span_attention = 0.0
            count = 0
            for i, idx_i in enumerate(span_indices):
                for idx_j in span_indices[i+1:]:
                    span_attention += (adjacency[idx_i, idx_j] + adjacency[idx_j, idx_i]) / 2
                    count += 1
            confidence = span_attention / count if count > 0 else 0.0

            span_tokens = [tokens[i] for i in span_indices]
            spans.append(Span(
                start=span_indices[0],
                end=span_indices[-1] + 1,
                tokens=span_tokens,
                text=" ".join(span_tokens),
                confidence=float(confidence),
            ))
            visited.update(span_indices)

    return spans


def detect_spans(
    tokens: List[str],
    attention: np.ndarray,
    threshold: float = 0.15,
) -> List[Span]:
    """Main entry point: detect semantic spans from attention.

    Args:
        tokens: List of tokens from tokenizer
        attention: Aggregated attention matrix (seq_len, seq_len)
        threshold: Attention threshold for connectivity

    Returns:
        List of detected spans, sorted by position
    """
    adjacency = build_adjacency(attention, threshold)
    spans = find_contiguous_spans(adjacency, tokens)
    return sorted(spans, key=lambda s: s.start)


def detect_hierarchical_spans(
    tokens: List[str],
    attention: np.ndarray,
    thresholds: List[float] = [0.25, 0.15, 0.10],
) -> List[List[Span]]:
    """Detect nested/hierarchical spans at different thresholds.

    Higher threshold = tighter binding = inner spans
    Lower threshold = looser binding = outer spans

    Example: "half (the price (of the cheese))"
    - threshold 0.25: ["the cheese"]
    - threshold 0.15: ["the price of the cheese"]
    - threshold 0.10: ["half the price of the cheese"]

    Args:
        tokens: List of tokens
        attention: Attention matrix
        thresholds: List of thresholds from tight to loose

    Returns:
        List of span lists, one per threshold level
    """
    return [
        detect_spans(tokens, attention, thresh)
        for thresh in sorted(thresholds, reverse=True)
    ]
