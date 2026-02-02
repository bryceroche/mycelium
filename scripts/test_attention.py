"""Test the attention-based decomposition pipeline.

This script tests the three stages of the attention pipeline:
1. extract_attention() - Pull attention matrices from transformer
2. detect_spans() - Cluster tokens into semantic spans
3. classify_span() - Map spans to mathematical operations

Usage:
    python scripts/test_attention.py
"""
import sys
sys.path.insert(0, '/Users/bryceroche/Desktop/mycelium/src')

import numpy as np
from typing import List

from mycelium.attention import extract_attention, detect_spans, classify_span
from mycelium.attention.span_detector import Span


def create_mock_attention_result(tokens: List[str]) -> np.ndarray:
    """Create mock attention matrix for testing span detection.

    Creates attention patterns that simulate semantic groupings:
    - Tokens within semantic units attend strongly to each other
    - Tokens across units have weaker attention
    """
    n = len(tokens)
    attention = np.random.uniform(0.05, 0.12, (n, n))

    # Find semantic groups based on token content
    # Group: "half the apples" (indices will vary based on tokenization)
    # Group: "that Mary has"

    # Simulate strong attention within phrases
    # For "John has half the apples that Mary has"
    # Let's assume rough groupings and add strong mutual attention

    for i in range(n):
        attention[i, i] = 1.0  # Self-attention is always strong

        # Adjacent tokens often have some attention
        if i > 0:
            attention[i, i-1] = max(attention[i, i-1], 0.2)
            attention[i-1, i] = max(attention[i-1, i], 0.2)

    # Find indices of key tokens and boost their mutual attention
    token_lower = [t.lower() for t in tokens]

    # "half the apples" should form a span
    half_idx = next((i for i, t in enumerate(token_lower) if 'half' in t), None)
    if half_idx is not None:
        # Boost attention in the 3-4 tokens after "half"
        for i in range(half_idx, min(half_idx + 4, n)):
            for j in range(half_idx, min(half_idx + 4, n)):
                if i != j:
                    attention[i, j] = 0.3 + np.random.uniform(0, 0.1)

    # "Mary has" could form a span
    mary_idx = next((i for i, t in enumerate(token_lower) if 'mary' in t), None)
    if mary_idx is not None:
        for i in range(mary_idx, min(mary_idx + 2, n)):
            for j in range(mary_idx, min(mary_idx + 2, n)):
                if i != j:
                    attention[i, j] = 0.25 + np.random.uniform(0, 0.1)

    return attention


def test_extract_attention(problem: str) -> tuple:
    """Test the attention extraction stage.

    Returns:
        Tuple of (tokens, attention) - either real or mock
    """
    print("=" * 60)
    print("STAGE 1: Attention Extraction")
    print("=" * 60)
    print(f"Problem: {problem}")
    print()

    try:
        result = extract_attention(problem)
        print("SUCCESS: Extracted attention from model")
        print(f"  Tokens ({len(result.tokens)}): {result.tokens}")
        print(f"  Attention shape: {result.attention.shape}")
        return result.tokens, result.attention

    except NotImplementedError:
        print("NOTE: extract_attention() not yet implemented")
        print("      Using mock tokens and attention for demonstration")
        print()

        # Create mock tokenization (simplified - real tokenizer would be different)
        tokens = problem.replace(",", " ,").replace(".", " .").split()
        print(f"  Mock tokens ({len(tokens)}): {tokens}")

        # Create mock attention that simulates semantic grouping
        attention = create_mock_attention_result(tokens)
        print(f"  Mock attention shape: {attention.shape}")

        return tokens, attention

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print()
        print("This may indicate the model needs to be downloaded.")
        print("Try: pip install transformers torch")
        print("     # Model will download on first use")
        return None, None


def test_detect_spans(tokens: List[str], attention: np.ndarray) -> List[Span]:
    """Test the span detection stage."""
    print()
    print("=" * 60)
    print("STAGE 2: Span Detection")
    print("=" * 60)

    if tokens is None:
        print("SKIPPED: No tokens available from previous stage")
        return []

    # Test with different thresholds
    thresholds = [0.25, 0.20, 0.15]
    all_spans = []

    for threshold in thresholds:
        spans = detect_spans(tokens, attention, threshold=threshold)
        print(f"\nThreshold {threshold}:")
        print(f"  Found {len(spans)} spans")

        for span in spans:
            print(f"    [{span.start}:{span.end}] '{span.text}' (conf: {span.confidence:.3f})")

        if threshold == 0.15:  # Use default threshold for classification
            all_spans = spans

    if not all_spans:
        print("\nNOTE: No spans detected. This could mean:")
        print("  - Attention threshold too high")
        print("  - Mock attention doesn't simulate semantic groups well")
        print("  - Try lower threshold or different problem text")

    return all_spans


def test_classify_spans(spans: List[Span]) -> None:
    """Test the span classification stage."""
    print()
    print("=" * 60)
    print("STAGE 3: Span Classification")
    print("=" * 60)

    if not spans:
        print("SKIPPED: No spans available from previous stage")
        print()
        print("Testing classifier with synthetic spans...")

        # Create synthetic spans for testing
        synthetic_spans = [
            Span(start=0, end=3, tokens=["half", "the", "apples"],
                 text="half the apples", confidence=0.8),
            Span(start=3, end=5, tokens=["that", "Mary"],
                 text="that Mary", confidence=0.6),
            Span(start=0, end=2, tokens=["twice", "the"],
                 text="twice the amount", confidence=0.75),
            Span(start=0, end=3, tokens=["50", "percent", "of"],
                 text="50 percent of sales", confidence=0.9),
        ]
        spans = synthetic_spans
        print(f"Created {len(spans)} synthetic spans for testing")
        print()

    for span in spans:
        operation = classify_span(span)

        print(f"\nSpan: '{span.text}'")
        if operation:
            print(f"  Operation type: {operation.op_type}")
            print(f"  Operands: {operation.operands}")
            print(f"  Confidence: {operation.confidence:.3f}")
        else:
            print("  Not classified (no matching pattern)")


def test_full_pipeline(problem: str) -> None:
    """Run the full attention pipeline on a problem."""
    print()
    print("#" * 60)
    print("FULL PIPELINE TEST")
    print("#" * 60)
    print()

    # Stage 1: Extract attention
    tokens, attention = test_extract_attention(problem)

    # Stage 2: Detect spans
    spans = test_detect_spans(tokens, attention)

    # Stage 3: Classify spans
    test_classify_spans(spans)

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def test_classifier_patterns() -> None:
    """Test the classifier with various mathematical phrases."""
    print()
    print("#" * 60)
    print("CLASSIFIER PATTERN TESTS")
    print("#" * 60)
    print()

    test_phrases = [
        "half the apples",
        "twice as many as Mary",
        "3 times the amount",
        "5 more than John",
        "10 less than total",
        "25 percent of the price",
        "50% of income",
        "the sum of x and y",
        "the difference between a and b",
        "the product of length and width",
        "total divided by count",
        "apples plus oranges",
        "revenue minus costs",
        "42",  # Just a number
        "John",  # Variable reference
    ]

    for phrase in test_phrases:
        span = Span(start=0, end=1, tokens=phrase.split(),
                   text=phrase, confidence=1.0)
        operation = classify_span(span)

        if operation:
            print(f"'{phrase}'")
            print(f"  -> {operation.op_type}({operation.operands})")
            print()


def main():
    """Main test runner."""
    print("Attention-Based Decomposition Pipeline Test")
    print("=" * 60)
    print()

    # Test problem
    problem = "John has half the apples that Mary has"

    # Run full pipeline test
    test_full_pipeline(problem)

    # Test classifier patterns
    test_classifier_patterns()

    # Test with additional problems
    additional_problems = [
        "The store sold twice as many shirts as pants",
        "Mary earned 25 percent more than John",
        "The total is the sum of all items",
    ]

    print()
    print("#" * 60)
    print("ADDITIONAL PROBLEM TESTS")
    print("#" * 60)

    for prob in additional_problems:
        print()
        print(f"Testing: {prob}")
        print("-" * 40)
        tokens, attention = test_extract_attention(prob)
        if tokens:
            spans = detect_spans(tokens, attention, threshold=0.15)
            print(f"Detected {len(spans)} spans")
            for span in spans:
                op = classify_span(span)
                if op:
                    print(f"  '{span.text}' -> {op.op_type}")


if __name__ == "__main__":
    main()
