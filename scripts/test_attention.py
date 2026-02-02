"""Test the attention-based decomposition pipeline.

This script tests the three stages of the attention pipeline:
1. extract_attention() - Pull attention matrices from transformer
2. detect_spans() - Cluster tokens into semantic spans
3. classify_span() - Map spans to mathematical operations

Usage:
    python scripts/test_attention.py              # Try real model (may download ~14GB)
    python scripts/test_attention.py --mock       # Use mock data (fast, no model needed)
    python scripts/test_attention.py --mock-only  # Skip model, test span/classify only
"""
import sys
import argparse

sys.path.insert(0, '/Users/bryceroche/Desktop/mycelium/src')

import numpy as np
from typing import List, Optional, Tuple

from mycelium.attention import extract_attention, detect_spans, classify_span
from mycelium.attention.span_detector import Span
from mycelium.attention.extractor import aggregate_attention


def create_mock_attention_result(tokens: List[str]) -> np.ndarray:
    """Create mock attention matrix for testing span detection.

    Creates attention patterns that simulate semantic groupings:
    - Tokens within semantic units attend strongly to each other
    - Tokens across units have weaker attention
    """
    np.random.seed(42)  # Reproducible results
    n = len(tokens)
    attention = np.random.uniform(0.05, 0.12, (n, n))

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

    # "twice" patterns
    twice_idx = next((i for i, t in enumerate(token_lower) if 'twice' in t), None)
    if twice_idx is not None:
        for i in range(twice_idx, min(twice_idx + 4, n)):
            for j in range(twice_idx, min(twice_idx + 4, n)):
                if i != j:
                    attention[i, j] = 0.28 + np.random.uniform(0, 0.1)

    return attention


def create_mock_tokens(problem: str) -> List[str]:
    """Create mock tokenization (simplified - real tokenizer would be different)."""
    return problem.replace(",", " ,").replace(".", " .").split()


def test_extract_attention(problem: str, use_mock: bool = False) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """Test the attention extraction stage.

    Args:
        problem: The math problem text
        use_mock: If True, skip model and use mock data

    Returns:
        Tuple of (tokens, attention) - either real or mock
    """
    print("=" * 60)
    print("STAGE 1: Attention Extraction")
    print("=" * 60)
    print(f"Problem: {problem}")
    print()

    if use_mock:
        print("MODE: Using mock tokens and attention (--mock flag)")
        print()
        tokens = create_mock_tokens(problem)
        print(f"  Mock tokens ({len(tokens)}): {tokens}")
        attention = create_mock_attention_result(tokens)
        print(f"  Mock attention shape: {attention.shape}")
        return tokens, attention

    print("MODE: Attempting to load real model...")
    print("      (This may download ~14GB on first run)")
    print()

    try:
        result = extract_attention(problem)
        print("SUCCESS: Extracted attention from model")
        print(f"  Tokens ({len(result.tokens)}): {result.tokens}")
        print(f"  Attention shape: {result.attention.shape}")
        print(f"  Semantic layers: {result.semantic_layers}")
        print(f"  Num heads: {len(result.semantic_heads)}")

        # Aggregate attention for span detection
        aggregated = aggregate_attention(
            result.attention,
            layers=list(range(result.attention.shape[0])),
            heads=list(range(result.attention.shape[1])),
            method="mean"
        )
        print(f"  Aggregated attention shape: {aggregated.shape}")

        return result.tokens, aggregated

    except NotImplementedError as e:
        print(f"NOTE: extract_attention() not yet implemented: {e}")
        print("      Falling back to mock data...")
        print()
        tokens = create_mock_tokens(problem)
        print(f"  Mock tokens ({len(tokens)}): {tokens}")
        attention = create_mock_attention_result(tokens)
        print(f"  Mock attention shape: {attention.shape}")
        return tokens, attention

    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print()
        print("Install required packages:")
        print("  pip install transformers torch")
        print()
        print("Falling back to mock data...")
        tokens = create_mock_tokens(problem)
        attention = create_mock_attention_result(tokens)
        return tokens, attention

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print()
        print("This may indicate:")
        print("  - Model download in progress (can take 10+ minutes)")
        print("  - Insufficient memory (7B model needs ~16GB RAM)")
        print("  - Network issues")
        print()
        print("Falling back to mock data...")
        tokens = create_mock_tokens(problem)
        attention = create_mock_attention_result(tokens)
        return tokens, attention


def test_detect_spans(tokens: List[str], attention: np.ndarray) -> List[Span]:
    """Test the span detection stage."""
    print()
    print("=" * 60)
    print("STAGE 2: Span Detection")
    print("=" * 60)

    if tokens is None:
        print("SKIPPED: No tokens available from previous stage")
        return []

    print(f"Input: {len(tokens)} tokens")
    print(f"Attention matrix: {attention.shape}")
    print()

    # Test with different thresholds
    thresholds = [0.25, 0.20, 0.15]
    all_spans = []

    for threshold in thresholds:
        spans = detect_spans(tokens, attention, threshold=threshold)
        print(f"Threshold {threshold}:")
        print(f"  Found {len(spans)} spans")

        for span in spans:
            print(f"    [{span.start}:{span.end}] '{span.text}' (conf: {span.confidence:.3f})")

        if threshold == 0.15:  # Use default threshold for classification
            all_spans = spans

        print()

    if not all_spans:
        print("NOTE: No spans detected. This could mean:")
        print("  - Attention threshold too high")
        print("  - Attention patterns don't show strong semantic grouping")
        print("  - Try lower threshold")

    return all_spans


def test_classify_spans(spans: List[Span]) -> None:
    """Test the span classification stage."""
    print()
    print("=" * 60)
    print("STAGE 3: Span Classification")
    print("=" * 60)

    if not spans:
        print("No spans from detection stage.")
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

        print(f"Span: '{span.text}'")
        if operation:
            print(f"  Operation type: {operation.op_type}")
            print(f"  Operands: {operation.operands}")
            print(f"  Confidence: {operation.confidence:.3f}")
        else:
            print("  Not classified (no matching pattern)")
        print()


def test_full_pipeline(problem: str, use_mock: bool = False) -> None:
    """Run the full attention pipeline on a problem."""
    print()
    print("#" * 60)
    print("FULL PIPELINE TEST")
    print("#" * 60)
    print()

    # Stage 1: Extract attention
    tokens, attention = test_extract_attention(problem, use_mock=use_mock)

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
    print("Testing pattern matching on mathematical phrases:")
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


def test_span_detection_only() -> None:
    """Test span detection with various mock attention patterns."""
    print()
    print("#" * 60)
    print("SPAN DETECTION TESTS")
    print("#" * 60)
    print()

    problems = [
        "John has half the apples that Mary has",
        "The store sold twice as many shirts as pants",
        "Mary earned 25 percent more than John",
    ]

    for problem in problems:
        print(f"Problem: {problem}")
        print("-" * 40)

        tokens = create_mock_tokens(problem)
        attention = create_mock_attention_result(tokens)

        spans = detect_spans(tokens, attention, threshold=0.15)
        print(f"Detected {len(spans)} spans:")
        for span in spans:
            op = classify_span(span)
            op_str = f" -> {op.op_type}" if op else ""
            print(f"  '{span.text}'{op_str}")
        print()


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test the attention-based decomposition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_attention.py              # Full test with model
  python scripts/test_attention.py --mock       # Use mock data (fast)
  python scripts/test_attention.py --mock-only  # Only test span/classify
        """
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock attention data instead of loading model"
    )
    parser.add_argument(
        "--mock-only", action="store_true",
        help="Skip model entirely, only test span detection and classification"
    )
    args = parser.parse_args()

    print("Attention-Based Decomposition Pipeline Test")
    print("=" * 60)

    if args.mock_only:
        print("MODE: --mock-only (skipping model, testing span/classify only)")
        print()
        test_span_detection_only()
        test_classifier_patterns()
        return

    use_mock = args.mock
    if use_mock:
        print("MODE: --mock (using mock attention data)")
    else:
        print("MODE: Full pipeline (will attempt to load model)")
        print()
        print("NOTE: First run downloads DeepSeek-Math-7B (~14GB).")
        print("      Use --mock flag for fast testing without model.")

    print()

    # Test problem
    problem = "John has half the apples that Mary has"

    # Run full pipeline test
    test_full_pipeline(problem, use_mock=use_mock)

    # Test classifier patterns (no model needed)
    test_classifier_patterns()

    # Additional problems (only if mock mode for speed)
    if use_mock:
        print()
        print("#" * 60)
        print("ADDITIONAL PROBLEM TESTS")
        print("#" * 60)

        additional_problems = [
            "The store sold twice as many shirts as pants",
            "Mary earned 25 percent more than John",
            "The total is the sum of all items",
        ]

        for prob in additional_problems:
            print()
            print(f"Testing: {prob}")
            print("-" * 40)
            tokens, attention = test_extract_attention(prob, use_mock=True)
            if tokens:
                spans = detect_spans(tokens, attention, threshold=0.15)
                print(f"Detected {len(spans)} spans")
                for span in spans:
                    op = classify_span(span)
                    if op:
                        print(f"  '{span.text}' -> {op.op_type}")


if __name__ == "__main__":
    main()
