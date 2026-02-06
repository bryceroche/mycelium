"""Debug span detection to understand why Panama Hats produces few spans."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from mycelium.dual_signal_templates import SpanDetector
from mycelium.attention_graph import AttentionGraphBuilder

def debug_span_detection():
    # Load detector
    print("Loading MiniLM...")
    detector = SpanDetector()

    # Test problem from GSM8K
    problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print(f"\nProblem: {problem[:80]}...")

    # Extract features
    embedding, attention_matrix, tokens = detector.extract_features(problem)

    print(f"\nTokens ({len(tokens)}): {tokens[:20]}...")
    print(f"Attention shape: {attention_matrix.shape}")

    # Average attention if multi-head
    if attention_matrix.ndim > 2:
        attention_matrix = attention_matrix.mean(axis=0)
        print(f"Averaged attention shape: {attention_matrix.shape}")

    # Build graph with default thresholds
    builder = AttentionGraphBuilder()

    print("\n" + "="*60)
    print("Testing with DEFAULT thresholds:")
    print(f"  boundary_drop_threshold = {builder.boundary_drop_threshold}")
    print(f"  min_span_size (in recursive) = 3")
    print("="*60)

    boundaries = builder.detect_span_boundaries(attention_matrix, tokens)
    print(f"\nBoundaries detected: {len(boundaries)}")
    for start, end in boundaries:
        span_text = ' '.join(tokens[start:end])
        print(f"  [{start}:{end}] {span_text[:60]}...")

    # Debug: show connectivity values at different split points
    print("\n" + "="*60)
    print("Connectivity analysis:")
    print("="*60)

    n = len(tokens)
    for split in range(6, n-6, 10):  # Every 10 tokens
        left_conn = builder.compute_span_connectivity(attention_matrix, 0, split)
        right_conn = builder.compute_span_connectivity(attention_matrix, split, n)
        cross_conn = builder._compute_cross_attention(attention_matrix, 0, split, split, n)

        internal_avg = (left_conn + right_conn) / 2
        ratio = cross_conn / internal_avg if internal_avg > 0 else 0

        print(f"  Split at {split}: left={left_conn:.4f}, right={right_conn:.4f}, "
              f"cross={cross_conn:.4f}, ratio={ratio:.4f}")

    # Test with more aggressive threshold
    print("\n" + "="*60)
    print("Testing with AGGRESSIVE thresholds (boundary_drop=0.7):")
    print("="*60)

    builder_aggressive = AttentionGraphBuilder(boundary_drop_threshold=0.7)
    boundaries_aggressive = builder_aggressive.detect_span_boundaries(attention_matrix, tokens)
    print(f"\nBoundaries detected: {len(boundaries_aggressive)}")
    for start, end in boundaries_aggressive:
        span_text = ' '.join(tokens[start:end])
        print(f"  [{start}:{end}] {span_text[:60]}...")

    # Test with very aggressive threshold
    print("\n" + "="*60)
    print("Testing with VERY AGGRESSIVE thresholds (boundary_drop=0.85):")
    print("="*60)

    builder_very = AttentionGraphBuilder(boundary_drop_threshold=0.85)
    boundaries_very = builder_very.detect_span_boundaries(attention_matrix, tokens)
    print(f"\nBoundaries detected: {len(boundaries_very)}")
    for start, end in boundaries_very:
        span_text = ' '.join(tokens[start:end])
        print(f"  [{start}:{end}] {span_text[:60]}...")

    # Show attention_received to understand entity detection
    print("\n" + "="*60)
    print("Attention received (entity detection signal):")
    print("="*60)

    attn_received = builder.compute_attention_received(attention_matrix)
    top_indices = np.argsort(attn_received)[-10:][::-1]
    for idx in top_indices:
        if idx < len(tokens):
            print(f"  {tokens[idx]:15s} -> {attn_received[idx]:.4f}")


if __name__ == "__main__":
    debug_span_detection()
