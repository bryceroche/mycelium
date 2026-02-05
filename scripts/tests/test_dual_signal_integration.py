#!/usr/bin/env python3
"""Test dual-signal integration for Mycelium.

This test demonstrates:
1. Span detection using attention patterns
2. Dual-signal (embedding + attention) operation matching
3. Welford statistics tracking
4. End-to-end problem solving

The dual-signal approach addresses the lexical vs operational similarity
problem by routing based on what operations DO, not what they SOUND LIKE.

Run: python -m mycelium.test_dual_signal_integration
"""

import sys
import traceback
from typing import List, Tuple


def test_mock_solver():
    """Test the solver with mock model (no GPU required)."""
    print("\n" + "=" * 60)
    print("TEST 1: Mock Solver (keyword-based classification)")
    print("=" * 60)

    from mycelium.dual_signal_solver import DualSignalSolver, SolverResult

    # Initialize with mock model
    solver = DualSignalSolver(mock_model=True)

    # Test problems with expected answers
    test_cases = [
        (
            "John has 5 apples. He gives 2 apples to Mary. How many apples does John have now?",
            3.0,  # 5 - 2 = 3
            "Basic SET then SUB"
        ),
        (
            "Lisa has 12 cookies. She ate 3 cookies.",
            9.0,  # 12 - 3 = 9
            "SET then SUB (ate = subtraction)"
        ),
        (
            "Tom had 8 dollars. He earned 5 more dollars mowing lawns.",
            13.0,  # 8 + 5 = 13
            "SET then ADD (earned = addition)"
        ),
    ]

    passed = 0
    for problem, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Problem: {problem}")
        print(f"Expected: {expected}")

        result = solver.solve(problem)

        print(f"Got: {result.answer}")
        print(f"State: {result.state}")
        print(f"Operations:")
        for op in result.operations:
            print(f"  {op.entity}: {op.dsl_expr}({op.value}) -> conf={op.confidence:.2f}")

        # Record outcome
        correct = abs(result.answer - expected) < 0.01
        solver.record_outcome(result, correct)

        if correct:
            print("[PASS]")
            passed += 1
        else:
            print(f"[FAIL] Expected {expected}, got {result.answer}")

    print(f"\n=== Mock Solver: {passed}/{len(test_cases)} passed ===")
    solver.print_stats()

    return passed == len(test_cases)


def test_template_store():
    """Test the template store directly."""
    print("\n" + "=" * 60)
    print("TEST 2: Template Store (dual-signal matching)")
    print("=" * 60)

    import numpy as np
    from mycelium.dual_signal_templates import (
        TemplateStore,
        DualSignalTemplate,
        WelfordStats,
    )

    # Create store
    store = TemplateStore(embedding_weight=0.6, attention_weight=0.4)

    # Create some templates with DSL expressions
    np.random.seed(123)  # Reproducible

    dsl_exprs = [("value", "value"), ("entity + value", "add"), ("entity - value", "sub")]
    templates = []
    for dsl_expr, label in dsl_exprs:
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        attention = np.random.randn(100).astype(np.float32)

        template = DualSignalTemplate(
            template_id=f"test_{label}",
            embedding_centroid=embedding,
            attention_signature=attention,
            dsl_expr=dsl_expr,
            span_examples=[f"Example for {label}"],
        )
        store.add_template(template)
        templates.append(template)

    print(f"Created {len(store.templates)} templates")

    # Test matching
    print("\nTesting matching...")

    # Create a query similar to first template (SET)
    set_template = templates[0]
    query_emb = set_template.embedding_centroid + 0.1 * np.random.randn(384)
    query_emb = query_emb.astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    query_att = set_template.attention_signature + 0.1 * np.random.randn(100)
    query_att = query_att.astype(np.float32)

    result = store.find_best_match(query_emb, query_att)

    if result:
        matched, score, emb_sim, att_sim = result
        print(f"Best match: {matched.template_id}")
        print(f"Combined score: {score:.4f}")
        print(f"Embedding similarity: {emb_sim:.4f}")
        print(f"Attention correlation: {att_sim:.4f}")

        # Should match value (SET) template
        if matched.template_id == "test_value":
            print("[PASS] Correctly matched value template")
        else:
            print(f"[FAIL] Expected test_value, got {matched.template_id}")
            return False
    else:
        print("[FAIL] No match found")
        return False

    # Test top-k matching
    print("\nTesting top-k matching...")
    top_k = store.find_top_k_matches(query_emb, query_att, k=3)
    print(f"Top {len(top_k)} matches:")
    for template, score, emb_sim, att_sim in top_k:
        print(f"  {template.template_id}: score={score:.4f}")

    print("\n=== Template Store: All tests passed ===")
    return True


def test_welford_stats():
    """Test Welford statistics tracking."""
    print("\n" + "=" * 60)
    print("TEST 3: Welford Statistics")
    print("=" * 60)

    from mycelium.dual_signal_templates import WelfordStats

    stats = WelfordStats()

    # Add some values
    values = [0.75, 0.82, 0.78, 0.90, 0.85, 0.72, 0.88]
    for v in values:
        stats.update(v)

    print(f"Values: {values}")
    print(f"Count: {stats.count}")
    print(f"Mean: {stats.mean:.4f}")
    print(f"Variance: {stats.variance:.4f}")
    print(f"Std: {stats.std:.4f}")

    # Verify against numpy
    import numpy as np
    expected_mean = np.mean(values)
    expected_var = np.var(values, ddof=1)  # Sample variance

    mean_ok = abs(stats.mean - expected_mean) < 0.0001
    var_ok = abs(stats.variance - expected_var) < 0.0001

    print(f"\nExpected mean: {expected_mean:.4f}, got: {stats.mean:.4f} {'[OK]' if mean_ok else '[FAIL]'}")
    print(f"Expected var: {expected_var:.4f}, got: {stats.variance:.4f} {'[OK]' if var_ok else '[FAIL]'}")

    # Test serialization
    d = stats.to_dict()
    stats2 = WelfordStats.from_dict(d)

    serial_ok = (stats2.count == stats.count and
                 abs(stats2.mean - stats.mean) < 0.0001 and
                 abs(stats2.variance - stats.variance) < 0.0001)
    print(f"Serialization: {'[OK]' if serial_ok else '[FAIL]'}")

    print("\n=== Welford Stats: All tests passed ===")
    return mean_ok and var_ok and serial_ok


def test_full_pipeline():
    """Test the full dual-signal pipeline."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline (with model if available)")
    print("=" * 60)

    try:
        from mycelium.dual_signal_pipeline import DualSignalPipeline

        # Try to initialize pipeline
        print("Initializing DualSignalPipeline...")
        pipeline = DualSignalPipeline()

        # Bootstrap with examples
        n_templates = pipeline.bootstrap_from_examples()
        print(f"Bootstrapped {n_templates} templates")

        # Test on a problem
        test_problem = "John has 5 apples. He gives 2 apples to Mary."

        print(f"\nProcessing: {test_problem}")
        output = pipeline.process_problem(test_problem)

        print(f"Spans detected: {output.spans_detected}")
        print(f"Templates available: {output.templates_available}")
        print(f"Matched operations:")

        for op in output.matched_operations:
            span_preview = op.span_text[:40] + "..." if len(op.span_text) > 40 else op.span_text
            print(f"  '{span_preview}'")
            print(f"    DSL: {op.dsl_expr}")
            print(f"    Confidence: {op.confidence:.3f}")
            print(f"    Embedding sim: {op.embedding_similarity:.3f}")
            print(f"    Attention sim: {op.attention_similarity:.3f}")

        # Record some outcomes
        if output.matched_operations:
            op = output.matched_operations[0]
            pipeline.record_outcome(
                op.template_id,
                success=True,
                embedding_sim=op.embedding_similarity,
                attention_sim=op.attention_similarity,
            )
            print(f"\nRecorded success for template: {op.template_id}")

        # Check decomposition candidates
        candidates = pipeline.get_decomposition_candidates()
        print(f"Decomposition candidates: {len(candidates)}")

        # Print stats
        pipeline.print_stats()

        print("\n=== Full Pipeline: Test completed ===")
        return True

    except Exception as e:
        print(f"Full pipeline test skipped: {e}")
        traceback.print_exc()
        return False


def test_span_detection():
    """Test attention-based span detection."""
    print("\n" + "=" * 60)
    print("TEST 5: Span Detection (requires model)")
    print("=" * 60)

    try:
        from mycelium.dual_signal_templates import SpanDetector

        print("Initializing SpanDetector...")
        detector = SpanDetector(model_path=None)  # Use base weights

        test_text = "John has 5 apples. He gives 2 apples to Mary. How many apples does John have now?"

        print(f"Text: {test_text}")
        print("\nExtracting features...")

        embedding, attention, tokens = detector.extract_features(test_text)

        print(f"Embedding shape: {embedding.shape}")
        print(f"Attention shape: {attention.shape}")
        print(f"Tokens: {len(tokens)}")

        print("\nDetecting spans (community method)...")
        spans = detector.extract_span_features(test_text, method="community")

        print(f"Detected {len(spans)} spans:")
        for i, span in enumerate(spans):
            text_preview = span['text'][:50] if len(span['text']) > 50 else span['text']
            print(f"  Span {i}: '{text_preview}'")
            print(f"    Density: {span['density']:.3f}")
            print(f"    Position: [{span['start']}, {span['end']})")

        print("\n=== Span Detection: Test completed ===")
        return True

    except Exception as e:
        print(f"Span detection test skipped: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DUAL-SIGNAL INTEGRATION TESTS")
    print("=" * 60)
    print("\nThese tests verify the dual-signal template system integration.")
    print("The system uses BOTH embedding similarity AND attention patterns")
    print("for operation classification.\n")

    results = {
        "Welford Stats": test_welford_stats(),
        "Template Store": test_template_store(),
        "Mock Solver": test_mock_solver(),
    }

    # These tests require model loading
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch:
        results["Span Detection"] = test_span_detection()
        results["Full Pipeline"] = test_full_pipeline()
    else:
        print("\n[SKIP] Span Detection and Full Pipeline tests require PyTorch")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
