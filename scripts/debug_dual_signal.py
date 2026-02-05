#!/usr/bin/env python3
"""Debug script for dual signal solver - run 6 random GSM8K spans with detailed output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import random
import json
from datasets import load_dataset
from mycelium.dual_signal_solver import DualSignalSolver
from mycelium.attention_graph import AttentionGraphBuilder

def extract_answer(answer_str: str) -> float:
    """Extract numeric answer from GSM8K answer string."""
    if "####" in answer_str:
        answer_str = answer_str.split("####")[-1].strip()
    try:
        return float(answer_str.replace(",", "").replace("$", ""))
    except ValueError:
        return None

def debug_solve(solver, question: str, expected: float, idx: int):
    """Run solve with detailed debug output."""
    print(f"\n{'='*70}")
    print(f"PROBLEM {idx+1}")
    print(f"{'='*70}")
    print(f"Question: {question[:200]}...")
    print(f"Expected answer: {expected}")
    print()

    # Get pipeline components
    solver._ensure_pipeline()
    pipeline = solver._pipeline
    detector = pipeline.detector
    store = pipeline.store

    # Extract attention features
    print("--- ATTENTION-BASED SPAN DETECTION ---")
    print(f"Original: {question[:100]}...")

    # Get attention matrix
    embedding, attention, tokens = detector.extract_features(question)

    # Average attention if multi-head
    if attention.ndim > 2:
        attention = attention.mean(axis=0)

    print(f"Tokens ({len(tokens)}): {tokens[:15]}...")
    print(f"Attention shape: {attention.shape}")

    # Build attention graph - NO HARDCODED LISTS
    graph_builder = AttentionGraphBuilder()
    graph = graph_builder.build_graph(attention, tokens, question)

    # Show detected entities (via attention_received, not hardcoded list)
    all_entities = []
    for span in graph.spans:
        all_entities.extend(span.entities)
    print(f"Entities detected (via attention_received): {[e.text for e in all_entities[:5]]}")

    # Show detected spans (via connectivity)
    print(f"Spans detected (via connectivity): {len(graph.spans)}")
    for i, span in enumerate(graph.spans[:3]):
        print(f"  Span {i+1}: '{span.text[:50]}...' (conn={span.connectivity:.3f})")
        if span.entities:
            print(f"    Entities: {[e.text for e in span.entities]}")

    # Show cross-span connections
    if graph.edges:
        print(f"Cross-span edges: {len(graph.edges)}")
        for src, dst, weight in graph.edges[:3]:
            print(f"  Span {src} -> Span {dst} (weight={weight:.3f})")

    # Match each span to templates
    print("\n--- SPAN → TEMPLATE MATCHES ---")
    for i, span in enumerate(graph.spans[:4]):  # Show first 4 spans
        # Get embedding for this span
        span_emb, span_att, span_tokens = detector.extract_features(span.text)
        if span_att.ndim > 2:
            span_att = span_att.mean(axis=0)
        attn_flat = span_att.flatten()

        # Find best matches for this span
        matches = store.find_top_k_matches(span_emb, attn_flat, k=3)

        print(f"\nSpan {i+1}: '{span.text[:50]}...'")
        if span.entities:
            print(f"  Entities: {[e.text for e in span.entities]}")

        if matches:
            for j, (template, score, emb_sim, att_sim) in enumerate(matches[:2]):
                print(f"  Match {j+1}: {template.operation_type.value} - {template.template_id[:30]}")
                print(f"    Score: {score:.4f} (emb={emb_sim:.4f}, att={att_sim:.4f})")
                print(f"    DSL: {template.dsl_expr}")
        else:
            print("  No matches found")

    # Run pipeline to get graph execution
    print("\n--- GRAPH EXECUTION ---")
    try:
        # Process problem through pipeline
        pipeline_output = pipeline._process_with_attention_graph(question)

        if pipeline_output.execution_result:
            exec_result = pipeline_output.execution_result
            print(f"Execution trace ({len(exec_result.execution_trace)} steps):")
            for step in exec_result.execution_trace[:5]:
                print(f"  {step}")
            if len(exec_result.execution_trace) > 5:
                print(f"  ... ({len(exec_result.execution_trace) - 5} more)")

            print(f"Entity values: {exec_result.entity_values}")
            print(f"Computed answer: {exec_result.answer}")

    except Exception as e:
        print(f"Graph execution error: {e}")
        import traceback
        traceback.print_exc()

    # Run actual solve
    print("\n--- SOLVE RESULT ---")
    try:
        result = solver.solve(question)
        answer = result.answer if result else None

        is_correct = False
        if answer is not None and expected is not None:
            if abs(expected) < 1e-6:
                is_correct = abs(answer - expected) < 1e-6
            else:
                is_correct = abs(answer - expected) / abs(expected) < 0.01

        print(f"Answer: {answer}")
        print(f"Correct: {'YES' if is_correct else 'NO'}")

        if result:
            if hasattr(result, 'confidence'):
                print(f"Confidence: {result.confidence:.4f}")
            if hasattr(result, 'method'):
                print(f"Method: {result.method}")
            if hasattr(result, 'steps') and result.steps:
                print(f"Steps: {len(result.steps)}")
                for step in result.steps[:3]:
                    print(f"  - {step}")

        return is_correct

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Load GSM8K
    print("\nLoading GSM8K test set...")
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    print(f"Total problems: {len(ds)}")

    # Initialize solver
    print("\nInitializing dual signal solver...")
    solver = DualSignalSolver()
    solver._ensure_pipeline()
    print(f"Templates loaded: {len(solver._pipeline.store.templates)}")

    # Sample random problems
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:args.num_problems]

    # Run debug on each
    correct = 0
    for i, idx in enumerate(sample_indices):
        problem = ds[idx]
        expected = extract_answer(problem['answer'])
        if expected is None:
            print(f"\nSkipping problem {idx} - could not extract answer")
            continue

        if debug_solve(solver, problem['question'], expected, i):
            correct += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {correct}/{args.num_problems} correct ({100*correct/args.num_problems:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
