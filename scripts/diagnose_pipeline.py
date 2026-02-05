#!/usr/bin/env python3
"""Diagnostic script: trace the dual-signal pipeline end-to-end.

Randomly samples GSM8K problems (fresh seed each run), runs through
the pipeline, and shows detailed span detection / template matching /
execution trace to debug wrong answers.

Usage:
    python scripts/diagnose_pipeline.py --templates /tmp/qwen_templates_smoke_120_v2.json -n 5
"""

import json
import random
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.dual_signal_pipeline import DualSignalPipeline
from mycelium.attention_graph import AttentionGraphBuilder


def load_gsm8k_problems(path: str, n: int, seed: int = None):
    """Load n random GSM8K problems with a fresh seed."""
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    print(f"Random seed: {seed}")
    rng = random.Random(seed)

    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))

    sampled = rng.sample(problems, min(n, len(problems)))
    return sampled, seed


def diagnose_problem(pipeline: DualSignalPipeline, problem_text: str, answer: str):
    """Run one problem through the pipeline with detailed tracing."""
    print(f"\n{'='*70}")
    print(f"PROBLEM: {problem_text[:120]}...")
    print(f"EXPECTED ANSWER: {answer}")
    print(f"{'='*70}")

    # Step 1: Extract features
    embedding, attention_matrix, tokens = pipeline.detector.extract_features(problem_text)
    if attention_matrix.ndim > 2:
        attention_matrix = attention_matrix.mean(axis=0)

    print(f"\n[1] TOKENIZATION: {len(tokens)} tokens")
    print(f"    Tokens: {tokens[:30]}{'...' if len(tokens) > 30 else ''}")

    # Step 2: Detect span boundaries
    builder = pipeline.graph_builder
    boundaries = builder.detect_span_boundaries(attention_matrix, tokens)
    print(f"\n[2] SPAN BOUNDARIES: {len(boundaries)} spans")
    from mycelium.attention_graph import _join_wordpiece_tokens
    for i, (start, end) in enumerate(boundaries):
        span_tokens = tokens[start:end]
        span_text = _join_wordpiece_tokens(span_tokens)
        connectivity = builder.compute_span_connectivity(attention_matrix, start, end)
        print(f"    Span {i}: [{start}:{end}] conn={connectivity:.4f}")
        print(f"      Tokens: {span_tokens}")
        print(f"      Text: '{span_text}'")

    # Step 3: Detect entities
    entities = builder.detect_entities(attention_matrix, tokens)
    print(f"\n[3] ENTITIES: {len(entities)} detected")
    for e in entities:
        print(f"    '{e.text}' attn_received={e.attention_received:.4f} indices={e.token_indices}")

    # Step 4: Build graph
    graph = builder.build_graph(attention_matrix, tokens, problem_text)
    print(f"\n[4] GRAPH: {len(graph.spans)} spans, {len(graph.edges)} edges")
    for i, span in enumerate(graph.spans):
        print(f"    Span {i}: '{span.text[:60]}' conn={span.connectivity:.4f}")
        if span.entities:
            for e in span.entities:
                print(f"      Entity: '{e.text}' attn={e.attention_received:.4f}")

    # Step 5: Match each span to template
    print(f"\n[5] TEMPLATE MATCHING:")
    span_templates = []
    for span_idx, span in enumerate(graph.spans):
        match = pipeline._match_span_to_template(span, attention_matrix, tokens)
        if match:
            print(f"    Span {span_idx}: '{span.text[:50]}' -> {match.dsl_expr} "
                  f"(conf={match.confidence:.3f}, emb={match.embedding_similarity:.3f}, "
                  f"att={match.attention_similarity:.3f})")
            print(f"      Template: {match.template_id}")
            print(f"      DSL: {match.dsl_expr}")
            span_templates.append((span_idx, match.dsl_expr))
        else:
            print(f"    Span {span_idx}: '{span.text[:50]}' -> NO MATCH")

    # Step 6: Execute graph
    exec_result = pipeline.graph_executor.execute_graph_with_attention(
        graph, span_templates, attention_matrix
    )
    print(f"\n[6] EXECUTION:")
    for step in exec_result.execution_trace:
        print(f"    {step}")
    print(f"    Answer: {exec_result.answer}")
    print(f"    Success: {exec_result.success}")
    if exec_result.error:
        print(f"    Error: {exec_result.error}")

    # Step 7: Compare
    try:
        expected = float(answer)
        computed = exec_result.answer
        correct = computed is not None and abs(computed - expected) < 0.01
        print(f"\n[7] RESULT: {'CORRECT' if correct else 'WRONG'}")
        if not correct and computed is not None:
            print(f"    Expected: {expected}, Got: {computed}, Diff: {abs(computed - expected):.2f}")
    except (ValueError, TypeError):
        print(f"\n[7] RESULT: Cannot compare (expected={answer})")

    return exec_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Diagnostics")
    parser.add_argument("--templates", default="/tmp/qwen_templates_smoke_120_v2.json",
                       help="Path to templates JSON")
    parser.add_argument("--data", default="data/gsm8k_gts/test.jsonl",
                       help="Path to GSM8K problems")
    parser.add_argument("-n", type=int, default=5, help="Number of problems to test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (fresh if omitted)")
    parser.add_argument("--problem", type=str, default=None, help="Specific problem text to test")
    args = parser.parse_args()

    # Initialize pipeline
    print("Loading pipeline...")
    pipeline = DualSignalPipeline(
        templates_path=args.templates,
        embedding_weight=0.5,
        attention_weight=0.5,
    )
    stats = pipeline.get_stats()
    print(f"Templates loaded: {stats['total_templates']}")
    print(f"Distribution: {stats['templates_by_operation']}")

    if args.problem:
        diagnose_problem(pipeline, args.problem, "?")
        return

    # Load random problems
    problems, seed = load_gsm8k_problems(args.data, args.n, seed=args.seed)

    correct = 0
    total = 0
    for p in problems:
        result = diagnose_problem(pipeline, p["question"], p["answer"])
        total += 1
        try:
            expected = float(p["answer"])
            if result.answer is not None and abs(result.answer - expected) < 0.01:
                correct += 1
        except (ValueError, TypeError):
            pass

    print(f"\n{'='*70}")
    print(f"SUMMARY: {correct}/{total} correct ({100*correct/total:.0f}%) | seed={seed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
