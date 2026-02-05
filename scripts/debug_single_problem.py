#!/usr/bin/env python3
"""Debug a single GSM8K problem through the full pipeline.

Shows exactly which spans were detected and which templates matched.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.dual_signal_pipeline import DualSignalPipeline
from mycelium.attention_graph import AttentionGraphBuilder, GraphExecutor

TEMPLATES_PATH = str(Path(__file__).parent.parent / "templates_1k_with_dsl.json")

PROBLEM = (
    "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells every remaining egg "
    "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she "
    "make every day at the farmers' market?"
)
EXPECTED = 18.0


def main():
    print("=" * 70)
    print("DEBUG: Single Problem Trace")
    print("=" * 70)
    print(f"\nProblem: {PROBLEM}")
    print(f"Expected answer: {EXPECTED}")

    # Initialize pipeline with our 1K templates
    print(f"\nLoading pipeline with templates from: {TEMPLATES_PATH}")
    pipeline = DualSignalPipeline()
    pipeline.load_templates(TEMPLATES_PATH, replace=True)
    print(f"Templates loaded: {len(pipeline.store.templates)}")

    # Step 1: Extract features
    print("\n" + "=" * 70)
    print("STEP 1: Feature Extraction (MiniLM)")
    print("=" * 70)
    embedding, attention_matrix, tokens = pipeline.detector.extract_features(PROBLEM)
    if attention_matrix.ndim > 2:
        attention_matrix = attention_matrix.mean(axis=0)
    print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Attention matrix shape: {attention_matrix.shape}")

    # Step 2: Build span graph
    print("\n" + "=" * 70)
    print("STEP 2: Span Graph (Attention-Based)")
    print("=" * 70)
    graph = pipeline.graph_builder.build_graph(attention_matrix, tokens, PROBLEM)
    print(f"Spans detected: {len(graph.spans)}")
    print(f"Edges: {len(graph.edges)}")

    for i, span in enumerate(graph.spans):
        entities_str = ", ".join(
            f"{e.text}(attn={e.attention_received:.4f})" for e in span.entities
        )
        print(f"\n  Span {i}: \"{span.text}\"")
        print(f"    Tokens: {span.token_indices[:10]}{'...' if len(span.token_indices) > 10 else ''}")
        print(f"    Connectivity: {span.connectivity:.4f}")
        print(f"    Entities: [{entities_str}]")

    print(f"\n  Edges (cross-span attention):")
    for src, dst, weight in graph.edges:
        print(f"    Span {src} -> Span {dst} (weight={weight:.4f})")

    # Step 3: Template matching per span
    print("\n" + "=" * 70)
    print("STEP 3: Template Matching (Dual-Signal)")
    print("=" * 70)

    span_templates = []
    for span_idx, span in enumerate(graph.spans):
        # Embed the span
        span_emb, span_attn, span_tokens = pipeline.detector.extract_features(span.text)
        if span_attn.ndim > 2:
            span_attn = span_attn.mean(axis=0)
        attn_flat = span_attn.flatten()

        # Find top-3 matches
        results = pipeline.store.find_top_k_matches(span_emb, attn_flat, k=3)

        print(f"\n  Span {span_idx}: \"{span.text}\"")
        if results:
            for rank, (template, combined, emb_sim, att_sim) in enumerate(results):
                marker = " <-- SELECTED" if rank == 0 else ""
                print(f"    #{rank+1}: template={template.template_id[:40]}...")
                print(f"         pattern=\"{template.pattern[:60]}\"")
                print(f"         dsl_expr=\"{template.dsl_expr}\"")
                print(f"         combined={combined:.4f} (emb={emb_sim:.4f}, att={att_sim:.4f}){marker}")
                if template.span_examples:
                    print(f"         examples: {template.span_examples[:2]}")

            best = results[0]
            span_templates.append((span_idx, best[0].dsl_expr))
        else:
            print(f"    NO MATCH")
            span_templates.append((span_idx, "value"))

    # Step 4: Graph execution
    print("\n" + "=" * 70)
    print("STEP 4: Graph Execution")
    print("=" * 70)

    result = pipeline.graph_executor.execute_graph_with_attention(
        graph, span_templates, attention_matrix
    )

    print(f"\nExecution trace:")
    for line in result.execution_trace:
        print(f"  {line}")
    print(f"\nEntity values: {result.entity_values}")
    print(f"Answer: {result.answer}")
    print(f"Expected: {EXPECTED}")
    print(f"Correct: {abs(result.answer - EXPECTED) < 0.01 if result.answer else False}")


if __name__ == "__main__":
    main()
