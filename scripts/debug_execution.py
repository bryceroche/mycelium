#!/usr/bin/env python3
"""Debug execution flow to trace which spans get DSLs and why."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.dual_signal_pipeline import DualSignalPipeline

def main():
    # Janet problem
    problem = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""

    print("=" * 60)
    print("DEBUG: Execution Flow")
    print("=" * 60)

    # Load pipeline
    print("\n1. Loading pipeline...")
    pipeline = DualSignalPipeline(
        templates_path="templates_with_attention.json"
    )

    print(f"   Templates: {len(pipeline.store.templates)}")
    print(f"   Weights: emb={pipeline.store.embedding_weight:.2f}, "
          f"att={pipeline.store.attention_weight:.2f}, "
          f"graph={pipeline.store.graph_weight:.2f}")

    # Extract spans
    print("\n2. Extracting features and building graph...")
    embedding, attention_matrix, tokens = pipeline.detector.extract_features(problem)
    if attention_matrix.ndim > 2:
        attention_matrix = attention_matrix.mean(axis=0)

    graph = pipeline.graph_builder.build_graph(attention_matrix, tokens, problem)
    print(f"   Spans detected: {len(graph.spans)}")
    print(f"   Graph edges: {len(graph.edges)}")
    for src, dst, weight in graph.edges[:10]:
        print(f"      [{src}] -> [{dst}]: {weight:.3f}")

    # Match each span
    print("\n3. Matching spans to templates...")
    span_subgraphs = []

    for span_idx, span in enumerate(graph.spans):
        # Compute backward attention: how much this span attends to earlier spans
        backward_attn = sum(
            weight for src, dst, weight in graph.edges
            if src == span_idx and dst < span_idx
        )
        print(f"\n   Span [{span_idx}]: '{span.text[:50]}...' " if len(span.text) > 50 else f"\n   Span [{span_idx}]: '{span.text}'")
        print(f"      Backward attention: {backward_attn:.3f} (needs_upstream: {backward_attn > 0.05})")

        # Get span embedding
        span_embedding, span_attention, _ = pipeline.detector.extract_features(span.text)
        if span_attention.ndim > 2:
            span_attention = span_attention.mean(axis=0)
        attention_flat = span_attention.flatten()

        # Find match
        result = pipeline.store.find_best_match(span_embedding, attention_flat, graph_embedding=None)

        if result:
            template, combined_score, emb_sim, att_sim, graph_sim = result
            print(f"      -> Matched: {template.template_id}")
            print(f"         Score: {combined_score:.3f} (emb={emb_sim:.3f}, att={att_sim:.3f}, graph={graph_sim:.3f})")
            print(f"         Pattern: {template.pattern[:50]}..." if len(template.pattern) > 50 else f"         Pattern: {template.pattern}")

            if template.subgraph:
                sg = template.subgraph
                ops = [s.get("op") if isinstance(s, dict) else s.op for s in sg.get("steps", []) if s]
                print(f"         Subgraph: {ops}")
                print(f"         Params: {list(sg.get('params', {}).keys())}")
                print(f"         Inputs: {list(sg.get('inputs', {}).keys())}")
                span_subgraphs.append((span_idx, sg))
            else:
                print("         Subgraph: NONE!")
        else:
            print("      -> NO MATCH!")

    print(f"\n4. Executing {len(span_subgraphs)} spans with DSLs...")

    # Execute
    execution_result = pipeline.graph_executor.execute_graph_with_subgraphs(
        graph, span_subgraphs, attention_matrix
    )

    print("\n5. Execution Result:")
    print(f"   Entity values: {execution_result.entity_values}")
    print(f"   Answer: {execution_result.answer}")
    print(f"   Success: {execution_result.success}")
    print("\n   Trace:")
    for line in execution_result.execution_trace:
        print(f"      {line}")

    # Expected answer
    expected = 18.0
    print(f"\n6. Expected: {expected}, Got: {execution_result.answer}")
    print(f"   Correct: {execution_result.answer == expected}")


if __name__ == "__main__":
    main()
