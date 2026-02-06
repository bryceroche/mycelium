"""Debug the full pipeline: span detection → template matching → execution."""

import sys
sys.path.insert(0, 'src')

import json
from mycelium.dual_signal_pipeline import DualSignalPipeline

def debug_full_pipeline():
    # Initialize pipeline with quad-signal
    print("Loading pipeline...")

    # Find templates file
    import os
    templates_candidates = [
        'data/atomic_templates_raw_centroids.json',  # Raw centroids - better matching
        'data/atomic_templates_final_with_attention.json',
        'data/atomic_templates_final.json',
        'data/atomic_templates_1k.json',
        'qwen_templates.json',
        'deduplicated_templates.json',
    ]

    templates_path = None
    for candidate in templates_candidates:
        if os.path.exists(candidate):
            templates_path = candidate
            break

    print(f"Using templates: {templates_path}")

    # Use new position-adaptive weights (defaults from DualSignalPipeline)
    pipeline = DualSignalPipeline(
        templates_path=templates_path,
    )

    print(f"Templates loaded: {len(pipeline.store.templates)}")
    print(f"Signal weights: emb={pipeline.store.embedding_weight}, att={pipeline.store.attention_weight}, "
          f"ba={pipeline.store.backward_attention_weight}, graph={pipeline.store.graph_weight}")

    # Test problem
    problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print(f"\nProblem: {problem[:80]}...")
    print("\n" + "="*60)
    print("Processing...")
    print("="*60)

    output = pipeline.process_problem(problem)

    print(f"\nSpans detected: {output.spans_detected}")
    print(f"Templates available: {output.templates_available}")
    print(f"Matched operations: {len(output.matched_operations)}")

    print("\n" + "="*60)
    print("Matched Operations:")
    print("="*60)

    for i, op in enumerate(output.matched_operations):
        print(f"\n[{i}] Span: '{op.span_text[:50]}...'")
        print(f"    Template: {op.template_id}")
        print(f"    Combined score: {op.combined_score:.3f}")
        print(f"    Embedding sim: {op.embedding_similarity:.3f}")
        print(f"    Attention sim: {op.attention_similarity:.3f}")
        print(f"    Confidence: {op.confidence:.3f}")
        if op.subgraph:
            steps = op.subgraph.get('steps', [])
            if steps:
                op_type = steps[-1].get('op', 'unknown')
                print(f"    Operation: {op_type}")
                print(f"    Params: {list(op.subgraph.get('params', {}).keys())}")
                print(f"    Inputs: {list(op.subgraph.get('inputs', {}).keys())}")

    print("\n" + "="*60)
    print("Execution Result:")
    print("="*60)

    if output.execution_result:
        print(f"Answer: {output.execution_result.answer}")
        print(f"Entity values: {output.execution_result.entity_values}")
        print(f"Success: {output.execution_result.success}")
        if output.execution_result.error:
            print(f"Error: {output.execution_result.error}")
        print("\nExecution trace:")
        for line in output.execution_result.execution_trace:
            print(f"  {line}")
    else:
        print("No execution result!")

    print(f"\nExpected answer: 18.0")
    print(f"Got: {output.answer}")


if __name__ == "__main__":
    debug_full_pipeline()
