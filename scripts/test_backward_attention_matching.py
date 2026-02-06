"""Test backward_attention matching in quad-signal template matching."""

import json
import numpy as np
import sys
sys.path.insert(0, 'src')

from mycelium.dual_signal_templates import TemplateStore, DualSignalTemplate

def test_backward_attention_matching():
    """Verify that backward_attention signal correctly routes to operation types."""

    # Load templates
    with open('data/atomic_templates_final_with_attention.json') as f:
        templates_data = json.load(f)

    print(f"Loaded {len(templates_data)} templates")

    # Create TemplateStore
    store = TemplateStore()

    # Group templates by operation type for analysis
    by_op = {}
    for t in templates_data:
        template = DualSignalTemplate(
            template_id=t['template_id'],
            embedding_centroid=np.array(t['embedding_centroid'], dtype=np.float32),
            attention_signature=np.zeros(384, dtype=np.float32),  # Placeholder
            pattern=t.get('pattern', ''),
            subgraph=t.get('subgraph'),
            backward_attention=t.get('backward_attention', 0.5),
            attention_entropy=t.get('attention_entropy', 0.5),
            span_position=t.get('span_position', 0.5),
            span_examples=t.get('pattern_examples', [])
        )
        store.add_template(template)

        # Track by operation
        sg = t.get('subgraph', {})
        if sg and 'steps' in sg:
            op = sg['steps'][0].get('op', 'unknown')
            if op not in by_op:
                by_op[op] = []
            by_op[op].append((t['template_id'], t.get('backward_attention', 0)))

    print(f"\nOperation distribution: {[(k, len(v)) for k, v in by_op.items()]}")
    print(f"\nBackward attention by operation:")
    for op, items in sorted(by_op.items()):
        ba_values = [x[1] for x in items]
        print(f"  {op}: mean={np.mean(ba_values):.3f}, min={min(ba_values):.3f}, max={max(ba_values):.3f}")

    # Test with synthetic queries
    # Use the first template's embedding as base query
    base_embedding = np.array(templates_data[0]['embedding_centroid'], dtype=np.float32)
    dummy_attention = np.zeros(384, dtype=np.float32)

    print("\n" + "="*60)
    print("TEST: Low backward_attention query (should match SET)")
    print("="*60)

    # Query with low backward_attention (like initial value setting)
    result = store.find_best_match(
        embedding=base_embedding,
        attention=dummy_attention,
        query_backward_attention=0.0  # Very low - initial value
    )

    if result:
        template, score, emb_sim, att_sim, graph_sim = result
        sg = template.subgraph
        op = sg['steps'][0].get('op', 'unknown') if sg and 'steps' in sg else 'unknown'
        print(f"Matched: {template.template_id}")
        print(f"  Operation: {op}")
        print(f"  Template backward_attention: {template.backward_attention:.3f}")
        print(f"  Combined score: {score:.3f}")
        print(f"  Pattern: {template.pattern[:60]}...")

    print("\n" + "="*60)
    print("TEST: High backward_attention query (should match consuming op)")
    print("="*60)

    # Query with high backward_attention (like subtraction/multiplication)
    result = store.find_best_match(
        embedding=base_embedding,
        attention=dummy_attention,
        query_backward_attention=6.0  # High - consuming operation
    )

    if result:
        template, score, emb_sim, att_sim, graph_sim = result
        sg = template.subgraph
        op = sg['steps'][0].get('op', 'unknown') if sg and 'steps' in sg else 'unknown'
        print(f"Matched: {template.template_id}")
        print(f"  Operation: {op}")
        print(f"  Template backward_attention: {template.backward_attention:.3f}")
        print(f"  Combined score: {score:.3f}")
        print(f"  Pattern: {template.pattern[:60]}...")

    print("\n" + "="*60)
    print("TEST: Compare top-k matches for both queries")
    print("="*60)

    # Get top-5 for low backward_attention
    print("\nTop-5 for query_backward_attention=0.0:")
    matches = store.find_top_k_matches(
        embedding=base_embedding,
        attention=dummy_attention,
        k=5,
        query_backward_attention=0.0
    )
    for template, score, emb_sim, att_sim, graph_sim in matches:
        sg = template.subgraph
        op = sg['steps'][0].get('op', 'unknown') if sg and 'steps' in sg else 'unknown'
        print(f"  {op:4s} ba={template.backward_attention:.2f} score={score:.3f}")

    # Get top-5 for high backward_attention
    print("\nTop-5 for query_backward_attention=6.0:")
    matches = store.find_top_k_matches(
        embedding=base_embedding,
        attention=dummy_attention,
        k=5,
        query_backward_attention=6.0
    )
    for template, score, emb_sim, att_sim, graph_sim in matches:
        sg = template.subgraph
        op = sg['steps'][0].get('op', 'unknown') if sg and 'steps' in sg else 'unknown'
        print(f"  {op:4s} ba={template.backward_attention:.2f} score={score:.3f}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Weights: emb={store.embedding_weight:.2f} att={store.attention_weight:.2f} ba={store.backward_attention_weight:.2f} graph={store.graph_weight:.2f}")
    print("If backward_attention routing is working, low query should prefer SET, high query should prefer SUB/MUL/ADD")


if __name__ == "__main__":
    test_backward_attention_matching()
