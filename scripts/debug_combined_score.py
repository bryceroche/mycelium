"""Debug the combined score calculation."""

import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from mycelium.dual_signal_templates import SpanDetector, TemplateStore, DualSignalTemplate

def debug_combined_score():
    # Load templates
    with open('data/atomic_templates_raw_centroids.json') as f:
        templates = json.load(f)

    print(f"Loaded {len(templates)} templates")

    # Load detector
    print("Loading MiniLM...")
    detector = SpanDetector()

    # Create template store with default weights (position-adaptive)
    store = TemplateStore()  # Uses new defaults: emb=0.55, att=0.15, ba=0.15, graph=0.15

    # Add templates to store
    for t in templates:
        template = DualSignalTemplate(
            template_id=t['template_id'],
            embedding_centroid=np.array(t['embedding_centroid'], dtype=np.float32),
            attention_signature=np.zeros(100, dtype=np.float32),  # No attention signature
            pattern=t.get('pattern', ''),
            subgraph=t.get('subgraph'),
            backward_attention=t.get('backward_attention', 0.5),
            attention_entropy=t.get('attention_entropy', 0.5),
            span_position=t.get('span_position', 0.5),
            span_examples=t.get('pattern_examples', [])
        )
        store.add_template(template)

    print(f"Templates in store: {len(store.templates)}")
    print(f"Weights: emb={store.embedding_weight}, att={store.attention_weight}, "
          f"ba={store.backward_attention_weight}, graph={store.graph_weight}")

    # Test span
    span_text = "16 eggs per day"
    print(f"\n--- Testing: '{span_text}' ---")

    # Embed span
    embedding, attention, _ = detector.extract_features(span_text)
    if attention.ndim > 2:
        attention = attention.mean(axis=0)
    attention_flat = attention.flatten()

    # Simulate backward_attention for this span (as computed in pipeline)
    # For the first few spans, this would be low
    query_ba_values = [0.0, 0.5, 1.0, 2.0, 3.0]

    for query_ba in query_ba_values:
        print(f"\n  Query backward_attention = {query_ba}")

        # Find best match
        result = store.find_best_match(
            embedding, attention_flat,
            query_backward_attention=query_ba
        )

        if result:
            template, combined, emb_sim, att_sim, graph_sim = result
            sg = template.subgraph or {}
            steps = sg.get('steps', [])
            op = steps[0].get('op', 'N/A') if steps else 'N/A'
            inputs = list(sg.get('inputs', {}).keys())
            input_str = f", needs: {inputs}" if inputs else " (self-contained)"

            print(f"    Best: {template.template_id} {op}{input_str}")
            print(f"    template_ba={template.backward_attention:.2f}")
            print(f"    combined={combined:.4f}, emb_sim={emb_sim:.4f}, att_sim={att_sim:.4f}")

    # Show top-5 for query_ba=0 (first span scenario)
    print(f"\n  Top-5 for query_ba=0.0:")
    matches = store.find_top_k_matches(
        embedding, attention_flat,
        k=5,
        query_backward_attention=0.0
    )
    for template, combined, emb_sim, att_sim, graph_sim in matches:
        sg = template.subgraph or {}
        steps = sg.get('steps', [])
        op = steps[0].get('op', 'N/A') if steps else 'N/A'
        print(f"    {combined:.4f} {template.template_id:12s} {op:4s} emb={emb_sim:.4f}, ba={template.backward_attention:.2f}")


if __name__ == "__main__":
    debug_combined_score()
