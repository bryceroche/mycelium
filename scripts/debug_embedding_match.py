"""Debug why wrong template is being selected."""

import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from mycelium.dual_signal_templates import SpanDetector

def debug_embedding_match():
    # Load templates
    with open('data/atomic_templates_raw_centroids.json') as f:
        templates = json.load(f)

    print(f"Loaded {len(templates)} templates")

    # Load detector
    print("Loading MiniLM...")
    detector = SpanDetector()

    # Test spans from the problem
    test_spans = [
        "16 eggs per day",
        "she eats three",
        "three for breakfast",
        "she sells",
    ]

    # Target templates we want to match
    target_templates = ['atomic_0366', 'atomic_0550', 'atomic_0042', 'atomic_0831']

    # Build template lookup
    template_lookup = {t['template_id']: t for t in templates}

    print("\n" + "="*70)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*70)

    for span_text in test_spans:
        print(f"\n--- Span: '{span_text}' ---")

        # Embed the span
        embedding, _, _ = detector.extract_features(span_text)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Compute similarity to all templates
        scores = []
        for t in templates:
            centroid = np.array(t['embedding_centroid'], dtype=np.float32)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            sim = np.dot(embedding, centroid)

            sg = t.get('subgraph', {})
            steps = sg.get('steps', [])
            op = steps[0].get('op', 'N/A') if steps else 'N/A'
            inputs = list(sg.get('inputs', {}).keys())
            ba = t.get('backward_attention', 0)

            scores.append((t['template_id'], sim, op, inputs, ba, t.get('pattern_examples', [])[:2]))

        scores.sort(key=lambda x: -x[1])

        print(f"\nTop 10 matches:")
        for tid, sim, op, inputs, ba, examples in scores[:10]:
            input_str = f", needs: {inputs}" if inputs else " (self-contained)"
            print(f"  {sim:.4f} {tid:12s} {op:4s} ba={ba:.2f}{input_str}")
            print(f"           examples: {examples}")

        # Show specific target templates
        print(f"\nTarget template scores:")
        for target_tid in target_templates:
            if target_tid in template_lookup:
                t = template_lookup[target_tid]
                centroid = np.array(t['embedding_centroid'], dtype=np.float32)
                centroid = centroid / np.linalg.norm(centroid)
                sim = np.dot(embedding, centroid)

                sg = t.get('subgraph', {})
                steps = sg.get('steps', [])
                op = steps[0].get('op', 'N/A') if steps else 'N/A'
                inputs = list(sg.get('inputs', {}).keys())
                ba = t.get('backward_attention', 0)

                input_str = f", needs: {inputs}" if inputs else " (self-contained)"
                print(f"  {sim:.4f} {target_tid:12s} {op:4s} ba={ba:.2f}{input_str}")


if __name__ == "__main__":
    debug_embedding_match()
