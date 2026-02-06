#!/usr/bin/env python3
"""Recompute template centroids from raw span_examples instead of vocab-constrained patterns.

This fixes the embedding mismatch:
- Before: centroid from "<NUM> eggs"
- After: centroid from average of ["16 eggs", "12 eggs", "5 apples"]

The raw span_examples match inference spans much better.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # Load templates
    with open('data/atomic_templates_final_with_attention.json') as f:
        templates = json.load(f)

    print(f"Loaded {len(templates)} templates")

    # Load embedding model
    print("Loading MiniLM...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Recompute centroids from raw span_examples
    print("Recomputing centroids from raw span_examples...")

    updated = 0
    for t in templates:
        examples = t.get('pattern_examples', t.get('span_examples', []))

        if not examples:
            continue

        # Take up to 20 examples for centroid computation
        sample = examples[:20]

        # Embed raw examples
        embeddings = model.encode(sample, convert_to_numpy=True)

        # Compute centroid (average)
        centroid = embeddings.mean(axis=0)

        # Normalize for cosine similarity
        centroid = centroid / np.linalg.norm(centroid)

        # Update template
        t['embedding_centroid'] = centroid.tolist()
        updated += 1

    print(f"Updated {updated} template centroids")

    # Save
    output_path = 'data/atomic_templates_raw_centroids.json'
    with open(output_path, 'w') as f:
        json.dump(templates, f)

    print(f"Saved to {output_path}")

    # Quick sanity check
    print("\nSanity check - test matching:")
    test_span = "16 eggs per day"
    test_emb = model.encode([test_span], convert_to_numpy=True)[0]
    test_emb = test_emb / np.linalg.norm(test_emb)

    # Find top matches
    scores = []
    for t in templates:
        centroid = np.array(t['embedding_centroid'])
        sim = np.dot(test_emb, centroid)
        scores.append((t['template_id'], t['pattern'], sim))

    scores.sort(key=lambda x: -x[2])
    print(f"\nTop matches for '{test_span}':")
    for tid, pattern, sim in scores[:5]:
        print(f"  {sim:.3f} {tid}: {pattern[:40]}")


if __name__ == "__main__":
    main()
