#!/usr/bin/env python3
"""
Recompute embedding centroids for deduplicated templates.

The current embeddings are nearly identical (0.99 cosine similarity) which makes
template matching useless. This script recomputes proper embeddings from the
actual span examples using MiniLM.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from sentence_transformers import SentenceTransformer


def recompute_embeddings(input_path: str, output_path: str):
    """Recompute embedding centroids from span examples."""
    print("=" * 60)
    print("RECOMPUTING TEMPLATE EMBEDDINGS")
    print("=" * 60)

    # Load templates
    print("\n[1] Loading templates...")
    with open(input_path, 'r') as f:
        templates = json.load(f)
    print(f"    Loaded {len(templates)} templates")

    # Load encoder
    print("\n[2] Loading MiniLM encoder...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"    Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Recompute embeddings
    print("\n[3] Recomputing embeddings from span examples...")
    updated = 0
    skipped = 0

    for i, template in enumerate(templates):
        spans = template.get('span_examples', [])

        if not spans:
            skipped += 1
            continue

        # Encode all spans
        embeddings = model.encode(spans, convert_to_numpy=True)

        # Compute centroid (mean of span embeddings)
        centroid = np.mean(embeddings, axis=0)

        # Normalize to unit vector
        centroid = centroid / np.linalg.norm(centroid)

        # Update template
        template['embedding_centroid'] = centroid.tolist()
        updated += 1

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(templates)}")

    print(f"    Updated: {updated}, Skipped: {skipped}")

    # Verify diversity
    print("\n[4] Verifying embedding diversity...")
    templates_by_op = {}
    for t in templates:
        op = t['operation']
        if op not in templates_by_op:
            templates_by_op[op] = t

    print("    Cross-operation similarity (should be < 0.95 for useful embeddings):")
    ops = list(templates_by_op.keys())
    for i, op1 in enumerate(ops):
        for op2 in ops[i+1:]:
            a = np.array(templates_by_op[op1]['embedding_centroid'])
            b = np.array(templates_by_op[op2]['embedding_centroid'])
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            status = "OK" if cos_sim < 0.95 else "SIMILAR"
            print(f"      {op1} vs {op2}: {cos_sim:.4f} [{status}]")

    # Check within-operation similarity (should be higher than cross-op)
    print("\n    Within-operation similarity (first 3 templates per op):")
    for op in ops:
        op_templates = [t for t in templates if t['operation'] == op][:3]
        if len(op_templates) >= 2:
            a = np.array(op_templates[0]['embedding_centroid'])
            b = np.array(op_templates[1]['embedding_centroid'])
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            print(f"      {op}: {cos_sim:.4f}")

    # Save updated templates
    print(f"\n[5] Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(templates, f, indent=2)
    print("    Done!")

    return templates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="deduplicated_templates.json")
    parser.add_argument("--output", default="deduplicated_templates.json")
    parser.add_argument("--backup", action="store_true", help="Create backup before overwriting")
    args = parser.parse_args()

    if args.backup and args.input == args.output:
        import shutil
        backup_path = args.input.replace('.json', '_backup.json')
        shutil.copy(args.input, backup_path)
        print(f"Created backup: {backup_path}")

    recompute_embeddings(args.input, args.output)
