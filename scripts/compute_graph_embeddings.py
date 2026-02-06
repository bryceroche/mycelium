#!/usr/bin/env python3
"""Compute graph embeddings for templates.

This script adds graph embeddings to templates based on their SubGraphDSL structure.
Graph embeddings capture operational semantics - what the computation DOES,
not what the text SOUNDS LIKE.

Usage:
    python scripts/compute_graph_embeddings.py \
        --templates templates_1k_trained.json \
        --output templates_with_graph_emb.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.graph_embedder import (
    graph_to_embedding,
    hash_graph_structure,
    GRAPH_EMBEDDING_DIM,
)


def main():
    parser = argparse.ArgumentParser(description="Compute graph embeddings for templates")
    parser.add_argument(
        "--templates",
        default="templates_1k_trained.json",
        help="Input templates JSON file"
    )
    parser.add_argument(
        "--output",
        default="templates_with_graph_emb.json",
        help="Output file with graph embeddings added"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Computing Graph Embeddings")
    print("=" * 60)

    # Load templates
    print(f"\n1. Loading templates from {args.templates}...")
    with open(args.templates, 'r') as f:
        templates = json.load(f)

    print(f"   Loaded {len(templates)} templates")

    # Count templates with/without subgraphs
    with_subgraph = sum(1 for t in templates if t.get('subgraph'))
    print(f"   Templates with subgraph: {with_subgraph}")

    # Compute graph embeddings
    print(f"\n2. Computing {GRAPH_EMBEDDING_DIM}-dim graph embeddings...")

    structure_counts = Counter()
    computed = 0
    skipped = 0

    for template in templates:
        subgraph = template.get('subgraph')

        if subgraph:
            # Compute graph embedding
            graph_emb = graph_to_embedding(subgraph)
            template['graph_embedding'] = graph_emb.tolist()

            # Track structure for statistics
            structure = hash_graph_structure(subgraph)
            structure_counts[structure] += 1
            computed += 1
        else:
            # No subgraph - skip or use zero embedding
            skipped += 1

    print(f"   Computed: {computed}")
    print(f"   Skipped (no subgraph): {skipped}")

    # Show structure distribution
    print(f"\n3. Graph structure distribution (top 20):")
    for structure, count in structure_counts.most_common(20):
        ops = structure.split('_')[0]
        has_inputs = '_True' in structure
        inputs_str = "+inputs" if has_inputs else ""
        print(f"   {count:4d}x {ops}{inputs_str}")

    # Count unique structures
    print(f"\n   Total unique structures: {len(structure_counts)}")

    # Save output
    print(f"\n4. Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(templates, f, indent=2)

    print(f"   Saved {len(templates)} templates with graph embeddings")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Templates processed: {len(templates)}")
    print(f"Graph embeddings added: {computed}")
    print(f"Unique graph structures: {len(structure_counts)}")

    # Show operation type breakdown
    print("\nOperation type breakdown:")
    op_counts = Counter()
    for t in templates:
        sg = t.get('subgraph', {})
        steps = sg.get('steps', [])
        for step in steps:
            op_counts[step.get('op', 'UNKNOWN')] += 1

    for op, count in op_counts.most_common():
        print(f"  {op}: {count}")


if __name__ == "__main__":
    main()
