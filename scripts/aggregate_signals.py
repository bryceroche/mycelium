#!/usr/bin/env python3
"""
Aggregate Qwen attention signals from 17k specialized templates to deduped templates.

Strategy:
1. Load 17,749 specialized templates (have Qwen signals: entropy, received, connection)
2. Load 207 deduped templates (no Qwen signals)
3. For each deduped template, find similar specialized templates by embedding cosine similarity
4. Aggregate signals: mean + variance (variance = routing confidence measure)
5. Save enhanced deduped templates with aggregated signals

USAGE:
    python scripts/aggregate_signals.py
    python scripts/aggregate_signals.py --output enhanced_dsl_library.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class AggregatedSignals:
    """Aggregated Qwen attention signals with variance."""
    # Mean signals (for routing)
    entropy_mean: float = 0.0
    received_mean: float = 0.0
    connection_mean: float = 0.0

    # Variance (for confidence - low variance = reliable signal)
    entropy_var: float = 0.0
    received_var: float = 0.0
    connection_var: float = 0.0

    # Min/max range (for sanity checks)
    entropy_min: float = 0.0
    entropy_max: float = 0.0
    received_min: float = 0.0
    received_max: float = 0.0
    connection_min: float = 0.0
    connection_max: float = 0.0

    # Metadata
    source_count: int = 0  # How many specialized templates contributed
    avg_similarity: float = 0.0  # Average cosine similarity to sources


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_specialized_templates(path: str) -> Dict[str, dict]:
    """Load specialized templates with Qwen signals."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both dict-of-dicts and list-of-dicts formats
    if isinstance(data, dict):
        if 'templates' in data:
            # {templates: [...]} format
            templates = {t.get('template_id', f'tmpl_{i}'): t for i, t in enumerate(data['templates'])}
        else:
            # {id: {...}, id: {...}} format
            templates = data
    else:
        # [{...}, {...}] format
        templates = {t.get('template_id', f'tmpl_{i}'): t for i, t in enumerate(data)}

    print(f"Loaded {len(templates)} specialized templates")
    return templates


def load_deduped_templates(path: str) -> Dict[str, dict]:
    """Load deduped templates."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Handle both dict-of-dicts and list-of-dicts formats
    if isinstance(data, dict):
        if 'templates' in data:
            templates = {t.get('template_id', f'dedup_{i}'): t for i, t in enumerate(data['templates'])}
        else:
            templates = data
    else:
        templates = {t.get('template_id', f'dedup_{i}'): t for i, t in enumerate(data)}

    print(f"Loaded {len(templates)} deduped templates")
    return templates


def aggregate_signals(
    deduped_templates: Dict[str, dict],
    specialized_templates: Dict[str, dict],
    similarity_threshold: float = 0.7,
    top_k: int = 50
) -> Dict[str, dict]:
    """
    For each deduped template, find similar specialized templates and aggregate their signals.

    Args:
        deduped_templates: Templates to enhance
        specialized_templates: Source templates with Qwen signals
        similarity_threshold: Minimum cosine similarity to include
        top_k: Maximum sources to aggregate per template

    Returns:
        Enhanced deduped templates with aggregated signals
    """
    print(f"\nAggregating signals (threshold={similarity_threshold}, top_k={top_k})...")

    # Extract embeddings from specialized templates
    spec_embeddings = {}
    spec_signals = {}

    for tid, tmpl in specialized_templates.items():
        emb = tmpl.get('embedding_centroid', [])
        if len(emb) >= 384:
            spec_embeddings[tid] = np.array(emb[:384])
            spec_signals[tid] = {
                'entropy': tmpl.get('attention_entropy', 0.0),
                'received': tmpl.get('attention_received', 0.0),
                'connection': tmpl.get('attention_connection', 0.0),
            }

    print(f"  {len(spec_embeddings)} specialized templates have valid embeddings")

    # Pre-compute normalized specialized embeddings for faster search
    spec_ids = list(spec_embeddings.keys())
    spec_matrix = np.stack([spec_embeddings[tid] for tid in spec_ids])
    spec_norms = np.linalg.norm(spec_matrix, axis=1, keepdims=True)
    spec_normalized = spec_matrix / (spec_norms + 1e-8)

    enhanced = {}
    stats = defaultdict(list)

    for tid, tmpl in deduped_templates.items():
        emb = tmpl.get('embedding_centroid', [])

        if len(emb) < 384:
            # No embedding - can't aggregate
            enhanced[tid] = tmpl.copy()
            enhanced[tid]['aggregated_signals'] = asdict(AggregatedSignals())
            stats['no_embedding'].append(tid)
            continue

        # Find similar specialized templates
        query = np.array(emb[:384])
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            enhanced[tid] = tmpl.copy()
            enhanced[tid]['aggregated_signals'] = asdict(AggregatedSignals())
            stats['zero_embedding'].append(tid)
            continue

        query_normalized = query / query_norm
        similarities = spec_normalized @ query_normalized

        # Get top-k above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Collect signals from similar templates
        entropy_vals = []
        received_vals = []
        connection_vals = []
        sim_vals = []

        for idx in top_indices:
            sim = similarities[idx]
            if sim < similarity_threshold:
                break

            src_tid = spec_ids[idx]
            signals = spec_signals[src_tid]

            entropy_vals.append(signals['entropy'])
            received_vals.append(signals['received'])
            connection_vals.append(signals['connection'])
            sim_vals.append(sim)

        if len(entropy_vals) == 0:
            # No similar templates found
            enhanced[tid] = tmpl.copy()
            enhanced[tid]['aggregated_signals'] = asdict(AggregatedSignals())
            stats['no_similar'].append(tid)
            continue

        # Compute aggregated signals
        entropy_arr = np.array(entropy_vals)
        received_arr = np.array(received_vals)
        connection_arr = np.array(connection_vals)

        agg = AggregatedSignals(
            entropy_mean=float(np.mean(entropy_arr)),
            received_mean=float(np.mean(received_arr)),
            connection_mean=float(np.mean(connection_arr)),
            entropy_var=float(np.var(entropy_arr)),
            received_var=float(np.var(received_arr)),
            connection_var=float(np.var(connection_arr)),
            entropy_min=float(np.min(entropy_arr)),
            entropy_max=float(np.max(entropy_arr)),
            received_min=float(np.min(received_arr)),
            received_max=float(np.max(received_arr)),
            connection_min=float(np.min(connection_arr)),
            connection_max=float(np.max(connection_arr)),
            source_count=len(entropy_vals),
            avg_similarity=float(np.mean(sim_vals)),
        )

        enhanced[tid] = tmpl.copy()
        enhanced[tid]['aggregated_signals'] = asdict(agg)

        stats['success'].append(tid)

    # Print summary
    print(f"\nAggregation results:")
    print(f"  Successful: {len(stats['success'])}")
    print(f"  No embedding: {len(stats['no_embedding'])}")
    print(f"  Zero embedding: {len(stats['zero_embedding'])}")
    print(f"  No similar templates: {len(stats['no_similar'])}")

    # Print signal statistics
    successful = [enhanced[tid] for tid in stats['success']]
    if successful:
        avg_sources = np.mean([t['aggregated_signals']['source_count'] for t in successful])
        avg_sim = np.mean([t['aggregated_signals']['avg_similarity'] for t in successful])
        avg_entropy_var = np.mean([t['aggregated_signals']['entropy_var'] for t in successful])

        print(f"\nSignal statistics (successful templates):")
        print(f"  Avg sources per template: {avg_sources:.1f}")
        print(f"  Avg similarity: {avg_sim:.3f}")
        print(f"  Avg entropy variance: {avg_entropy_var:.4f}")

    return enhanced


def save_enhanced_templates(templates: Dict[str, dict], output_path: str):
    """Save enhanced templates to JSON."""
    # Convert to list format matching dsl_library.json structure
    output = {"templates": list(templates.values())}

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(templates)} enhanced templates to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Qwen signals to deduped templates")

    parser.add_argument(
        "--specialized", "-s",
        default="specialized_templates.json",
        help="Path to specialized templates with Qwen signals"
    )
    parser.add_argument(
        "--deduped", "-d",
        default="dsl_library.json",
        help="Path to deduped templates"
    )
    parser.add_argument(
        "--output", "-o",
        default="enhanced_dsl_library.json",
        help="Output path for enhanced templates"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Minimum cosine similarity threshold"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=50,
        help="Maximum sources to aggregate per template"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    specialized_path = project_root / args.specialized
    deduped_path = project_root / args.deduped
    output_path = project_root / args.output

    print("=" * 60)
    print("Signal Aggregation: 17k → Deduped Templates")
    print("=" * 60)
    print(f"\nSpecialized: {specialized_path}")
    print(f"Deduped: {deduped_path}")
    print(f"Output: {output_path}")

    # Load templates
    specialized = load_specialized_templates(str(specialized_path))
    deduped = load_deduped_templates(str(deduped_path))

    # Aggregate signals
    enhanced = aggregate_signals(
        deduped,
        specialized,
        similarity_threshold=args.threshold,
        top_k=args.top_k
    )

    # Save
    save_enhanced_templates(enhanced, str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
