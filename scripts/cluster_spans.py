#!/usr/bin/env python3
"""Cluster normalized spans into centroid templates.

Loads labeled spans from DB, normalizes them, embeds normalized spans,
clusters by cosine similarity, computes centroids, and creates SpanTemplates.

Usage:
    python scripts/cluster_spans.py [--threshold 0.85] [--min-cluster 2]
    python scripts/cluster_spans.py --dry-run  # Test without DB writes
"""

import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentence_transformers import SentenceTransformer

from mycelium.db import get_labeled_spans, store_centroid, LabeledSpanRow
from mycelium.span_normalizer import normalize_span
from mycelium.span_templates import SpanTemplate, register_template


def load_and_normalize_spans(limit: int = 5000) -> List[Tuple[str, str, str, dict]]:
    """Load spans from DB and normalize them.

    Returns:
        List of (original_text, normalized_text, operation, extracted_values)
    """
    spans: List[LabeledSpanRow] = get_labeled_spans(limit=limit)
    results = []

    for span in spans:
        if span.operation:
            try:
                normalized, extracted = normalize_span(span.span_text)
                results.append((span.span_text, normalized, span.operation, extracted))
            except Exception as e:
                print(f"  Warning: Failed to normalize '{span.span_text[:50]}...': {e}")

    return results


def embed_normalized_spans(normalized_texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Embed all normalized spans.

    Args:
        normalized_texts: List of normalized span texts
        model: SentenceTransformer model

    Returns:
        (N, D) normalized embeddings array
    """
    embeddings = model.encode(normalized_texts, convert_to_numpy=True, show_progress_bar=True)
    # L2 normalize to unit vectors for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def greedy_cluster(embeddings: np.ndarray, threshold: float = 0.85) -> List[List[int]]:
    """Greedy clustering by cosine similarity.

    Iterates through embeddings and creates clusters by assigning each
    unassigned embedding to the first cluster whose centroid is within
    the similarity threshold.

    Args:
        embeddings: (N, D) normalized embeddings
        threshold: Minimum similarity to join cluster

    Returns:
        List of clusters, each cluster is a list of indices
    """
    n = len(embeddings)
    assigned = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new cluster with this span
        cluster = [i]
        assigned[i] = True

        # Find all similar unassigned spans
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    return clusters


def compute_centroid(embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
    """Compute centroid of cluster.

    Args:
        embeddings: (N, D) embeddings array
        indices: Indices of embeddings in this cluster

    Returns:
        (D,) normalized centroid vector as float32
    """
    cluster_embs = embeddings[indices]
    centroid = np.mean(cluster_embs, axis=0)
    # L2 normalize
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroid.astype(np.float32)


def detect_dsl_type(normalized_text: str) -> str:
    """Detect if span requires simple or complex DSL.

    Complex patterns involve:
    - Comparisons (more than, less than)
    - Ratios (twice as many, half of)
    - Percentages

    Args:
        normalized_text: The normalized span pattern

    Returns:
        "simple" or "complex"
    """
    text_lower = normalized_text.lower()

    # Patterns that indicate complex DSL
    complex_patterns = [
        "more than",
        "less than",
        "fewer than",
        "twice",
        "triple",
        "half",
        "double",
        "percent",
        "%",
        "as many as",
        "times as",
    ]

    for pattern in complex_patterns:
        if pattern in text_lower:
            return "complex"

    return "simple"


def create_templates(
    spans: List[Tuple[str, str, str, dict]],
    embeddings: np.ndarray,
    clusters: List[List[int]],
    min_cluster_size: int = 2
) -> List[SpanTemplate]:
    """Create SpanTemplate for each cluster.

    Args:
        spans: List of (original_text, normalized_text, operation, extracted_values)
        embeddings: (N, D) normalized embeddings
        clusters: List of clusters (each cluster is list of indices)
        min_cluster_size: Minimum spans required to form a template

    Returns:
        List of SpanTemplate objects
    """
    templates = []

    for cluster_idx, indices in enumerate(clusters):
        if len(indices) < min_cluster_size:
            continue

        # Get cluster info
        cluster_spans = [spans[i] for i in indices]
        operations = [s[2] for s in cluster_spans]
        normalized_texts = [s[1] for s in cluster_spans]
        original_texts = [s[0] for s in cluster_spans]

        # Most common operation in cluster
        op_counts: Dict[str, int] = defaultdict(int)
        for op in operations:
            op_counts[op] += 1
        primary_op = max(op_counts, key=op_counts.get)

        # Most common normalized pattern
        pattern_counts: Dict[str, int] = defaultdict(int)
        for norm in normalized_texts:
            pattern_counts[norm] += 1
        primary_pattern = max(pattern_counts, key=pattern_counts.get)

        # Compute centroid
        centroid = compute_centroid(embeddings, indices)

        # Detect DSL type from pattern
        dsl_type = detect_dsl_type(primary_pattern)

        # Create unique template ID
        # Use operation + simplified pattern hash
        pattern_slug = primary_pattern.lower()
        pattern_slug = pattern_slug.replace("[name]", "n").replace("[n]", "v")
        pattern_slug = pattern_slug.replace("[item]", "i").replace("[subj]", "s")
        pattern_slug = pattern_slug.replace("[obj]", "o").replace("[poss]", "p")
        pattern_slug = "".join(c for c in pattern_slug if c.isalnum() or c == "_")[:20]
        template_id = f"{primary_op.lower()}_{pattern_slug}_{cluster_idx}"

        # Create template
        template = SpanTemplate(
            template_id=template_id,
            pattern=primary_pattern,
            centroid=centroid,
            operation=primary_op,
            dsl_type=dsl_type,
            examples=original_texts[:5],  # Keep top 5 examples
            count=len(indices),
        )
        templates.append(template)

    return templates


def print_cluster_stats(clusters: List[List[int]], spans: List[Tuple[str, str, str, dict]]):
    """Print detailed statistics about clusters."""
    print("\n  Cluster size distribution:")
    sizes = [len(c) for c in clusters]

    # Histogram of sizes
    size_counts: Dict[str, int] = defaultdict(int)
    for size in sizes:
        if size == 1:
            size_counts["1 (singleton)"] += 1
        elif size <= 3:
            size_counts["2-3"] += 1
        elif size <= 10:
            size_counts["4-10"] += 1
        elif size <= 50:
            size_counts["11-50"] += 1
        else:
            size_counts["50+"] += 1

    for bucket, count in sorted(size_counts.items()):
        print(f"    {bucket}: {count} clusters")

    print(f"\n  Largest clusters:")
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)
    for idx, cluster in sorted_clusters[:5]:
        # Get operation distribution
        ops = [spans[i][2] for i in cluster]
        op_counts: Dict[str, int] = defaultdict(int)
        for op in ops:
            op_counts[op] += 1
        op_str = ", ".join(f"{op}:{count}" for op, count in sorted(op_counts.items(), key=lambda x: -x[1]))

        # Get sample pattern
        sample_pattern = spans[cluster[0]][1]

        print(f"    Cluster {idx}: {len(cluster)} spans, ops=[{op_str}]")
        print(f"      Pattern: {sample_pattern}")


def cluster_by_operation(
    spans: List[Tuple[str, str, str, dict]],
    embeddings: np.ndarray,
    threshold: float,
    min_cluster_size: int
) -> Tuple[List[List[int]], List[SpanTemplate]]:
    """Cluster spans WITHIN each operation type.

    This ensures spans are only clustered with other spans of the same operation,
    avoiding mixing of operations in clusters.

    Args:
        spans: List of (original_text, normalized_text, operation, extracted_values)
        embeddings: (N, D) normalized embeddings
        threshold: Similarity threshold for clustering
        min_cluster_size: Minimum spans required to form a template

    Returns:
        Tuple of (all_clusters, all_templates)
    """
    # Group spans by operation
    op_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, (_, _, op, _) in enumerate(spans):
        op_to_indices[op].append(i)

    all_clusters = []
    all_templates = []

    print(f"\n  Clustering within each operation type:")

    for op in sorted(op_to_indices.keys()):
        indices = op_to_indices[op]
        if len(indices) < min_cluster_size:
            print(f"    {op}: {len(indices)} spans (skipped - below min size)")
            continue

        # Get embeddings for this operation's spans
        op_embeddings = embeddings[indices]

        # Cluster within this operation
        local_clusters = greedy_cluster(op_embeddings, threshold=threshold)

        # Convert local indices back to global indices
        global_clusters = []
        for local_cluster in local_clusters:
            global_cluster = [indices[i] for i in local_cluster]
            global_clusters.append(global_cluster)

        # Create templates for this operation's clusters
        op_templates = create_templates_for_op(
            spans, embeddings, global_clusters, op, min_cluster_size
        )

        all_clusters.extend(global_clusters)
        all_templates.extend(op_templates)

        # Stats for this operation
        sizes = [len(c) for c in global_clusters]
        non_singleton = sum(1 for s in sizes if s >= min_cluster_size)
        print(f"    {op}: {len(indices)} spans → {len(global_clusters)} clusters → {len(op_templates)} templates")

    return all_clusters, all_templates


def create_templates_for_op(
    spans: List[Tuple[str, str, str, dict]],
    embeddings: np.ndarray,
    clusters: List[List[int]],
    operation: str,
    min_cluster_size: int = 2
) -> List[SpanTemplate]:
    """Create SpanTemplate for clusters of a single operation type.

    Since all spans in these clusters have the same operation, we don't need
    to vote on the operation.
    """
    templates = []

    for cluster_idx, indices in enumerate(clusters):
        if len(indices) < min_cluster_size:
            continue

        # Get cluster info
        cluster_spans = [spans[i] for i in indices]
        normalized_texts = [s[1] for s in cluster_spans]
        original_texts = [s[0] for s in cluster_spans]

        # Most common normalized pattern
        pattern_counts: Dict[str, int] = defaultdict(int)
        for norm in normalized_texts:
            pattern_counts[norm] += 1
        primary_pattern = max(pattern_counts, key=pattern_counts.get)

        # Compute centroid
        centroid = compute_centroid(embeddings, indices)

        # Detect DSL type from pattern
        dsl_type = detect_dsl_type(primary_pattern)

        # Create unique template ID
        pattern_slug = primary_pattern.lower()
        pattern_slug = pattern_slug.replace("[name]", "n").replace("[n]", "v")
        pattern_slug = pattern_slug.replace("[item]", "i").replace("[subj]", "s")
        pattern_slug = pattern_slug.replace("[obj]", "o").replace("[poss]", "p")
        pattern_slug = "".join(c for c in pattern_slug if c.isalnum() or c == "_")[:20]
        template_id = f"{operation.lower()}_{pattern_slug}_{cluster_idx}"

        # Create template
        template = SpanTemplate(
            template_id=template_id,
            pattern=primary_pattern,
            centroid=centroid,
            operation=operation,  # All spans have same operation
            dsl_type=dsl_type,
            examples=original_texts[:5],
            count=len(indices),
        )
        templates.append(template)

    return templates


def main():
    parser = argparse.ArgumentParser(description="Cluster spans into templates")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similarity threshold for clustering (default: 0.85)")
    parser.add_argument("--min-cluster", type=int, default=2,
                        help="Minimum cluster size to create template (default: 2)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Maximum spans to load from DB (default: 5000)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model to use (default: all-MiniLM-L6-v2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't save to DB, just print statistics")
    args = parser.parse_args()

    print("=" * 60)
    print("Span Clustering Pipeline (by Operation)")
    print("=" * 60)
    print(f"  Threshold: {args.threshold}")
    print(f"  Min cluster size: {args.min_cluster}")
    print(f"  Model: {args.model}")
    print(f"  Dry run: {args.dry_run}")

    # Load model
    print("\n[1/4] Loading sentence-transformers model...")
    model = SentenceTransformer(args.model)
    print(f"  Model loaded: {args.model}")

    # Load and normalize
    print("\n[2/4] Loading and normalizing spans from database...")
    spans = load_and_normalize_spans(limit=args.limit)
    print(f"  Loaded {len(spans)} labeled spans with operations")

    if len(spans) == 0:
        print("\n  ERROR: No labeled spans found in database.")
        print("  Run span labeling first to populate the database.")
        return

    # Show some examples
    print("\n  Examples of normalization:")
    for orig, norm, op, extracted in spans[:3]:
        print(f"    Original:   {orig}")
        print(f"    Normalized: {norm}")
        print(f"    Operation:  {op}")
        print(f"    Extracted:  {extracted}")
        print()

    # Get unique operations
    ops = set(s[2] for s in spans)
    print(f"  Operations in data: {sorted(ops)}")

    # Count by operation
    op_counts: Dict[str, int] = defaultdict(int)
    for _, _, op, _ in spans:
        op_counts[op] += 1
    print(f"  Distribution: {dict(sorted(op_counts.items()))}")

    # Embed normalized spans
    print("\n[3/4] Embedding normalized spans...")
    normalized_texts = [s[1] for s in spans]
    embeddings = embed_normalized_spans(normalized_texts, model)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Cluster BY OPERATION
    print(f"\n[4/4] Clustering by operation (threshold={args.threshold})...")
    clusters, templates = cluster_by_operation(
        spans, embeddings, args.threshold, args.min_cluster
    )
    print(f"\n  Total clusters: {len(clusters)}")
    print(f"  Total templates: {len(templates)}")

    # Show templates by operation
    print("\n  Templates by operation:")
    template_op_counts: Dict[str, int] = defaultdict(int)
    for t in templates:
        template_op_counts[t.operation] += 1
    for op, count in sorted(template_op_counts.items()):
        print(f"    {op}: {count} templates")

    # Show DSL type distribution
    simple_count = sum(1 for t in templates if t.dsl_type == "simple")
    complex_count = sum(1 for t in templates if t.dsl_type == "complex")
    print(f"\n  DSL types: simple={simple_count}, complex={complex_count}")

    # Show top templates per operation
    print("\n  Top templates by operation:")
    for op in sorted(template_op_counts.keys()):
        op_templates = [t for t in templates if t.operation == op]
        op_templates.sort(key=lambda x: x.count, reverse=True)
        print(f"\n    {op}:")
        for t in op_templates[:3]:
            print(f"      {t.pattern} (n={t.count})")
            if t.examples:
                print(f"        e.g., {t.examples[0]}")

    # Save to DB
    if not args.dry_run:
        print("\n  Saving to database...")
        saved_count = 0
        for t in templates:
            try:
                # Store centroid with template_id as the operation key
                store_centroid(t.template_id, t.centroid, t.count)
                # Register template in memory
                register_template(t)
                saved_count += 1
            except Exception as e:
                print(f"    Warning: Failed to save template {t.template_id}: {e}")
        print(f"  Saved {saved_count} templates to database")
    else:
        print("\n  [DRY RUN] Not saving to database")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Spans processed: {len(spans)}")
    print(f"  Clusters formed: {len(clusters)}")
    print(f"  Templates created: {len(templates)}")
    coverage = sum(t.count for t in templates)
    print(f"  Coverage: {coverage}/{len(spans)} spans ({100*coverage/len(spans):.1f}%)")
    print("\n  By operation:")
    for op in sorted(template_op_counts.keys()):
        op_coverage = sum(t.count for t in templates if t.operation == op)
        op_total = op_counts[op]
        print(f"    {op}: {op_coverage}/{op_total} spans ({100*op_coverage/op_total:.1f}%)")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
