#!/usr/bin/env python3
"""Cluster 23K templates down to ~1K and save as deduplicated_1k_templates.json.

Uses greedy cosine similarity clustering at threshold ~0.698.
For each cluster, picks the template with highest count as representative,
averages the centroid across all members, and collects all pattern_examples.
"""

import json
import numpy as np
import time
from pathlib import Path


def greedy_cluster(centroids_norm: np.ndarray, threshold: float) -> list:
    """Greedy cosine similarity clustering."""
    n = len(centroids_norm)
    used = set()
    clusters = []

    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        remaining = [j for j in range(i + 1, n) if j not in used]
        if remaining:
            remaining_arr = np.array(remaining)
            sims = centroids_norm[remaining_arr] @ centroids_norm[i]
            for idx, sim in zip(remaining_arr, sims):
                if sim >= threshold:
                    cluster.append(idx)
                    used.add(idx)

        clusters.append(cluster)

    return clusters


def main():
    project_root = Path(__file__).parent.parent
    templates_path = project_root / "qwen_templates.json"
    output_path = project_root / "deduplicated_1k_templates.json"

    THRESHOLD = 0.698

    print(f"Loading templates from {templates_path}...")
    t0 = time.time()
    with open(templates_path) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates in {time.time() - t0:.1f}s")

    # Extract embeddings
    embeddings = []
    valid_indices = []
    for i, tpl in enumerate(templates):
        emb = tpl.get("embedding_centroid")
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Got {len(embeddings)} valid embeddings, dim={embeddings.shape[1]}")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    centroids_norm = embeddings / np.maximum(norms, 1e-8)

    # Cluster
    print(f"Clustering at threshold={THRESHOLD}...")
    t0 = time.time()
    clusters = greedy_cluster(centroids_norm, THRESHOLD)
    print(f"Got {len(clusters)} clusters in {time.time() - t0:.1f}s")

    # Build deduplicated templates
    deduped = []
    for cluster_idx, cluster in enumerate(clusters):
        # Map cluster indices back to template indices
        tpl_indices = [valid_indices[i] for i in cluster]
        cluster_templates = [templates[i] for i in tpl_indices]

        # Pick representative: highest count
        best = max(cluster_templates, key=lambda t: t.get("count", 1))

        # Average centroid across cluster members
        cluster_embs = embeddings[cluster]
        avg_centroid = np.mean(cluster_embs, axis=0)
        avg_centroid = avg_centroid / (np.linalg.norm(avg_centroid) + 1e-8)

        # Collect all pattern examples (up to 20)
        all_examples = []
        for t in cluster_templates:
            all_examples.extend(t.get("pattern_examples", []))
        all_examples = all_examples[:20]

        # Collect all unique patterns
        all_patterns = list(set(t.get("pattern", "") for t in cluster_templates))

        deduped.append({
            "template_id": f"dedup_{cluster_idx:04d}",
            "pattern": best.get("pattern", ""),
            "embedding_centroid": avg_centroid.tolist(),
            "count": sum(t.get("count", 1) for t in cluster_templates),
            "cluster_size": len(cluster),
            "pattern_examples": all_examples,
            "member_patterns": all_patterns[:10],  # Top 10 unique patterns in cluster
            "qwen_fail_rate": best.get("qwen_fail_rate", 0.0),
        })

    # Sort by count (most common first)
    deduped.sort(key=lambda t: t["count"], reverse=True)

    # Stats
    total_count = sum(t["count"] for t in deduped)
    sizes = [t["cluster_size"] for t in deduped]
    print(f"\n=== Deduplication Stats ===")
    print(f"Original templates: {len(templates)}")
    print(f"Deduplicated templates: {len(deduped)}")
    print(f"Compression ratio: {len(deduped)/len(templates)*100:.1f}%")
    print(f"Total span count: {total_count}")
    print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}, median={np.median(sizes):.1f}")
    print(f"\nTop 10 templates by count:")
    for t in deduped[:10]:
        print(f"  {t['pattern'][:60]:<60} count={t['count']:>5}  cluster_size={t['cluster_size']}")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"Saved {len(deduped)} templates ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
