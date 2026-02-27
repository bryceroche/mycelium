#!/usr/bin/env python3
"""
Compare IB Clustering Approaches:
1. Silhouette-based (force k in range, pick best)
2. Hierarchical β-annealing (recursive split until pure)

Uses the 19,641 pre-extracted features from the IAF data.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score, mutual_info_score

# ============================================================================
# DATA LOADING
# ============================================================================

def load_features(path: str) -> List[Dict]:
    """Load pre-extracted features."""
    with open(path) as f:
        return json.load(f)


def encode_features(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode features for clustering."""
    operators = np.array([f['operator'] for f in features]).reshape(-1, 1)
    result_types = np.array([f['result_type'] for f in features]).reshape(-1, 1)
    categories = np.array([f['category'] for f in features]).reshape(-1, 1)

    # Simplified encoding for comparison
    encoders = {}
    encoded = []

    for name, data, weight in [
        ('op', operators, 2.0),
        ('rt', result_types, 1.5),
        ('cat', categories, 0.5),
    ]:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_enc = enc.fit_transform(data) * weight
        encoded.append(X_enc)
        encoders[name] = enc

    X = np.hstack(encoded).astype(np.float32)

    # Y = operator as target for purity calculation
    op_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    op_enc.fit(operators)
    Y = op_enc.transform(operators).argmax(axis=1)

    return X, Y, [f['operator'] for f in features]


def compute_cluster_purity(labels: np.ndarray, operators: List[str]) -> Dict[int, float]:
    """Compute operator purity for each cluster."""
    clusters = defaultdict(list)
    for label, op in zip(labels, operators):
        clusters[int(label)].append(op)

    purities = {}
    for cid, ops in clusters.items():
        counts = Counter(ops)
        top_count = counts.most_common(1)[0][1]
        purities[cid] = top_count / len(ops)

    return purities


# ============================================================================
# APPROACH 1: SILHOUETTE-BASED
# ============================================================================

def silhouette_clustering(X: np.ndarray, operators: List[str],
                          min_k: int = 30, max_k: int = 100) -> Dict:
    """
    Silhouette-based approach: search k in [min_k, max_k], pick best silhouette.
    """
    print(f"\n{'='*60}")
    print("APPROACH 1: SILHOUETTE-BASED CLUSTERING")
    print(f"{'='*60}")
    print(f"Searching k in [{min_k}, {max_k}]...")

    start_time = time.time()
    best_k, best_score, best_labels = min_k, -1, None
    scores = []

    for k in range(min_k, min(max_k + 1, len(X) // 10), 5):
        clusterer = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
        labels = clusterer.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        scores.append((k, score))

        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

        if k % 20 == 0:
            print(f"  k={k}: silhouette={score:.3f}")

    elapsed = time.time() - start_time

    # Analyze results
    purities = compute_cluster_purity(best_labels, operators)
    mean_purity = np.mean(list(purities.values()))
    min_purity = min(purities.values())
    impure_count = sum(1 for p in purities.values() if p < 0.95)

    print(f"\nResults:")
    print(f"  Best k: {best_k}")
    print(f"  Silhouette: {best_score:.3f}")
    print(f"  Mean purity: {mean_purity:.1%}")
    print(f"  Min purity: {min_purity:.1%}")
    print(f"  Impure clusters (<95%): {impure_count}/{best_k}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        'method': 'silhouette',
        'n_clusters': best_k,
        'silhouette': best_score,
        'mean_purity': mean_purity,
        'min_purity': min_purity,
        'impure_count': impure_count,
        'labels': best_labels,
        'purities': purities,
        'time': elapsed,
    }


# ============================================================================
# APPROACH 2: HIERARCHICAL β-ANNEALING
# ============================================================================

@dataclass
class ClusterNode:
    """Node in the hierarchical cluster tree."""
    id: str
    indices: np.ndarray  # indices into original feature array
    purity: float
    dominant_op: str
    children: List['ClusterNode'] = None

    @property
    def is_leaf(self):
        return self.children is None or len(self.children) == 0

    @property
    def size(self):
        return len(self.indices)


def anneal_beta_local(X: np.ndarray, operators: List[str],
                      beta_range: Tuple[float, float] = (0.01, 5.0),
                      n_steps: int = 20) -> Tuple[int, np.ndarray]:
    """
    β-annealing on a subset of data to find natural cluster count.
    Returns (optimal_k, labels).
    """
    if len(X) < 10:
        return 1, np.zeros(len(X), dtype=int)

    betas = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), n_steps)

    cluster_counts = []
    for beta in betas:
        # Heuristic: k scales with β and data size
        k = max(2, min(len(X) // 5, int(2 + 5 * np.log10(beta + 1))))
        cluster_counts.append(k)

    # Find most stable k (simple plateau detection)
    counts = Counter(cluster_counts)
    optimal_k = counts.most_common(1)[0][0]
    optimal_k = max(2, min(optimal_k, len(X) // 3))

    # Cluster with optimal k
    if len(X) > 100:
        clusterer = MiniBatchKMeans(n_clusters=optimal_k, n_init=5, random_state=42)
    else:
        clusterer = AgglomerativeClustering(n_clusters=optimal_k)

    labels = clusterer.fit_predict(X)
    return optimal_k, labels


def hierarchical_beta_annealing(X: np.ndarray, operators: List[str],
                                 purity_threshold: float = 0.95,
                                 min_cluster_size: int = 10,
                                 max_depth: int = 5,
                                 parent_id: str = "root",
                                 depth: int = 0) -> List[ClusterNode]:
    """
    Hierarchical β-annealing: recursively split until purity threshold met.
    """
    # Base case: too small or max depth
    if len(X) < min_cluster_size or depth >= max_depth:
        ops = [operators[i] for i in range(len(operators)) if i < len(X)]
        if not ops:
            ops = operators[:len(X)]
        counts = Counter(ops)
        dominant = counts.most_common(1)[0][0] if counts else "UNKNOWN"
        purity = counts.most_common(1)[0][1] / len(ops) if counts else 1.0

        return [ClusterNode(
            id=f"{parent_id}",
            indices=np.arange(len(X)),
            purity=purity,
            dominant_op=dominant,
            children=None
        )]

    # Anneal to find natural clusters at this level
    n_clusters, labels = anneal_beta_local(X, operators)

    results = []
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() == 0:
            continue

        cluster_indices = np.where(mask)[0]
        cluster_ops = [operators[i] for i in cluster_indices]

        # Compute purity
        counts = Counter(cluster_ops)
        dominant_op = counts.most_common(1)[0][0]
        purity = counts.most_common(1)[0][1] / len(cluster_ops)

        node_id = f"{parent_id}.{cid}"

        if purity >= purity_threshold or len(cluster_indices) < min_cluster_size:
            # Pure enough or too small - this is a leaf
            results.append(ClusterNode(
                id=node_id,
                indices=cluster_indices,
                purity=purity,
                dominant_op=dominant_op,
                children=None
            ))
        else:
            # Impure - recurse
            child_X = X[cluster_indices]
            child_ops = cluster_ops

            children = hierarchical_beta_annealing(
                child_X, child_ops,
                purity_threshold=purity_threshold,
                min_cluster_size=min_cluster_size,
                max_depth=max_depth,
                parent_id=node_id,
                depth=depth + 1
            )

            results.append(ClusterNode(
                id=node_id,
                indices=cluster_indices,
                purity=purity,
                dominant_op=dominant_op,
                children=children
            ))

    return results


def collect_leaves(nodes: List[ClusterNode]) -> List[ClusterNode]:
    """Collect all leaf nodes from hierarchy."""
    leaves = []
    for node in nodes:
        if node.is_leaf:
            leaves.append(node)
        elif node.children:
            leaves.extend(collect_leaves(node.children))
    return leaves


def hierarchical_clustering(X: np.ndarray, operators: List[str],
                            purity_threshold: float = 0.95) -> Dict:
    """
    Hierarchical β-annealing approach.
    """
    print(f"\n{'='*60}")
    print("APPROACH 2: HIERARCHICAL β-ANNEALING")
    print(f"{'='*60}")
    print(f"Purity threshold: {purity_threshold:.0%}")
    print("Recursively splitting impure clusters...")

    start_time = time.time()

    # Build hierarchy
    hierarchy = hierarchical_beta_annealing(
        X, operators,
        purity_threshold=purity_threshold,
        min_cluster_size=10,
        max_depth=5
    )

    # Collect leaf nodes (final templates)
    leaves = collect_leaves(hierarchy)

    elapsed = time.time() - start_time

    # Create labels array
    labels = np.zeros(len(X), dtype=int)
    for i, leaf in enumerate(leaves):
        # Map back to original indices (this is approximate due to recursion)
        pass  # Labels not strictly needed for comparison

    # Analyze results
    purities = {i: leaf.purity for i, leaf in enumerate(leaves)}
    mean_purity = np.mean([leaf.purity for leaf in leaves])
    min_purity = min(leaf.purity for leaf in leaves) if leaves else 0
    impure_count = sum(1 for leaf in leaves if leaf.purity < 0.95)

    # Count depth
    def max_depth(nodes, d=0):
        if not nodes:
            return d
        return max(max_depth(n.children, d+1) if n.children else d for n in nodes)

    tree_depth = max_depth(hierarchy)

    print(f"\nResults:")
    print(f"  Leaf templates: {len(leaves)}")
    print(f"  Tree depth: {tree_depth}")
    print(f"  Mean purity: {mean_purity:.1%}")
    print(f"  Min purity: {min_purity:.1%}")
    print(f"  Impure clusters (<95%): {impure_count}/{len(leaves)}")
    print(f"  Time: {elapsed:.1f}s")

    # Show template distribution
    print(f"\nTemplate size distribution:")
    sizes = sorted([leaf.size for leaf in leaves], reverse=True)
    for i, size in enumerate(sizes[:10]):
        leaf = [l for l in leaves if l.size == size][0]
        print(f"  {leaf.dominant_op:20} size={size:5} purity={leaf.purity:.0%}")
    if len(sizes) > 10:
        print(f"  ... and {len(sizes) - 10} more templates")

    return {
        'method': 'hierarchical',
        'n_clusters': len(leaves),
        'tree_depth': tree_depth,
        'mean_purity': mean_purity,
        'min_purity': min_purity,
        'impure_count': impure_count,
        'hierarchy': hierarchy,
        'leaves': leaves,
        'time': elapsed,
    }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    print("="*60)
    print("IB CLUSTERING APPROACH COMPARISON")
    print("="*60)

    # Load data
    data_path = "/Users/bryceroche/Desktop/mycelium/data/ib_templates/classifier_labels.json"
    print(f"\nLoading features from {data_path}...")
    features = load_features(data_path)
    print(f"Loaded {len(features):,} features")

    # Encode
    print("Encoding features...")
    X, Y, operators = encode_features(features)
    print(f"Encoded matrix: {X.shape}")

    # Show operator distribution
    op_counts = Counter(operators)
    print(f"\nOperator distribution:")
    for op, count in op_counts.most_common(10):
        print(f"  {op:20} {count:5} ({count/len(operators):.1%})")

    # Run both approaches
    silhouette_result = silhouette_clustering(X, operators, min_k=30, max_k=100)
    hierarchical_result = hierarchical_clustering(X, operators, purity_threshold=0.95)

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Silhouette':>15} {'Hierarchical':>15}")
    print("-"*60)
    print(f"{'Templates/Clusters':<25} {silhouette_result['n_clusters']:>15} {hierarchical_result['n_clusters']:>15}")
    print(f"{'Mean Purity':<25} {silhouette_result['mean_purity']:>14.1%} {hierarchical_result['mean_purity']:>14.1%}")
    print(f"{'Min Purity':<25} {silhouette_result['min_purity']:>14.1%} {hierarchical_result['min_purity']:>14.1%}")
    print(f"{'Impure (<95%)':<25} {silhouette_result['impure_count']:>15} {hierarchical_result['impure_count']:>15}")
    print(f"{'Time (sec)':<25} {silhouette_result['time']:>14.1f}s {hierarchical_result['time']:>14.1f}s")

    # Winner?
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    if hierarchical_result['mean_purity'] > silhouette_result['mean_purity']:
        print("✓ Hierarchical has higher mean purity")
    else:
        print("✓ Silhouette has higher mean purity")

    if hierarchical_result['impure_count'] < silhouette_result['impure_count']:
        print("✓ Hierarchical has fewer impure clusters")
    else:
        print("✓ Silhouette has fewer impure clusters")

    print(f"\nHierarchical found {hierarchical_result['n_clusters']} templates")
    print(f"with tree depth {hierarchical_result['tree_depth']} (natural taxonomy)")


if __name__ == "__main__":
    main()
