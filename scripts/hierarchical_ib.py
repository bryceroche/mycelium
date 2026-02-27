#!/usr/bin/env python3
"""
Hierarchical β-Annealing for IB Template Discovery

The prism analogy:
- First pass: split white light into spectral bands (coarse families)
- Recursive passes: split each impure band until spectral lines emerge

Stops when every cluster reaches 95%+ operator purity.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TemplateNode:
    """Node in the template hierarchy tree."""
    id: str
    depth: int
    indices: List[int]  # indices into original feature array
    operator_counts: Dict[str, int]
    children: List['TemplateNode'] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def dominant_op(self) -> str:
        if not self.operator_counts:
            return "UNKNOWN"
        return max(self.operator_counts.items(), key=lambda x: x[1])[0]

    @property
    def purity(self) -> float:
        if not self.operator_counts or self.size == 0:
            return 0.0
        return max(self.operator_counts.values()) / self.size

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self):
        return f"Template({self.id}, {self.dominant_op}, n={self.size}, purity={self.purity:.0%})"


def load_features(path: str) -> List[Dict]:
    """Load pre-extracted features."""
    with open(path) as f:
        return json.load(f)


def encode_features(features: List[Dict]) -> np.ndarray:
    """Encode features for clustering."""
    operators = np.array([f['operator'] for f in features]).reshape(-1, 1)
    result_types = np.array([f['result_type'] for f in features]).reshape(-1, 1)
    categories = np.array([f['category'] for f in features]).reshape(-1, 1)

    encoded = []
    for data, weight in [(operators, 2.0), (result_types, 1.5), (categories, 0.5)]:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded.append(enc.fit_transform(data) * weight)

    return np.hstack(encoded).astype(np.float32)


def find_optimal_k(X: np.ndarray, max_k: int = 10) -> int:
    """
    Simple β-inspired heuristic: try a few k values, pick one that
    creates reasonably sized clusters.
    """
    n = len(X)
    if n < 20:
        return min(2, n // 5 + 1)

    # Try k from 2 to max_k, pick the one with best balance
    best_k = 2
    best_score = -1

    for k in range(2, min(max_k + 1, n // 5)):
        clusterer = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
        labels = clusterer.fit_predict(X)

        # Score: prefer k that creates non-tiny clusters
        sizes = [np.sum(labels == i) for i in range(k)]
        min_size = min(sizes)
        balance = min_size / (n / k)  # How balanced are clusters?

        if balance > best_score:
            best_score = balance
            best_k = k

    return best_k


def split_cluster(X: np.ndarray, indices: List[int], operators: List[str],
                  depth: int, parent_id: str, node_counter: List[int]) -> List[TemplateNode]:
    """
    Split a cluster using β-annealing principles.
    Returns list of child nodes.
    """
    n = len(indices)

    if n < 10:
        # Too small to split
        op_counts = Counter(operators[i] for i in indices)
        return [TemplateNode(
            id=f"{parent_id}",
            depth=depth,
            indices=indices,
            operator_counts=dict(op_counts),
        )]

    # Find natural split
    X_subset = X[indices]
    k = find_optimal_k(X_subset, max_k=5)

    clusterer = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42)
    labels = clusterer.fit_predict(X_subset)

    children = []
    for cid in range(k):
        mask = labels == cid
        if mask.sum() == 0:
            continue

        child_indices = [indices[i] for i in range(len(indices)) if mask[i]]
        op_counts = Counter(operators[i] for i in child_indices)

        node_counter[0] += 1
        child = TemplateNode(
            id=f"{parent_id}.{cid}",
            depth=depth + 1,
            indices=child_indices,
            operator_counts=dict(op_counts),
        )
        children.append(child)

    return children


def hierarchical_annealing(X: np.ndarray, operators: List[str],
                           purity_threshold: float = 0.95,
                           min_size: int = 10,
                           max_depth: int = 6,
                           verbose: bool = True) -> TemplateNode:
    """
    Build hierarchical template tree via recursive β-annealing.

    Algorithm:
    1. Start with all features in one cluster
    2. Split into natural sub-clusters
    3. For each impure sub-cluster (< threshold), recurse
    4. Stop when pure or too small or max depth
    """
    if verbose:
        print(f"\nHierarchical β-Annealing")
        print(f"  Purity threshold: {purity_threshold:.0%}")
        print(f"  Min cluster size: {min_size}")
        print(f"  Max depth: {max_depth}")

    all_indices = list(range(len(X)))
    op_counts = Counter(operators)

    # Root node
    root = TemplateNode(
        id="root",
        depth=0,
        indices=all_indices,
        operator_counts=dict(op_counts),
    )

    # Track statistics
    node_counter = [0]
    splits_by_depth = defaultdict(int)

    # BFS queue: nodes to potentially split
    queue = [root]

    while queue:
        node = queue.pop(0)

        # Check stopping conditions
        if node.purity >= purity_threshold:
            if verbose and node.depth > 0:
                print(f"  {'  '*node.depth}[{node.id}] PURE: {node.dominant_op} "
                      f"(n={node.size}, purity={node.purity:.0%})")
            continue

        if node.size < min_size:
            if verbose:
                print(f"  {'  '*node.depth}[{node.id}] TOO SMALL: {node.dominant_op} "
                      f"(n={node.size}, purity={node.purity:.0%})")
            continue

        if node.depth >= max_depth:
            if verbose:
                print(f"  {'  '*node.depth}[{node.id}] MAX DEPTH: {node.dominant_op} "
                      f"(n={node.size}, purity={node.purity:.0%})")
            continue

        # Split this impure node
        if verbose:
            print(f"  {'  '*node.depth}[{node.id}] SPLITTING: {node.dominant_op} "
                  f"(n={node.size}, purity={node.purity:.0%})")

        children = split_cluster(
            X, node.indices, operators,
            node.depth, node.id, node_counter
        )

        node.children = children
        splits_by_depth[node.depth] += 1

        # Add impure children to queue
        for child in children:
            if child.purity < purity_threshold and child.size >= min_size:
                queue.append(child)

    if verbose:
        print(f"\n  Splits by depth: {dict(splits_by_depth)}")

    return root


def collect_leaves(node: TemplateNode) -> List[TemplateNode]:
    """Collect all leaf nodes (final templates)."""
    if node.is_leaf:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(collect_leaves(child))
    return leaves


def print_tree(node: TemplateNode, indent: int = 0):
    """Pretty print the template tree."""
    prefix = "  " * indent
    status = "LEAF" if node.is_leaf else f"{len(node.children)} children"
    purity_bar = "█" * int(node.purity * 10)
    print(f"{prefix}{node.dominant_op:15} n={node.size:5} purity={node.purity:5.0%} {purity_bar} [{status}]")

    for child in node.children:
        print_tree(child, indent + 1)


def main():
    print("=" * 70)
    print("HIERARCHICAL β-ANNEALING FOR IB TEMPLATES")
    print("=" * 70)

    # Load data
    data_path = "/Users/bryceroche/Desktop/mycelium/data/ib_templates/classifier_labels.json"
    print(f"\nLoading {data_path}...")
    features = load_features(data_path)
    operators = [f['operator'] for f in features]
    print(f"Loaded {len(features):,} features")

    # Encode
    print("Encoding...")
    X = encode_features(features)
    print(f"Encoded: {X.shape}")

    # Show initial distribution
    print(f"\nInitial operator distribution:")
    for op, count in Counter(operators).most_common():
        pct = count / len(operators) * 100
        print(f"  {op:15} {count:5} ({pct:5.1f}%)")

    # Run hierarchical annealing
    start_time = time.time()
    root = hierarchical_annealing(
        X, operators,
        purity_threshold=0.95,
        min_size=10,
        max_depth=6,
        verbose=True
    )
    elapsed = time.time() - start_time

    # Collect leaves
    leaves = collect_leaves(root)
    leaves_sorted = sorted(leaves, key=lambda x: x.size, reverse=True)

    # Statistics
    purities = [leaf.purity for leaf in leaves]
    mean_purity = np.mean(purities)
    min_purity = min(purities)
    impure_count = sum(1 for p in purities if p < 0.95)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total leaf templates: {len(leaves)}")
    print(f"  Mean purity: {mean_purity:.1%}")
    print(f"  Min purity: {min_purity:.1%}")
    print(f"  Impure (<95%): {impure_count}")
    print(f"  Time: {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("TEMPLATE HIERARCHY")
    print(f"{'='*70}")
    print_tree(root)

    print(f"\n{'='*70}")
    print("FINAL TEMPLATES (sorted by size)")
    print(f"{'='*70}")
    for i, leaf in enumerate(leaves_sorted[:30]):
        print(f"  T{i:02d}: {leaf.dominant_op:15} n={leaf.size:5} purity={leaf.purity:.0%}")

    if len(leaves) > 30:
        print(f"  ... and {len(leaves) - 30} more")

    # Save results
    output = {
        'n_templates': len(leaves),
        'mean_purity': mean_purity,
        'min_purity': min_purity,
        'templates': [
            {
                'id': leaf.id,
                'operator': leaf.dominant_op,
                'size': leaf.size,
                'purity': leaf.purity,
                'operator_counts': leaf.operator_counts,
            }
            for leaf in leaves_sorted
        ]
    }

    output_path = "/Users/bryceroche/Desktop/mycelium/data/ib_templates/hierarchical_templates.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
