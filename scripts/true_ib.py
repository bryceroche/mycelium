#!/usr/bin/env python3
"""
True Information Bottleneck with Target Variable Y

This is the ACTUAL IB algorithm: min I(X;T) - β·I(T;Y)

- X = Qwen embeddings of CoT steps (continuous, 896-dim)
- Y = sympy operator labels (discrete: Add, Mul, Pow, sqrt, sin, etc.)
- T = cluster assignments (what we're learning)
- β = tradeoff parameter (low β = compress, high β = preserve Y info)

β-annealing: Start with β≈0 (one cluster), increase β, watch for
phase transitions where clusters spontaneously bifurcate. The data
decides when to split based on information-theoretic pressure.

Reference: Tishby et al., "The Information Bottleneck Method"
"""

import json
import numpy as np
import boto3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import time
import argparse

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'


@dataclass
class IBCluster:
    """A cluster in the IB solution."""
    id: int
    indices: np.ndarray
    y_distribution: Dict[str, int]  # operator -> count

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def dominant_y(self) -> str:
        if not self.y_distribution:
            return "UNKNOWN"
        return max(self.y_distribution.items(), key=lambda x: x[1])[0]

    @property
    def purity(self) -> float:
        """Fraction of steps with the dominant Y."""
        if not self.y_distribution or self.size == 0:
            return 0.0
        return max(self.y_distribution.values()) / self.size


def load_data() -> Tuple[np.ndarray, List[Set[str]], List[Dict]]:
    """
    Load embeddings (X) and operator labels (Y).

    Returns:
        X: (n_steps, 896) embedding matrix
        Y: list of operator sets per step
        steps: step metadata
    """
    print("Loading embeddings (X)...")
    s3.download_file(BUCKET, 'ib_results_v2/step_embeddings_math.npz', '/tmp/embeddings.npz')
    X = np.load('/tmp/embeddings.npz')['embeddings']
    print(f"  X shape: {X.shape}")

    print("Loading operator labels (Y)...")
    response = s3.get_object(Bucket=BUCKET, Key='ib_y_labels/aggregated_y_labels.json')
    y_data = json.loads(response['Body'].read().decode('utf-8'))

    print("Loading step metadata...")
    response = s3.get_object(Bucket=BUCKET, Key='ib_results_v2/aggregated_steps.json')
    steps_data = json.loads(response['Body'].read().decode('utf-8'))
    steps = steps_data['steps']

    # Build Y array aligned with X
    # Y data has results per step, need to match by content_hash
    y_by_hash = {}
    for result in y_data['results']:
        h = result.get('content_hash')
        y_ops = result.get('Y')
        if h and y_ops:
            y_by_hash[h] = set(y_ops)

    Y = []
    valid_indices = []
    for i, step in enumerate(steps):
        h = step.get('content_hash')
        if h in y_by_hash:
            Y.append(y_by_hash[h])
            valid_indices.append(i)
        else:
            Y.append(None)

    # Filter to only steps with Y labels
    valid_indices = np.array(valid_indices)
    X_valid = X[valid_indices]
    Y_valid = [Y[i] for i in valid_indices]
    steps_valid = [steps[i] for i in valid_indices]

    print(f"  Steps with Y labels: {len(Y_valid)}")

    # Count Y distribution
    y_counts = Counter()
    for y_set in Y_valid:
        for op in y_set:
            y_counts[op] += 1
    print(f"  Y distribution (top 10):")
    for op, count in y_counts.most_common(10):
        print(f"    {op}: {count}")

    return X_valid, Y_valid, steps_valid


def compute_I_TY(assignments: np.ndarray, Y: List[Set[str]]) -> float:
    """
    Compute I(T;Y) - mutual information between clusters and operators.

    I(T;Y) = H(Y) - H(Y|T)
           = sum over t,y of p(t,y) * log(p(t,y) / (p(t)*p(y)))

    For multi-label Y (sets of operators), we compute MI for each operator
    independently and sum.
    """
    n = len(Y)
    unique_clusters = np.unique(assignments)

    # Get all unique operators
    all_ops = set()
    for y_set in Y:
        all_ops.update(y_set)

    total_mi = 0.0

    for op in all_ops:
        # Binary indicator: does step have this operator?
        y_binary = np.array([1 if op in y_set else 0 for y_set in Y])

        # p(y=1) and p(y=0)
        p_y1 = y_binary.sum() / n
        p_y0 = 1 - p_y1

        if p_y1 == 0 or p_y1 == 1:
            continue  # No information

        # H(Y) for this operator
        H_Y = -p_y1 * np.log(p_y1 + 1e-10) - p_y0 * np.log(p_y0 + 1e-10)

        # H(Y|T) = sum_t p(t) * H(Y|T=t)
        H_Y_given_T = 0.0
        for t in unique_clusters:
            mask = assignments == t
            p_t = mask.sum() / n
            if p_t == 0:
                continue

            # p(y=1|t)
            p_y1_given_t = y_binary[mask].sum() / mask.sum()
            p_y0_given_t = 1 - p_y1_given_t

            if p_y1_given_t > 0 and p_y1_given_t < 1:
                H_Y_given_t = -p_y1_given_t * np.log(p_y1_given_t + 1e-10) \
                              -p_y0_given_t * np.log(p_y0_given_t + 1e-10)
                H_Y_given_T += p_t * H_Y_given_t

        # I(T;Y) for this operator
        mi_op = H_Y - H_Y_given_T
        total_mi += mi_op

    return total_mi


def compute_I_XT(X: np.ndarray, assignments: np.ndarray) -> float:
    """
    Compute I(X;T) - mutual information between embeddings and clusters.

    For continuous X, approximate via variance reduction:
    I(X;T) ≈ log(var(X)) - E_t[log(var(X|T=t))]

    Higher I(X;T) = more information preserved = less compression.
    """
    n = len(X)
    unique_clusters = np.unique(assignments)

    # Total variance (trace of covariance)
    total_var = np.var(X, axis=0).sum()

    # Conditional variance
    conditional_var = 0.0
    for t in unique_clusters:
        mask = assignments == t
        p_t = mask.sum() / n
        if mask.sum() > 1:
            cluster_var = np.var(X[mask], axis=0).sum()
            conditional_var += p_t * cluster_var

    # I(X;T) ≈ reduction in variance
    if total_var > 0:
        return np.log(total_var + 1e-10) - np.log(conditional_var + 1e-10)
    return 0.0


def compute_ib_objective(X: np.ndarray, Y: List[Set[str]],
                         assignments: np.ndarray, beta: float) -> float:
    """
    Compute IB objective: I(X;T) - β·I(T;Y)

    We want to MINIMIZE this (minimize compression cost, maximize relevance).
    """
    I_XT = compute_I_XT(X, assignments)
    I_TY = compute_I_TY(assignments, Y)
    return I_XT - beta * I_TY


def try_split(X: np.ndarray, Y: List[Set[str]],
              cluster_indices: np.ndarray, beta: float) -> Optional[Tuple[np.ndarray, float]]:
    """
    Try to split a cluster. Returns new assignments if split improves objective.

    Uses PCA to find split direction, then finds optimal split point
    that maximizes I(T;Y) gain.
    """
    n = len(cluster_indices)
    if n < 20:
        return None

    X_cluster = X[cluster_indices]
    Y_cluster = [Y[i] for i in cluster_indices]

    # Find principal direction
    centroid = X_cluster.mean(axis=0)
    centered = X_cluster - centroid

    try:
        # Use SVD for numerical stability
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        split_dir = Vt[0]  # First principal component
    except:
        return None

    # Project onto split direction
    projections = centered @ split_dir

    # Try multiple split points and pick best by I(T;Y) improvement
    sorted_indices = np.argsort(projections)
    best_split = None
    best_I_TY_gain = 0

    # Try splits at 10%, 20%, ..., 90%
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        split_idx = int(n * frac)
        if split_idx < 10 or n - split_idx < 10:
            continue

        # Create split assignment
        split = np.zeros(n, dtype=int)
        split[sorted_indices[split_idx:]] = 1

        # Compute I(T;Y) for this split
        I_TY_split = 0
        for op in set().union(*Y_cluster):
            y_binary = np.array([1 if op in y_set else 0 for y_set in Y_cluster])

            for t in [0, 1]:
                mask = split == t
                if mask.sum() == 0:
                    continue
                p_t = mask.sum() / n
                p_y1_given_t = y_binary[mask].sum() / mask.sum()
                p_y1 = y_binary.sum() / n

                if p_y1_given_t > 0 and p_y1 > 0:
                    # Contribution to MI
                    I_TY_split += (p_t * p_y1_given_t) * np.log((p_y1_given_t + 1e-10) / (p_y1 + 1e-10))

        if I_TY_split > best_I_TY_gain:
            best_I_TY_gain = I_TY_split
            best_split = split.copy()

    if best_split is None or best_I_TY_gain < 0.01:
        return None

    # Check if split is justified by IB objective
    # Compute variance reduction (approximates I(X;T) change)
    c0 = X_cluster[best_split == 0].mean(axis=0) if (best_split == 0).sum() > 0 else centroid
    c1 = X_cluster[best_split == 1].mean(axis=0) if (best_split == 1).sum() > 0 else centroid

    dist_before = np.linalg.norm(X_cluster - centroid, axis=1).mean()
    dist_after = 0
    for t, c in [(0, c0), (1, c1)]:
        mask = best_split == t
        if mask.sum() > 0:
            dist_after += mask.sum() / n * np.linalg.norm(X_cluster[mask] - c, axis=1).mean()

    variance_reduction = dist_before - dist_after

    # Split if: β * I(T;Y)_gain > I(X;T)_cost
    # Approximation: variance_reduction ≈ I(X;T) cost of splitting
    if beta * best_I_TY_gain > variance_reduction * 0.1:  # Threshold factor
        return best_split, best_I_TY_gain

    return None


def ib_annealing(X: np.ndarray, Y: List[Set[str]],
                 beta_start: float = 0.01,
                 beta_end: float = 100.0,
                 beta_steps: int = 50,
                 min_cluster_size: int = 50,
                 verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    β-annealing for Information Bottleneck.

    Start with one cluster (maximum compression).
    Increase β gradually.
    At each β, try to split clusters where it improves the IB objective.
    Record bifurcations (phase transitions).
    """
    n = len(X)
    assignments = np.zeros(n, dtype=int)
    next_cluster_id = 1

    betas = np.logspace(np.log10(beta_start), np.log10(beta_end), beta_steps)
    history = []

    if verbose:
        print(f"\n{'='*70}")
        print("TRUE INFORMATION BOTTLENECK")
        print(f"{'='*70}")
        print(f"  X: {X.shape}")
        print(f"  Y: {len(Y)} steps with operator labels")
        print(f"  β range: {beta_start} → {beta_end}")
        print(f"  Min cluster size: {min_cluster_size}")

    for beta_idx, beta in enumerate(betas):
        unique_clusters = np.unique(assignments)
        n_clusters_before = len(unique_clusters)
        splits_this_round = 0

        for cid in unique_clusters:
            mask = assignments == cid
            cluster_size = mask.sum()

            if cluster_size < min_cluster_size * 2:
                continue

            cluster_indices = np.where(mask)[0]

            # Try to split
            result = try_split(X, Y, cluster_indices, beta)

            if result is not None:
                split_assignments, I_TY_gain = result

                # Apply split
                new_cid = next_cluster_id
                next_cluster_id += 1

                for i, idx in enumerate(cluster_indices):
                    if split_assignments[i] == 1:
                        assignments[idx] = new_cid

                splits_this_round += 1

                # Record bifurcation
                history.append({
                    'beta': float(beta),
                    'parent_cluster': int(cid),
                    'new_cluster': int(new_cid),
                    'I_TY_gain': float(I_TY_gain),
                    'parent_size': int((assignments == cid).sum()),
                    'child_size': int((assignments == new_cid).sum()),
                })

        n_clusters_after = len(np.unique(assignments))

        if verbose and (splits_this_round > 0 or beta_idx % 10 == 0):
            I_TY = compute_I_TY(assignments, Y)
            print(f"  β={beta:.3f}: {n_clusters_after} clusters, {splits_this_round} splits, I(T;Y)={I_TY:.3f}")

    return assignments, history


def build_clusters(X: np.ndarray, Y: List[Set[str]],
                   assignments: np.ndarray, steps: List[Dict]) -> List[IBCluster]:
    """Build cluster objects from assignments."""
    clusters = []

    for cid in np.unique(assignments):
        mask = assignments == cid
        indices = np.where(mask)[0]

        # Count Y distribution
        y_dist = Counter()
        for idx in indices:
            for op in Y[idx]:
                y_dist[op] += 1

        clusters.append(IBCluster(
            id=int(cid),
            indices=indices,
            y_distribution=dict(y_dist),
        ))

    # Sort by size
    clusters.sort(key=lambda c: c.size, reverse=True)

    return clusters


def save_results(clusters: List[IBCluster], history: List[Dict],
                 steps: List[Dict], upload: bool = True):
    """Save IB results."""
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal clusters: {len(clusters)}")

    # Compute stats
    sizes = [c.size for c in clusters]
    purities = [c.purity for c in clusters]

    print(f"Size distribution: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f}")
    print(f"Purity distribution: min={min(purities):.2f}, max={max(purities):.2f}, mean={np.mean(purities):.2f}")

    print(f"\nTop 20 clusters:")
    for c in clusters[:20]:
        top_ops = sorted(c.y_distribution.items(), key=lambda x: -x[1])[:3]
        top_ops_str = ', '.join(f"{op}:{cnt}" for op, cnt in top_ops)
        print(f"  C{c.id:03d}: n={c.size:5}, purity={c.purity:.2f}, ops=[{top_ops_str}]")

    # Build output
    templates = []
    for i, c in enumerate(clusters):
        templates.append({
            'template_id': i,
            'cluster_id': int(c.id),
            'size': c.size,
            'purity': round(c.purity, 4),
            'dominant_op': c.dominant_y,
            'y_distribution': c.y_distribution,
            'step_indices': c.indices.tolist(),
        })

    output = {
        'method': 'true_information_bottleneck',
        'n_clusters': len(clusters),
        'total_steps': sum(c.size for c in clusters),
        'mean_purity': round(float(np.mean(purities)), 4),
        'bifurcation_history': history,
        'templates': templates,
    }

    # Save locally
    with open('/tmp/ib_templates.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to /tmp/ib_templates.json")

    if upload:
        s3.upload_file('/tmp/ib_templates.json', BUCKET, 'ib_true/templates.json')
        print(f"Uploaded to s3://{BUCKET}/ib_true/templates.json")

    return output


def analyze_elbow(history: List[Dict], gain_threshold: float = None) -> Tuple[int, float]:
    """
    Analyze bifurcation history to find the elbow.

    The elbow is where I(T;Y) gains drop from meaningful to noise.
    Uses gain threshold method: only count splits with gain >= threshold.

    If no threshold provided, uses 0.15 which typically gives ~30 clusters.

    Returns: (optimal_n_clusters, elbow_gain_threshold)
    """
    if not history:
        return 1, 0.0

    gains = np.array([h['I_TY_gain'] for h in history])
    n_clusters = np.array([i + 2 for i in range(len(history))])
    cumulative_I_TY = np.cumsum(gains)

    print(f"\n{'='*70}")
    print("ELBOW ANALYSIS")
    print(f"{'='*70}")
    print(f"Total splits: {len(history)}")
    print(f"Gain range: {gains.min():.4f} - {gains.max():.4f}")
    print(f"Mean gain: {gains.mean():.4f}")
    print(f"Median gain: {np.median(gains):.4f}")

    # Threshold-based analysis - most reliable for IB
    print(f"\nThreshold-based cluster counts:")
    thresholds = [0.30, 0.20, 0.15, 0.10, 0.08, 0.05]
    for t in thresholds:
        count = (gains >= t).sum()
        marker = " <-- RECOMMENDED" if t == 0.15 else ""
        print(f"  gain >= {t:.2f}: {count + 1:3} clusters{marker}")

    # Use provided threshold or default to 0.15
    if gain_threshold is None:
        gain_threshold = 0.15
        print(f"\nUsing default threshold: {gain_threshold}")

    elbow_n = (gains >= gain_threshold).sum() + 1
    elbow_gain = gain_threshold

    print(f"\n⚡ ELBOW: {elbow_n} clusters (gain threshold = {gain_threshold})")

    # Show cumulative I(T;Y) captured
    if elbow_n - 2 < len(cumulative_I_TY):
        captured = cumulative_I_TY[elbow_n - 2]
        total = cumulative_I_TY[-1]
        pct = 100 * captured / total
        print(f"   Cumulative I(T;Y): {captured:.3f} / {total:.3f} ({pct:.1f}%)")

    # Also show geometric elbow for comparison
    x_norm = (n_clusters - n_clusters.min()) / (n_clusters.max() - n_clusters.min() + 1e-10)
    y_norm = (cumulative_I_TY - cumulative_I_TY.min()) / (cumulative_I_TY.max() - cumulative_I_TY.min() + 1e-10)
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-10)

    max_dist = 0
    geom_elbow_idx = 0
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        proj = start + np.dot(point - start, line_unit) * line_unit
        dist = np.linalg.norm(point - proj)
        if dist > max_dist:
            max_dist = dist
            geom_elbow_idx = i

    print(f"\n(Geometric elbow: {n_clusters[geom_elbow_idx]} clusters - for reference)")

    return elbow_n, elbow_gain


def plot_elbow(history: List[Dict], output_path: str = '/tmp/ib_elbow.png'):
    """
    Plot I(T;Y) gain vs number of clusters to visualize the elbow.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    if not history:
        print("No bifurcation history to plot")
        return

    n_clusters = np.array([i + 2 for i in range(len(history))])
    gains = np.array([h['I_TY_gain'] for h in history])
    cumulative_I_TY = np.cumsum(gains)

    # Find elbow
    x_norm = (n_clusters - n_clusters.min()) / (n_clusters.max() - n_clusters.min() + 1e-10)
    y_norm = (cumulative_I_TY - cumulative_I_TY.min()) / (cumulative_I_TY.max() - cumulative_I_TY.min() + 1e-10)

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-10)

    max_dist = 0
    elbow_idx = 0
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        proj = start + np.dot(point - start, line_unit) * line_unit
        dist = np.linalg.norm(point - proj)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: I(T;Y) gain per split
    ax1 = axes[0, 0]
    ax1.bar(n_clusters, gains, alpha=0.7, color='steelblue')
    ax1.axvline(x=n_clusters[elbow_idx], color='red', linestyle='--', linewidth=2, label=f'Elbow at n={n_clusters[elbow_idx]}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('I(T;Y) Gain per Split')
    ax1.set_title('Per-Split I(T;Y) Gain')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative I(T;Y)
    ax2 = axes[0, 1]
    ax2.plot(n_clusters, cumulative_I_TY, 'b-', linewidth=2, marker='o', markersize=3)
    ax2.axvline(x=n_clusters[elbow_idx], color='red', linestyle='--', linewidth=2, label=f'Elbow at n={n_clusters[elbow_idx]}')
    ax2.scatter([n_clusters[elbow_idx]], [cumulative_I_TY[elbow_idx]], color='red', s=100, zorder=5)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Cumulative I(T;Y)')
    ax2.set_title('Cumulative I(T;Y) vs Cluster Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log-scale gain to see falloff
    ax3 = axes[1, 0]
    ax3.semilogy(n_clusters, gains + 1e-6, 'g-', linewidth=2, marker='s', markersize=3)
    ax3.axvline(x=n_clusters[elbow_idx], color='red', linestyle='--', linewidth=2, label=f'Elbow at n={n_clusters[elbow_idx]}')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('I(T;Y) Gain (log scale)')
    ax3.set_title('Per-Split Gain (Log Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gain sorted by magnitude
    ax4 = axes[1, 1]
    sorted_gains = np.sort(gains)[::-1]
    ax4.bar(range(len(sorted_gains)), sorted_gains, alpha=0.7, color='orange')
    # Find where elbow's gain falls in sorted order
    elbow_gain = gains[elbow_idx]
    elbow_rank = np.searchsorted(-sorted_gains, -elbow_gain)
    ax4.axvline(x=elbow_rank, color='red', linestyle='--', linewidth=2, label=f'Elbow gain rank: {elbow_rank+1}')
    ax4.set_xlabel('Split Rank (by gain)')
    ax4.set_ylabel('I(T;Y) Gain')
    ax4.set_title('Gains Sorted by Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'IB Elbow Analysis: Optimal = {n_clusters[elbow_idx]} clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    return n_clusters[elbow_idx]


def truncate_to_elbow(clusters: List['IBCluster'], history: List[Dict],
                      elbow_n: int) -> Tuple[List['IBCluster'], List[Dict]]:
    """
    Truncate clusters to the elbow point.
    Merges clusters that were created after the elbow.
    """
    # Find which clusters to keep
    # Clusters created after elbow_n splits should be merged back
    n_splits_to_keep = elbow_n - 1  # e.g., 50 clusters = 49 splits

    if n_splits_to_keep >= len(history):
        return clusters, history

    # Build merge map from bifurcation history
    # Later clusters should merge back to their parents
    merge_to = {}
    for i, h in enumerate(history):
        if i >= n_splits_to_keep:
            new_cid = h['new_cluster']
            parent_cid = h['parent_cluster']
            # Trace parent up if it was also created after elbow
            while parent_cid in merge_to:
                parent_cid = merge_to[parent_cid]
            merge_to[new_cid] = parent_cid

    print(f"\nTruncating to {elbow_n} clusters (merging {len(merge_to)} clusters)")

    # Rebuild clusters with merges
    merged_clusters = {}
    for c in clusters:
        cid = c.id
        while cid in merge_to:
            cid = merge_to[cid]

        if cid not in merged_clusters:
            merged_clusters[cid] = {
                'id': cid,
                'indices': [],
                'y_distribution': Counter(),
            }

        merged_clusters[cid]['indices'].extend(c.indices.tolist())
        for op, cnt in c.y_distribution.items():
            merged_clusters[cid]['y_distribution'][op] += cnt

    # Convert back to IBCluster objects
    new_clusters = []
    for cid, data in merged_clusters.items():
        new_clusters.append(IBCluster(
            id=cid,
            indices=np.array(data['indices']),
            y_distribution=dict(data['y_distribution']),
        ))

    new_clusters.sort(key=lambda c: c.size, reverse=True)
    truncated_history = history[:n_splits_to_keep]

    return new_clusters, truncated_history


def main():
    parser = argparse.ArgumentParser(description='True Information Bottleneck')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results (find elbow)')
    parser.add_argument('--truncate', type=int, default=None,
                        help='Truncate to N clusters')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Gain threshold for elbow (default: 0.15 → ~30 clusters)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate elbow plot')
    args = parser.parse_args()

    print("=" * 70)
    print("TRUE INFORMATION BOTTLENECK")
    print("min I(X;T) - β·I(T;Y)")
    print("=" * 70)

    if args.analyze_only:
        # Load existing results
        print("Loading existing IB results...")
        response = s3.get_object(Bucket=BUCKET, Key='ib_true/templates.json')
        data = json.loads(response['Body'].read().decode('utf-8'))
        history = data['bifurcation_history']

        # Analyze elbow
        elbow_n, elbow_gain = analyze_elbow(history, gain_threshold=args.threshold)

        if args.plot:
            plot_elbow(history)

        if args.truncate:
            print(f"\nTruncating to {args.truncate} clusters...")
            # Would need to reload X, Y to rebuild clusters properly
            print("(Full truncation requires re-running with --truncate flag)")

    else:
        # Load data
        X, Y, steps = load_data()

        # Normalize embeddings
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Run IB
        start_time = time.time()
        assignments, history = ib_annealing(
            X_norm, Y,
            beta_start=0.01,
            beta_end=100.0,
            beta_steps=50,
            min_cluster_size=50,
            verbose=True,
        )
        elapsed = time.time() - start_time
        print(f"\nIB completed in {elapsed:.1f}s")

        # Build clusters
        clusters = build_clusters(X_norm, Y, assignments, steps)

        # Analyze elbow
        elbow_n, elbow_gain = analyze_elbow(history, gain_threshold=args.threshold)

        if args.plot:
            plot_elbow(history)

        # Truncate to elbow if requested
        if args.truncate:
            clusters, history = truncate_to_elbow(clusters, history, args.truncate)
        elif elbow_n < len(clusters):
            print(f"\n⚠️  Elbow suggests {elbow_n} clusters, but we have {len(clusters)}")
            print(f"   Run with --truncate {elbow_n} to use elbow-optimal clustering")

        # Save
        output = save_results(clusters, history, steps, upload=True)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
