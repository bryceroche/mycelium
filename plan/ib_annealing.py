"""
Multi-Dimensional Information Bottleneck Annealing for C2 Cluster Discovery

Finds clusters in X (attention features) that maximally predict Y (structural properties).
Clusters become C2's label set.

Usage:
    python ib_annealing.py --sweep      # Run full β sweep
    python ib_annealing.py --evaluate   # Evaluate saved clusters
"""

import json
import argparse
import io
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"
INPUT_PREFIX = "ib_ready/"
OUTPUT_PREFIX = "ib_results_math/"

# Y dimension metadata
Y_DIM_NAMES = [
    "step_type", "complexity_change", "n_operands", "has_dependency",
    "output_type", "step_position", "reference_distance"
]

Y_DIM_CARDINALITIES = [12, 3, 4, 2, 5, 3, 4]

STEP_TYPE_NAMES = [
    "evaluate", "simplify", "substitute", "solve_equation",
    "factor", "expand", "apply_theorem", "count",
    "compare", "convert", "setup", "other"
]

COMPLEXITY_NAMES = ["reduces", "neutral", "increases"]
OUTPUT_TYPE_NAMES = ["number", "expression", "equation", "boolean", "set"]
POSITION_NAMES = ["early", "middle", "late"]
DISTANCE_NAMES = ["none", "local", "medium", "distant"]

Y_VALUE_NAMES = {
    "step_type": STEP_TYPE_NAMES,
    "complexity_change": COMPLEXITY_NAMES,
    "n_operands": ["1", "2", "3", "4+"],
    "has_dependency": ["false", "true"],
    "output_type": OUTPUT_TYPE_NAMES,
    "step_position": POSITION_NAMES,
    "reference_distance": DISTANCE_NAMES,
}

# Marginal weights for IB objective
MARGINAL_WEIGHTS = {
    "step_type": 1.0,
    "has_dependency": 0.9,
    "n_operands": 0.8,
    "reference_distance": 0.8,
    "complexity_change": 0.7,
    "output_type": 0.5,
    "step_position": 0.4,
}


def load_data():
    """Load X and Y matrices from S3."""
    print("Loading data from S3...")

    # Load X
    resp = s3.get_object(Bucket=BUCKET, Key=f"{INPUT_PREFIX}X_matrix.npy")
    X = np.load(io.BytesIO(resp["Body"].read()))

    # Load Y
    resp = s3.get_object(Bucket=BUCKET, Key=f"{INPUT_PREFIX}Y_matrix.npy")
    Y = np.load(io.BytesIO(resp["Body"].read()))

    print(f"Loaded X: {X.shape}, Y: {Y.shape}")
    return X, Y


def compute_weighted_nmi(assignments, Y, alpha=0.3):
    """
    Compute weighted NMI combining joint and marginal relevance.

    R(T;Y) = α · I(T; Y_joint) + (1-α) · Σ_i w_i · I(T; Y_i)
    """
    n_samples = len(assignments)

    # Marginal NMIs
    marginal_nmis = {}
    for dim_idx, dim_name in enumerate(Y_DIM_NAMES):
        nmi = normalized_mutual_info_score(assignments, Y[:, dim_idx])
        marginal_nmis[dim_name] = nmi

    # Weighted marginal average
    total_weight = sum(MARGINAL_WEIGHTS.values())
    weighted_marginal = sum(
        MARGINAL_WEIGHTS[dim] * marginal_nmis[dim]
        for dim in Y_DIM_NAMES
    ) / total_weight

    # Joint NMI (encode Y as single integer)
    # Use only top dimensions to avoid sparse space
    top_dims = [0, 1, 3, 4]  # step_type, complexity, has_dep, output_type
    y_joint = np.zeros(n_samples, dtype=int)
    multiplier = 1
    for dim_idx in top_dims:
        y_joint += Y[:, dim_idx] * multiplier
        multiplier *= Y_DIM_CARDINALITIES[dim_idx]

    joint_nmi = normalized_mutual_info_score(assignments, y_joint)

    # Combined relevance
    combined_nmi = alpha * joint_nmi + (1 - alpha) * weighted_marginal

    return {
        "combined_nmi": combined_nmi,
        "joint_nmi": joint_nmi,
        "weighted_marginal_nmi": weighted_marginal,
        "per_dim_nmi": marginal_nmis,
    }


def ib_clustering(X, Y, n_clusters, beta, n_restarts=10, max_iter=100, alpha=0.3):
    """
    Information Bottleneck clustering using modified k-means.

    Objective: minimize I(X;T) - β · R(T;Y)

    Approximation: use distance-based compression with Y-weighted relevance.
    """
    n_samples, x_dim = X.shape

    # Standardize X for fair distance computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_assignments = None
    best_objective = float('-inf')  # We want to MAXIMIZE relevance - compression

    for restart in range(n_restarts):
        # Initialize with k-means++
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=10, random_state=restart)
        assignments = kmeans.fit_predict(X_scaled)

        for iteration in range(max_iter):
            old_assignments = assignments.copy()

            # Compute cluster statistics
            cluster_centroids = np.zeros((n_clusters, x_dim))
            cluster_y_dists = {}  # (cluster_id, dim_idx) -> distribution
            cluster_sizes = np.zeros(n_clusters)

            for c in range(n_clusters):
                mask = assignments == c
                cluster_sizes[c] = mask.sum()
                if cluster_sizes[c] == 0:
                    # Reinitialize empty cluster
                    cluster_centroids[c] = X_scaled[np.random.randint(n_samples)]
                    continue

                cluster_centroids[c] = X_scaled[mask].mean(axis=0)

                # Y distributions per dimension
                for dim_idx in range(Y.shape[1]):
                    n_classes = Y_DIM_CARDINALITIES[dim_idx]
                    dist = np.bincount(Y[mask, dim_idx], minlength=n_classes).astype(float)
                    dist = (dist + 0.1) / (dist.sum() + 0.1 * n_classes)  # Laplace smoothing
                    cluster_y_dists[(c, dim_idx)] = dist

            # Reassign samples
            new_assignments = np.zeros(n_samples, dtype=int)

            # Compute all distances at once for efficiency
            distances = np.zeros((n_samples, n_clusters))
            for c in range(n_clusters):
                distances[:, c] = np.sum((X_scaled - cluster_centroids[c]) ** 2, axis=1)

            # Compute Y relevance for each sample-cluster pair
            relevance = np.zeros((n_samples, n_clusters))
            for c in range(n_clusters):
                if cluster_sizes[c] == 0:
                    relevance[:, c] = -1000  # Very low relevance for empty clusters
                    continue

                for dim_idx, dim_name in enumerate(Y_DIM_NAMES):
                    weight = MARGINAL_WEIGHTS[dim_name]
                    dist = cluster_y_dists[(c, dim_idx)]
                    y_vals = Y[:, dim_idx]
                    # Log probability of observing this Y value in this cluster
                    relevance[:, c] += weight * np.log(dist[y_vals] + 1e-10)

            # Total cost: compression (distance) - β * relevance
            # We want to MINIMIZE cost, so relevance should be subtracted from distance
            costs = distances - beta * relevance
            new_assignments = np.argmin(costs, axis=1)

            # Check convergence
            changed = (new_assignments != old_assignments).sum()
            assignments = new_assignments

            if changed < n_samples * 0.001:
                break

        # Evaluate this solution
        nmi_result = compute_weighted_nmi(assignments, Y, alpha)
        objective = nmi_result["combined_nmi"]

        if objective > best_objective:
            best_objective = objective
            best_assignments = assignments.copy()

    return best_assignments


def run_single_config(args):
    """Run IB clustering for a single (beta, n_clusters) configuration."""
    X, Y, beta, n_clusters, alpha = args

    try:
        assignments = ib_clustering(
            X, Y, n_clusters, beta,
            n_restarts=10, max_iter=100, alpha=alpha
        )

        # Evaluate
        n_effective = len(set(assignments))
        nmi_result = compute_weighted_nmi(assignments, Y, alpha)

        # Cluster size distribution
        cluster_sizes = np.bincount(assignments, minlength=n_clusters)
        cluster_fractions = cluster_sizes / len(assignments)

        # Cluster purity (mean dominant Y fraction per cluster)
        purities = []
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() == 0:
                continue
            # Dominant step_type fraction
            step_type_dist = np.bincount(Y[mask, 0], minlength=12)
            purity = step_type_dist.max() / mask.sum()
            purities.append(purity)

        return {
            "beta": beta,
            "n_clusters": n_clusters,
            "n_effective_clusters": n_effective,
            "combined_nmi": nmi_result["combined_nmi"],
            "joint_nmi": nmi_result["joint_nmi"],
            "weighted_marginal_nmi": nmi_result["weighted_marginal_nmi"],
            "per_dim_nmi": nmi_result["per_dim_nmi"],
            "cluster_sizes": cluster_sizes.tolist(),
            "min_cluster_size": int(cluster_sizes[cluster_sizes > 0].min()) if (cluster_sizes > 0).any() else 0,
            "max_cluster_size": int(cluster_sizes.max()),
            "mean_purity": float(np.mean(purities)) if purities else 0.0,
            "assignments": assignments,
        }
    except Exception as e:
        return {
            "beta": beta,
            "n_clusters": n_clusters,
            "error": str(e),
        }


def run_beta_sweep(X, Y, alpha=0.3):
    """Run full β and n_clusters sweep."""
    beta_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    n_cluster_values = [5, 8, 10, 12, 15, 20, 25]

    print(f"Running β sweep: {len(beta_values)} × {len(n_cluster_values)} = {len(beta_values) * len(n_cluster_values)} configs")

    results = []
    best_result = None
    best_score = -1

    total = len(beta_values) * len(n_cluster_values)
    completed = 0

    for beta in beta_values:
        for n_clusters in n_cluster_values:
            result = run_single_config((X, Y, beta, n_clusters, alpha))

            if "error" not in result:
                results.append(result)

                # Track best based on combined NMI with size constraints
                n_eff = result["n_effective_clusters"]
                min_size = result["min_cluster_size"]
                max_frac = result["max_cluster_size"] / len(X)

                # Score: combined NMI, penalized for bad cluster distribution
                score = result["combined_nmi"]
                if n_eff < 8 or n_eff > 25:
                    score *= 0.9  # Penalty for too few/many clusters
                if min_size < len(X) * 0.01:
                    score *= 0.95  # Penalty for tiny clusters
                if max_frac > 0.3:
                    score *= 0.95  # Penalty for dominant cluster

                if score > best_score:
                    best_score = score
                    best_result = result

            completed += 1
            if completed % 5 == 0:
                print(f"  [{completed}/{total}] β={beta:.1f}, k={n_clusters}: "
                      f"NMI={result.get('combined_nmi', 0):.3f}, "
                      f"eff={result.get('n_effective_clusters', 0)}")

    return results, best_result


def generate_cluster_profiles(X, Y, assignments):
    """Generate detailed profiles for each cluster."""
    n_clusters = assignments.max() + 1
    profiles = []

    for c in range(n_clusters):
        mask = assignments == c
        size = mask.sum()

        if size == 0:
            continue

        profile = {
            "cluster_id": int(c),
            "size": int(size),
            "fraction": float(size / len(assignments)),
        }

        # Compute dominant values for each Y dimension
        for dim_idx, dim_name in enumerate(Y_DIM_NAMES):
            y_vals = Y[mask, dim_idx]
            dist = np.bincount(y_vals, minlength=Y_DIM_CARDINALITIES[dim_idx])
            dist_normed = dist / dist.sum()
            dominant_idx = dist.argmax()

            value_names = Y_VALUE_NAMES[dim_name]
            dominant_name = value_names[dominant_idx] if dominant_idx < len(value_names) else str(dominant_idx)

            profile[f"dominant_{dim_name}"] = dominant_name
            profile[f"{dim_name}_distribution"] = {
                value_names[i]: float(dist_normed[i])
                for i in range(min(len(value_names), len(dist_normed)))
            }

        # Mean X vector (attention features)
        profile["mean_x_vector"] = X[mask].mean(axis=0).tolist()

        profiles.append(profile)

    # Sort by size descending
    profiles.sort(key=lambda p: -p["size"])

    return profiles


def make_cluster_label(profile):
    """Generate human-readable label from cluster profile."""
    parts = []

    ref_dist = profile.get("dominant_reference_distance", "none")
    if ref_dist != "none":
        parts.append(ref_dist)

    step_type = profile.get("dominant_step_type", "other")
    parts.append(step_type)

    complexity = profile.get("dominant_complexity_change", "neutral")
    parts.append(complexity)

    return "_".join(parts)


def save_results(results, best_result, profiles, X, Y):
    """Save all results to S3."""
    print("\nSaving results to S3...")

    # Remove assignments from results for JSON serialization
    results_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "assignments"}
        results_json.append(r_copy)

    # Beta sweep results
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}beta_sweep_results.json",
        Body=json.dumps(results_json, indent=2).encode("utf-8"),
    )

    # Optimal config
    optimal_config = {
        "beta": best_result["beta"],
        "n_clusters": best_result["n_clusters"],
        "n_effective_clusters": best_result["n_effective_clusters"],
        "combined_nmi": best_result["combined_nmi"],
        "joint_nmi": best_result["joint_nmi"],
        "weighted_marginal_nmi": best_result["weighted_marginal_nmi"],
        "per_dim_nmi": best_result["per_dim_nmi"],
        "mean_purity": best_result["mean_purity"],
        "min_cluster_size": best_result["min_cluster_size"],
        "max_cluster_size": best_result["max_cluster_size"],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}optimal_config.json",
        Body=json.dumps(optimal_config, indent=2).encode("utf-8"),
    )

    # Cluster assignments
    assignments_buffer = io.BytesIO()
    np.save(assignments_buffer, best_result["assignments"])
    assignments_buffer.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}cluster_assignments.npy",
        Body=assignments_buffer.read(),
    )

    # Cluster profiles
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}cluster_profiles.json",
        Body=json.dumps(profiles, indent=2).encode("utf-8"),
    )

    # Cluster labels
    labels = {
        p["cluster_id"]: make_cluster_label(p)
        for p in profiles
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}cluster_labels.json",
        Body=json.dumps(labels, indent=2).encode("utf-8"),
    )

    # C2 label mapping (cluster_id -> label name)
    c2_mapping = {
        str(p["cluster_id"]): {
            "label": make_cluster_label(p),
            "dominant_step_type": p["dominant_step_type"],
            "dominant_complexity": p["dominant_complexity_change"],
            "dominant_output_type": p["dominant_output_type"],
            "size": p["size"],
            "fraction": p["fraction"],
        }
        for p in profiles
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}c2_label_mapping.json",
        Body=json.dumps(c2_mapping, indent=2).encode("utf-8"),
    )

    print(f"Saved to s3://{BUCKET}/{OUTPUT_PREFIX}")


def print_validation_report(best_result, profiles, X, Y):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"  β = {best_result['beta']}")
    print(f"  n_clusters = {best_result['n_clusters']}")
    print(f"  n_effective = {best_result['n_effective_clusters']}")
    print(f"  Combined NMI = {best_result['combined_nmi']:.4f}")

    print("\nPer-dimension NMI:")
    for dim, nmi in sorted(best_result["per_dim_nmi"].items(),
                           key=lambda x: -x[1]):
        print(f"  {dim:<20} {nmi:.4f}")

    print("\nCluster size distribution:")
    sizes = np.array(best_result["cluster_sizes"])
    sizes = sizes[sizes > 0]
    print(f"  Smallest: {sizes.min()} ({sizes.min()/len(X)*100:.1f}%)")
    print(f"  Largest:  {sizes.max()} ({sizes.max()/len(X)*100:.1f}%)")
    print(f"  Median:   {int(np.median(sizes))} ({np.median(sizes)/len(X)*100:.1f}%)")

    print(f"\nCluster purity (mean dominant step_type fraction): {best_result['mean_purity']*100:.1f}%")

    # Check inter/intra cluster distances
    assignments = best_result["assignments"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    intra_dists = []
    centroids = []
    for c in range(best_result["n_clusters"]):
        mask = assignments == c
        if mask.sum() < 2:
            continue
        cluster_x = X_scaled[mask]
        centroid = cluster_x.mean(axis=0)
        centroids.append(centroid)
        dists = np.sqrt(np.sum((cluster_x - centroid) ** 2, axis=1))
        intra_dists.extend(dists)

    inter_dists = []
    for i, c1 in enumerate(centroids):
        for c2 in centroids[i+1:]:
            inter_dists.append(np.sqrt(np.sum((c1 - c2) ** 2)))

    if intra_dists and inter_dists:
        mean_intra = np.mean(intra_dists)
        mean_inter = np.mean(inter_dists)
        print(f"\nCluster separation:")
        print(f"  Inter-cluster distance (mean): {mean_inter:.3f}")
        print(f"  Intra-cluster distance (mean): {mean_intra:.3f}")
        print(f"  Ratio (should be >2.0): {mean_inter/mean_intra:.2f}")

    print("\nTop clusters by size:")
    for p in profiles[:5]:
        print(f"  {p['cluster_id']}: {p['size']:>5} ({p['fraction']*100:>5.1f}%) "
              f"- {make_cluster_label(p)}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run full β sweep")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved clusters")
    parser.add_argument("--alpha", type=float, default=0.3, help="Joint vs marginal weight")
    args = parser.parse_args()

    X, Y = load_data()

    if args.sweep:
        print(f"\nRunning IB annealing sweep (alpha={args.alpha})...")
        results, best_result = run_beta_sweep(X, Y, alpha=args.alpha)

        # Generate profiles for best result
        profiles = generate_cluster_profiles(X, Y, best_result["assignments"])

        # Print validation report
        print_validation_report(best_result, profiles, X, Y)

        # Save results
        save_results(results, best_result, profiles, X, Y)

    elif args.evaluate:
        # Load and evaluate saved assignments
        print("Loading saved cluster assignments...")
        resp = s3.get_object(Bucket=BUCKET, Key=f"{OUTPUT_PREFIX}cluster_assignments.npy")
        assignments = np.load(io.BytesIO(resp["Body"].read()))

        nmi_result = compute_weighted_nmi(assignments, Y)
        print(f"Combined NMI: {nmi_result['combined_nmi']:.4f}")
        for dim, nmi in nmi_result["per_dim_nmi"].items():
            print(f"  {dim}: {nmi:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
