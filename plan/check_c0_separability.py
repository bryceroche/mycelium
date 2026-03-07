"""
Check if C0's 4-dim attention features separate IB clusters better than C1-A's 896-dim features.
"""

import json
import io
import numpy as np
import boto3
from collections import defaultdict

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Window parameters
W = 16
S = 8

print("Loading IB step map and cluster assignments...")
# Load IB assignments
resp = s3.get_object(Bucket=BUCKET, Key="ib_results_math/cluster_assignments.npy")
ib_assignments = np.load(io.BytesIO(resp["Body"].read()))

# Load step map (maps step_idx to problem_id)
resp = s3.get_object(Bucket=BUCKET, Key="ib_ready/step_map.json")
step_map = json.loads(resp["Body"].read().decode("utf-8"))

print(f"  {len(ib_assignments)} IB assignments")
print(f"  {len(step_map)} steps in map")

# Create step -> cluster mapping
step_to_cluster = {}
for i, step_info in enumerate(step_map):
    key = (step_info["problem_id"], step_info["step_idx"])
    step_to_cluster[key] = int(ib_assignments[i])

# Load C0 hint vectors from multiple chunks
print("\nLoading C0 hint vectors...")
c0_hints = {}
n_loaded = 0

for gpu in range(4):
    for chunk in range(20):
        try:
            key = f"c0_training_data/hint_vectors/instance1_iaf_v3_gpu{gpu}_valid_chunk_{chunk:03d}_hints.jsonl"
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            content = resp["Body"].read().decode("utf-8")

            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)

                # Try different problem ID formats to match step_map
                for prefix in ["chunk0_", "chunk1_", "chunk2_", "chunk3_", ""]:
                    prob_id = f"{prefix}{d['problem_id']}"

                    n_tokens = len(d["hint_vectors"]["entropy"])
                    features = np.zeros((n_tokens, 4))
                    features[:, 0] = d["hint_vectors"]["entropy"][:n_tokens]
                    features[:, 1] = d["hint_vectors"]["tension"][:n_tokens]
                    features[:, 2] = d["hint_vectors"]["telegraph"][:n_tokens]

                    # Handle both "connectivity" and "received" field names
                    if "connectivity" in d["hint_vectors"]:
                        features[:, 3] = d["hint_vectors"]["connectivity"][:n_tokens]
                    elif "received" in d["hint_vectors"]:
                        features[:, 3] = d["hint_vectors"]["received"][:n_tokens]

                    c0_hints[prob_id] = features
                    n_loaded += 1

        except Exception as e:
            continue

print(f"  Loaded {n_loaded} problem variants with C0 hints")

# Show sample problem IDs from both sources
step_prob_ids = set(prob_id for prob_id, _ in step_to_cluster.keys())
c0_prob_ids = set(c0_hints.keys())

print(f"\nSample step_map problem IDs: {list(step_prob_ids)[:5]}")
print(f"Sample C0 problem IDs: {list(c0_prob_ids)[:5]}")

# Find overlap
overlap = step_prob_ids & c0_prob_ids
print(f"Overlap: {len(overlap)} problems")

# Compute per-cluster C0 features
print("\nComputing C0 features per IB cluster...")
cluster_features = defaultdict(list)
n_matched = 0

for (prob_id, step_idx), cluster_id in step_to_cluster.items():
    if prob_id not in c0_hints:
        continue

    c0_feats = c0_hints[prob_id]

    # Mean pool all tokens as approximation
    pooled = c0_feats.mean(axis=0)  # (4,)
    cluster_features[cluster_id].append(pooled)
    n_matched += 1

print(f"  Matched {n_matched} steps to C0 features")
print(f"  Clusters with data: {len(cluster_features)}")

if n_matched == 0:
    print("\nNo matches found. Let's try direct problem_id matching...")

    # Try using just the numeric part
    step_to_cluster_simple = {}
    for (prob_id, step_idx), cluster in step_to_cluster.items():
        # Extract numeric part
        if "_" in prob_id:
            numeric = prob_id.split("_")[-1]
        else:
            numeric = prob_id
        step_to_cluster_simple[(numeric, step_idx)] = cluster

    for prob_id in c0_hints:
        # Extract numeric part
        if "_" in prob_id:
            numeric = prob_id.split("_")[-1]
        else:
            numeric = prob_id

        # Check if this matches any step
        for (step_prob, step_idx), cluster_id in step_to_cluster_simple.items():
            if step_prob == numeric:
                c0_feats = c0_hints[prob_id]
                pooled = c0_feats.mean(axis=0)
                cluster_features[cluster_id].append(pooled)
                n_matched += 1

    print(f"  After simple matching: {n_matched} steps matched")

# Compute separability
print("\n" + "="*50)
print("C0 FEATURE SEPARABILITY (4-dim attention space)")
print("="*50)

if len(cluster_features) < 2:
    print("Not enough clusters with data to compute separability")
    exit(1)

# Compute centroids
centroids = {}
for c_id, feats in cluster_features.items():
    if len(feats) >= 5:
        centroids[c_id] = np.mean(feats, axis=0)

print(f"Clusters with >= 5 samples: {len(centroids)}")

if len(centroids) < 2:
    print("Not enough clusters with sufficient samples")
    exit(1)

# Intra-cluster distances
intra_dists = []
for c_id, feats in cluster_features.items():
    if c_id in centroids and len(feats) >= 5:
        centroid = centroids[c_id]
        dists = [np.linalg.norm(f - centroid) for f in feats]
        intra_dists.append(np.mean(dists))

avg_intra = np.mean(intra_dists)

# Inter-cluster distances
c_ids = list(centroids.keys())
inter_dists = []
for i, c1 in enumerate(c_ids):
    for c2 in c_ids[i+1:]:
        inter_dists.append(np.linalg.norm(centroids[c1] - centroids[c2]))

avg_inter = np.mean(inter_dists)
ratio = avg_inter / avg_intra

print(f"Avg intra-cluster distance: {avg_intra:.4f}")
print(f"Avg inter-cluster distance: {avg_inter:.4f}")
print(f"Separability ratio: {ratio:.2f}")
print()

if ratio >= 2.0:
    print("VERDICT: C0 features ARE separable (ratio >= 2.0)")
    print("Proceed with C0-based text energy!")
elif ratio >= 1.5:
    print("VERDICT: C0 features show marginal separability (1.5 <= ratio < 2.0)")
    print("Worth trying C0-based energy")
else:
    print("VERDICT: C0 features still poorly separable (ratio < 1.5)")

print()
print("For comparison:")
print("  C1-A 896-dim space: ratio = 0.22-0.46")
