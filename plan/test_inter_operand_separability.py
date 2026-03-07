"""
Test Inter-Operand Feature Separability (REAL TEST)

Hypothesis: Operation signal is in tokens BETWEEN operands,
not in the mean-pooled window.

Using Sonnet-extracted operand positions from c2c3_sonnet_labels.
"""

import io
import json
import numpy as np
import boto3
from collections import defaultdict

s3 = boto3.client("s3")
BUCKET = "mycelium-data"


def load_sonnet_labels(max_files=10):
    """Load Sonnet-extracted labels with operand positions."""
    print("Loading Sonnet labels with operand positions...")

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="c2c3_sonnet_labels/instance1_iaf_v3")

    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".jsonl")]
    print(f"  Found {len(files)} label files")

    all_segments = []
    for key in files[:max_files]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            content = resp["Body"].read().decode("utf-8")

            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)

                prob_id = d["problem_id"]
                for seg in d.get("segments", []):
                    seg["problem_id"] = prob_id
                    seg["c2_label"] = seg.get("c2_label", "UNKNOWN")
                    all_segments.append(seg)

        except Exception as e:
            continue

    print(f"  Loaded {len(all_segments)} segments with operand data")
    return all_segments


def load_cached_features():
    """Load cached per-token features."""
    print("\nLoading cached features...")

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="c2c3_training_ready/cached_features/")

    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".npy")]
    print(f"  Found {len(files)} feature files")

    # Build mapping: problem_idx -> features
    prob_to_features = {}
    for key in files:  # Load ALL files
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            features = np.load(io.BytesIO(resp["Body"].read()))

            # Filename is just "0.npy", "1.npy", etc.
            prob_idx = key.split("/")[-1].replace(".npy", "")
            prob_to_features[prob_idx] = features

        except Exception:
            continue

    print(f"  Loaded {len(prob_to_features)} feature sets")
    return prob_to_features


def extract_inter_operand(window_features, operand_positions):
    """
    Extract mean of tokens BETWEEN operands.

    window_features: (16, 896)
    operand_positions: list of (start, end) tuples

    Returns: (896,) mean of inter-operand tokens, or None
    """
    if len(operand_positions) < 2:
        return None

    # Sort by start position
    positions = sorted(operand_positions, key=lambda x: x[0])

    # Get region between first and second operand
    first_end = positions[0][1]  # End of first operand
    second_start = positions[1][0]  # Start of second operand

    # Clamp to valid range
    start = min(max(first_end, 0), 15)
    end = min(max(second_start, 0), 16)

    if start >= end:
        # Operands overlap or adjacent - use the span itself
        all_start = min(p[0] for p in positions[:2])
        all_end = max(p[1] for p in positions[:2])
        return window_features[all_start:min(all_end+1, 16)].mean(axis=0)

    # Mean of inter-operand tokens
    return window_features[start:end].mean(axis=0)


def compute_separability(features_by_label):
    """Compute inter/intra label distance ratio."""
    valid = {k: np.array(v) for k, v in features_by_label.items() if len(v) >= 10}

    if len(valid) < 2:
        return 0.0, {"error": "not enough labels", "n_labels": len(valid)}

    # Centroids
    centroids = {k: v.mean(axis=0) for k, v in valid.items()}

    # Intra-cluster distances
    intra = []
    for k, feats in valid.items():
        c = centroids[k]
        intra.extend(np.linalg.norm(feats - c, axis=1))

    # Inter-cluster distances
    labels = list(centroids.keys())
    inter = []
    for i, k1 in enumerate(labels):
        for k2 in labels[i+1:]:
            inter.append(np.linalg.norm(centroids[k1] - centroids[k2]))

    avg_intra = np.mean(intra)
    avg_inter = np.mean(inter)
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0.0

    return ratio, {
        "n_labels": len(valid),
        "n_samples": sum(len(v) for v in valid.values()),
        "avg_intra": avg_intra,
        "avg_inter": avg_inter,
        "labels": list(valid.keys()),
    }


def main():
    print("="*60)
    print("INTER-OPERAND FEATURE SEPARABILITY (REAL TEST)")
    print("="*60)

    # Load data
    segments = load_sonnet_labels(max_files=62)  # Load all
    prob_to_features = load_cached_features()

    # Extract features for each approach
    mean_pooled_by_label = defaultdict(list)
    inter_operand_by_label = defaultdict(list)

    n_processed = 0
    n_with_features = 0
    n_with_operands = 0

    # Track which problem IDs match
    segment_prob_ids = set(s["problem_id"] for s in segments[:100])
    feature_prob_ids = set(prob_to_features.keys())
    print(f"\nSample segment problem IDs: {list(segment_prob_ids)[:5]}")
    print(f"Sample feature problem IDs: {list(feature_prob_ids)[:5]}")

    # Need to map between them
    # Segments use "chunk2_0", features use "0"
    def normalize_prob_id(prob_id):
        """Extract numeric part from problem ID."""
        if "_" in str(prob_id):
            return str(prob_id).split("_")[-1]
        return str(prob_id)

    for seg in segments:
        n_processed += 1

        prob_id = normalize_prob_id(seg["problem_id"])
        w_idx = seg.get("window_idx", 0)
        c2_label = seg.get("c2_label", "UNKNOWN")

        # Skip non-operation labels
        if c2_label in ["UNKNOWN", "NO_OP", "OTHER"]:
            continue

        if prob_id not in prob_to_features:
            continue

        features = prob_to_features[prob_id]
        if w_idx >= len(features):
            continue

        window_features = features[w_idx]  # (16, 896)
        n_with_features += 1

        # Method 1: Mean-pooled (baseline)
        mean_pooled = window_features.mean(axis=0)
        mean_pooled_by_label[c2_label].append(mean_pooled)

        # Method 2: Inter-operand extraction
        operand_positions = []
        for op in seg.get("c3_operands", []):
            if isinstance(op, dict) and "start" in op and "end" in op:
                start = op["start"]
                end = op["end"]
                if 0 <= start < 16 and 0 <= end <= 16:
                    operand_positions.append((start, end))

        if len(operand_positions) >= 2:
            n_with_operands += 1
            inter_feat = extract_inter_operand(window_features, operand_positions)
            if inter_feat is not None:
                inter_operand_by_label[c2_label].append(inter_feat)

    print(f"\nProcessed: {n_processed} segments")
    print(f"With features: {n_with_features}")
    print(f"With 2+ operands: {n_with_operands}")

    print(f"\nLabels found (mean-pooled): {dict((k, len(v)) for k, v in mean_pooled_by_label.items())}")
    print(f"Labels found (inter-operand): {dict((k, len(v)) for k, v in inter_operand_by_label.items())}")

    # Compute separability
    print("\n" + "="*60)
    print("SEPARABILITY RESULTS")
    print("="*60)

    mp_ratio, mp_stats = compute_separability(mean_pooled_by_label)
    io_ratio, io_stats = compute_separability(inter_operand_by_label)

    print(f"\nMEAN-POOLED FEATURES (baseline):")
    print(f"  Labels with 10+ samples: {mp_stats.get('n_labels', 0)}")
    print(f"  Total samples: {mp_stats.get('n_samples', 0)}")
    print(f"  Labels: {mp_stats.get('labels', [])}")
    print(f"  Avg intra-label dist: {mp_stats.get('avg_intra', 0):.4f}")
    print(f"  Avg inter-label dist: {mp_stats.get('avg_inter', 0):.4f}")
    print(f"  SEPARABILITY RATIO: {mp_ratio:.4f}")

    print(f"\nINTER-OPERAND FEATURES (hypothesis):")
    print(f"  Labels with 10+ samples: {io_stats.get('n_labels', 0)}")
    print(f"  Total samples: {io_stats.get('n_samples', 0)}")
    print(f"  Labels: {io_stats.get('labels', [])}")
    print(f"  Avg intra-label dist: {io_stats.get('avg_intra', 0):.4f}")
    print(f"  Avg inter-label dist: {io_stats.get('avg_inter', 0):.4f}")
    print(f"  SEPARABILITY RATIO: {io_ratio:.4f}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if mp_ratio > 0 and io_ratio > 0:
        improvement = io_ratio / mp_ratio
        print(f"\nImprovement: {improvement:.2f}x")

        if io_ratio > 1.0:
            print(f"\n✓ SUCCESS: Inter-operand ratio {io_ratio:.2f} > 1.0")
            print("  Operation labels ARE separable in inter-operand feature space!")
            print("  The 'of' vs 'plus' signal IS in the tokens between operands.")
            print("\n  Next: Train C2 classifier on inter-operand features.")
        elif io_ratio > mp_ratio:
            print(f"\n~ PARTIAL: Inter-operand improved ({io_ratio:.2f} vs {mp_ratio:.2f})")
            print("  But still < 1.0, clusters overlap.")
        else:
            print(f"\n✗ NO IMPROVEMENT: Ratios similar or worse")
    elif io_ratio > 0:
        print(f"\nInter-operand ratio: {io_ratio:.4f}")
        if io_ratio > 1.0:
            print("✓ SUCCESS: Operations separable!")
        elif io_ratio > 0.5:
            print("~ MARGINAL: Some separation")
        else:
            print("✗ LOW: Poor separation")
    else:
        print("\nInsufficient data to compute ratios")

    # Save results
    print("\nSaving results...")
    results = {
        "mean_pooled": {"ratio": float(mp_ratio), **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in mp_stats.items()}},
        "inter_operand": {"ratio": float(io_ratio), **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in io_stats.items()}},
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/inter_operand_separability.json",
        Body=json.dumps(results, indent=2, default=str).encode("utf-8")
    )


if __name__ == "__main__":
    main()
