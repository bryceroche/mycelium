"""
Create Scaffold Training Data

Maps Sonnet-extracted step_types to scaffold categories.
Creates training data for step-type prediction from C1-A features.

Scaffold step types (consolidating Sonnet's 12 types):
    SETUP:      establish equation/expression (EQUATION c2_label)
    SUBSTITUTE: substitute, evaluate
    SIMPLIFY:   simplify, expand, factor
    SOLVE:      solve_equation
    COMPUTE:    fraction, sqrt, power operations
    THEOREM:    apply_theorem
    OTHER:      other, compare, NO_OP
"""

import json
import io
import numpy as np
from collections import Counter, defaultdict
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"


# Mapping from Sonnet step_types to scaffold categories
STEP_TYPE_MAP = {
    # Setup
    "setup": "SETUP",

    # Substitute/Evaluate (plugging values)
    "substitute": "SUBSTITUTE",
    "evaluate": "SUBSTITUTE",

    # Simplification (algebraic manipulation)
    "simplify": "SIMPLIFY",
    "expand": "SIMPLIFY",
    "factor": "SIMPLIFY",

    # Solving
    "solve_equation": "SOLVE",

    # Computation (specific operations)
    "fraction": "COMPUTE",
    "sqrt": "COMPUTE",
    "power": "COMPUTE",
    "count": "COMPUTE",

    # Theorem application
    "apply_theorem": "THEOREM",

    # Other
    "other": "OTHER",
    "compare": "OTHER",
}

# C2 labels that suggest scaffold types
C2_TO_SCAFFOLD = {
    "EQUATION": "SETUP",
    "ADD": "COMPUTE",
    "MUL": "COMPUTE",
    "SUB": "COMPUTE",
    "DIV": "COMPUTE",
    "NO_OP": "OTHER",
    "OTHER": "OTHER",
}


def get_window_scaffold_type(segment):
    """
    Determine scaffold type for a window from Sonnet labels.
    Uses majority voting over operand step_types, with c2_label as tiebreaker.
    """
    # Check c2_label first for SETUP (EQUATION)
    c2 = segment.get("c2_label", "OTHER")
    if c2 == "EQUATION":
        return "SETUP"

    # Count step_types from operands
    step_type_counts = Counter()
    for op in segment.get("c3_operands", []):
        st = op.get("step_type", "other")
        scaffold_type = STEP_TYPE_MAP.get(st, "OTHER")
        step_type_counts[scaffold_type] += 1

    if not step_type_counts:
        # No operands - use c2_label
        return C2_TO_SCAFFOLD.get(c2, "OTHER")

    # Get most common
    most_common = step_type_counts.most_common(1)[0][0]
    return most_common


def load_sonnet_labels():
    """Load all Sonnet label files."""
    print("Loading Sonnet labels...")

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="c2c3_sonnet_labels/instance1_iaf_v3")
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".jsonl")]
    print(f"  Found {len(files)} label files")

    all_segments = []
    problems_seen = set()

    for key in files:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            content = resp["Body"].read().decode("utf-8")

            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)

                prob_id = d["problem_id"]
                if prob_id in problems_seen:
                    continue
                problems_seen.add(prob_id)

                for seg in d.get("segments", []):
                    seg["problem_id"] = prob_id
                    seg["scaffold_type"] = get_window_scaffold_type(seg)
                    all_segments.append(seg)

        except Exception as e:
            continue

    print(f"  Loaded {len(all_segments)} segments from {len(problems_seen)} problems")
    return all_segments


def load_cached_features():
    """Load cached C1-A features."""
    print("\nLoading cached C1-A features...")

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="c2c3_training_ready/cached_features/")
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".npy")]
    print(f"  Found {len(files)} feature files")

    prob_to_features = {}
    for key in files:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            features = np.load(io.BytesIO(resp["Body"].read()))

            prob_idx = key.split("/")[-1].replace(".npy", "")
            prob_to_features[prob_idx] = features
        except:
            continue

    print(f"  Loaded {len(prob_to_features)} feature sets")
    return prob_to_features


def create_training_data(segments, prob_to_features):
    """
    Create training data for scaffold type prediction.

    For each window:
        X: mean-pooled C1-A features (896-dim)
        Y: scaffold type label
    """
    print("\nCreating training data...")

    # Label mapping
    SCAFFOLD_TYPES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"]
    label_to_idx = {t: i for i, t in enumerate(SCAFFOLD_TYPES)}

    X_list = []
    y_list = []
    meta_list = []

    type_counts = Counter()
    matched = 0
    unmatched = 0

    for seg in segments:
        prob_id = str(seg["problem_id"])
        window_idx = seg.get("window_idx", 0)
        scaffold_type = seg.get("scaffold_type", "OTHER")

        if prob_id not in prob_to_features:
            unmatched += 1
            continue

        features = prob_to_features[prob_id]
        if window_idx >= len(features):
            unmatched += 1
            continue

        # Get window features (16 tokens × 896 dims)
        window_features = features[window_idx]

        # Mean pool to get single vector
        pooled = window_features.mean(axis=0)  # (896,)

        X_list.append(pooled)
        y_list.append(label_to_idx[scaffold_type])
        meta_list.append({
            "problem_id": prob_id,
            "window_idx": window_idx,
            "scaffold_type": scaffold_type,
        })

        type_counts[scaffold_type] += 1
        matched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    print(f"\n  Scaffold type distribution:")
    for t, count in type_counts.most_common():
        pct = count / matched * 100 if matched > 0 else 0
        print(f"    {t}: {count} ({pct:.1f}%)")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y, meta_list, SCAFFOLD_TYPES


def evaluate_baseline(X, y, scaffold_types):
    """Quick baseline: logistic regression on mean-pooled features."""
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train simple logistic regression
    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.1%}")
    print("\nClassification Report:")
    # Use labels parameter to handle missing classes
    unique_labels = sorted(set(y_test) | set(y_pred))
    target_names_filtered = [scaffold_types[i] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels,
                               target_names=target_names_filtered, zero_division=0))

    return acc


def save_training_data(X, y, meta, scaffold_types):
    """Save training data to S3."""
    print("\nSaving training data to S3...")

    # Save features
    buf = io.BytesIO()
    np.save(buf, X)
    buf.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key="scaffold_training/features.npy",
        Body=buf.getvalue()
    )

    # Save labels
    buf = io.BytesIO()
    np.save(buf, y)
    buf.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key="scaffold_training/labels.npy",
        Body=buf.getvalue()
    )

    # Save metadata
    s3.put_object(
        Bucket=BUCKET,
        Key="scaffold_training/metadata.json",
        Body=json.dumps({
            "scaffold_types": scaffold_types,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(scaffold_types),
            "meta": meta[:100],  # Sample of metadata
        }, indent=2).encode("utf-8")
    )

    print(f"  Saved to s3://{BUCKET}/scaffold_training/")


def main():
    print("="*60)
    print("SCAFFOLD TRAINING DATA CREATION")
    print("="*60)

    # Load data
    segments = load_sonnet_labels()
    prob_to_features = load_cached_features()

    # Create training data
    X, y, meta, scaffold_types = create_training_data(segments, prob_to_features)

    if len(X) < 100:
        print("\n⚠ Not enough data for meaningful training")
        return

    # Baseline evaluation
    acc = evaluate_baseline(X, y, scaffold_types)

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    random_baseline = 1.0 / len(scaffold_types)
    improvement = acc / random_baseline

    print(f"\nRandom baseline: {random_baseline:.1%}")
    print(f"LogReg accuracy: {acc:.1%}")
    print(f"Improvement over random: {improvement:.2f}x")

    if acc > 0.40:
        print("\n✓ Scaffold types ARE predictable from C1-A features")
        print("  Even 40%+ is useful - narrows LLM's generation space")
    elif acc > 0.25:
        print("\n~ MARGINAL: Some signal, but weak")
        print("  May need better features or coarser categories")
    else:
        print("\n✗ Scaffold types NOT predictable from C1-A features")
        print("  Need different approach")

    # Save if useful
    if acc > 0.25:
        save_training_data(X, y, meta, scaffold_types)

    print("\nDone.")


if __name__ == "__main__":
    main()
