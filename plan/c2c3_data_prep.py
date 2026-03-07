"""
C2/C3 Training Data Preparation

Maps IB clusters to C1-A windows and caches hidden states.

Usage:
    python c2c3_data_prep.py --map-windows       # Map IB clusters to windows
    python c2c3_data_prep.py --cache-features    # Cache C1-A hidden states (GPU)
    python c2c3_data_prep.py --assemble          # Assemble final training data
    python c2c3_data_prep.py --stats             # Print dataset statistics
"""

import json
import re
import argparse
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import boto3
from botocore.config import Config

config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=config)
BUCKET = "mycelium-data"

# Window parameters (must match C1-A training)
W = 16  # Window size
S = 8   # Stride

# C2 labels (25 IB clusters + NO_OP)
NO_OP_LABEL = 0  # Reserve cluster 0 for NO_OP


def preprocess_latex(text: str) -> str:
    """Apply LaTeX preprocessing to match C1-A tokenization."""
    # Basic normalization - match C1-A preprocessing
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\begin{array}", "").replace("\\end{array}", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_ib_data():
    """Load IB cluster assignments and step mapping."""
    print("Loading IB data...")

    # Cluster assignments
    resp = s3.get_object(Bucket=BUCKET, Key="ib_results_math/cluster_assignments.npy")
    assignments = np.load(io.BytesIO(resp["Body"].read()))

    # Step map
    resp = s3.get_object(Bucket=BUCKET, Key="ib_ready/step_map.json")
    step_map = json.loads(resp["Body"].read().decode("utf-8"))

    # Cluster labels
    resp = s3.get_object(Bucket=BUCKET, Key="ib_results_math/c2_label_mapping.json")
    c2_labels = json.loads(resp["Body"].read().decode("utf-8"))

    print(f"  Assignments: {len(assignments)}")
    print(f"  Step map: {len(step_map)}")
    print(f"  C2 labels: {len(c2_labels)} clusters")

    return assignments, step_map, c2_labels


def load_sonnet_steps():
    """Load Sonnet parsed steps indexed by (problem_id, step_idx)."""
    print("Loading Sonnet parsed steps...")

    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_data_v2/parsed_steps.jsonl")
    content = resp["Body"].read().decode("utf-8")

    steps_index = {}
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        step = json.loads(line)
        key = (step["problem_id"], step["step_idx"])
        steps_index[key] = step

    print(f"  Indexed {len(steps_index)} steps")
    return steps_index


def load_c1_problems():
    """Load C1 training problems indexed by problem_idx (as string)."""
    print("Loading C1 training problems...")

    resp = s3.get_object(Bucket=BUCKET, Key="c1_training_v6/merged_training.jsonl")

    problems = {}
    for line in resp["Body"].iter_lines():
        p = json.loads(line.decode("utf-8"))
        # Use string key to match extract_problem_idx output
        key = str(p["problem_idx"])
        problems[key] = {
            "problem_idx": key,
            "problem_text": p.get("original_text", p.get("problem_text", "")),
            "bio_labels": p.get("bio_labels", []),
        }

    print(f"  Loaded {len(problems)} C1 problems")
    return problems


def extract_problem_idx(problem_id: str) -> str:
    """Extract problem_idx from chunk problem_id like 'chunk2_102' -> '102'."""
    parts = problem_id.split("_")
    if len(parts) >= 2:
        return parts[-1]  # Return as string to match C1 keys
    return ""


def find_text_in_problem(text_reference: str, problem_text: str) -> int:
    """Find character position of text_reference in problem_text."""
    if not text_reference:
        return -1

    preprocessed = preprocess_latex(problem_text)
    ref_clean = preprocess_latex(text_reference)

    # Exact match
    pos = preprocessed.lower().find(ref_clean.lower())
    if pos >= 0:
        return pos

    # Try first 50 chars
    if len(ref_clean) > 50:
        pos = preprocessed.lower().find(ref_clean[:50].lower())
        if pos >= 0:
            return pos

    # Try without LaTeX delimiters
    ref_simple = re.sub(r'[\$\\{}]', '', ref_clean)
    text_simple = re.sub(r'[\$\\{}]', '', preprocessed)
    pos = text_simple.lower().find(ref_simple[:30].lower())
    if pos >= 0:
        return pos

    return -1


def char_to_window_idx(char_pos: int, problem_text: str, n_windows: int) -> int:
    """Map character position to window index."""
    if char_pos < 0:
        return -1

    # Approximate: assume uniform char-to-token ratio
    total_chars = len(problem_text)
    if total_chars == 0:
        return -1

    # Rough estimate of token position
    # Assuming ~4 chars per token on average
    approx_tok = char_pos // 4

    # Map to window
    # Windows start at positions 0, S, 2S, ...
    window_idx = approx_tok // S

    return min(window_idx, n_windows - 1) if n_windows > 0 else -1


def build_window_labels(c1_problems, sonnet_steps, ib_assignments, step_map, c2_labels):
    """Build per-window labels for each C1 problem."""
    print("\nBuilding window labels...")

    # Create step -> cluster mapping
    step_to_cluster = {}
    for i, step_info in enumerate(step_map):
        key = (step_info["problem_id"], step_info["step_idx"])
        step_to_cluster[key] = int(ib_assignments[i])

    # Group Sonnet steps by problem_idx
    steps_by_problem = defaultdict(list)
    for key, step in sonnet_steps.items():
        problem_id, step_idx = key
        prob_idx = extract_problem_idx(problem_id)
        if prob_idx:  # Non-empty string
            steps_by_problem[prob_idx].append({
                **step,
                "cluster_id": step_to_cluster.get(key, -1)
            })

    print(f"  Steps grouped by problem: {len(steps_by_problem)} problems")

    # Build window labels for each C1 problem
    window_labels_all = {}
    stats = {
        "problems_with_steps": 0,
        "problems_without_steps": 0,
        "total_windows": 0,
        "windows_with_ops": 0,
        "windows_no_op": 0,
        "steps_mapped": 0,
        "steps_unmapped": 0,
    }

    for prob_idx, problem in c1_problems.items():
        problem_text = problem["problem_text"]
        preprocessed = preprocess_latex(problem_text)

        # Estimate number of windows based on text length
        # Assuming ~4 chars per token, 512 max tokens
        approx_tokens = min(len(preprocessed) // 4, 512)
        n_windows = max(1, (approx_tokens - W) // S + 1)

        # Initialize all windows as NO_OP
        window_labels = [
            {
                "c2_cluster_id": NO_OP_LABEL,
                "c2_cluster": "NO_OP",
                "c3_operands": [],
                "step_type": None,
                "n_operands": 0,
                "has_dependency": False,
                "reference_distance": "none"
            }
            for _ in range(n_windows)
        ]

        # Map steps to windows
        steps = steps_by_problem.get(prob_idx, [])
        if steps:
            stats["problems_with_steps"] += 1

            for step in steps:
                text_ref = step.get("text_reference")
                cluster_id = step.get("cluster_id", -1)

                if cluster_id < 0:
                    stats["steps_unmapped"] += 1
                    continue

                if text_ref is None:
                    # Computational step with no text reference
                    # Could try to map based on position in solution
                    stats["steps_unmapped"] += 1
                    continue

                char_pos = find_text_in_problem(text_ref, problem_text)
                if char_pos < 0:
                    stats["steps_unmapped"] += 1
                    continue

                w_idx = char_to_window_idx(char_pos, problem_text, n_windows)
                if w_idx < 0 or w_idx >= n_windows:
                    stats["steps_unmapped"] += 1
                    continue

                # Update window label
                # Note: cluster IDs are 0-24, we shift by 1 to reserve 0 for NO_OP
                cluster_label = c2_labels.get(str(cluster_id), {}).get("label", f"cluster_{cluster_id}")

                window_labels[w_idx] = {
                    "c2_cluster_id": cluster_id + 1,  # Shift to reserve 0 for NO_OP
                    "c2_cluster": cluster_label,
                    "c3_operands": step.get("operands", []),
                    "step_type": step.get("step_type"),
                    "n_operands": step.get("n_operands", 0),
                    "has_dependency": step.get("has_dependency", False),
                    "reference_distance": step.get("reference_distance", "none")
                }
                stats["steps_mapped"] += 1
        else:
            stats["problems_without_steps"] += 1

        # Count statistics
        for wl in window_labels:
            stats["total_windows"] += 1
            if wl["c2_cluster_id"] == NO_OP_LABEL:
                stats["windows_no_op"] += 1
            else:
                stats["windows_with_ops"] += 1

        window_labels_all[prob_idx] = window_labels

    return window_labels_all, stats


def save_window_labels(window_labels, stats, c2_labels):
    """Save window labels to S3."""
    print("\nSaving window labels...")

    # Save window labels (too large for single JSON, save per-problem)
    output_prefix = "c2c3_training_ready/"

    # Create label mapping with NO_OP
    full_c2_mapping = {"0": {"label": "NO_OP", "dominant_step_type": None, "size": 0}}
    for cid, info in c2_labels.items():
        new_id = str(int(cid) + 1)  # Shift by 1
        full_c2_mapping[new_id] = info

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{output_prefix}c2_label_mapping.json",
        Body=json.dumps(full_c2_mapping, indent=2).encode("utf-8")
    )

    # Save window labels as single JSONL
    labels_jsonl = []
    for prob_idx, labels in window_labels.items():
        labels_jsonl.append(json.dumps({
            "problem_idx": prob_idx,
            "window_labels": labels
        }))

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{output_prefix}window_labels.jsonl",
        Body="\n".join(labels_jsonl).encode("utf-8")
    )

    # Save stats
    stats["no_op_rate"] = stats["windows_no_op"] / stats["total_windows"] if stats["total_windows"] > 0 else 0
    stats["mapping_rate"] = stats["steps_mapped"] / (stats["steps_mapped"] + stats["steps_unmapped"]) if (stats["steps_mapped"] + stats["steps_unmapped"]) > 0 else 0

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{output_prefix}window_label_stats.json",
        Body=json.dumps(stats, indent=2).encode("utf-8")
    )

    print(f"  Saved to s3://{BUCKET}/{output_prefix}")
    print(f"  Total windows: {stats['total_windows']}")
    print(f"  NO_OP windows: {stats['windows_no_op']} ({stats['no_op_rate']*100:.1f}%)")
    print(f"  Operation windows: {stats['windows_with_ops']} ({100-stats['no_op_rate']*100:.1f}%)")
    print(f"  Steps mapped: {stats['steps_mapped']} ({stats['mapping_rate']*100:.1f}%)")


def map_windows():
    """Main function to map IB clusters to windows."""
    # Load data
    ib_assignments, step_map, c2_labels = load_ib_data()
    sonnet_steps = load_sonnet_steps()
    c1_problems = load_c1_problems()

    # Build window labels
    window_labels, stats = build_window_labels(
        c1_problems, sonnet_steps, ib_assignments, step_map, c2_labels
    )

    # Save results
    save_window_labels(window_labels, stats, c2_labels)


def print_stats():
    """Print dataset statistics."""
    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_ready/window_label_stats.json")
    stats = json.loads(resp["Body"].read().decode("utf-8"))

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-windows", action="store_true", help="Map IB clusters to windows")
    parser.add_argument("--cache-features", action="store_true", help="Cache C1-A hidden states")
    parser.add_argument("--assemble", action="store_true", help="Assemble training data")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    if args.map_windows:
        map_windows()
    elif args.cache_features:
        print("Feature caching requires GPU - run on g5.xlarge")
        print("See cache_c1a_features.py for implementation")
    elif args.assemble:
        print("Assembly requires cached features - run after --cache-features")
    elif args.stats:
        print_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
