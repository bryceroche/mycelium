"""
Lambda function: Per-Token Entropy Map Extraction

Computes per-token entropy from teacher attention patterns.
For each input token, measures how uncertain the teacher is when attending to that region.

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/analysis/c0_entropy/maps/{chunk_id}.json

Each output record:
{
    "problem_id": str,
    "n_input_tokens": int,
    "entropy_map_weighted": [float],     # Method 1: attention-weighted entropy per token
    "entropy_map_received": [float],     # Method 2: entropy when token is heavily attended
    "mean_entropy": float,
    "max_entropy": float,
    "entropy_std": float,
    "high_entropy_regions": [(start, end, mean_entropy)],
    "high_entropy_fraction": float,
}

Memory: 3GB Lambda
"""

import json
import math
import boto3
from collections import defaultdict

s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Key heads for entropy computation
# Telegraph heads (track reading vs computing transitions)
# Alarm heads (attend during key operations)
ENTROPY_HEADS = ["L22H3", "L22H4", "L23H11", "L23H23", "L24H6"]

# Smoothing window for high-entropy region detection
SMOOTH_WINDOW = 5
HIGH_ENTROPY_PERCENTILE = 75


def compute_shannon_entropy(weights):
    """Compute Shannon entropy of a probability distribution."""
    entropy = 0.0
    for w in weights:
        if w > 1e-10:
            entropy -= w * math.log(w + 1e-10)
    return entropy


def extract_entropy_maps(record):
    """
    Extract per-token entropy maps from a single IAF record.

    Method 1 (weighted): For each input token, compute weighted average of
    attention entropy at each step, weighted by how much that step attends to the token.

    Method 2 (received): For each input token, compute average entropy of steps
    where the token receives significant attention (>threshold).
    """
    top_positions = record.get("top_positions", [])
    if not top_positions:
        return None

    # Infer n_input_tokens from max position seen
    n_input = 0
    for step_data in top_positions:
        for head_key, positions in step_data.items():
            if head_key not in ENTROPY_HEADS:
                continue
            for p in positions:
                pos = p.get("pos", 0)
                n_input = max(n_input, pos + 1)

    if n_input == 0:
        return None

    n_steps = len(top_positions)

    # Method 1: Attention-weighted entropy per token
    # For each token, accumulate (attention_weight * step_entropy)
    weighted_entropy_sum = [0.0] * n_input
    attention_sum = [0.0] * n_input

    # Method 2: Received entropy (entropy when token is attended)
    received_entropy_sum = [0.0] * n_input
    received_count = [0] * n_input
    RECEIVE_THRESHOLD = 0.1  # Token must receive >10% attention from a head

    for step_idx, step_data in enumerate(top_positions):
        for head_key in ENTROPY_HEADS:
            if head_key not in step_data:
                continue

            positions = step_data[head_key]

            # Build attention distribution for this head at this step
            attn_dict = {}
            total_weight = 0.0
            for p in positions:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                if pos < n_input:
                    attn_dict[pos] = attn_dict.get(pos, 0.0) + weight
                    total_weight += weight

            if total_weight < 1e-10:
                continue

            # Normalize to get probability distribution
            for pos in attn_dict:
                attn_dict[pos] /= total_weight

            # Compute entropy of this attention distribution
            attn_values = list(attn_dict.values())
            step_entropy = compute_shannon_entropy(attn_values)

            # Method 1: Weight by attention to each token
            for pos, attn in attn_dict.items():
                weighted_entropy_sum[pos] += attn * step_entropy
                attention_sum[pos] += attn

            # Method 2: Record entropy for heavily-attended tokens
            for pos, attn in attn_dict.items():
                if attn > RECEIVE_THRESHOLD:
                    received_entropy_sum[pos] += step_entropy
                    received_count[pos] += 1

    # Compute final entropy maps
    entropy_map_weighted = []
    entropy_map_received = []

    for i in range(n_input):
        # Method 1
        if attention_sum[i] > 1e-10:
            entropy_map_weighted.append(weighted_entropy_sum[i] / attention_sum[i])
        else:
            entropy_map_weighted.append(0.0)

        # Method 2
        if received_count[i] > 0:
            entropy_map_received.append(received_entropy_sum[i] / received_count[i])
        else:
            entropy_map_received.append(0.0)

    # Compute summary statistics
    if entropy_map_weighted:
        mean_entropy = sum(entropy_map_weighted) / len(entropy_map_weighted)
        max_entropy = max(entropy_map_weighted)
        variance = sum((e - mean_entropy) ** 2 for e in entropy_map_weighted) / len(entropy_map_weighted)
        entropy_std = math.sqrt(variance)
    else:
        mean_entropy = max_entropy = entropy_std = 0.0

    # Detect high-entropy regions
    high_entropy_regions = detect_high_entropy_regions(
        entropy_map_weighted, SMOOTH_WINDOW, HIGH_ENTROPY_PERCENTILE
    )

    # Compute high-entropy fraction
    if entropy_map_weighted:
        threshold = sorted(entropy_map_weighted)[int(len(entropy_map_weighted) * HIGH_ENTROPY_PERCENTILE / 100)]
        high_count = sum(1 for e in entropy_map_weighted if e > threshold)
        high_entropy_fraction = high_count / len(entropy_map_weighted)
    else:
        high_entropy_fraction = 0.0

    return {
        "n_input_tokens": n_input,
        "n_decode_steps": n_steps,
        "entropy_map_weighted": entropy_map_weighted,
        "entropy_map_received": entropy_map_received,
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "entropy_std": entropy_std,
        "high_entropy_regions": high_entropy_regions,
        "high_entropy_fraction": high_entropy_fraction,
    }


def detect_high_entropy_regions(entropy_map, smooth_window, percentile):
    """
    Detect contiguous spans of high-entropy tokens.

    1. Smooth the entropy map with moving average
    2. Threshold at percentile within this problem
    3. Find contiguous spans above threshold
    """
    if len(entropy_map) < smooth_window:
        return []

    # Smooth with moving average
    smoothed = []
    half_w = smooth_window // 2
    for i in range(len(entropy_map)):
        start = max(0, i - half_w)
        end = min(len(entropy_map), i + half_w + 1)
        smoothed.append(sum(entropy_map[start:end]) / (end - start))

    # Threshold at percentile
    sorted_vals = sorted(smoothed)
    threshold_idx = int(len(sorted_vals) * percentile / 100)
    threshold = sorted_vals[min(threshold_idx, len(sorted_vals) - 1)]

    # Find contiguous spans above threshold
    regions = []
    in_region = False
    region_start = 0

    for i, val in enumerate(smoothed):
        if val > threshold:
            if not in_region:
                in_region = True
                region_start = i
        else:
            if in_region:
                region_end = i
                region_mean = sum(entropy_map[region_start:region_end]) / (region_end - region_start)
                regions.append((region_start, region_end, region_mean))
                in_region = False

    # Handle region that extends to end
    if in_region:
        region_end = len(smoothed)
        region_mean = sum(entropy_map[region_start:region_end]) / (region_end - region_start)
        regions.append((region_start, region_end, region_mean))

    return regions


def lambda_handler(event, context):
    """
    Process one IAF chunk and extract entropy maps for all problems.
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "analysis/c0_entropy/maps/")

    # Load chunk
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))

    # Process each problem
    results = []
    stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "total_entropy": 0.0,
        "total_high_regions": 0,
    }

    for record in chunk_data:
        stats["total"] += 1

        # Get problem ID
        problem_id = record.get("problem_id", record.get("problem_idx", f"unknown_{stats['total']}"))

        # Extract entropy maps
        entropy_data = extract_entropy_maps(record)

        if entropy_data is None:
            stats["skipped"] += 1
            continue

        entropy_data["problem_id"] = problem_id
        results.append(entropy_data)

        stats["processed"] += 1
        stats["total_entropy"] += entropy_data["mean_entropy"]
        stats["total_high_regions"] += len(entropy_data["high_entropy_regions"])

    # Compute averages
    if stats["processed"] > 0:
        stats["avg_entropy"] = stats["total_entropy"] / stats["processed"]
        stats["avg_high_regions"] = stats["total_high_regions"] / stats["processed"]
    else:
        stats["avg_entropy"] = 0.0
        stats["avg_high_regions"] = 0.0

    # Save results
    chunk_name = chunk_key.split("/")[-1].replace(".json", "_entropy.json")
    output_key = f"{output_prefix}{chunk_name}"

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(results).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "chunk": chunk_key,
        "output": output_key,
        "stats": stats,
    }
