"""
Lambda function: C0 Hint Vector Extraction (Map phase)

Extracts 5-dimensional per-token hint vectors from IAF attention data:
  1. Entropy - attention entropy weighted by how much each token is attended
  2. Received - total attention received by each token (normalized)
  3. Tension - computational relatedness vs text distance
  4. Telegraph - L22H3 reading signal mapped to input tokens
  5. Connectivity - effective attention breadth when attending to each token

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/c0_training_data/hint_vectors/{chunk_id}.jsonl

Memory: 3GB Lambda
Pure Python implementation (no numpy)
"""

import json
import math
import boto3

s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Key heads for entropy/tension/connectivity (computation-focused)
KEY_HEADS = ["L22H3", "L22H4", "L23H11", "L23H23", "L24H6"]

# All 10 heads for received attention
ALL_HEADS = ["L5H19", "L14H0", "L22H3", "L22H4", "L23H11", "L23H23", "L24H4", "L24H6", "L24H16", "L25H1"]


def compute_entropy_per_token(top_positions, n_input, heads):
    """
    Compute entropy per input token, weighted by attention received.
    """
    entropy_map = [0.0] * n_input
    weight_map = [0.0] * n_input
    n_steps = len(top_positions)

    for step_data in top_positions:
        for head in heads:
            if head not in step_data:
                continue

            positions = step_data[head]
            total = sum(p.get("weight", 0.0) for p in positions if p.get("pos", 0) < n_input)
            if total < 1e-10:
                continue

            # Build normalized attention distribution
            attn = {}
            for p in positions:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                if pos < n_input:
                    attn[pos] = weight / total

            # Shannon entropy
            H = 0.0
            for prob in attn.values():
                if prob > 1e-10:
                    H -= prob * math.log(prob)

            # Distribute entropy to tokens by attention
            for pos, prob in attn.items():
                entropy_map[pos] += H * prob
                weight_map[pos] += prob

    # Normalize
    result = [entropy_map[i] / (weight_map[i] + 1e-10) for i in range(n_input)]
    return result


def compute_received_attention(top_positions, n_input, heads):
    """
    Total attention each input token receives across all heads and steps.
    """
    received = [0.0] * n_input

    for step_data in top_positions:
        for head in heads:
            if head not in step_data:
                continue
            for p in step_data[head]:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                if pos < n_input:
                    received[pos] += weight

    # Normalize to [0, 1]
    max_val = max(received) if received else 0
    if max_val > 1e-10:
        received = [r / max_val for r in received]

    return received


def compute_tension(top_positions, n_input, heads):
    """
    Tension measures gap between computational relatedness and text distance.
    Memory-efficient: compute per-step contribution without storing full matrix.
    """
    tension_per_token = [0.0] * n_input

    for step_data in top_positions:
        for head in heads:
            if head not in step_data:
                continue

            positions = step_data[head]
            total = sum(p.get("weight", 0.0) for p in positions if p.get("pos", 0) < n_input)
            if total < 1e-10:
                continue

            # Build normalized attention
            attn = {}
            for p in positions:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                if pos < n_input:
                    attn[pos] = weight / total

            # For each token i, tension contribution = sum_j(attn_i * attn_j * |i-j|)
            # = attn_i * weighted_avg_distance
            for i, attn_i in attn.items():
                expected_dist = sum(attn_j * abs(i - j) for j, attn_j in attn.items())
                tension_per_token[i] += attn_i * expected_dist

    # Normalize to [0, 1]
    max_val = max(tension_per_token) if tension_per_token else 0
    if max_val > 1e-10:
        tension_per_token = [t / max_val for t in tension_per_token]

    return tension_per_token


def compute_telegraph(top_positions, n_input):
    """
    Map L22H3's telegraph signal to input token space.
    """
    telegraph_map = [0.0] * n_input
    weight_map = [0.0] * n_input

    head = "L22H3"

    for step_data in top_positions:
        if head not in step_data:
            continue

        positions = step_data[head]

        # Reading signal = sum of attention to input tokens
        reading_signal = sum(p.get("weight", 0.0) for p in positions if p.get("pos", 0) < n_input)

        # Distribute to attended tokens
        for p in positions:
            pos = p.get("pos", 0)
            weight = p.get("weight", 0.0)
            if pos < n_input:
                telegraph_map[pos] += reading_signal * weight
                weight_map[pos] += weight

    result = [telegraph_map[i] / (weight_map[i] + 1e-10) for i in range(n_input)]

    # Normalize to [0, 1]
    max_val = max(result) if result else 0
    if max_val > 1e-10:
        result = [r / max_val for r in result]

    return result


def compute_connectivity(top_positions, n_input, heads):
    """
    Effective attention breadth when attending to each token.
    """
    connectivity_map = [0.0] * n_input
    weight_map = [0.0] * n_input

    for step_data in top_positions:
        for head in heads:
            if head not in step_data:
                continue

            positions = step_data[head]
            total = sum(p.get("weight", 0.0) for p in positions if p.get("pos", 0) < n_input)
            if total < 1e-10:
                continue

            # Build normalized attention
            attn = {}
            for p in positions:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                if pos < n_input:
                    attn[pos] = weight / total

            # Entropy of attention distribution
            H = 0.0
            for prob in attn.values():
                if prob > 1e-10:
                    H -= prob * math.log(prob)

            # Effective number of tokens
            effective_breadth = math.exp(H)

            # Distribute to tokens by attention
            for pos, prob in attn.items():
                connectivity_map[pos] += effective_breadth * prob
                weight_map[pos] += prob

    result = [connectivity_map[i] / (weight_map[i] + 1e-10) for i in range(n_input)]

    # Normalize to [0, 1]
    max_val = max(result) if result else 0
    if max_val > 1e-10:
        result = [r / max_val for r in result]

    return result


def extract_hint_vectors(record):
    """Extract all 5 hint features for a single problem."""
    top_positions = record.get("top_positions", [])
    n_input = record.get("input_len", 0)

    if not top_positions or n_input == 0:
        return None

    # Extract all 5 features
    entropy = compute_entropy_per_token(top_positions, n_input, KEY_HEADS)
    received = compute_received_attention(top_positions, n_input, ALL_HEADS)
    tension = compute_tension(top_positions, n_input, KEY_HEADS)
    telegraph = compute_telegraph(top_positions, n_input)
    connectivity = compute_connectivity(top_positions, n_input, KEY_HEADS)

    # Summary statistics
    mean_entropy = sum(entropy) / len(entropy) if entropy else 0
    mean_tension = sum(tension) / len(tension) if tension else 0
    mean_received = sum(received) / len(received) if received else 0

    return {
        "problem_id": str(record.get("problem_idx", "")),
        "problem_text": record.get("problem_text", ""),
        "n_input_tokens": n_input,
        "n_decode_steps": len(top_positions),
        "hint_vectors": {
            "entropy": entropy,
            "received": received,
            "tension": tension,
            "telegraph": telegraph,
            "connectivity": connectivity,
        },
        "summary_stats": {
            "mean_entropy": mean_entropy,
            "mean_tension": mean_tension,
            "mean_received": mean_received,
        }
    }


def lambda_handler(event, context):
    """Process one IAF chunk and extract hint vectors for all problems."""
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c0_training_data/hint_vectors/")

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
        "total_tension": 0.0,
        "total_received": 0.0,
    }

    for record in chunk_data:
        stats["total"] += 1

        hint_data = extract_hint_vectors(record)

        if hint_data is None:
            stats["skipped"] += 1
            continue

        results.append(hint_data)
        stats["processed"] += 1

        summary = hint_data["summary_stats"]
        stats["total_entropy"] += summary["mean_entropy"]
        stats["total_tension"] += summary["mean_tension"]
        stats["total_received"] += summary["mean_received"]

    # Compute averages
    if stats["processed"] > 0:
        stats["avg_entropy"] = stats["total_entropy"] / stats["processed"]
        stats["avg_tension"] = stats["total_tension"] / stats["processed"]
        stats["avg_received"] = stats["total_received"] / stats["processed"]

    # Save results as JSONL
    chunk_name = chunk_key.split("/")[-1].replace(".json", "_hints.jsonl")
    output_key = f"{output_prefix}{chunk_name}"

    lines = [json.dumps(r) for r in results]
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body="\n".join(lines).encode("utf-8"),
        ContentType="application/json",
    )

    # Save chunk stats
    stats_key = f"{output_prefix}stats/{chunk_name.replace('.jsonl', '_stats.json')}"
    s3.put_object(
        Bucket=bucket,
        Key=stats_key,
        Body=json.dumps(stats, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "chunk": chunk_key,
        "output": output_key,
        "stats": stats,
    }
