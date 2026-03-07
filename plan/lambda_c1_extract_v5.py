"""
Lambda function: C1 Training Data Extraction v5

Key fix from v4: Proper decode-to-input coordinate transformation for boundaries.

v4 aggregated attention over entire COMPUTING spans — wrong coordinate system.
v5 looks at attention DELTA at each transition point to find exactly which
input token the model shifts attention TO.

At transition step t:
    delta = attention[t, :] - attention[t-1, :]
    boundary_token = argmax(delta)

This is the principled decode-to-input projection.

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/c1_training_v5/*.json
"""

import json
import boto3
import numpy as np
from typing import Optional

s3 = boto3.client("s3")

BUCKET = "mycelium-data"

TELEGRAPH_HEAD = "L22H3"
ALARM_HEAD = "L22H4"

TELEGRAPH_THRESHOLD = 0.48
ALARM_THRESHOLD = 0.57


def extract_head_signals(problem: dict) -> Optional[dict]:
    """Extract reading signals for telegraph and alarm heads."""
    top_positions = problem.get("top_positions", [])
    n_steps = len(top_positions)

    if n_steps < 10:
        return None

    signals = {
        TELEGRAPH_HEAD: np.zeros(n_steps, dtype=np.float64),
        ALARM_HEAD: np.zeros(n_steps, dtype=np.float64),
    }

    for step, entry in enumerate(top_positions):
        for head in [TELEGRAPH_HEAD, ALARM_HEAD]:
            if head in entry:
                positions = entry[head]
                total_weight = sum(p.get("weight", 0.0) for p in positions)
                signals[head][step] = total_weight

    return signals


def fit_heartbeat_hmm(signal: np.ndarray, threshold: float) -> dict:
    """Fit two-state HMM to heartbeat signal."""
    n = len(signal)
    raw_states = (signal > threshold).astype(np.int32)

    # Smooth: remove runs shorter than 3 steps
    smoothed = raw_states.copy()
    min_run = 3

    i = 0
    while i < n:
        state = smoothed[i]
        j = i + 1
        while j < n and smoothed[j] == state:
            j += 1

        run_len = j - i
        if run_len < min_run and i > 0 and j < n:
            for k in range(i, j):
                smoothed[k] = smoothed[i - 1]
        i = j

    # Find transition points
    transitions = []
    for t in range(1, n):
        if smoothed[t] != smoothed[t - 1]:
            transitions.append(t)

    return {
        "states": smoothed,
        "transitions": transitions,
        "n_transitions": len(transitions),
    }


def get_attention_distribution(entry: dict, head: str, n_input: int) -> np.ndarray:
    """Extract attention distribution over input tokens at one decode step."""
    attn = np.zeros(n_input, dtype=np.float64)

    if head in entry:
        for p in entry[head]:
            pos = p.get("pos", 0)
            weight = p.get("weight", 0.0)
            if 0 <= pos < n_input:
                attn[pos] = weight

    return attn


def compute_boundary_probs_v5(
    heartbeat: dict,
    top_positions: list[dict],
    n_input: int,
    min_delta_threshold: float = 0.05,
) -> tuple[list[float], int, dict]:
    """
    Compute boundary probabilities using transition-point attention deltas.

    For each transition step t:
    1. Compute attention delta: attn[t] - attn[t-1]
    2. Find input token with max positive delta
    3. Mark that token as a boundary

    Returns:
        boundary_probs: per-token boundary probabilities
        n_boundaries: count of detected boundaries
        stats: extraction statistics
    """
    transitions = heartbeat["transitions"]
    n_steps = len(top_positions)

    boundary_probs = [0.0] * n_input
    boundary_positions = []

    stats = {
        "n_transitions": len(transitions),
        "n_mapped": 0,
        "n_failed": 0,
        "max_deltas": [],
    }

    for t in transitions:
        if t < 1 or t >= n_steps:
            stats["n_failed"] += 1
            continue

        # Get attention at t and t-1
        attn_before = get_attention_distribution(top_positions[t-1], TELEGRAPH_HEAD, n_input)
        attn_after = get_attention_distribution(top_positions[t], TELEGRAPH_HEAD, n_input)

        # Compute delta
        delta = attn_after - attn_before

        # Find token with max positive delta
        max_delta = delta.max()
        stats["max_deltas"].append(float(max_delta))

        if max_delta < min_delta_threshold:
            # Attention is diffuse, no clear shift
            stats["n_failed"] += 1
            continue

        boundary_token = int(np.argmax(delta))
        boundary_positions.append(boundary_token)
        stats["n_mapped"] += 1

    # Merge nearby boundaries
    unique_boundaries = _merge_nearby_boundaries(boundary_positions, gap=2)

    for b in unique_boundaries:
        if 0 <= b < n_input:
            boundary_probs[b] = 1.0

    return boundary_probs, len(unique_boundaries), stats


def _merge_nearby_boundaries(boundaries: list[int], gap: int = 2) -> list[int]:
    """Merge boundaries within `gap` positions."""
    if not boundaries:
        return []

    sorted_b = sorted(set(boundaries))
    merged = [sorted_b[0]]

    for b in sorted_b[1:]:
        if b - merged[-1] > gap:
            merged.append(b)

    return merged


def detect_co_transitions(
    heartbeat_transitions: list[int],
    heartbeat_states: np.ndarray,
    alarm_events: list[int],
    window: int = 2,
    min_gap: int = 5,
) -> list[dict]:
    """Detect co-transitions: telegraph change + alarm fires."""
    alarm_set = set(alarm_events)
    co_transitions = []
    last_step = -min_gap - 1

    for t in heartbeat_transitions:
        if t - last_step < min_gap:
            continue

        has_alarm = False
        for offset in range(-window, window + 1):
            if (t + offset) in alarm_set:
                has_alarm = True
                break

        if has_alarm:
            state_after = int(heartbeat_states[t]) if t < len(heartbeat_states) else 0
            co_transitions.append({"step": t, "state_after": state_after})
            last_step = t

    return co_transitions


def compute_co_transition_stats(co_transitions: list[dict], n_steps: int) -> dict:
    """Compute co-transition statistics."""
    n = len(co_transitions)

    if n == 0:
        return {
            "n_co_transitions": 0,
            "reading_ratio": 0.5,
            "mean_spacing": float(n_steps),
            "burstiness": 0.0,
        }

    n_reading = sum(1 for c in co_transitions if c["state_after"] == 1)
    reading_ratio = n_reading / n

    if n == 1:
        return {
            "n_co_transitions": 1,
            "reading_ratio": float(reading_ratio),
            "mean_spacing": float(n_steps),
            "burstiness": 0.0,
        }

    steps = [c["step"] for c in co_transitions]
    spacings = [steps[i+1] - steps[i] for i in range(len(steps) - 1)]

    mean_spacing = np.mean(spacings)
    var_spacing = np.var(spacings)
    burstiness = var_spacing / mean_spacing if mean_spacing > 0 else 0.0

    return {
        "n_co_transitions": n,
        "reading_ratio": float(reading_ratio),
        "mean_spacing": float(mean_spacing),
        "burstiness": float(burstiness),
    }


def detect_alarm_events(signal: np.ndarray, threshold: float) -> list[int]:
    """Detect alarm events (signal dips below threshold)."""
    return np.where(signal < threshold)[0].tolist()


def _cluster_events(events: list[int], gap: int = 3) -> list[tuple[int, int]]:
    """Cluster nearby events into spans."""
    if not events:
        return []

    spans = []
    start = events[0]
    end = events[0]

    for t in events[1:]:
        if t <= end + gap:
            end = t
        else:
            spans.append((start, end))
            start = t
            end = t
    spans.append((start, end))

    return spans


def compute_bp_depth(n_alarm_spans: int) -> int:
    """Compute BP depth from alarm span count."""
    if n_alarm_spans <= 15:
        return 1
    elif n_alarm_spans <= 24:
        return 2
    else:
        return 3


def generate_c1_record(problem: dict) -> Optional[dict]:
    """Generate one C1 training record in v5 schema."""
    top_positions = problem.get("top_positions", [])
    problem_text = problem.get("problem_text", problem.get("question", ""))
    problem_id = str(problem.get("problem_idx", problem.get("problem_id", "unknown")))

    if not top_positions or not problem_text:
        return None

    n_gen = len(top_positions)

    # Determine n_input
    max_pos = 0
    for entry in top_positions:
        for head_key, positions in entry.items():
            for p in positions:
                max_pos = max(max_pos, p.get("pos", 0))
    n_input = max_pos + 1

    if n_input < 5:
        return None

    # Extract signals
    signals = extract_head_signals(problem)
    if signals is None:
        return None

    telegraph_signal = signals[TELEGRAPH_HEAD]
    alarm_signal = signals[ALARM_HEAD]

    # Heartbeat for transitions
    heartbeat = fit_heartbeat_hmm(telegraph_signal, TELEGRAPH_THRESHOLD)

    # v5: Boundary probs using transition-point delta
    boundary_probs, n_boundaries, boundary_stats = compute_boundary_probs_v5(
        heartbeat, top_positions, n_input
    )

    # Alarm events
    alarm_events = detect_alarm_events(alarm_signal, ALARM_THRESHOLD)

    # Co-transitions
    co_transitions = detect_co_transitions(
        heartbeat["transitions"],
        heartbeat["states"],
        alarm_events,
        window=2,
        min_gap=5,
    )
    co_stats = compute_co_transition_stats(co_transitions, n_gen)

    # BP depth
    n_alarm_spans = len(_cluster_events(alarm_events, gap=3))
    bp_depth = compute_bp_depth(n_alarm_spans)

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "n_input_tokens": n_input,

        # Head 1: Boundary detection (per-token) - v5 fix
        "boundary_probs": boundary_probs,
        "n_boundaries": n_boundaries,

        # Boundary extraction stats (for analysis)
        "boundary_extraction": {
            "n_transitions": boundary_stats["n_transitions"],
            "n_mapped": boundary_stats["n_mapped"],
            "n_failed": boundary_stats["n_failed"],
            "mapping_rate": boundary_stats["n_mapped"] / max(1, boundary_stats["n_transitions"]),
        },

        # Head 2: Co-transition statistics
        "n_co_transitions": co_stats["n_co_transitions"],
        "reading_ratio": co_stats["reading_ratio"],
        "mean_spacing": co_stats["mean_spacing"],
        "burstiness": co_stats["burstiness"],

        # Head 3: BP depth
        "bp_depth": bp_depth,
        "n_alarm_spans": n_alarm_spans,
    }


def lambda_handler(event, context):
    """Process one IAF chunk, generate C1 v5 training data."""
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c1_training_v5/")

    # Download IAF chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    # Process
    records = []
    stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "co_transition_counts": [],
        "boundary_counts": [],
        "mapping_rates": [],
        "bp_depth_dist": {1: 0, 2: 0, 3: 0},
    }

    for idx, problem in enumerate(problems):
        stats["total"] += 1

        if "problem_idx" not in problem and "problem_id" not in problem:
            problem["problem_idx"] = f"{chunk_key}_{idx}"

        record = generate_c1_record(problem)

        if record is None:
            stats["skipped"] += 1
            continue

        records.append(record)
        stats["generated"] += 1

        stats["co_transition_counts"].append(record["n_co_transitions"])
        stats["boundary_counts"].append(record["n_boundaries"])
        stats["mapping_rates"].append(record["boundary_extraction"]["mapping_rate"])
        stats["bp_depth_dist"][record["bp_depth"]] += 1

    # Compute summary stats
    if stats["generated"] > 0:
        ct = stats["co_transition_counts"]
        stats["co_transition_mean"] = float(np.mean(ct))
        stats["co_transition_median"] = float(np.median(ct))

        bc = stats["boundary_counts"]
        stats["boundary_mean"] = float(np.mean(bc))

        mr = stats["mapping_rates"]
        stats["mapping_rate_mean"] = float(np.mean(mr))
        stats["mapping_rate_min"] = float(np.min(mr))

        total_tokens = sum(r["n_input_tokens"] for r in records)
        total_boundaries = sum(bc)
        stats["boundary_density"] = total_boundaries / max(1, total_tokens)

    # Clean up large lists
    del stats["co_transition_counts"]
    del stats["boundary_counts"]
    del stats["mapping_rates"]

    # Upload records
    chunk_name = chunk_key.split("/")[-1]
    output_key = f"{output_prefix}{chunk_name}"

    output = {
        "chunk_key": chunk_key,
        "stats": stats,
        "records": records,
    }

    print(
        f"Uploading s3://{bucket}/{output_key} | "
        f"{stats['generated']} records | "
        f"boundaries: {stats.get('boundary_mean', 0):.1f} | "
        f"mapping: {stats.get('mapping_rate_mean', 0):.1%}"
    )

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(output, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    # Stats for reduce
    stats_key = f"{output_prefix}stats/{chunk_name}"
    s3.put_object(
        Bucket=bucket,
        Key=stats_key,
        Body=json.dumps(stats, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "chunk_key": chunk_key,
        "output_key": output_key,
        "stats": stats,
    }
