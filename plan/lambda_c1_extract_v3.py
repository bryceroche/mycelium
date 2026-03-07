"""
Lambda function: C1 Training Data Extraction (Validated Heads)

Extracts C1 multi-task training data using validated attention heads:
- L22H3: Primary heartbeat (47% duty cycle, 0.08/0.87 separation)
- L23H11: Secondary heartbeat for crossing detection
- L22H4: Rare event alarm (major phase boundaries)

Outputs per problem:
1. Heartbeat signal and HMM state sequence
2. BIO labels for input token spans
3. Knot crossings from L22H3 × L23H11 intersections
4. BP depth from recomputed unknotting number
5. Alarm events from L22H4

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/c1_training_v3/*.json

Memory: 3GB Lambda
"""

import json
import boto3
import numpy as np
from collections import defaultdict
from typing import Optional


s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Validated heads from analysis
HEARTBEAT_HEAD = "L22H3"      # Primary telegraph (47% duty cycle)
SECONDARY_HEAD = "L23H11"     # Secondary telegraph for crossings
ALARM_HEAD = "L22H4"          # Rare event detector (77% high)

# Otsu thresholds from analysis
HEARTBEAT_THRESHOLD = 0.48    # Midpoint of 0.08/0.87
SECONDARY_THRESHOLD = 0.45    # Midpoint of 0.12/0.79
ALARM_THRESHOLD = 0.57        # Midpoint for L22H4 (lower = alarm firing)

# Knot signature mapping (31 classes)
# Classes 0-14: 0-3 crossings (exhaustive)
# Classes 15-30: 4 crossings (all 16 patterns)
KNOT_CLASSES = {
    # 0 crossings
    "": 0,
    # 1 crossing
    "U": 1, "O": 2,
    # 2 crossings
    "UU": 3, "UO": 4, "OU": 5, "OO": 6,
    # 3 crossings
    "UUU": 7, "UUO": 8, "UOU": 9, "UOO": 10,
    "OUU": 11, "OUO": 12, "OOU": 13, "OOO": 14,
    # 4 crossings (all 16 patterns)
    "UUUU": 15, "UUUO": 16, "UUOU": 17, "UUOO": 18,
    "UOUU": 19, "UOUO": 20, "UOOU": 21, "UOOO": 22,
    "OUUU": 23, "OUUO": 24, "OUOU": 25, "OUOO": 26,
    "OOUU": 27, "OOUO": 28, "OOOU": 29, "OOOO": 30,
}
N_KNOT_CLASSES = 31  # 0-30 are valid classes, no RARE needed


def extract_head_signals(problem: dict) -> Optional[dict]:
    """
    Extract reading signals for validated heads.

    Returns dict with signal arrays for each head.
    """
    top_positions = problem.get("top_positions", [])
    n_steps = len(top_positions)

    if n_steps < 10:
        return None

    signals = {
        HEARTBEAT_HEAD: np.zeros(n_steps, dtype=np.float64),
        SECONDARY_HEAD: np.zeros(n_steps, dtype=np.float64),
        ALARM_HEAD: np.zeros(n_steps, dtype=np.float64),
    }

    for step, entry in enumerate(top_positions):
        for head in [HEARTBEAT_HEAD, SECONDARY_HEAD, ALARM_HEAD]:
            if head in entry:
                positions = entry[head]
                total_weight = sum(p.get("weight", 0.0) for p in positions)
                signals[head][step] = total_weight

    return signals


def fit_heartbeat_hmm(signal: np.ndarray, threshold: float) -> dict:
    """
    Fit two-state HMM to heartbeat signal.

    Simple approach: Otsu threshold + smoothing to remove single-step noise.
    States: 0 = COMPUTING (low), 1 = READING (high)
    """
    n = len(signal)

    # Initial binary classification
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
            # Short run - absorb into surrounding
            for k in range(i, j):
                smoothed[k] = smoothed[i - 1]
        i = j

    # Find transition points
    transitions = []
    for t in range(1, n):
        if smoothed[t] != smoothed[t - 1]:
            transitions.append(t)

    # Compute state statistics
    computing_steps = np.sum(smoothed == 0)
    reading_steps = np.sum(smoothed == 1)

    return {
        "states": smoothed,
        "transitions": transitions,
        "n_transitions": len(transitions),
        "computing_ratio": float(computing_steps / n),
        "reading_ratio": float(reading_steps / n),
    }


def project_to_bio_labels(
    heartbeat: dict,
    top_positions: list[dict],
    n_input: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Project heartbeat COMPUTING phases onto input tokens as BIO labels.

    For each COMPUTING span, find which input tokens are attended.
    """
    states = heartbeat["states"]
    n_steps = len(states)

    bio = ["O"] * n_input
    boundaries = []

    # Find COMPUTING spans (state == 0)
    i = 0
    while i < n_steps:
        if states[i] == 0:  # COMPUTING
            start = i
            while i < n_steps and states[i] == 0:
                i += 1
            end = i

            # Aggregate attention over input during this span
            span_attn = np.zeros(n_input, dtype=np.float64)
            count = 0

            for t in range(start, min(end, len(top_positions))):
                entry = top_positions[t]
                if HEARTBEAT_HEAD in entry:
                    for p in entry[HEARTBEAT_HEAD]:
                        pos = p.get("pos", 0)
                        weight = p.get("weight", 0.0)
                        if 0 <= pos < n_input:
                            span_attn[pos] += weight
                    count += 1

            if count > 0:
                span_attn /= count

            # Find dominant region (top 70th percentile)
            if span_attn.sum() > 1e-6:
                region = _find_dominant_region(span_attn)
                boundaries.append(region)
        else:
            i += 1

    # Merge overlapping spans
    merged = _merge_spans(boundaries)

    # Assign BIO labels
    for start, end in merged:
        for j in range(start, min(end + 1, n_input)):
            if j == start:
                bio[j] = "B-COMP"
            else:
                bio[j] = "I-COMP"

    return bio, merged


def _find_dominant_region(attn: np.ndarray) -> tuple[int, int]:
    """Find contiguous region with highest attention."""
    threshold = np.percentile(attn, 70)
    if threshold < 1e-6:
        peak = int(np.argmax(attn))
        return (max(0, peak - 2), min(len(attn) - 1, peak + 2))

    above = (attn >= threshold).astype(int)

    best_start, best_end, best_len = 0, 0, 0
    curr_start = None

    for i in range(len(above)):
        if above[i]:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                length = i - curr_start
                if length > best_len:
                    best_start, best_end, best_len = curr_start, i - 1, length
                curr_start = None

    if curr_start is not None:
        length = len(above) - curr_start
        if length > best_len:
            best_start, best_end = curr_start, len(above) - 1

    return (best_start, best_end)


def _merge_spans(spans: list[tuple[int, int]], gap: int = 2) -> list[tuple[int, int]]:
    """Merge overlapping or nearby spans."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _get_dominant_input_region(
    top_positions: list[dict],
    step: int,
    head: str,
    n_input: int,
) -> tuple[int, int]:
    """
    Get dominant input region from head's attention at a step.
    Uses weighted average ± 1 std dev for focused region.
    """
    if step >= len(top_positions):
        return (0, n_input // 2)

    entry = top_positions[step]
    if head not in entry:
        return (0, n_input // 2)

    positions = entry[head]
    if not positions:
        return (0, n_input // 2)

    # Get positions and weights
    pos_list = []
    weight_list = []
    for p in positions:
        pos = p.get("pos", 0)
        w = p.get("weight", 0.0)
        if 0 <= pos < n_input and w > 0:
            pos_list.append(pos)
            weight_list.append(w)

    if not pos_list:
        return (0, n_input // 2)

    pos_arr = np.array(pos_list)
    weight_arr = np.array(weight_list)
    weight_arr = weight_arr / weight_arr.sum()  # Normalize

    # Weighted mean and std
    mean = np.sum(pos_arr * weight_arr)
    variance = np.sum(weight_arr * (pos_arr - mean) ** 2)
    std = max(np.sqrt(variance), 3)  # Min std of 3

    start = int(max(0, mean - std))
    end = int(min(n_input - 1, mean + std))

    return (start, end)


def detect_crossings(
    signal1: np.ndarray,
    signal2: np.ndarray,
    top_positions: list[dict],
    n_input: int,
    min_gap: int = 5,
    min_divergence: float = 0.1,
) -> list[dict]:
    """
    Detect crossings between two reading signals.

    A crossing occurs when the relative ordering changes:
    signal1 > signal2 → signal1 < signal2 (or vice versa)

    Args:
        min_gap: Minimum steps between crossings (debouncing)
        min_divergence: Minimum signal divergence after crossing
    """
    n = min(len(signal1), len(signal2))
    crossings = []
    last_crossing_step = -min_gap - 1  # Allow first crossing

    for t in range(n - 1):
        # Debounce: skip if too close to last crossing
        if t - last_crossing_step < min_gap:
            continue

        # Check for sign change in (signal1 - signal2)
        diff_t = signal1[t] - signal2[t]
        diff_t1 = signal1[t + 1] - signal2[t + 1]

        # Skip if either is near zero (ambiguous)
        if abs(diff_t) < 0.05 or abs(diff_t1) < 0.05:
            continue

        if (diff_t > 0) != (diff_t1 > 0):
            # Check for minimum divergence after crossing
            # Look ahead a few steps to ensure signals actually diverged
            max_divergence = abs(diff_t1)
            for lookahead in range(2, min(6, n - t)):
                future_diff = signal1[t + lookahead] - signal2[t + lookahead]
                if (diff_t1 > 0) == (future_diff > 0):  # Same direction
                    max_divergence = max(max_divergence, abs(future_diff))

            if max_divergence < min_divergence:
                continue  # Signals didn't really diverge

            # Crossing detected
            crossing_type = "over" if diff_t1 > 0 else "under"

            # Get focused input region using dominant positions
            input_region = _get_dominant_input_region(
                top_positions, t, HEARTBEAT_HEAD, n_input
            )

            crossings.append({
                "step": t,
                "input_region": list(input_region),
                "type": crossing_type,
                "signal1_val": float(signal1[t]),
                "signal2_val": float(signal2[t]),
                "divergence": float(max_divergence),
            })
            last_crossing_step = t

    return crossings


def _overlap_fraction(r1: list[int], r2: list[int]) -> float:
    """
    Compute overlap fraction between two regions.
    Returns overlap / min(len(r1), len(r2))
    """
    overlap_start = max(r1[0], r2[0])
    overlap_end = min(r1[1], r2[1])
    overlap_len = max(0, overlap_end - overlap_start + 1)

    len1 = r1[1] - r1[0] + 1
    len2 = r2[1] - r2[0] + 1
    min_len = max(1, min(len1, len2))

    return overlap_len / min_len


def compute_knot_signature(crossings: list[dict], min_overlap: float = 0.3) -> str:
    """
    Compute knot signature from crossing sequence.

    Only count crossings where input regions meaningfully overlap
    (overlap fraction > min_overlap with at least one other crossing).
    """
    if not crossings:
        return ""

    if len(crossings) == 1:
        c = crossings[0]
        return "O" if c["type"] == "over" else "U"

    # Filter to crossings with meaningful overlap
    valid_crossings = []
    for i, c in enumerate(crossings):
        r1 = c["input_region"]
        for j, other in enumerate(crossings):
            if i != j:
                r2 = other["input_region"]
                if _overlap_fraction(r1, r2) >= min_overlap:
                    valid_crossings.append(c)
                    break

    # Build signature
    sig = "".join("O" if c["type"] == "over" else "U" for c in valid_crossings)
    return sig


def compute_bp_depth_from_major_boundaries(n_major: int) -> int:
    """
    Compute BP depth from major boundary count.

    Major boundaries are where heartbeat transitions coincide with alarm events,
    indicating significant phase transitions.

    Thresholds based on terciles of the distribution (mean ~11, median ~10):
    - 1: Simple problems (≤7 major boundaries) ~33%
    - 2: Moderate (8-12 major boundaries) ~33%
    - 3: Complex (>12 major boundaries) ~33%
    """
    if n_major <= 7:
        return 1
    elif n_major <= 12:
        return 2
    else:
        return 3


def detect_alarm_events(signal: np.ndarray, threshold: float) -> dict:
    """
    Detect alarm events from L22H4 (rare dips below threshold).
    """
    n = len(signal)

    # Find steps where alarm fires (signal below threshold)
    alarm_steps = np.where(signal < threshold)[0].tolist()

    # Cluster into spans
    spans = []
    if alarm_steps:
        start = alarm_steps[0]
        end = alarm_steps[0]

        for t in alarm_steps[1:]:
            if t <= end + 2:  # Allow gap of 2
                end = t
            else:
                spans.append((start, end))
                start = t
                end = t
        spans.append((start, end))

    return {
        "events": alarm_steps,
        "spans": spans,
        "n_events": len(alarm_steps),
        "n_spans": len(spans),
    }


def find_major_boundaries(
    heartbeat_transitions: list[int],
    alarm_events: list[int],
    window: int = 2,
) -> list[int]:
    """
    Find major boundaries where heartbeat and alarm co-transition.
    """
    major = []
    alarm_set = set(alarm_events)

    for t in heartbeat_transitions:
        # Check if alarm fires within ±window steps
        for offset in range(-window, window + 1):
            if (t + offset) in alarm_set:
                major.append(t)
                break

    return major


def generate_c1_record(problem: dict) -> Optional[dict]:
    """
    Generate one C1 training record from an IAF problem.
    """
    top_positions = problem.get("top_positions", [])
    problem_text = problem.get("problem_text", problem.get("question", ""))
    problem_id = str(problem.get("problem_idx", problem.get("problem_id", "unknown")))

    if not top_positions or not problem_text:
        return None

    n_gen = len(top_positions)

    # Determine n_input from data
    max_pos = 0
    for entry in top_positions:
        for head_key, positions in entry.items():
            for p in positions:
                max_pos = max(max_pos, p.get("pos", 0))
    n_input = max_pos + 1

    if n_input < 5:
        return None

    # Extract head signals
    signals = extract_head_signals(problem)
    if signals is None:
        return None

    heartbeat_signal = signals[HEARTBEAT_HEAD]
    secondary_signal = signals[SECONDARY_HEAD]
    alarm_signal = signals[ALARM_HEAD]

    # --- 1. Heartbeat HMM ---
    heartbeat = fit_heartbeat_hmm(heartbeat_signal, HEARTBEAT_THRESHOLD)

    # --- 2. BIO labels ---
    bio_labels, phase_boundaries = project_to_bio_labels(
        heartbeat, top_positions, n_input
    )

    # --- 3. Alarm events ---
    alarms = detect_alarm_events(alarm_signal, ALARM_THRESHOLD)

    # --- 4. Major boundaries ---
    major_boundaries = find_major_boundaries(
        heartbeat["transitions"], alarms["events"]
    )

    # --- 5. Knot crossings ---
    crossings = detect_crossings(
        heartbeat_signal, secondary_signal, top_positions, n_input
    )
    knot_sig = compute_knot_signature(crossings)
    n_crossings = len(crossings)

    # Use first 4 chars for classification (covers all 31 classes)
    sig_prefix = knot_sig[:4] if len(knot_sig) >= 4 else knot_sig
    knot_class = KNOT_CLASSES.get(sig_prefix, KNOT_CLASSES.get(knot_sig[:3], 0))

    # O/U ratio as additional feature
    o_ratio = knot_sig.count("O") / max(1, len(knot_sig)) if knot_sig else 0.5

    # --- 6. BP depth from major boundaries ---
    bp_depth = compute_bp_depth_from_major_boundaries(len(major_boundaries))

    # BIO counts
    n_b = sum(1 for l in bio_labels if l == "B-COMP")
    n_i = sum(1 for l in bio_labels if l == "I-COMP")
    n_o = sum(1 for l in bio_labels if l == "O")

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "n_input_tokens": n_input,
        "n_generated_tokens": n_gen,

        # Head 1: Phase segmentation
        "bio_labels": bio_labels,
        "phase_boundaries": phase_boundaries,
        "major_boundaries": major_boundaries,
        "heartbeat_states": heartbeat["states"].tolist(),
        "heartbeat_raw": heartbeat_signal.tolist(),
        "n_transitions": heartbeat["n_transitions"],
        "computing_ratio": heartbeat["computing_ratio"],

        # Head 2: Knot classification (first 4 chars -> 31 classes)
        "crossings": crossings,
        "knot_signature": knot_sig,
        "knot_prefix": sig_prefix,
        "knot_class": knot_class,
        "n_crossings": n_crossings,
        "o_ratio": o_ratio,

        # Head 3: BP depth (from major boundaries)
        "bp_depth": bp_depth,
        "n_major_boundaries": len(major_boundaries),

        # Alarm events (use spans for meaningful count)
        "alarm_events": alarms["events"],
        "alarm_spans": alarms["spans"],
        "n_alarm_events": alarms["n_spans"],  # Count spans, not individual steps

        # Metadata
        "n_spans": len(phase_boundaries),
        "bio_counts": {"B-COMP": n_b, "I-COMP": n_i, "O": n_o},
    }


def lambda_handler(event, context):
    """
    Process one IAF chunk, generate C1 training data.
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c1_training_v3/")

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
        "bio_label_dist": {"B-COMP": 0, "I-COMP": 0, "O": 0},
        "knot_class_dist": defaultdict(int),
        "bp_depth_dist": {1: 0, 2: 0, 3: 0},
        "avg_transitions": 0.0,
        "avg_crossings": 0.0,
        "avg_spans": 0.0,
        "avg_alarms": 0.0,
    }

    total_trans = 0
    total_cross = 0
    total_spans = 0
    total_alarms = 0

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

        # Accumulate stats
        bio = record.get("bio_counts", {})
        stats["bio_label_dist"]["B-COMP"] += bio.get("B-COMP", 0)
        stats["bio_label_dist"]["I-COMP"] += bio.get("I-COMP", 0)
        stats["bio_label_dist"]["O"] += bio.get("O", 0)
        stats["knot_class_dist"][record["knot_class"]] += 1
        stats["bp_depth_dist"][record["bp_depth"]] += 1
        total_trans += record["n_transitions"]
        total_cross += record["n_crossings"]
        total_spans += record["n_spans"]
        total_alarms += record.get("n_alarm_events", 0)

    if stats["generated"] > 0:
        stats["avg_transitions"] = total_trans / stats["generated"]
        stats["avg_crossings"] = total_cross / stats["generated"]
        stats["avg_spans"] = total_spans / stats["generated"]
        stats["avg_alarms"] = total_alarms / stats["generated"]

    stats["knot_class_dist"] = dict(stats["knot_class_dist"])

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
        f"avg_trans: {stats['avg_transitions']:.1f} | "
        f"avg_cross: {stats['avg_crossings']:.1f} | "
        f"avg_spans: {stats['avg_spans']:.1f}"
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
