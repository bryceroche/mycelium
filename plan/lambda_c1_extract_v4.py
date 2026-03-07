"""
Lambda function: C1 Training Data Extraction v4 (Shadow Reader)

Key changes from v3:
1. Telegraph x alarm crossings (L22H3 x L22H4) instead of telegraph x telegraph
2. Boundary probabilities instead of BIO labels
3. Raw crossing statistics instead of knot classification

Outputs per record:
- boundary_probs: per-token boundary detection target
- n_crossings, o_ratio, mean_spacing, burstiness: crossing statistics
- bp_depth, n_major_boundaries: complexity measures

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/c1_training_v4/*.json

Memory: 3GB Lambda
"""

import json
import boto3
import numpy as np
from typing import Optional


s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Validated heads from analysis
TELEGRAPH_HEAD = "L22H3"      # Primary telegraph (47% duty cycle, 74:1 contrast)
ALARM_HEAD = "L22H4"          # Rare event detector (77% high, sharp dips)

# Otsu thresholds from head analysis
TELEGRAPH_THRESHOLD = 0.48   # Midpoint of 0.08/0.87 for L22H3
ALARM_THRESHOLD = 0.57       # Threshold for L22H4


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
    """
    Fit two-state HMM to heartbeat signal.
    States: 0 = COMPUTING (low), 1 = READING (high)
    """
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


def compute_boundary_probs(
    heartbeat: dict,
    top_positions: list[dict],
    n_input: int,
) -> tuple[list[float], int]:
    """
    Compute boundary probabilities for input tokens.

    For each COMPUTING span, find the dominant input region and mark
    the START of each region with 1.0 (boundary), else 0.0.
    """
    states = heartbeat["states"]
    n_steps = len(states)

    boundary_probs = [0.0] * n_input
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
                if TELEGRAPH_HEAD in entry:
                    for p in entry[TELEGRAPH_HEAD]:
                        pos = p.get("pos", 0)
                        weight = p.get("weight", 0.0)
                        if 0 <= pos < n_input:
                            span_attn[pos] += weight
                    count += 1

            if count > 0:
                span_attn /= count

            # Find dominant region start
            if span_attn.sum() > 1e-6:
                region_start = _find_region_start(span_attn)
                if region_start is not None:
                    boundaries.append(region_start)
        else:
            i += 1

    # Merge nearby boundaries and mark in output
    unique_boundaries = _merge_nearby_boundaries(boundaries, gap=2)
    for b in unique_boundaries:
        if 0 <= b < n_input:
            boundary_probs[b] = 1.0

    return boundary_probs, len(unique_boundaries)


def _find_region_start(attn: np.ndarray) -> Optional[int]:
    """Find start of dominant attention region."""
    threshold = np.percentile(attn, 70)
    if threshold < 1e-6:
        return int(np.argmax(attn))

    above = (attn >= threshold).astype(int)

    # Find first position above threshold in longest run
    best_start, best_len = None, 0
    curr_start = None

    for i in range(len(above)):
        if above[i]:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                length = i - curr_start
                if length > best_len:
                    best_start, best_len = curr_start, length
                curr_start = None

    if curr_start is not None:
        length = len(above) - curr_start
        if length > best_len:
            best_start = curr_start

    return best_start


def _merge_nearby_boundaries(boundaries: list[int], gap: int = 2) -> list[int]:
    """Merge boundaries that are within `gap` positions of each other."""
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
    """
    Detect co-transitions: moments where telegraph changes state AND alarm fires.

    This is conceptually different from signal crossings:
    - Crossing: two lines intersect (depends on absolute levels)
    - Co-transition: two systems change state together (depends on derivatives)

    A co-transition marks a structural boundary where the problem "surprised"
    the model - the regular heartbeat shifted AND the alarm fired.

    Args:
        heartbeat_transitions: step indices where L22H3 changes state
        heartbeat_states: binary state array (0=computing, 1=reading)
        alarm_events: step indices where L22H4 dips (alarm fires)
        window: max steps between telegraph transition and alarm event
        min_gap: minimum steps between co-transitions (debouncing)
    """
    alarm_set = set(alarm_events)
    co_transitions = []
    last_step = -min_gap - 1

    for t in heartbeat_transitions:
        # Debounce
        if t - last_step < min_gap:
            continue

        # Check if alarm fires within ±window steps
        has_alarm = False
        for offset in range(-window, window + 1):
            if (t + offset) in alarm_set:
                has_alarm = True
                break

        if has_alarm:
            # Record what state L22H3 is transitioning TO
            # If states[t] == 1, we're entering READING state
            # If states[t] == 0, we're entering COMPUTING state
            state_after = int(heartbeat_states[t]) if t < len(heartbeat_states) else 0

            co_transitions.append({
                "step": t,
                "state_after": state_after,  # 0=computing, 1=reading
            })
            last_step = t

    return co_transitions


def compute_co_transition_stats(co_transitions: list[dict], n_steps: int) -> dict:
    """
    Compute co-transition statistics for regression targets.

    Returns:
        n_co_transitions: total count (expect 1-5 per problem)
        reading_ratio: fraction where alarm fires during READING state
                       (vs COMPUTING state) - indicates what phase triggers alarms
        mean_spacing: average steps between co-transitions
        burstiness: variance/mean of spacing (0 = perfectly regular)
    """
    n = len(co_transitions)

    if n == 0:
        return {
            "n_co_transitions": 0,
            "reading_ratio": 0.5,  # neutral when no data
            "mean_spacing": float(n_steps),
            "burstiness": 0.0,
        }

    # Reading ratio: fraction transitioning TO reading state when alarm fires
    n_reading = sum(1 for c in co_transitions if c["state_after"] == 1)
    reading_ratio = n_reading / n

    if n == 1:
        return {
            "n_co_transitions": 1,
            "reading_ratio": float(reading_ratio),
            "mean_spacing": float(n_steps),
            "burstiness": 0.0,
        }

    # Spacing between consecutive co-transitions
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


def find_major_boundaries(
    heartbeat_transitions: list[int],
    alarm_events: list[int],
    window: int = 2,
) -> list[int]:
    """Find major boundaries where heartbeat and alarm co-transition."""
    major = []
    alarm_set = set(alarm_events)

    for t in heartbeat_transitions:
        for offset in range(-window, window + 1):
            if (t + offset) in alarm_set:
                major.append(t)
                break

    return major


def compute_bp_depth(n_alarm_spans: int) -> int:
    """
    Compute BP depth from alarm span count.

    Tercile-based thresholds calibrated on v4 data:
    Mean alarm spans ~21, median ~20
    - 1: Simple (≤15 alarm spans) ~33%
    - 2: Moderate (16-24 spans) ~33%
    - 3: Complex (>24 spans) ~33%
    """
    if n_alarm_spans <= 15:
        return 1
    elif n_alarm_spans <= 24:
        return 2
    else:
        return 3


def generate_c1_record(problem: dict) -> Optional[dict]:
    """Generate one C1 training record in v4 schema."""
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

    # Extract signals
    signals = extract_head_signals(problem)
    if signals is None:
        return None

    telegraph_signal = signals[TELEGRAPH_HEAD]
    alarm_signal = signals[ALARM_HEAD]

    # --- 1. Heartbeat for boundary detection ---
    heartbeat = fit_heartbeat_hmm(telegraph_signal, TELEGRAPH_THRESHOLD)

    # --- 2. Boundary probabilities ---
    boundary_probs, n_boundaries = compute_boundary_probs(
        heartbeat, top_positions, n_input
    )

    # --- 3. Alarm events ---
    alarm_events = detect_alarm_events(alarm_signal, ALARM_THRESHOLD)

    # --- 4. Co-transitions (telegraph state change + alarm fires) ---
    co_transitions = detect_co_transitions(
        heartbeat["transitions"],
        heartbeat["states"],
        alarm_events,
        window=2,
        min_gap=5,
    )
    co_stats = compute_co_transition_stats(co_transitions, n_gen)

    # --- 5. BP depth (based on total alarm spans, not co-transitions) ---
    # Use number of alarm event clusters for complexity
    n_alarm_spans = len(_cluster_events(alarm_events, gap=3))
    bp_depth = compute_bp_depth(n_alarm_spans)

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "n_input_tokens": n_input,

        # Head 1: Boundary detection (per-token)
        "boundary_probs": boundary_probs,
        "n_boundaries": n_boundaries,

        # Head 2: Co-transition statistics (sequence-level)
        # Co-transitions = telegraph state change + alarm fires within ±2 steps
        "n_co_transitions": co_stats["n_co_transitions"],
        "reading_ratio": co_stats["reading_ratio"],  # fraction entering reading state
        "mean_spacing": co_stats["mean_spacing"],
        "burstiness": co_stats["burstiness"],

        # Head 3: BP depth (sequence-level)
        "bp_depth": bp_depth,
        "n_alarm_spans": n_alarm_spans,
    }


def lambda_handler(event, context):
    """Process one IAF chunk, generate C1 v4 training data."""
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c1_training_v4/")

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
        "reading_ratios": [],
        "spacings": [],
        "burstinesses": [],
        "boundary_counts": [],
        "alarm_span_counts": [],
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

        # Collect stats for reduce
        stats["co_transition_counts"].append(record["n_co_transitions"])
        stats["reading_ratios"].append(record["reading_ratio"])
        stats["spacings"].append(record["mean_spacing"])
        stats["burstinesses"].append(record["burstiness"])
        stats["boundary_counts"].append(record["n_boundaries"])
        stats["alarm_span_counts"].append(record["n_alarm_spans"])
        stats["bp_depth_dist"][record["bp_depth"]] += 1

    # Compute summary stats
    if stats["generated"] > 0:
        ct = stats["co_transition_counts"]
        stats["co_transition_mean"] = float(np.mean(ct))
        stats["co_transition_median"] = float(np.median(ct))
        stats["co_transition_p25"] = float(np.percentile(ct, 25))
        stats["co_transition_p75"] = float(np.percentile(ct, 75))
        stats["co_transition_max"] = int(max(ct))
        stats["zero_co_transition_count"] = sum(1 for c in ct if c == 0)

        stats["reading_ratio_mean"] = float(np.mean(stats["reading_ratios"]))
        stats["reading_ratio_std"] = float(np.std(stats["reading_ratios"]))

        stats["boundary_mean"] = float(np.mean(stats["boundary_counts"]))
        bc = stats["boundary_counts"]
        total_tokens = sum(r["n_input_tokens"] for r in records)
        total_boundaries = sum(bc)
        stats["boundary_density"] = total_boundaries / max(1, total_tokens)

    # Clean up large lists before upload
    del stats["co_transition_counts"]
    del stats["reading_ratios"]
    del stats["spacings"]
    del stats["burstinesses"]
    del stats["boundary_counts"]
    del stats["alarm_span_counts"]

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
        f"co_trans: {stats.get('co_transition_mean', 0):.2f} | "
        f"median: {stats.get('co_transition_median', 0):.1f}"
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
