"""
Lambda function: C1 Training Data Generation (Map phase)

Extracts three training signals from IAF data for the C1 structural analyzer:

  Head 1 — Phase segmentation (token-level BIO over input tokens)
    Source: Heartbeat telegraph signal (read/compute HMM state transitions)
    Projects generation-phase boundaries back onto input token positions
    via attention mapping.

  Head 2 — Knot signature (sequence-level, 24 classes)
    Source: Crossing detection from v2 pipeline (pre-computed or inline)

  Head 3 — BP depth prediction (1/2/3 rounds)
    Source: Unknotting number from crossing analysis

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
        s3://mycelium-data/crossings_v2/*.json (pre-computed knot data)
Output: s3://mycelium-data/c1_training/{chunk_id}.json

Each output record:
{
    "problem_id": str,
    "input_token_ids": [int],          # tokenized problem text
    "bio_labels": [str],               # B-COMP, I-COMP, O per input token
    "phase_boundaries": [(int, int)],  # (start, end) of computation spans
    "heartbeat_signal": [int],         # raw 0/1 read/compute per gen token
    "knot_signature": str,             # e.g. "OU", "" for unknot
    "knot_class": int,                 # categorical index (0-23)
    "bp_depth": int,                   # recommended rounds: 1, 2, or 3
    "n_crossings": int,
    "n_steps": int,
}

Memory: 3GB Lambda
"""

import json
import re
import boto3
import numpy as np
from collections import defaultdict
from typing import Optional


s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Layer filtering for attention reconstruction
MIN_LAYER = 14

# Knot signature → class index mapping
# Built from the v2 crossing detection results (24 learnable classes)
KNOT_CLASSES = {
    "": 0,        # unknot
    "U": 1,
    "O": 2,
    "UU": 3,
    "OU": 4,
    "OO": 5,
    "UO": 6,
    "UUU": 7,
    "OUU": 8,
    "OOU": 9,
    "OOO": 10,
    "UOU": 11,
    "UUO": 12,
    "UOO": 13,
    "UOU": 14,
    "OUO": 15,
    "UUUU": 16,
    "OUUU": 17,
    "OOUU": 18,
    "OOOU": 19,
    "OOOO": 20,
    "UUUO": 21,
    "UUOO": 22,
}
RARE_CLASS = 23
N_KNOT_CLASSES = 24


# ---------------------------------------------------------------------------
# 1. Heartbeat extraction (Read/Compute HMM)
# ---------------------------------------------------------------------------

def extract_heartbeat(
    top_positions: list[dict],
    n_input: int,
) -> np.ndarray:
    """
    Extract the heartbeat signal: per generated token, classify as
    READING (attending to input) or COMPUTING (attending to recent
    generated context / own output).

    The teacher model's attention alternates between reading input tokens
    (gathering operands) and attending to recently generated tokens
    (performing computation). This creates a telegraph-like binary signal.

    Classification signal:
      - High attention mass on input tokens → READING (0)
      - Low attention mass on input tokens → COMPUTING (1)
      
    The threshold is adaptive: computed from each problem's attention
    distribution, not hardcoded.

    Returns: (n_generated,) binary array, 0=read, 1=compute
    """
    n_gen = len(top_positions)
    heartbeat = np.zeros(n_gen, dtype=np.int32)

    # Compute input attention mass per generated token
    input_mass = np.zeros(n_gen, dtype=np.float64)

    for t, entry in enumerate(top_positions):
        total_weight = 0.0
        input_weight = 0.0

        for head_key, positions in entry.items():
            match = re.match(r"L(\d+)H(\d+)", head_key)
            if not match:
                continue
            layer_idx = int(match.group(1))
            if layer_idx < MIN_LAYER:
                continue

            for p in positions:
                pos = p.get("pos", 0)
                weight = p.get("weight", 0.0)
                total_weight += weight
                if pos < n_input:
                    input_weight += weight

        input_mass[t] = input_weight / max(total_weight, 1e-10)

    # Adaptive threshold via Otsu's method
    # Separates the bimodal distribution of input attention mass
    # into reading (high mass) and computing (low mass)
    threshold = _otsu_threshold(input_mass)

    heartbeat = (input_mass < threshold).astype(np.int32)

    return heartbeat, input_mass, threshold


def _otsu_threshold(values: np.ndarray, n_bins: int = 50) -> float:
    """
    Otsu's method: find threshold that minimizes intra-class variance.
    
    Optimal for bimodal distributions (read vs compute attention mass).
    """
    if len(values) < 2:
        return 0.5

    hist, bin_edges = np.histogram(values, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()

    if total == 0:
        return 0.5

    best_threshold = 0.5
    best_variance = float("inf")

    cum_sum = 0
    cum_weight = 0

    total_sum = (hist * bin_centers).sum()

    for i in range(n_bins):
        cum_weight += hist[i]
        if cum_weight == 0:
            continue

        bg_weight = total - cum_weight
        if bg_weight == 0:
            break

        cum_sum += hist[i] * bin_centers[i]

        mean_fg = cum_sum / cum_weight
        mean_bg = (total_sum - cum_sum) / bg_weight

        between_var = cum_weight * bg_weight * (mean_fg - mean_bg) ** 2
        # Maximize between-class variance = minimize within-class
        if between_var > (1.0 / max(best_variance, 1e-10)):
            best_variance = 1.0 / between_var
            best_threshold = bin_centers[i]

    return best_threshold


# ---------------------------------------------------------------------------
# 2. HMM smoothing of heartbeat signal
# ---------------------------------------------------------------------------

def smooth_heartbeat_hmm(
    raw_heartbeat: np.ndarray,
    input_mass: np.ndarray,
    min_run_length: int = 3,
) -> np.ndarray:
    """
    Smooth the raw heartbeat signal using a simple HMM-like approach.
    
    The raw binary signal has noise — single-token flickers between
    read and compute states. Real phase transitions last multiple tokens.
    
    Approach: Viterbi-like smoothing with minimum run length constraint.
    Short runs (< min_run_length tokens) get absorbed into the
    surrounding state.
    
    Also uses input_mass confidence: transitions at high-confidence
    points (mass near 0 or 1) are preserved; transitions at ambiguous
    points (mass near threshold) are smoothed away.
    """
    n = len(raw_heartbeat)
    if n < 2:
        return raw_heartbeat.copy()

    smoothed = raw_heartbeat.copy()

    # Pass 1: Remove short runs
    i = 0
    while i < n:
        # Find run of same state
        state = smoothed[i]
        j = i + 1
        while j < n and smoothed[j] == state:
            j += 1

        run_length = j - i

        if run_length < min_run_length and i > 0 and j < n:
            # Short run — absorb into surrounding state
            surrounding_state = smoothed[i - 1]
            for k in range(i, j):
                smoothed[k] = surrounding_state

        i = j

    # Pass 2: Confidence-based smoothing
    # At each transition point, check if the attention mass actually shifted
    # significantly. If not, smooth away the transition.
    for t in range(1, n):
        if smoothed[t] != smoothed[t - 1]:
            # Transition detected — is it confident?
            mass_delta = abs(input_mass[t] - input_mass[t - 1])
            if mass_delta < 0.1:
                # Low-confidence transition — keep previous state
                smoothed[t] = smoothed[t - 1]

    return smoothed


# ---------------------------------------------------------------------------
# 3. Project heartbeat onto input tokens (BIO labels)
# ---------------------------------------------------------------------------

def project_heartbeat_to_input(
    heartbeat: np.ndarray,
    top_positions: list[dict],
    n_input: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Project generation-phase heartbeat signal back onto input tokens
    to produce BIO labels for span segmentation.
    
    For each COMPUTE phase in the heartbeat:
      1. Find which input tokens are attended to during this phase
      2. Mark those input tokens as part of a computation span
      3. Assign BIO tags: B-COMP at span start, I-COMP continuation, O outside
    
    Returns: (bio_labels, phase_boundaries)
      bio_labels: per input token BIO tag
      phase_boundaries: list of (start, end) computation spans in input
    """
    bio = ["O"] * n_input

    # Find compute phases (contiguous runs of state=1)
    phases = _find_runs(heartbeat, target_state=1)

    # For each compute phase, find which input tokens it attends to
    span_regions = []

    for phase_start, phase_end in phases:
        # Aggregate attention over input during this compute phase
        phase_attn = np.zeros(n_input, dtype=np.float64)
        n_tokens = 0

        for t in range(phase_start, min(phase_end, len(top_positions))):
            entry = top_positions[t]
            for head_key, positions in entry.items():
                match = re.match(r"L(\d+)H(\d+)", head_key)
                if not match:
                    continue
                layer_idx = int(match.group(1))
                if layer_idx < MIN_LAYER:
                    continue

                for p in positions:
                    pos = p.get("pos", 0)
                    weight = p.get("weight", 0.0)
                    if 0 <= pos < n_input:
                        phase_attn[pos] += weight
            n_tokens += 1

        if n_tokens > 0:
            phase_attn /= n_tokens

        # Find dominant input region for this compute phase
        if phase_attn.sum() > 1e-10:
            region = _find_dominant_region(phase_attn, threshold_quantile=0.7)
            span_regions.append(region)

    # Merge overlapping spans and assign BIO labels
    merged_spans = _merge_overlapping_spans(span_regions)

    for span_start, span_end in merged_spans:
        for i in range(span_start, min(span_end + 1, n_input)):
            if i == span_start:
                bio[i] = "B-COMP"
            else:
                bio[i] = "I-COMP"

    return bio, merged_spans


def _find_runs(
    signal: np.ndarray,
    target_state: int,
) -> list[tuple[int, int]]:
    """Find contiguous runs of target_state in signal."""
    runs = []
    i = 0
    while i < len(signal):
        if signal[i] == target_state:
            start = i
            while i < len(signal) and signal[i] == target_state:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _find_dominant_region(
    attn_map: np.ndarray,
    threshold_quantile: float = 0.7,
) -> tuple[int, int]:
    """Find dominant contiguous attention region."""
    threshold = np.quantile(attn_map, threshold_quantile)
    if threshold < 1e-10:
        peak = int(np.argmax(attn_map))
        return (max(0, peak - 2), min(len(attn_map) - 1, peak + 2))

    above = (attn_map >= threshold).astype(int)

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


def _merge_overlapping_spans(
    spans: list[tuple[int, int]],
    gap_tolerance: int = 2,
) -> list[tuple[int, int]]:
    """Merge spans that overlap or are within gap_tolerance tokens."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


# ---------------------------------------------------------------------------
# 4. Crossing detection (inline, adapted from v2)
# ---------------------------------------------------------------------------

def detect_crossings_inline(
    top_positions: list[dict],
    n_input: int,
    heartbeat: np.ndarray,
) -> dict:
    """
    Detect knot crossings using heartbeat-defined step boundaries.
    
    Uses the smoothed heartbeat to define computation steps, then
    checks for overlapping attention regions between steps.
    
    This is more principled than the v2 JSD-based segmentation because
    the heartbeat directly measures cognitive phase transitions.
    """
    # Use compute phases as step definitions
    compute_phases = _find_runs(heartbeat, target_state=1)

    if len(compute_phases) < 2:
        return {"n_crossings": 0, "crossings": [], "knot_signature": "",
                "unknotting_number": 0}

    # Compute per-step attention profiles
    step_profiles = []
    for phase_start, phase_end in compute_phases:
        phase_attn = np.zeros(n_input, dtype=np.float64)
        count = 0
        for t in range(phase_start, min(phase_end, len(top_positions))):
            entry = top_positions[t]
            for head_key, positions in entry.items():
                match = re.match(r"L(\d+)H(\d+)", head_key)
                if not match:
                    continue
                if int(match.group(1)) < MIN_LAYER:
                    continue
                for p in positions:
                    pos = p.get("pos", 0)
                    weight = p.get("weight", 0.0)
                    if 0 <= pos < n_input:
                        phase_attn[pos] += weight
            count += 1

        if count > 0:
            phase_attn /= count

        region = _find_dominant_region(phase_attn)
        step_profiles.append({
            "phase": (phase_start, phase_end),
            "region_start": region[0],
            "region_end": region[1],
            "attn_map": phase_attn,
        })

    # Detect crossings between step pairs
    crossings = []
    for i in range(len(step_profiles)):
        for j in range(i + 1, len(step_profiles)):
            pi, pj = step_profiles[i], step_profiles[j]

            # Check overlap
            o_start = max(pi["region_start"], pj["region_start"])
            o_end = min(pi["region_end"], pj["region_end"])
            overlap_len = o_end - o_start + 1

            if overlap_len < 2:
                continue

            len_i = pi["region_end"] - pi["region_start"] + 1
            len_j = pj["region_end"] - pj["region_start"] + 1
            strength = overlap_len / max(1, min(len_i, len_j))

            if strength < 0.15:
                continue

            # Classify over/under using temporal attention dynamics
            crossing_type = _classify_crossing_temporal(
                top_positions, pi, pj, (o_start, o_end), n_input
            )

            crossings.append({
                "step_i": i,
                "step_j": j,
                "crossing_type": crossing_type,
                "overlap_strength": float(strength),
            })

    # Compute knot invariants
    knot_sig = "".join(
        "O" if c["crossing_type"] == "over" else "U"
        for c in sorted(crossings, key=lambda c: (c["step_i"], c["step_j"]))
    )

    unknotting = _compute_unknotting(crossings, len(step_profiles))

    return {
        "n_crossings": len(crossings),
        "crossings": crossings,
        "knot_signature": knot_sig,
        "unknotting_number": unknotting,
    }


def _classify_crossing_temporal(
    top_positions, prof_i, prof_j, overlap_region, n_input
) -> str:
    """Classify crossing using per-token temporal dynamics within step_j."""
    o_start, o_end = overlap_region
    j_start, j_end = prof_j["phase"]

    i_excl_start = prof_i["region_start"]
    i_excl_end = min(o_start - 1, prof_i["region_end"])
    has_exclusive = i_excl_end >= i_excl_start

    if not has_exclusive:
        i_mass = prof_i["attn_map"][o_start:o_end + 1].sum()
        j_mass = prof_j["attn_map"][o_start:o_end + 1].sum()
        return "under" if j_mass > i_mass * 1.5 else "over"

    # Find onset of overlap vs exclusive attention within step_j
    overlap_onset = None
    excl_onset = None

    for t in range(j_start, min(j_end, len(top_positions))):
        entry = top_positions[t]
        o_mass = 0.0
        e_mass = 0.0

        for head_key, positions in entry.items():
            match = re.match(r"L(\d+)H(\d+)", head_key)
            if not match:
                continue
            if int(match.group(1)) < MIN_LAYER:
                continue
            for p in positions:
                pos = p.get("pos", 0)
                w = p.get("weight", 0.0)
                if o_start <= pos <= o_end:
                    o_mass += w
                if i_excl_start <= pos <= i_excl_end:
                    e_mass += w

        if overlap_onset is None and o_mass > 0.1:
            overlap_onset = t - j_start
        if excl_onset is None and e_mass > 0.1:
            excl_onset = t - j_start

    if excl_onset is not None and overlap_onset is not None:
        return "over" if excl_onset < overlap_onset else "under"

    return "under"


def _compute_unknotting(crossings, n_steps):
    """Minimum crossings to remove for linearization."""
    if not crossings:
        return 0
    remaining = list(range(len(crossings)))
    cuts = 0
    while remaining:
        step_counts = defaultdict(list)
        for idx in remaining:
            c = crossings[idx]
            step_counts[c["step_i"]].append(idx)
            step_counts[c["step_j"]].append(idx)
        if not step_counts:
            break
        busiest = max(step_counts, key=lambda s: len(step_counts[s]))
        removed = set(step_counts[busiest])
        remaining = [i for i in remaining if i not in removed]
        cuts += 1
    return cuts


# ---------------------------------------------------------------------------
# 5. Generate C1 training record
# ---------------------------------------------------------------------------

def derive_bio_from_step_regions(
    step_regions: list[dict],
    n_input: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Derive BIO labels from pre-computed step regions.

    Each step has an input_region (start, end) indicating which input
    tokens were attended during that computation step. Convert these
    to token-level BIO tags.
    """
    bio = ["O"] * n_input
    spans = []

    for step in step_regions:
        region = step.get("input_region", [])
        if len(region) >= 2:
            start, end = int(region[0]), int(region[1])
            # Clamp to valid range
            start = max(0, min(start, n_input - 1))
            end = max(start, min(end, n_input - 1))
            spans.append((start, end))

    # Merge overlapping spans
    merged = _merge_overlapping_spans(spans)

    # Assign BIO labels
    for span_start, span_end in merged:
        for i in range(span_start, min(span_end + 1, n_input)):
            if i == span_start:
                bio[i] = "B-COMP"
            else:
                bio[i] = "I-COMP"

    return bio, merged


def derive_bio_from_attention(
    top_positions: list[dict],
    n_input: int,
    n_steps: int = 5,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Fallback: Derive BIO labels by segmenting attention patterns.

    Splits generation into n_steps segments and finds the dominant
    input region attended in each segment.
    """
    n_gen = len(top_positions)
    if n_gen < n_steps:
        n_steps = max(1, n_gen)

    step_size = n_gen // n_steps
    spans = []

    for step_idx in range(n_steps):
        t_start = step_idx * step_size
        t_end = (step_idx + 1) * step_size if step_idx < n_steps - 1 else n_gen

        # Aggregate attention over this step
        step_attn = np.zeros(n_input, dtype=np.float64)
        count = 0

        for t in range(t_start, t_end):
            entry = top_positions[t]
            for head_key, positions in entry.items():
                match = re.match(r"L(\d+)H(\d+)", head_key)
                if not match:
                    continue
                if int(match.group(1)) < MIN_LAYER:
                    continue
                for p in positions:
                    pos = p.get("pos", 0)
                    weight = p.get("weight", 0.0)
                    if 0 <= pos < n_input:
                        step_attn[pos] += weight
            count += 1

        if count > 0:
            step_attn /= count

        # Find dominant region
        if step_attn.sum() > 1e-10:
            region = _find_dominant_region(step_attn, threshold_quantile=0.7)
            spans.append(region)

    # Merge and assign BIO
    merged = _merge_overlapping_spans(spans)
    bio = ["O"] * n_input

    for span_start, span_end in merged:
        for i in range(span_start, min(span_end + 1, n_input)):
            if i == span_start:
                bio[i] = "B-COMP"
            else:
                bio[i] = "I-COMP"

    return bio, merged


def generate_c1_record(
    problem: dict,
    precomputed_crossings: Optional[dict] = None,
) -> Optional[dict]:
    """
    Generate one C1 training record from an IAF problem.

    Combines phase segmentation (from step regions) + knot crossings
    into a single multi-task training example.
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

    # --- Phase segmentation (BIO labels) ---
    # Prefer pre-computed step regions if available
    if precomputed_crossings is not None and "step_regions" in precomputed_crossings:
        step_regions = precomputed_crossings["step_regions"]
        bio_labels, phase_boundaries = derive_bio_from_step_regions(
            step_regions, n_input
        )
        n_steps = len(step_regions)
    else:
        # Fallback: derive from attention patterns
        bio_labels, phase_boundaries = derive_bio_from_attention(
            top_positions, n_input
        )
        n_steps = len(phase_boundaries)

    # --- Knot crossings ---
    if precomputed_crossings is not None:
        knot_sig = precomputed_crossings.get("knot_signature", "")
        n_crossings = precomputed_crossings.get("n_crossings", 0)
        unknotting_num = precomputed_crossings.get("unknotting_number", 0)
    else:
        # No pre-computed data - compute inline
        # Use attention-based step detection
        crossing_result = detect_crossings_inline(
            top_positions, n_input, np.zeros(n_gen, dtype=np.int32)
        )
        knot_sig = crossing_result["knot_signature"]
        n_crossings = crossing_result["n_crossings"]
        unknotting_num = crossing_result["unknotting_number"]

    # Knot class index
    knot_class = KNOT_CLASSES.get(knot_sig, RARE_CLASS)

    # BP depth from unknotting number
    if unknotting_num == 0:
        bp_depth = 1
    elif unknotting_num == 1:
        bp_depth = 2
    else:
        bp_depth = 3

    # Count BIO label distribution
    n_b_comp = sum(1 for l in bio_labels if l == "B-COMP")
    n_i_comp = sum(1 for l in bio_labels if l == "I-COMP")
    n_outside = sum(1 for l in bio_labels if l == "O")

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "n_input_tokens": n_input,
        "n_generated_tokens": n_gen,

        # Head 1: Phase segmentation
        "bio_labels": bio_labels,
        "phase_boundaries": phase_boundaries,
        "n_steps": n_steps,

        # Head 2: Knot classification
        "knot_signature": knot_sig,
        "knot_class": knot_class,
        "n_crossings": n_crossings,

        # Head 3: BP depth
        "bp_depth": bp_depth,
        "unknotting_number": unknotting_num,

        # Metadata
        "n_spans": len(phase_boundaries),
        "bio_counts": {"B-COMP": n_b_comp, "I-COMP": n_i_comp, "O": n_outside},
    }


# ---------------------------------------------------------------------------
# 6. Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """
    Process one IAF chunk, generate C1 training data.
    
    Event:
    {
        "chunk_key": "iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_000.json",
        "crossings_prefix": "crossings_v2/",  # optional: pre-computed crossings
        "output_prefix": "c1_training/",
        "bucket": "mycelium-data"
    }
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c1_training/")
    crossings_prefix = event.get("crossings_prefix", None)

    # Download IAF chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    # Optionally load pre-computed crossings
    precomputed = {}
    if crossings_prefix:
        chunk_name = chunk_key.split("/")[-1]
        crossings_key = f"{crossings_prefix}{chunk_name}"
        try:
            resp = s3.get_object(Bucket=bucket, Key=crossings_key)
            crossings_data = json.loads(resp["Body"].read().decode("utf-8"))
            for result in crossings_data.get("results", []):
                pid = str(result.get("problem_id", ""))
                precomputed[pid] = result
            print(f"Loaded {len(precomputed)} pre-computed crossings")
        except Exception as e:
            print(f"No pre-computed crossings found ({e}), computing inline")

    # Process
    records = []
    stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "bio_label_dist": {"B-COMP": 0, "I-COMP": 0, "O": 0},
        "knot_class_dist": defaultdict(int),
        "bp_depth_dist": {1: 0, 2: 0, 3: 0},
        "avg_steps": 0.0,
        "avg_spans": 0.0,
    }

    total_steps = 0
    total_spans = 0

    for idx, problem in enumerate(problems):
        stats["total"] += 1

        if "problem_idx" not in problem and "problem_id" not in problem:
            problem["problem_idx"] = f"{chunk_key}_{idx}"

        pid = str(problem.get("problem_idx", problem.get("problem_id", "")))
        crossing_data = precomputed.get(pid, None)

        record = generate_c1_record(problem, crossing_data)

        if record is None:
            stats["skipped"] += 1
            continue

        records.append(record)
        stats["generated"] += 1

        # Accumulate stats
        bio_counts = record.get("bio_counts", {})
        stats["bio_label_dist"]["B-COMP"] += bio_counts.get("B-COMP", 0)
        stats["bio_label_dist"]["I-COMP"] += bio_counts.get("I-COMP", 0)
        stats["bio_label_dist"]["O"] += bio_counts.get("O", 0)
        stats["knot_class_dist"][record["knot_class"]] += 1
        stats["bp_depth_dist"][record["bp_depth"]] += 1
        total_steps += record.get("n_steps", 0)
        total_spans += record["n_spans"]

    if stats["generated"] > 0:
        stats["avg_steps"] = total_steps / stats["generated"]
        stats["avg_spans"] = total_spans / stats["generated"]

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
        f"avg steps: {stats['avg_steps']:.1f} | "
        f"avg spans: {stats['avg_spans']:.1f}"
    )

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(output, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    # Stats for fast reduce
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
