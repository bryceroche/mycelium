"""
Lambda function: IAF Knot Crossing Detection (Map phase) — v2

Adapted to actual IAF data format:
{
    "generated_cot": str,
    "problem_text": str,      # may or may not be present
    "top_positions": [         # one entry per generated token
        {
            "L22H4": [{"pos": 2, "weight": 0.984}, ...],
            "L22H3": [...],
            ...  # top 10 heads per generated token
        }
    ]
}

Processing pipeline:
1. Reconstruct per-token input attention from sparse top_positions
2. Segment generated tokens into CoT steps (sentence/equation boundaries)
3. Compute per-step dominant attention regions
4. Detect crossings (overlapping regions between steps)
5. Classify crossing type (over/under) from temporal attention dynamics

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

# Heads from later layers carry more semantic signal.
# Weight by layer depth — L25 matters more than L5 for crossing detection.
# This is soft weighting, not a hard filter.
MIN_LAYER = 14  # ignore heads below this (syntactic, not semantic)


# ---------------------------------------------------------------------------
# 1. Reconstruct attention from sparse top_positions
# ---------------------------------------------------------------------------

def reconstruct_token_attention(
    top_positions_entry: dict,
    n_input: int,
) -> np.ndarray:
    """
    Reconstruct a dense attention vector over input from one generated token's
    sparse top_positions.
    
    Aggregates across heads with layer-depth weighting:
      weight_factor = (layer_idx - MIN_LAYER + 1) / (max_layer - MIN_LAYER + 1)
    
    Returns: (n_input,) attention distribution
    """
    attn = np.zeros(n_input, dtype=np.float64)
    total_head_weight = 0.0
    
    for head_key, positions in top_positions_entry.items():
        # Parse layer index from key like "L22H4"
        match = re.match(r"L(\d+)H(\d+)", head_key)
        if not match:
            continue
        
        layer_idx = int(match.group(1))
        
        # Skip early layers — syntactic, not semantic
        if layer_idx < MIN_LAYER:
            continue
        
        # Layer-depth weighting (deeper = more weight for semantic crossings)
        layer_weight = (layer_idx - MIN_LAYER + 1) / 15.0  # normalize roughly
        
        for entry in positions:
            pos = entry.get("pos", 0)
            weight = entry.get("weight", 0.0)
            
            if 0 <= pos < n_input:
                attn[pos] += weight * layer_weight
        
        total_head_weight += layer_weight
    
    # Normalize
    if total_head_weight > 0:
        attn /= total_head_weight
    
    return attn


def reconstruct_all_attention(
    top_positions: list[dict],
    n_input: int,
) -> np.ndarray:
    """
    Reconstruct attention for all generated tokens.
    
    Returns: (n_generated, n_input) attention matrix
    """
    n_gen = len(top_positions)
    attn_matrix = np.zeros((n_gen, n_input), dtype=np.float64)
    
    for t, entry in enumerate(top_positions):
        attn_matrix[t] = reconstruct_token_attention(entry, n_input)
    
    return attn_matrix


# ---------------------------------------------------------------------------
# 2. Segment CoT into computation steps
# ---------------------------------------------------------------------------

def segment_cot_steps(generated_cot: str) -> list[tuple[int, int]]:
    """
    Segment CoT text into computation steps.
    
    Returns list of (start_token_idx, end_token_idx) into the generated
    token sequence. Each segment = one computation step.
    
    Segmentation signals (from the CoT text):
    - Newlines (steps are usually on separate lines)
    - Equation boundaries (= sign followed by result)
    - Step markers ("Step 1:", "First,", numbered lists)
    - LaTeX display math boundaries (\\[ ... \\])
    
    This is approximate — the crossing detection is robust to
    over-segmentation (same as C1 segmenter).
    """
    if not generated_cot:
        return []
    
    # Split into rough tokens (whitespace) to get positional indices
    # These are approximate — actual tokenizer alignment happens via
    # the top_positions array length
    lines = generated_cot.split("\n")
    
    steps = []
    current_pos = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_pos += len(line) + 1  # +1 for newline
            continue
        
        # Approximate token count for this line
        # (rough: ~1.3 tokens per word for math-heavy text)
        word_count = len(line_stripped.split())
        approx_tokens = max(1, int(word_count * 1.3))
        
        # Check if this line contains computation
        has_equation = bool(re.search(r'[=≈><]', line_stripped))
        has_math_op = bool(re.search(r'[\+\-\*/×÷]', line_stripped))
        has_number = bool(re.search(r'\d+', line_stripped))
        has_latex = bool(re.search(r'\\(?:frac|cdot|times|div|sqrt)', line_stripped))
        
        is_computation = (has_equation or has_latex) and has_number
        is_operation = has_math_op and has_number
        
        if is_computation or is_operation:
            steps.append((current_pos, current_pos + approx_tokens))
        
        current_pos += approx_tokens
    
    return steps


def segment_by_attention_shift(
    attn_matrix: np.ndarray,
    min_step_tokens: int = 5,
    shift_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Alternative segmentation: detect step boundaries from attention shifts.
    
    When the dominant attention region jumps significantly between consecutive
    generated tokens, that's a step boundary. This is purely data-driven,
    no text parsing needed.
    
    Uses Jensen-Shannon divergence between consecutive attention distributions
    to detect boundaries (same signal as C1 segmenter training).
    """
    n_gen = attn_matrix.shape[0]
    if n_gen < 2:
        return [(0, n_gen)]
    
    # Compute JSD between consecutive tokens
    jsd_values = np.zeros(n_gen - 1)
    for t in range(n_gen - 1):
        p = attn_matrix[t] + 1e-10
        q = attn_matrix[t + 1] + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
        jsd_values[t] = jsd
    
    # Find boundaries: peaks in JSD above threshold
    # Use adaptive threshold based on distribution
    if len(jsd_values) > 0:
        threshold = max(
            shift_threshold,
            np.mean(jsd_values) + 1.0 * np.std(jsd_values),
        )
    else:
        threshold = shift_threshold
    
    boundaries = [0]
    for t in range(len(jsd_values)):
        if jsd_values[t] > threshold:
            # Enforce minimum step length
            if (t + 1) - boundaries[-1] >= min_step_tokens:
                boundaries.append(t + 1)
    boundaries.append(n_gen)
    
    # Convert to step ranges
    steps = []
    for i in range(len(boundaries) - 1):
        steps.append((boundaries[i], boundaries[i + 1]))
    
    return steps


# ---------------------------------------------------------------------------
# 3. Core crossing detection
# ---------------------------------------------------------------------------

def detect_crossings(problem: dict) -> dict:
    """
    Detect knot crossings in a single problem's IAF data.
    
    Actual IAF format:
    {
        "generated_cot": str,
        "top_positions": [
            {"L22H4": [{"pos": 2, "weight": 0.984}, ...], ...},
            ...  # one per generated token
        ],
        "problem_text": str (optional),
        "question": str (optional),
        "problem_idx": int (optional),
    }
    """
    top_positions = problem.get("top_positions", [])
    generated_cot = problem.get("generated_cot", "")
    problem_id = str(problem.get("problem_idx", problem.get("problem_id", "unknown")))
    
    if not top_positions:
        return _empty_result(problem_id, 0)
    
    n_gen = len(top_positions)
    
    # --- Determine n_input from the data ---
    # Find max input position referenced across all tokens/heads
    max_pos = 0
    for entry in top_positions:
        for head_key, positions in entry.items():
            for p in positions:
                max_pos = max(max_pos, p.get("pos", 0))
    n_input = max_pos + 1
    
    if n_input < 2:
        return _empty_result(problem_id, 0)
    
    # --- Reconstruct full attention matrix ---
    attn_matrix = reconstruct_all_attention(top_positions, n_input)
    
    # --- Segment into computation steps ---
    # Use attention-based segmentation (data-driven, no text heuristics)
    steps = segment_by_attention_shift(attn_matrix)
    
    if len(steps) < 2:
        return _empty_result(problem_id, len(steps))
    
    # --- Compute per-step attention profiles ---
    step_profiles = []
    for step_start, step_end in steps:
        # Aggregate attention across all tokens in this step
        step_attn = attn_matrix[step_start:step_end].mean(axis=0)
        
        # Find dominant region
        region = _find_dominant_region(step_attn)
        
        # Track temporal dynamics within the step
        step_profiles.append({
            "step_range": (step_start, step_end),
            "attn_map": step_attn,
            "region_start": region[0],
            "region_end": region[1],
            "attn_mass": float(step_attn[region[0]:region[1] + 1].sum()),
            "n_tokens": step_end - step_start,
        })
    
    # --- Detect crossings ---
    crossings = []
    
    for i in range(len(step_profiles)):
        for j in range(i + 1, len(step_profiles)):
            prof_i = step_profiles[i]
            prof_j = step_profiles[j]
            
            # Check overlap
            overlap = _compute_overlap(prof_i, prof_j)
            if overlap is None:
                continue
            
            overlap_start, overlap_end, overlap_strength = overlap
            
            # Classify crossing type using temporal dynamics
            crossing_type = _classify_crossing_v2(
                attn_matrix=attn_matrix,
                step_i=steps[i],
                step_j=steps[j],
                prof_i=prof_i,
                prof_j=prof_j,
                overlap_region=(overlap_start, overlap_end),
            )
            
            crossings.append({
                "step_i": i,
                "step_j": j,
                "step_i_range": list(steps[i]),
                "step_j_range": list(steps[j]),
                "overlap_region": [overlap_start, overlap_end],
                "crossing_type": crossing_type,
                "overlap_strength": float(overlap_strength),
                "temporal_gap": j - i,
            })
    
    # --- Compute knot invariants ---
    crossing_number = len(crossings)
    unknotting_number = _compute_unknotting_number(crossings, len(steps))
    knot_signature = _compute_knot_signature(crossings)
    
    return {
        "problem_id": problem_id,
        "n_generated_tokens": n_gen,
        "n_input_tokens": n_input,
        "n_steps": len(steps),
        "n_crossings": crossing_number,
        "crossing_number": crossing_number,
        "unknotting_number": unknotting_number,
        "crossings": crossings,
        "knot_signature": knot_signature,
        "step_regions": [
            {
                "step": idx,
                "token_range": list(steps[idx]),
                "input_region": [p["region_start"], p["region_end"]],
                "attn_mass": p["attn_mass"],
            }
            for idx, p in enumerate(step_profiles)
        ],
    }


# ---------------------------------------------------------------------------
# Attention region analysis
# ---------------------------------------------------------------------------

def _find_dominant_region(
    attn_map: np.ndarray,
    threshold_quantile: float = 0.75,
    min_region_len: int = 1,
) -> tuple[int, int]:
    """
    Find dominant contiguous attention region using quantile thresholding.
    """
    if attn_map.sum() < 1e-10:
        return (0, max(0, len(attn_map) - 1))
    
    threshold = np.quantile(attn_map, threshold_quantile)
    if threshold < 1e-10:
        # Fallback: use top positions directly
        peak = int(np.argmax(attn_map))
        return (max(0, peak - 2), min(len(attn_map) - 1, peak + 2))
    
    above = (attn_map >= threshold).astype(int)
    
    # Find largest contiguous region
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
    
    if best_len < min_region_len:
        peak = int(np.argmax(attn_map))
        return (max(0, peak - 2), min(len(attn_map) - 1, peak + 2))
    
    return (best_start, best_end)


# ---------------------------------------------------------------------------
# Crossing detection and classification
# ---------------------------------------------------------------------------

def _compute_overlap(
    prof_i: dict,
    prof_j: dict,
    min_overlap_tokens: int = 2,
    min_overlap_strength: float = 0.15,
) -> Optional[tuple[int, int, float]]:
    """
    Compute overlap between two step attention regions.
    Returns (start, end, strength) or None.
    """
    start = max(prof_i["region_start"], prof_j["region_start"])
    end = min(prof_i["region_end"], prof_j["region_end"])
    
    overlap_len = end - start + 1
    if overlap_len < min_overlap_tokens:
        return None
    
    len_i = prof_i["region_end"] - prof_i["region_start"] + 1
    len_j = prof_j["region_end"] - prof_j["region_start"] + 1
    strength = overlap_len / max(1, min(len_i, len_j))
    
    if strength < min_overlap_strength:
        return None
    
    return (start, end, float(strength))


def _classify_crossing_v2(
    attn_matrix: np.ndarray,
    step_i: tuple[int, int],
    step_j: tuple[int, int],
    prof_i: dict,
    prof_j: dict,
    overlap_region: tuple[int, int],
) -> str:
    """
    Classify crossing as over/under using per-token temporal dynamics.
    
    We have the full per-token attention matrix, so we can look at
    exactly WHEN within step_j's generation the overlap region gets
    attended to.
    
    Over:  step_j references step_i's exclusive region BEFORE the overlap
           → forward substitution (used step_i's intermediate result)
    
    Under: step_j hits the overlap region BEFORE or without referencing
           step_i's exclusive region → independent read
    """
    overlap_start, overlap_end = overlap_region
    
    # Step i's exclusive region (parts of i's region NOT in the overlap)
    i_excl_start = prof_i["region_start"]
    i_excl_end = min(overlap_start - 1, prof_i["region_end"])
    
    has_exclusive = i_excl_end >= i_excl_start
    
    j_start, j_end = step_j
    j_tokens = attn_matrix[j_start:j_end]  # (n_j_tokens, n_input)
    
    if len(j_tokens) == 0:
        return "under"
    
    # For each token in step_j, compute attention mass on:
    # (a) the overlap region
    # (b) step_i's exclusive region
    overlap_mass = j_tokens[:, overlap_start:overlap_end + 1].sum(axis=1)
    
    if has_exclusive and i_excl_end >= i_excl_start:
        excl_mass = j_tokens[:, i_excl_start:i_excl_end + 1].sum(axis=1)
    else:
        # No exclusive region — can't determine directionality
        # Use attention magnitude comparison as fallback
        i_overlap_mass = prof_i["attn_map"][overlap_start:overlap_end + 1].sum()
        j_overlap_mass = prof_j["attn_map"][overlap_start:overlap_end + 1].sum()
        
        if j_overlap_mass > i_overlap_mass * 1.5:
            return "under"  # step_j reads overlap more heavily → independent
        return "over"       # step_j reads less → using intermediate
    
    # Find onset times (first token where mass exceeds 30% of peak)
    overlap_onset = _find_onset(overlap_mass)
    excl_onset = _find_onset(excl_mass)
    
    if overlap_onset is None and excl_onset is None:
        return "under"  # no strong attention to either → independent
    
    if excl_onset is not None and overlap_onset is not None:
        if excl_onset < overlap_onset:
            return "over"   # referenced step_i first, then processed overlap
        else:
            return "under"  # hit overlap first → independent read
    
    if excl_onset is not None and overlap_onset is None:
        return "over"   # referenced step_i but not overlap directly
    
    return "under"  # hit overlap but not step_i's exclusive region


def _find_onset(
    mass_sequence: np.ndarray,
    threshold_ratio: float = 0.3,
) -> Optional[int]:
    """
    Find earliest position where attention mass exceeds threshold.
    Returns token index or None.
    """
    if len(mass_sequence) == 0:
        return None
    
    peak = mass_sequence.max()
    if peak < 1e-6:
        return None
    
    threshold = peak * threshold_ratio
    indices = np.where(mass_sequence >= threshold)[0]
    
    return int(indices[0]) if len(indices) > 0 else None


# ---------------------------------------------------------------------------
# Knot invariants
# ---------------------------------------------------------------------------

def _compute_unknotting_number(crossings: list[dict], n_steps: int) -> int:
    """Minimum crossings to remove to linearize the dependency structure."""
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
        
        busiest_step = max(step_counts, key=lambda s: len(step_counts[s]))
        removed = set(step_counts[busiest_step])
        remaining = [i for i in remaining if i not in removed]
        cuts += 1
    
    return cuts


def _compute_knot_signature(crossings: list[dict]) -> str:
    """Canonical crossing sequence: O for over, U for under."""
    if not crossings:
        return ""
    
    sorted_crossings = sorted(crossings, key=lambda c: (c["step_i"], c["step_j"]))
    return "".join("O" if c["crossing_type"] == "over" else "U" for c in sorted_crossings)


def _empty_result(problem_id: str, n_steps: int) -> dict:
    return {
        "problem_id": problem_id,
        "n_generated_tokens": 0,
        "n_input_tokens": 0,
        "n_steps": n_steps,
        "n_crossings": 0,
        "crossing_number": 0,
        "unknotting_number": 0,
        "crossings": [],
        "knot_signature": "",
        "step_regions": [],
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """
    Process one IAF chunk from S3.
    
    Event:
    {
        "chunk_key": "iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_000.json",
        "output_prefix": "crossings/",
        "bucket": "mycelium-data"
    }
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "crossings/")
    
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    raw = response["Body"].read().decode("utf-8")
    chunk_data = json.loads(raw)
    
    # Handle both list and dict formats
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])
    
    results = []
    stats = {
        "total": 0,
        "unknotted": 0,
        "knotted": 0,
        "max_crossings": 0,
        "crossing_type_counts": {"over": 0, "under": 0},
        "signature_counts": defaultdict(int),
        "n_steps_histogram": defaultdict(int),
        "crossing_number_histogram": defaultdict(int),
        "avg_input_tokens": 0,
        "avg_generated_tokens": 0,
    }
    
    total_input_tokens = 0
    total_gen_tokens = 0
    
    for idx, problem in enumerate(problems):
        # Give each problem an ID if it doesn't have one
        if "problem_idx" not in problem and "problem_id" not in problem:
            problem["problem_idx"] = f"{chunk_key}_{idx}"
        
        result = detect_crossings(problem)
        results.append(result)
        
        # Accumulate stats
        stats["total"] += 1
        total_input_tokens += result.get("n_input_tokens", 0)
        total_gen_tokens += result.get("n_generated_tokens", 0)
        stats["n_steps_histogram"][result["n_steps"]] += 1
        stats["crossing_number_histogram"][result["crossing_number"]] += 1
        
        if result["n_crossings"] == 0:
            stats["unknotted"] += 1
        else:
            stats["knotted"] += 1
            stats["max_crossings"] = max(stats["max_crossings"], result["n_crossings"])
            for c in result["crossings"]:
                stats["crossing_type_counts"][c["crossing_type"]] += 1
        
        stats["signature_counts"][result["knot_signature"]] += 1
    
    if stats["total"] > 0:
        stats["avg_input_tokens"] = total_input_tokens / stats["total"]
        stats["avg_generated_tokens"] = total_gen_tokens / stats["total"]
    
    # Convert defaultdicts for JSON
    stats["signature_counts"] = dict(stats["signature_counts"])
    stats["n_steps_histogram"] = {str(k): v for k, v in stats["n_steps_histogram"].items()}
    stats["crossing_number_histogram"] = {str(k): v for k, v in stats["crossing_number_histogram"].items()}
    
    # Upload results
    chunk_name = chunk_key.split("/")[-1]
    output_key = f"{output_prefix}{chunk_name}"
    
    output = {
        "chunk_key": chunk_key,
        "stats": stats,
        "results": results,
    }
    
    print(
        f"Uploading s3://{bucket}/{output_key} | "
        f"{len(results)} problems | "
        f"{stats['knotted']} knotted | "
        f"max_cross={stats['max_crossings']}"
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
