"""
Lambda function: IAF Knot Crossing Detection (Map phase) — v2 Pure Python

Adapted to actual IAF data format, no numpy dependency for Lambda.
"""

import json
import re
import math
import boto3
from collections import defaultdict
from typing import Optional


s3 = boto3.client("s3")

BUCKET = "mycelium-data"
MIN_LAYER = 14  # ignore heads below this (syntactic, not semantic)


# ---------------------------------------------------------------------------
# Pure Python math helpers
# ---------------------------------------------------------------------------

def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _zeros_2d(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _sum(arr: list[float]) -> float:
    return sum(arr)


def _mean(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0


def _std(arr: list[float]) -> float:
    if len(arr) < 2:
        return 0.0
    m = _mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))


def _argmax(arr: list[float]) -> int:
    if not arr:
        return 0
    max_val = arr[0]
    max_idx = 0
    for i, v in enumerate(arr):
        if v > max_val:
            max_val = v
            max_idx = i
    return max_idx


def _max(arr: list[float]) -> float:
    return max(arr) if arr else 0.0


def _quantile(arr: list[float], q: float) -> float:
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    idx = q * (len(sorted_arr) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_arr) - 1)
    frac = idx - lower
    return sorted_arr[lower] * (1 - frac) + sorted_arr[upper] * frac


def _where_ge(arr: list[float], threshold: float) -> list[int]:
    """Return indices where arr[i] >= threshold."""
    return [i for i, v in enumerate(arr) if v >= threshold]


def _slice_sum(arr: list[float], start: int, end: int) -> float:
    """Sum arr[start:end+1]."""
    return sum(arr[start:end + 1])


def _row_slice_sum(matrix: list[list[float]], row: int, col_start: int, col_end: int) -> float:
    """Sum matrix[row][col_start:col_end+1]."""
    return sum(matrix[row][col_start:col_end + 1])


def _mean_axis0(matrix: list[list[float]]) -> list[float]:
    """Compute column-wise mean of a 2D matrix."""
    if not matrix:
        return []
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    return [sum(matrix[r][c] for r in range(n_rows)) / n_rows for c in range(n_cols)]


def _col_sums(matrix: list[list[float]], col_start: int, col_end: int) -> list[float]:
    """For each row, sum columns from col_start to col_end inclusive."""
    return [sum(row[col_start:col_end + 1]) for row in matrix]


# ---------------------------------------------------------------------------
# 1. Reconstruct attention from sparse top_positions
# ---------------------------------------------------------------------------

def reconstruct_token_attention(
    top_positions_entry: dict,
    n_input: int,
) -> list[float]:
    """
    Reconstruct dense attention vector over input from sparse top_positions.
    """
    attn = _zeros(n_input)
    total_head_weight = 0.0

    for head_key, positions in top_positions_entry.items():
        match = re.match(r"L(\d+)H(\d+)", head_key)
        if not match:
            continue

        layer_idx = int(match.group(1))
        if layer_idx < MIN_LAYER:
            continue

        layer_weight = (layer_idx - MIN_LAYER + 1) / 15.0

        for entry in positions:
            pos = entry.get("pos", 0)
            weight = entry.get("weight", 0.0)

            if 0 <= pos < n_input:
                attn[pos] += weight * layer_weight

        total_head_weight += layer_weight

    if total_head_weight > 0:
        attn = [v / total_head_weight for v in attn]

    return attn


def reconstruct_all_attention(
    top_positions: list[dict],
    n_input: int,
) -> list[list[float]]:
    """
    Reconstruct attention for all generated tokens.
    Returns: (n_generated, n_input) attention matrix as nested lists.
    """
    return [reconstruct_token_attention(entry, n_input) for entry in top_positions]


# ---------------------------------------------------------------------------
# 2. Segment CoT into computation steps
# ---------------------------------------------------------------------------

def segment_by_attention_shift(
    attn_matrix: list[list[float]],
    min_step_tokens: int = 5,
    shift_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Detect step boundaries from attention shifts using JSD.
    """
    n_gen = len(attn_matrix)
    if n_gen < 2:
        return [(0, n_gen)]

    n_input = len(attn_matrix[0]) if attn_matrix else 0

    # Compute JSD between consecutive tokens
    jsd_values = []
    for t in range(n_gen - 1):
        p = [v + 1e-10 for v in attn_matrix[t]]
        q = [v + 1e-10 for v in attn_matrix[t + 1]]

        p_sum = sum(p)
        q_sum = sum(q)
        p = [v / p_sum for v in p]
        q = [v / q_sum for v in q]

        # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q)
        m = [0.5 * (p[i] + q[i]) for i in range(n_input)]

        kl_pm = sum(p[i] * math.log(p[i] / m[i]) for i in range(n_input) if p[i] > 0 and m[i] > 0)
        kl_qm = sum(q[i] * math.log(q[i] / m[i]) for i in range(n_input) if q[i] > 0 and m[i] > 0)
        jsd = 0.5 * (kl_pm + kl_qm)
        jsd_values.append(jsd)

    # Adaptive threshold
    if jsd_values:
        threshold = max(shift_threshold, _mean(jsd_values) + 1.0 * _std(jsd_values))
    else:
        threshold = shift_threshold

    # Find boundaries
    boundaries = [0]
    for t, jsd in enumerate(jsd_values):
        if jsd > threshold:
            if (t + 1) - boundaries[-1] >= min_step_tokens:
                boundaries.append(t + 1)
    boundaries.append(n_gen)

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# 3. Core crossing detection
# ---------------------------------------------------------------------------

def detect_crossings(problem: dict) -> dict:
    """Detect knot crossings in a single problem's IAF data."""
    top_positions = problem.get("top_positions", [])
    problem_id = str(problem.get("problem_idx", problem.get("problem_id", "unknown")))

    if not top_positions:
        return _empty_result(problem_id, 0)

    n_gen = len(top_positions)

    # Determine n_input from data
    max_pos = 0
    for entry in top_positions:
        for head_key, positions in entry.items():
            for p in positions:
                max_pos = max(max_pos, p.get("pos", 0))
    n_input = max_pos + 1

    if n_input < 2:
        return _empty_result(problem_id, 0)

    # Reconstruct full attention matrix
    attn_matrix = reconstruct_all_attention(top_positions, n_input)

    # Segment into computation steps
    steps = segment_by_attention_shift(attn_matrix)

    if len(steps) < 2:
        return _empty_result(problem_id, len(steps))

    # Compute per-step attention profiles
    step_profiles = []
    for step_start, step_end in steps:
        step_rows = attn_matrix[step_start:step_end]
        step_attn = _mean_axis0(step_rows)

        region = _find_dominant_region(step_attn)

        step_profiles.append({
            "step_range": (step_start, step_end),
            "attn_map": step_attn,
            "region_start": region[0],
            "region_end": region[1],
            "attn_mass": float(_slice_sum(step_attn, region[0], region[1])),
            "n_tokens": step_end - step_start,
        })

    # Detect crossings
    crossings = []

    for i in range(len(step_profiles)):
        for j in range(i + 1, len(step_profiles)):
            prof_i = step_profiles[i]
            prof_j = step_profiles[j]

            overlap = _compute_overlap(prof_i, prof_j)
            if overlap is None:
                continue

            overlap_start, overlap_end, overlap_strength = overlap

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
    attn_map: list[float],
    threshold_quantile: float = 0.75,
    min_region_len: int = 1,
) -> tuple[int, int]:
    """Find dominant contiguous attention region."""
    if _sum(attn_map) < 1e-10:
        return (0, max(0, len(attn_map) - 1))

    threshold = _quantile(attn_map, threshold_quantile)
    if threshold < 1e-10:
        peak = _argmax(attn_map)
        return (max(0, peak - 2), min(len(attn_map) - 1, peak + 2))

    above = [1 if v >= threshold else 0 for v in attn_map]

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
        peak = _argmax(attn_map)
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
    """Compute overlap between two step attention regions."""
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
    attn_matrix: list[list[float]],
    step_i: tuple[int, int],
    step_j: tuple[int, int],
    prof_i: dict,
    prof_j: dict,
    overlap_region: tuple[int, int],
) -> str:
    """Classify crossing as over/under using per-token temporal dynamics."""
    overlap_start, overlap_end = overlap_region

    i_excl_start = prof_i["region_start"]
    i_excl_end = min(overlap_start - 1, prof_i["region_end"])
    has_exclusive = i_excl_end >= i_excl_start

    j_start, j_end = step_j
    j_tokens = attn_matrix[j_start:j_end]

    if len(j_tokens) == 0:
        return "under"

    # Compute attention mass on overlap and exclusive regions per token
    overlap_mass = _col_sums(j_tokens, overlap_start, overlap_end)

    if has_exclusive:
        excl_mass = _col_sums(j_tokens, i_excl_start, i_excl_end)
    else:
        i_overlap_mass = _slice_sum(prof_i["attn_map"], overlap_start, overlap_end)
        j_overlap_mass = _slice_sum(prof_j["attn_map"], overlap_start, overlap_end)

        if j_overlap_mass > i_overlap_mass * 1.5:
            return "under"
        return "over"

    overlap_onset = _find_onset(overlap_mass)
    excl_onset = _find_onset(excl_mass)

    if overlap_onset is None and excl_onset is None:
        return "under"

    if excl_onset is not None and overlap_onset is not None:
        if excl_onset < overlap_onset:
            return "over"
        else:
            return "under"

    if excl_onset is not None and overlap_onset is None:
        return "over"

    return "under"


def _find_onset(
    mass_sequence: list[float],
    threshold_ratio: float = 0.3,
) -> Optional[int]:
    """Find earliest position where attention mass exceeds threshold."""
    if not mass_sequence:
        return None

    peak = _max(mass_sequence)
    if peak < 1e-6:
        return None

    threshold = peak * threshold_ratio
    indices = _where_ge(mass_sequence, threshold)

    return indices[0] if indices else None


# ---------------------------------------------------------------------------
# Knot invariants
# ---------------------------------------------------------------------------

def _compute_unknotting_number(crossings: list[dict], n_steps: int) -> int:
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
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "crossings_v2/")

    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    raw = response["Body"].read().decode("utf-8")
    chunk_data = json.loads(raw)

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
        if "problem_idx" not in problem and "problem_id" not in problem:
            problem["problem_idx"] = f"{chunk_key}_{idx}"

        result = detect_crossings(problem)
        results.append(result)

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

    stats["signature_counts"] = dict(stats["signature_counts"])
    stats["n_steps_histogram"] = {str(k): v for k, v in stats["n_steps_histogram"].items()}
    stats["crossing_number_histogram"] = {str(k): v for k, v in stats["crossing_number_histogram"].items()}

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
