"""
Lambda function: IAF Knot Crossing Detection (Map phase)

Processes one chunk of IAF data from S3, detects crossings in the
teacher's attention trace, and classifies crossing type (over/under).

A "crossing" occurs when two CoT steps attend to overlapping regions
of the input problem text. The temporal order of attention peaks
determines over vs under.

Input:  s3://mycelium-data/iaf/{chunk_id}.json  (200MB chunks)
Output: s3://mycelium-data/crossings/{chunk_id}.json

Memory: 3GB Lambda
"""

import json
import boto3
from collections import defaultdict
from typing import Optional


# Pure Python replacements for numpy operations
def _mean_axis0(vectors: list[list[float]]) -> list[float]:
    """Compute mean along axis 0 (column-wise mean)."""
    if not vectors:
        return []
    n_cols = len(vectors[0])
    n_rows = len(vectors)
    return [sum(v[i] for v in vectors) / n_rows for i in range(n_cols)]


def _quantile(arr: list[float], q: float) -> float:
    """Compute quantile of a list."""
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    idx = q * (len(sorted_arr) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_arr) - 1)
    frac = idx - lower
    return sorted_arr[lower] * (1 - frac) + sorted_arr[upper] * frac


s3 = boto3.client("s3")

BUCKET = "mycelium-data"


# ---------------------------------------------------------------------------
# Core crossing detection
# ---------------------------------------------------------------------------

def detect_crossings(iaf_trace: dict) -> dict:
    """
    Detect knot crossings in a single problem's IAF trace.
    
    IAF trace structure (per problem):
    {
        "problem_id": str,
        "problem_text": str,
        "cot_steps": [
            {
                "step_idx": int,
                "tokens": [str],           # generated CoT tokens
                "iaf_per_token": [          # per generated token:
                    {
                        "token_idx": int,
                        "input_attention": [float]  # attention over input tokens
                    }
                ]
            }
        ],
        "input_tokens": [str],
        "span_boundaries": [(int, int)],   # JSD-derived span boundaries
    }
    
    Returns:
    {
        "problem_id": str,
        "n_steps": int,
        "n_crossings": int,
        "crossing_number": int,            # topological crossing number
        "unknotting_number": int,           # min cuts to linearize
        "crossings": [
            {
                "step_i": int,
                "step_j": int,
                "overlap_region": (int, int),
                "crossing_type": "over" | "under",
                "overlap_strength": float,
                "temporal_gap": int,
            }
        ],
        "knot_signature": str,             # canonical crossing sequence
        "attention_regions": [             # per-step dominant attention region
            {"step": int, "start": int, "end": int, "peak_token_idx": int}
        ],
    }
    """
    steps = iaf_trace.get("cot_steps", [])
    n_input = len(iaf_trace.get("input_tokens", []))
    
    if not steps or n_input == 0:
        return _empty_result(iaf_trace.get("problem_id", "unknown"), len(steps))
    
    # --- Step 1: Compute dominant attention region per CoT step ---
    # For each step, aggregate per-token IAF into a step-level attention map
    # over the input, then find the dominant contiguous region.
    
    step_regions = []
    step_attention_maps = []
    
    for step in steps:
        iaf_tokens = step.get("iaf_per_token", [])
        if not iaf_tokens:
            step_regions.append(None)
            step_attention_maps.append(None)
            continue
        
        # Aggregate: mean attention over all generated tokens in this step
        attn_vectors = []
        for tok_data in iaf_tokens:
            attn = tok_data.get("input_attention", [])
            if len(attn) == n_input:
                attn_vectors.append(attn)
        
        if not attn_vectors:
            step_regions.append(None)
            step_attention_maps.append(None)
            continue
        
        # Step-level attention map over input
        attn_map = _mean_axis0(attn_vectors)  # (n_input,)
        step_attention_maps.append(attn_map)
        
        # Find dominant region: contiguous span above threshold
        region = _find_dominant_region(attn_map, threshold_quantile=0.75)
        
        # Also track peak token index (which generated token had max attention)
        peak_token_idx = _find_peak_token(iaf_tokens, region, n_input)
        
        step_regions.append({
            "step": step["step_idx"],
            "start": region[0],
            "end": region[1],
            "peak_token_idx": peak_token_idx,
            "attn_mass": float(sum(attn_map[region[0]:region[1] + 1])),
        })
    
    # --- Step 2: Detect crossings (overlapping attention regions) ---
    
    crossings = []
    valid_regions = [(i, r) for i, r in enumerate(step_regions) if r is not None]
    
    for idx_a in range(len(valid_regions)):
        for idx_b in range(idx_a + 1, len(valid_regions)):
            i, region_i = valid_regions[idx_a]
            j, region_j = valid_regions[idx_b]
            
            # Check overlap
            overlap = _compute_overlap(region_i, region_j)
            if overlap is None:
                continue
            
            overlap_start, overlap_end, overlap_strength = overlap
            
            # --- Step 3: Classify crossing type (over/under) ---
            # "Over": step_j resolves the shared region using step_i's
            #         intermediate result (forward substitution)
            # "Under": step_j re-reads the original span independently
            #          (parallel computation, later merged)
            #
            # Signal: look at step_j's per-token IAF sequence in the 
            # overlap region. If attention peaks AFTER step_j has already
            # attended to step_i's result region → "over" (it used the
            # intermediate). If attention peaks EARLY before any intermediate
            # references → "under" (independent read).
            
            crossing_type = _classify_crossing(
                step_i_data=steps[i],
                step_j_data=steps[j],
                overlap_region=(overlap_start, overlap_end),
                step_i_region=region_i,
                step_attention_maps=step_attention_maps,
                i=i, j=j,
                n_input=n_input,
            )
            
            crossings.append({
                "step_i": steps[i]["step_idx"],
                "step_j": steps[j]["step_idx"],
                "overlap_region": (overlap_start, overlap_end),
                "crossing_type": crossing_type,
                "overlap_strength": overlap_strength,
                "temporal_gap": j - i,
            })
    
    # --- Step 4: Compute knot invariants ---
    
    crossing_number = len(crossings)
    unknotting_number = _compute_unknotting_number(crossings, len(steps))
    knot_signature = _compute_knot_signature(crossings)
    
    return {
        "problem_id": iaf_trace.get("problem_id", "unknown"),
        "n_steps": len(steps),
        "n_crossings": crossing_number,
        "crossing_number": crossing_number,
        "unknotting_number": unknotting_number,
        "crossings": crossings,
        "knot_signature": knot_signature,
        "attention_regions": [r for r in step_regions if r is not None],
    }


# ---------------------------------------------------------------------------
# Attention region analysis
# ---------------------------------------------------------------------------

def _find_dominant_region(
    attn_map: list[float],
    threshold_quantile: float = 0.75,
) -> tuple[int, int]:
    """
    Find the dominant contiguous attention region.
    
    Uses quantile thresholding + largest connected component.
    Learned threshold from data distribution, not hardcoded
    (the quantile adapts to each step's attention shape).
    """
    threshold = _quantile(attn_map, threshold_quantile)
    above = (attn_map >= threshold).astype(int)
    
    # Find largest contiguous region above threshold
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
    
    # Handle region extending to end
    if curr_start is not None:
        length = len(above) - curr_start
        if length > best_len:
            best_start, best_end = curr_start, len(above) - 1
    
    return (best_start, best_end)


def _find_peak_token(
    iaf_tokens: list[dict],
    region: tuple[int, int],
    n_input: int,
) -> int:
    """Find which generated token has maximum attention in the region."""
    best_idx = 0
    best_mass = 0.0
    
    for tok_data in iaf_tokens:
        attn = tok_data.get("input_attention", [])
        if len(attn) != n_input:
            continue
        mass = sum(attn[region[0]:region[1] + 1])
        if mass > best_mass:
            best_mass = mass
            best_idx = tok_data["token_idx"]
    
    return best_idx


# ---------------------------------------------------------------------------
# Crossing detection and classification
# ---------------------------------------------------------------------------

def _compute_overlap(
    region_i: dict,
    region_j: dict,
    min_overlap_tokens: int = 2,
) -> Optional[tuple[int, int, float]]:
    """
    Compute overlap between two attention regions.
    
    Returns (start, end, strength) or None if no meaningful overlap.
    Strength = overlap length / min(region_i length, region_j length).
    """
    start = max(region_i["start"], region_j["start"])
    end = min(region_i["end"], region_j["end"])
    
    overlap_len = end - start + 1
    if overlap_len < min_overlap_tokens:
        return None
    
    len_i = region_i["end"] - region_i["start"] + 1
    len_j = region_j["end"] - region_j["start"] + 1
    strength = overlap_len / min(len_i, len_j)
    
    return (start, end, strength)


def _classify_crossing(
    step_i_data: dict,
    step_j_data: dict,
    overlap_region: tuple[int, int],
    step_i_region: dict,
    step_attention_maps: list,
    i: int,
    j: int,
    n_input: int,
) -> str:
    """
    Classify a crossing as "over" or "under" based on temporal IAF dynamics.
    
    Over:  step_j attends to the overlap region AFTER it has already referenced
           step_i's output region. This means step_j is using step_i's 
           intermediate result → forward substitution → the strand goes "over."
    
    Under: step_j attends to the overlap region BEFORE or independently of 
           step_i's output. This means step_j reads the original input 
           independently → parallel computation → the strand goes "under."
    
    Signal: temporal position of overlap attention peak within step_j's
            generation sequence, relative to step_i reference attention.
    """
    j_tokens = step_j_data.get("iaf_per_token", [])
    if not j_tokens:
        return "under"  # default: no data → assume independent
    
    overlap_start, overlap_end = overlap_region
    
    # Find when step_j first strongly attends to the overlap region
    overlap_peak_time = _find_attention_onset(
        j_tokens, overlap_start, overlap_end, n_input
    )
    
    # Find when step_j references step_i's output region
    # (if step_i produced a result, step_j might attend to where step_i
    # wrote its output in the CoT — this shows up as attention to CoT tokens
    # that were generated by step_i)
    #
    # Since we only have input attention (not self-attention over CoT),
    # we use a proxy: does step_j attend to step_i's FULL region
    # (not just overlap) at a different time than the overlap?
    
    i_region_start = step_i_region["start"]
    i_region_end = step_i_region["end"]
    
    # Attention to step_i's non-overlapping region
    i_exclusive_start = i_region_start
    i_exclusive_end = min(overlap_start - 1, i_region_end)
    
    if i_exclusive_end > i_exclusive_start:
        ref_peak_time = _find_attention_onset(
            j_tokens, i_exclusive_start, i_exclusive_end, n_input
        )
        
        # Over: step_j references step_i's region BEFORE the overlap
        # (it read step_i's output first, then processed the shared region)
        if ref_peak_time is not None and overlap_peak_time is not None:
            if ref_peak_time < overlap_peak_time:
                return "over"
            else:
                return "under"
    
    # Fallback: use attention mass ratio
    # If step_j has higher attention on the overlap than step_i did,
    # it's likely re-reading independently (under).
    # If lower, it's likely just referencing through step_i (over).
    if step_attention_maps[i] is not None and step_attention_maps[j] is not None:
        i_mass = sum(step_attention_maps[i][overlap_start:overlap_end + 1])
        j_mass = sum(step_attention_maps[j][overlap_start:overlap_end + 1])
        
        if i_mass > 0 and j_mass / (i_mass + 1e-8) > 1.5:
            return "under"  # step_j reads more heavily → independent
        else:
            return "over"   # step_j reads less → using intermediate
    
    return "under"  # conservative default


def _find_attention_onset(
    iaf_tokens: list[dict],
    region_start: int,
    region_end: int,
    n_input: int,
    threshold_ratio: float = 0.5,
) -> Optional[int]:
    """
    Find the earliest generated token that strongly attends to a region.
    
    Returns the token index, or None if no strong attention found.
    "Strong" = attention mass in region exceeds threshold_ratio of peak.
    """
    # First find the peak attention mass across all tokens
    peak_mass = 0.0
    for tok_data in iaf_tokens:
        attn = tok_data.get("input_attention", [])
        if len(attn) != n_input:
            continue
        mass = sum(attn[region_start:region_end + 1])
        peak_mass = max(peak_mass, mass)
    
    if peak_mass < 1e-6:
        return None
    
    # Find earliest token exceeding threshold
    threshold = peak_mass * threshold_ratio
    for tok_data in iaf_tokens:
        attn = tok_data.get("input_attention", [])
        if len(attn) != n_input:
            continue
        mass = sum(attn[region_start:region_end + 1])
        if mass >= threshold:
            return tok_data["token_idx"]
    
    return None


# ---------------------------------------------------------------------------
# Knot invariants
# ---------------------------------------------------------------------------

def _compute_unknotting_number(crossings: list[dict], n_steps: int) -> int:
    """
    Estimate unknotting number: minimum crossings to remove to make
    the dependency graph acyclic / linear.
    
    Uses greedy approach: iteratively remove the crossing that 
    eliminates the most other crossings (shared overlap regions
    create dependent crossing clusters).
    """
    if not crossings:
        return 0
    
    # Build crossing dependency graph
    # Two crossings are "linked" if they share a step
    remaining = list(range(len(crossings)))
    cuts = 0
    
    while remaining:
        # Find step involved in most remaining crossings
        step_counts = defaultdict(list)
        for idx in remaining:
            c = crossings[idx]
            step_counts[c["step_i"]].append(idx)
            step_counts[c["step_j"]].append(idx)
        
        if not step_counts:
            break
        
        # Cut the most-connected step
        busiest_step = max(step_counts, key=lambda s: len(step_counts[s]))
        removed = set(step_counts[busiest_step])
        remaining = [i for i in remaining if i not in removed]
        cuts += 1
    
    return cuts


def _compute_knot_signature(crossings: list[dict]) -> str:
    """
    Compute canonical knot signature from crossing sequence.
    
    Format: "O" for over, "U" for under, ordered by (step_i, step_j).
    Examples:
      ""        → unknot (trivial, no crossings)
      "O"       → trefoil-like (one over crossing)
      "OU"      → figure-eight-like (one over, one under)
      "OOO"     → complex coupled system
    
    This signature is the knot invariant that gets predicted by C2.
    Problems with the same signature have isomorphic computational topology.
    """
    if not crossings:
        return ""
    
    # Sort by temporal order (step_i, then step_j)
    sorted_crossings = sorted(crossings, key=lambda c: (c["step_i"], c["step_j"]))
    
    # Build signature string
    sig = ""
    for c in sorted_crossings:
        sig += "O" if c["crossing_type"] == "over" else "U"
    
    return sig


def _empty_result(problem_id: str, n_steps: int) -> dict:
    return {
        "problem_id": problem_id,
        "n_steps": n_steps,
        "n_crossings": 0,
        "crossing_number": 0,
        "unknotting_number": 0,
        "crossings": [],
        "knot_signature": "",
        "attention_regions": [],
    }


# ---------------------------------------------------------------------------
# Lambda handler (Map phase)
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """
    Process one IAF chunk from S3.
    
    Event:
    {
        "chunk_key": "iaf/chunk_000.json",
        "output_prefix": "crossings/",
        "bucket": "mycelium-data"
    }
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "crossings/")
    
    # Download chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    
    # Process each problem
    results = []
    stats = {
        "total": 0,
        "unknotted": 0,
        "knotted": 0,
        "max_crossings": 0,
        "crossing_type_counts": {"over": 0, "under": 0},
        "signature_counts": defaultdict(int),
    }
    
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])
    
    for problem in problems:
        result = detect_crossings(problem)
        results.append(result)
        
        # Accumulate stats
        stats["total"] += 1
        if result["n_crossings"] == 0:
            stats["unknotted"] += 1
        else:
            stats["knotted"] += 1
            stats["max_crossings"] = max(stats["max_crossings"], result["n_crossings"])
            for c in result["crossings"]:
                stats["crossing_type_counts"][c["crossing_type"]] += 1
        
        stats["signature_counts"][result["knot_signature"]] += 1
    
    # Convert defaultdict for JSON serialization
    stats["signature_counts"] = dict(stats["signature_counts"])
    
    # Upload results
    chunk_name = chunk_key.split("/")[-1]
    output_key = f"{output_prefix}{chunk_name}"
    
    output = {
        "chunk_key": chunk_key,
        "stats": stats,
        "results": results,
    }
    
    print(f"Uploading s3://{bucket}/{output_key} ({len(results)} problems)")
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(output, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    
    # Upload stats separately for fast reduce
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
