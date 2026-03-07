"""
Lambda function: Attention Head Telegraph Analysis (Map phase)

Analyzes the 10 pre-selected attention heads in IAF data to determine
which exhibit clean telegraph behavior (alternating read/compute states).

For each head, computes:
- Reading signal: sum of attention to input tokens per decode step
- Bimodality: how well the signal separates into two states
- Contrast ratio: separation between high/low states
- Transition statistics: frequency, dwell times, regularity
- Cross-head correlations and synchronization

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/head_analysis/per_chunk_stats/*.json

Memory: 3GB Lambda
"""

import json
import boto3
import numpy as np
from collections import defaultdict


s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# The 10 pre-selected heads in the IAF data
HEAD_KEYS = [
    "L5H19", "L14H0", "L22H3", "L22H4", "L23H11",
    "L23H23", "L24H4", "L24H6", "L24H16", "L25H1"
]
N_HEADS = len(HEAD_KEYS)


def extract_reading_signals(problem: dict) -> dict:
    """
    Extract reading signal time series for each head.

    reading_signal[head, step] = sum(attention weights to input tokens)

    High value = head is reading input
    Low value = head is attending to generated tokens (computing)
    """
    top_positions = problem.get("top_positions", [])
    n_steps = len(top_positions)

    if n_steps == 0:
        return None

    # Initialize signals for all heads
    signals = {head: np.zeros(n_steps, dtype=np.float64) for head in HEAD_KEYS}

    for step, entry in enumerate(top_positions):
        for head_key in HEAD_KEYS:
            if head_key in entry:
                positions = entry[head_key]
                total_weight = sum(p.get("weight", 0.0) for p in positions)
                signals[head_key][step] = total_weight

    return signals


def compute_bimodality(signal: np.ndarray) -> dict:
    """
    Compute bimodality metrics for a signal.

    Uses coefficient of bimodality: (skewness^2 + 1) / kurtosis
    Values > 0.555 suggest bimodality.

    Also computes simple two-mode statistics using median split.
    """
    if len(signal) < 10:
        return {"bimodal_coef": 0.0, "mode_separation": 0.0, "low_mean": 0.0, "high_mean": 0.0}

    # Remove zeros (steps where head wasn't active)
    active = signal[signal > 1e-6]
    if len(active) < 10:
        return {"bimodal_coef": 0.0, "mode_separation": 0.0, "low_mean": 0.0, "high_mean": 0.0}

    # Bimodality coefficient
    n = len(active)
    mean = np.mean(active)
    std = np.std(active)

    if std < 1e-10:
        return {"bimodal_coef": 0.0, "mode_separation": 0.0, "low_mean": mean, "high_mean": mean}

    skewness = np.mean(((active - mean) / std) ** 3)
    kurtosis = np.mean(((active - mean) / std) ** 4)

    # Bimodality coefficient (adjusted for sample size)
    if kurtosis > 0:
        bimodal_coef = (skewness ** 2 + 1) / kurtosis
    else:
        bimodal_coef = 0.0

    # Two-mode statistics using median split
    median = np.median(active)
    low_mode = active[active <= median]
    high_mode = active[active > median]

    low_mean = np.mean(low_mode) if len(low_mode) > 0 else 0.0
    high_mean = np.mean(high_mode) if len(high_mode) > 0 else 0.0

    # Mode separation (normalized)
    mode_separation = (high_mean - low_mean) / (high_mean + low_mean + 1e-10)

    return {
        "bimodal_coef": float(bimodal_coef),
        "mode_separation": float(mode_separation),
        "low_mean": float(low_mean),
        "high_mean": float(high_mean),
    }


def compute_contrast_ratio(signal: np.ndarray) -> float:
    """
    Contrast ratio: mean(above median) / mean(below median)
    Higher = cleaner separation between states.
    """
    active = signal[signal > 1e-6]
    if len(active) < 4:
        return 1.0

    median = np.median(active)
    below = active[active <= median]
    above = active[active > median]

    if len(below) == 0 or len(above) == 0:
        return 1.0

    mean_below = np.mean(below)
    mean_above = np.mean(above)

    if mean_below < 1e-10:
        return float(mean_above) if mean_above > 0 else 1.0

    return float(mean_above / mean_below)


def compute_transitions(signal: np.ndarray) -> dict:
    """
    Compute transition statistics for telegraph behavior.

    Uses Otsu's threshold (not median) to separate high/low states.
    This captures actual bimodal structure rather than arbitrary 50/50 split.

    - frac_high: fraction of time in high state (above Otsu threshold)
    - state_balance: 1 - |frac_high - 0.5| * 2 (1.0 = perfect 50/50, 0.0 = all one state)
    - low_mean, high_mean: mean signal in each state
    """
    active = signal[signal > 1e-6]
    if len(active) < 4:
        return {"n_transitions": 0, "mean_dwell": 0.0, "dwell_variance": 0.0,
                "transition_rate": 0.0, "frac_high": 0.5, "state_balance": 1.0,
                "low_mean": 0.0, "high_mean": 0.0, "otsu_threshold": 0.5}

    # Otsu's method to find optimal threshold
    threshold = _otsu_threshold(active)
    binary = (active > threshold).astype(int)

    # Fraction in high state (using Otsu threshold, not median)
    frac_high = float(np.mean(binary))
    state_balance = 1.0 - abs(frac_high - 0.5) * 2

    # Mean signal in each state
    low_vals = active[active <= threshold]
    high_vals = active[active > threshold]
    low_mean = float(np.mean(low_vals)) if len(low_vals) > 0 else 0.0
    high_mean = float(np.mean(high_vals)) if len(high_vals) > 0 else 0.0

    # Count transitions
    transitions = np.sum(np.abs(np.diff(binary)))

    # Compute dwell times (run lengths)
    dwells = []
    current_run = 1
    for i in range(1, len(binary)):
        if binary[i] == binary[i-1]:
            current_run += 1
        else:
            dwells.append(current_run)
            current_run = 1
    dwells.append(current_run)

    dwells = np.array(dwells)
    mean_dwell = float(np.mean(dwells))
    dwell_variance = float(np.var(dwells)) if len(dwells) > 1 else 0.0

    # Transition rate (per step)
    transition_rate = float(transitions / max(len(active) - 1, 1))

    return {
        "n_transitions": int(transitions),
        "mean_dwell": mean_dwell,
        "dwell_variance": dwell_variance,
        "transition_rate": transition_rate,
        "frac_high": frac_high,
        "state_balance": state_balance,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "otsu_threshold": float(threshold),
    }


def _otsu_threshold(values: np.ndarray, n_bins: int = 50) -> float:
    """
    Otsu's method: find threshold that maximizes between-class variance.
    Returns the threshold that best separates bimodal distribution.
    """
    if len(values) < 2:
        return 0.5

    # Normalize to [0, 1]
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-10:
        return (v_min + v_max) / 2

    normalized = (values - v_min) / (v_max - v_min)

    hist, bin_edges = np.histogram(normalized, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()

    if total == 0:
        return (v_min + v_max) / 2

    best_threshold = 0.5
    best_variance = 0.0

    cum_sum = 0.0
    cum_weight = 0

    total_sum = (hist * bin_centers).sum()

    for i in range(n_bins - 1):
        cum_weight += hist[i]
        if cum_weight == 0:
            continue

        bg_weight = total - cum_weight
        if bg_weight == 0:
            break

        cum_sum += hist[i] * bin_centers[i]

        mean_fg = cum_sum / cum_weight
        mean_bg = (total_sum - cum_sum) / bg_weight

        # Between-class variance
        between_var = cum_weight * bg_weight * (mean_fg - mean_bg) ** 2

        if between_var > best_variance:
            best_variance = between_var
            best_threshold = bin_centers[i]

    # Convert back to original scale
    return v_min + best_threshold * (v_max - v_min)


def compute_head_correlation(signals: dict, head1: str, head2: str) -> float:
    """Pearson correlation between two heads' reading signals."""
    s1 = signals[head1]
    s2 = signals[head2]

    # Only compare where both are active
    mask = (s1 > 1e-6) & (s2 > 1e-6)
    if np.sum(mask) < 10:
        return 0.0

    s1_active = s1[mask]
    s2_active = s2[mask]

    # Pearson correlation
    if np.std(s1_active) < 1e-10 or np.std(s2_active) < 1e-10:
        return 0.0

    corr = np.corrcoef(s1_active, s2_active)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_sync_score(signals: dict, head1: str, head2: str) -> float:
    """
    Synchronization score: fraction of transitions that co-occur within ±1 step.
    """
    s1 = signals[head1]
    s2 = signals[head2]

    # Find transitions for each head
    def get_transitions(signal):
        active = signal[signal > 1e-6]
        if len(active) < 4:
            return []
        median = np.median(active)
        binary = (signal > median).astype(int)
        trans_idx = np.where(np.abs(np.diff(binary)) > 0)[0]
        return trans_idx.tolist()

    t1 = set(get_transitions(s1))
    t2 = set(get_transitions(s2))

    if len(t1) == 0 or len(t2) == 0:
        return 0.0

    # Count co-occurring transitions (within ±1 step)
    co_occur = 0
    for t in t1:
        if t in t2 or (t-1) in t2 or (t+1) in t2:
            co_occur += 1

    # Normalize by smaller set
    sync = co_occur / min(len(t1), len(t2))
    return float(sync)


def analyze_problem(problem: dict) -> dict:
    """
    Analyze one problem, return per-head stats and cross-head metrics.
    """
    signals = extract_reading_signals(problem)
    if signals is None:
        return None

    # Per-head metrics
    head_stats = {}
    for head in HEAD_KEYS:
        signal = signals[head]
        bimodality = compute_bimodality(signal)
        contrast = compute_contrast_ratio(signal)
        transitions = compute_transitions(signal)

        head_stats[head] = {
            "bimodal_coef": bimodality["bimodal_coef"],
            "mode_separation": bimodality["mode_separation"],
            "low_mean": bimodality["low_mean"],
            "high_mean": bimodality["high_mean"],
            "contrast_ratio": contrast,
            **transitions,
            "signal_mean": float(np.mean(signal)),
            "signal_std": float(np.std(signal)),
            "n_active_steps": int(np.sum(signal > 1e-6)),
        }

    # Cross-head correlation matrix (upper triangle)
    correlations = {}
    for i, h1 in enumerate(HEAD_KEYS):
        for j, h2 in enumerate(HEAD_KEYS):
            if i < j:
                key = f"{h1}_{h2}"
                correlations[key] = compute_head_correlation(signals, h1, h2)

    # Cross-head synchronization matrix (upper triangle)
    sync_scores = {}
    for i, h1 in enumerate(HEAD_KEYS):
        for j, h2 in enumerate(HEAD_KEYS):
            if i < j:
                key = f"{h1}_{h2}"
                sync_scores[key] = compute_sync_score(signals, h1, h2)

    return {
        "head_stats": head_stats,
        "correlations": correlations,
        "sync_scores": sync_scores,
    }


def lambda_handler(event, context):
    """
    Process one IAF chunk, compute head analysis statistics.

    Event:
    {
        "chunk_key": "iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_000.json",
        "output_prefix": "head_analysis/per_chunk_stats/",
        "bucket": "mycelium-data"
    }
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "head_analysis/per_chunk_stats/")

    # Download IAF chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    # Aggregate stats across problems
    agg_head_stats = {head: defaultdict(list) for head in HEAD_KEYS}
    agg_correlations = defaultdict(list)
    agg_sync_scores = defaultdict(list)

    # Store example waveforms (first 5 problems with good data)
    example_waveforms = []

    n_processed = 0
    for idx, problem in enumerate(problems):
        result = analyze_problem(problem)
        if result is None:
            continue

        n_processed += 1

        # Aggregate head stats
        for head, stats in result["head_stats"].items():
            for key, val in stats.items():
                agg_head_stats[head][key].append(val)

        # Aggregate cross-head metrics
        for key, val in result["correlations"].items():
            agg_correlations[key].append(val)
        for key, val in result["sync_scores"].items():
            agg_sync_scores[key].append(val)

        # Store example waveforms (first 5)
        if len(example_waveforms) < 5:
            signals = extract_reading_signals(problem)
            if signals:
                example_waveforms.append({
                    "problem_idx": idx,
                    "problem_id": str(problem.get("problem_idx", problem.get("problem_id", idx))),
                    "signals": {head: signals[head].tolist() for head in HEAD_KEYS}
                })

    # Compute chunk-level aggregates
    chunk_head_stats = {}
    for head in HEAD_KEYS:
        stats = agg_head_stats[head]
        chunk_head_stats[head] = {
            key: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            for key, vals in stats.items() if vals
        }

    chunk_correlations = {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for key, vals in agg_correlations.items() if vals
    }

    chunk_sync = {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for key, vals in agg_sync_scores.items() if vals
    }

    # Upload results
    chunk_name = chunk_key.split("/")[-1]
    output_key = f"{output_prefix}{chunk_name}"

    output = {
        "chunk_key": chunk_key,
        "n_problems": len(problems),
        "n_processed": n_processed,
        "head_stats": chunk_head_stats,
        "correlations": chunk_correlations,
        "sync_scores": chunk_sync,
        "example_waveforms": example_waveforms,
    }

    print(f"Uploading s3://{bucket}/{output_key} | {n_processed} problems")

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(output, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    # Summary for orchestrator
    summary = {head: chunk_head_stats[head].get("contrast_ratio", {}).get("mean", 0)
               for head in HEAD_KEYS}

    return {
        "statusCode": 200,
        "chunk_key": chunk_key,
        "output_key": output_key,
        "n_processed": n_processed,
        "contrast_summary": summary,
    }
