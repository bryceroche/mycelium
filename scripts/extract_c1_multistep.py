#!/usr/bin/env python3
"""
Extract C1 Multi-Step Training Data from IAF top_positions

OPTION A: Multi-channel output where each channel = one computation step's relevance.

Instead of global aggregation (which makes ~90% of tokens "relevant"),
we preserve per-step attention maps. Each step attends to 3-10 tokens.

The raw IAF data has one top_positions entry per GENERATED TOKEN (e.g. 907 entries).
We identify "computation steps" as entries where:
1. >70% of attention weight goes to problem text (not CoT)
2. The attention is focused (top positions have high weights)

We cluster these into N_CHANNELS distinct computation steps.

Output format:
{
    "text": "problem text...",
    "step_labels": [
        [0, 0, 0.9, 0.8, 0, 0, ...],  # Step 0: attends to tokens 2,3
        [0, 0, 0, 0.7, 0, 0.9, ...],  # Step 1: attends to tokens 3,5
        [0, 0, 0, 0, 0, 0, ...],      # Step 2: inactive (padding)
        ...                           # Up to N_CHANNELS steps
    ],
    "n_active_steps": 2,
}

Usage:
  python extract_c1_multistep.py --input iaf_data.json --output c1_multistep.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import numpy as np

N_CHANNELS = 8  # Max number of computation steps we support


def extract_problem_info(record: Dict) -> Tuple[str, int]:
    """Extract problem text and token length from IAF record."""
    problem_text = (record.get('problem_text') or record.get('question') or
                    record.get('problem') or record.get('input', ''))

    problem_len = (record.get('input_len') or record.get('problem_len') or
                   record.get('n_input_tokens'))

    if problem_len is None:
        input_tokens = record.get('input_tokens', [])
        if input_tokens:
            problem_len = len(input_tokens)
        else:
            input_ids = record.get('input_ids', [])
            problem_len = len(input_ids) if input_ids else len(problem_text) // 4

    return problem_text, problem_len


def extract_step_attention(step_data: Dict, problem_len: int,
                           top_k: int = 5, weight_threshold: float = 0.1) -> np.ndarray:
    """
    Extract attention map for a single step.
    Returns array of shape (problem_len,) with relevance scores 0-1.
    """
    relevance = np.zeros(problem_len, dtype=np.float32)

    if not isinstance(step_data, dict):
        return relevance

    position_weights = defaultdict(list)

    for head_name, positions in step_data.items():
        if isinstance(positions, list):
            for pos_data in positions[:top_k]:
                if isinstance(pos_data, dict):
                    pos = pos_data.get('pos', -1)
                    weight = pos_data.get('weight', 0)

                    # Only problem text positions
                    if 0 <= pos < problem_len and weight >= weight_threshold:
                        position_weights[pos].append(weight)

    # Max weight across heads
    for pos, weights in position_weights.items():
        relevance[pos] = max(weights)

    return relevance


def compute_problem_attention_ratio(step_data: Dict, problem_len: int, top_k: int = 5) -> float:
    """Compute what fraction of attention goes to problem text vs CoT."""
    if not isinstance(step_data, dict):
        return 0.0

    prob_weight = 0.0
    total_weight = 0.0

    for head_name, positions in step_data.items():
        if isinstance(positions, list):
            for pos_data in positions[:top_k]:
                if isinstance(pos_data, dict):
                    weight = pos_data.get('weight', 0)
                    pos = pos_data.get('pos', -1)
                    total_weight += weight
                    if 0 <= pos < problem_len:
                        prob_weight += weight

    return prob_weight / total_weight if total_weight > 0 else 0.0


def find_computation_steps(top_positions: List[Dict], problem_len: int,
                           min_ratio: float = 0.6, max_steps: int = 20) -> List[int]:
    """
    Find steps that are likely "computation" (attending to problem text).
    Returns list of step indices.
    """
    computation_steps = []

    for step_idx, step_data in enumerate(top_positions):
        ratio = compute_problem_attention_ratio(step_data, problem_len)
        if ratio >= min_ratio:
            computation_steps.append(step_idx)

        if len(computation_steps) >= max_steps:
            break

    return computation_steps


def cluster_into_channels(relevance_maps: List[np.ndarray], n_channels: int) -> List[np.ndarray]:
    """
    Cluster similar relevance maps into N channels.
    Uses simple approach: take top N diverse maps based on peak positions.
    """
    if len(relevance_maps) <= n_channels:
        return relevance_maps

    # Get peak positions for each map
    peak_positions = []
    for rel in relevance_maps:
        peaks = np.where(rel > 0.3)[0]
        peak_positions.append(set(peaks.tolist()))

    # Greedy selection: pick maps with different peak positions
    selected = [0]  # Start with first
    for i in range(1, len(relevance_maps)):
        # Check if this map has different peaks than already selected
        current_peaks = peak_positions[i]
        is_different = True
        for sel_idx in selected:
            overlap = len(current_peaks & peak_positions[sel_idx])
            union = len(current_peaks | peak_positions[sel_idx])
            if union > 0 and overlap / union > 0.5:  # >50% overlap = too similar
                is_different = False
                break

        if is_different:
            selected.append(i)
            if len(selected) >= n_channels:
                break

    return [relevance_maps[i] for i in selected]


def process_record(record: Dict, top_k: int = 5,
                   weight_threshold: float = 0.1,
                   min_ratio: float = 0.6) -> Optional[Dict]:
    """
    Process IAF record into multi-step C1 training data.
    """
    try:
        problem_text, problem_len = extract_problem_info(record)

        if not problem_text or problem_len == 0:
            return None

        top_positions = record.get('top_positions', [])
        if not top_positions:
            return None

        # Find computation steps (those attending to problem text)
        comp_step_indices = find_computation_steps(
            top_positions, problem_len, min_ratio=min_ratio, max_steps=30
        )

        if len(comp_step_indices) == 0:
            return None

        # Extract relevance maps for computation steps
        relevance_maps = []
        for step_idx in comp_step_indices:
            rel = extract_step_attention(
                top_positions[step_idx], problem_len, top_k, weight_threshold
            )
            if np.max(rel) > 0.3:  # Has meaningful attention
                relevance_maps.append(rel)

        if len(relevance_maps) == 0:
            return None

        # Cluster into N_CHANNELS
        clustered = cluster_into_channels(relevance_maps, N_CHANNELS)

        # Build step_labels
        step_labels = [rel.tolist() for rel in clustered]
        n_active = len(step_labels)

        # Pad to N_CHANNELS
        while len(step_labels) < N_CHANNELS:
            step_labels.append([0.0] * problem_len)

        # Stats
        tokens_per_step = [int(np.sum(np.array(sl) > 0.3)) for sl in step_labels[:n_active]]

        return {
            'text': problem_text,
            'step_labels': step_labels,
            'n_active_steps': n_active,
            'problem_len': problem_len,
            'tokens_per_step': tokens_per_step,
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract multi-step C1 training data')
    parser.add_argument('--input', required=True, help='Input IAF JSON path')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--limit', type=int, help='Limit records')
    parser.add_argument('--top-k', type=int, default=5, help='Top K positions per head')
    parser.add_argument('--weight-threshold', type=float, default=0.1,
                        help='Min attention weight')
    args = parser.parse_args()

    print("=" * 60)
    print("C1 MULTI-STEP EXTRACTION (OPTION A)")
    print(f"Channels: {N_CHANNELS}")
    print("=" * 60)

    # Load
    print(f"\nLoading {args.input}...")
    with open(args.input) as f:
        records = json.load(f)

    if args.limit:
        records = records[:args.limit]

    print(f"Loaded {len(records)} records")

    # Process
    results = []
    for i, record in enumerate(records):
        result = process_record(record, args.top_k, args.weight_threshold)
        if result:
            results.append(result)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}, extracted {len(results)}")

    print(f"\nExtracted {len(results)} samples")

    # Stats
    if results:
        avg_steps = np.mean([r['n_active_steps'] for r in results])
        all_tokens_per_step = []
        for r in results:
            all_tokens_per_step.extend(r['tokens_per_step'])

        print(f"\nStats:")
        print(f"  Avg active steps: {avg_steps:.1f}")
        print(f"  Avg tokens per step: {np.mean(all_tokens_per_step):.1f}")
        print(f"  Max tokens per step: {max(all_tokens_per_step)}")

        # Distribution of active steps
        step_counts = [r['n_active_steps'] for r in results]
        print(f"\n  Step distribution:")
        for n in range(1, N_CHANNELS + 1):
            count = sum(1 for s in step_counts if s == n)
            if count > 0:
                print(f"    {n} steps: {count} ({100*count/len(results):.1f}%)")

    # Save (remove debug field)
    for r in results:
        del r['tokens_per_step']

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f)

    print("Done!")


if __name__ == "__main__":
    main()
