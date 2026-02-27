#!/usr/bin/env python3
"""
Extract C1 Multi-Step Training Data from IAF v2 format

IAF v2 has one record per computation step with:
- text: problem text
- step_idx: which step this is
- per_token_iaf: [{step, top_positions, top_weights}, ...]

This script groups by problem text and extracts per-step relevance maps.

Output format for C1 multi-head model:
{
    "text": "problem text",
    "step_labels": [[0, 0.9, 0, ...], [0, 0, 0.8, ...], ...],  # N_CHANNELS arrays
    "n_active_steps": 3,
}

Usage:
  python extract_c1_from_iaf_v2.py --input iaf_v2.json --output c1_multistep.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

N_CHANNELS = 8


def group_by_problem(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by problem text."""
    grouped = defaultdict(list)
    for rec in records:
        text = rec.get('text', '')
        if text:
            grouped[text].append(rec)
    return grouped


def extract_step_relevance(step_data: Dict, problem_len: int,
                           use_weights: bool = True) -> np.ndarray:
    """
    Extract relevance map for one step from per_token_iaf entry.

    step_data has: {step, token_id, iaf, top_positions, top_weights}
    """
    relevance = np.zeros(problem_len, dtype=np.float32)

    positions = step_data.get('top_positions', [])
    weights = step_data.get('top_weights', [])

    for i, pos in enumerate(positions):
        if 0 <= pos < problem_len:
            if use_weights and i < len(weights):
                # Normalize weight to 0-1 range
                w = min(weights[i] * 20, 1.0)  # Scale up since weights are small
            else:
                w = 1.0 / (i + 1)  # Decay by rank if no weights
            relevance[pos] = max(relevance[pos], w)

    return relevance


def estimate_problem_len(records: List[Dict]) -> int:
    """Estimate problem token length from attention positions."""
    all_positions = []
    for rec in records:
        for step_data in rec.get('per_token_iaf', []):
            all_positions.extend(step_data.get('top_positions', []))

    if all_positions:
        # Problem length is roughly the range of attended positions
        return max(all_positions) + 10
    return 100  # Default


def process_problem(problem_text: str, records: List[Dict]) -> Optional[Dict]:
    """Process all records for one problem into multi-step labels."""
    if not records:
        return None

    # Sort by step_idx
    records = sorted(records, key=lambda r: r.get('step_idx', 0))

    # Estimate problem length
    problem_len = estimate_problem_len(records)

    # Collect all step relevance maps
    step_maps = []
    seen_steps = set()

    for rec in records:
        step_idx = rec.get('step_idx', 0)
        if step_idx in seen_steps:
            continue  # Skip duplicate steps
        seen_steps.add(step_idx)

        # Get the first entry from per_token_iaf (usually the most relevant)
        per_token = rec.get('per_token_iaf', [])
        if per_token:
            relevance = extract_step_relevance(per_token[0], problem_len)
            if np.max(relevance) > 0.1:  # Has meaningful attention
                step_maps.append(relevance)

    if len(step_maps) == 0:
        return None

    # Take up to N_CHANNELS steps
    step_maps = step_maps[:N_CHANNELS]
    n_active = len(step_maps)

    # Pad to N_CHANNELS
    while len(step_maps) < N_CHANNELS:
        step_maps.append(np.zeros(problem_len, dtype=np.float32))

    # Convert to lists
    step_labels = [sm.tolist() for sm in step_maps]

    # Stats
    tokens_per_step = [int(np.sum(np.array(sl) > 0.1)) for sl in step_labels[:n_active]]

    return {
        'text': problem_text,
        'step_labels': step_labels,
        'n_active_steps': n_active,
        'problem_len': problem_len,
        'tokens_per_step': tokens_per_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input IAF v2 JSON')
    parser.add_argument('--output', required=True, help='Output JSON')
    parser.add_argument('--limit', type=int, help='Limit problems')
    args = parser.parse_args()

    print("=" * 60)
    print(f"C1 MULTI-STEP EXTRACTION FROM IAF V2")
    print(f"N_CHANNELS: {N_CHANNELS}")
    print("=" * 60)

    # Load
    print(f"\nLoading {args.input}...")
    with open(args.input) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    # Group by problem
    grouped = group_by_problem(records)
    print(f"Unique problems: {len(grouped)}")

    # Process
    results = []
    for i, (problem_text, recs) in enumerate(grouped.items()):
        if args.limit and i >= args.limit:
            break

        result = process_problem(problem_text, recs)
        if result:
            results.append(result)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1} problems, extracted {len(results)}")

    print(f"\nExtracted {len(results)} problems")

    # Stats
    if results:
        avg_steps = np.mean([r['n_active_steps'] for r in results])
        all_tokens = []
        for r in results:
            all_tokens.extend(r['tokens_per_step'])

        print(f"\nStats:")
        print(f"  Avg active steps: {avg_steps:.1f}")
        print(f"  Avg tokens per step: {np.mean(all_tokens):.1f}")
        print(f"  Median tokens per step: {np.median(all_tokens):.0f}")

        # Step distribution
        step_counts = [r['n_active_steps'] for r in results]
        print(f"\n  Step distribution:")
        for n in range(1, N_CHANNELS + 1):
            count = sum(1 for s in step_counts if s == n)
            if count > 0:
                print(f"    {n} steps: {count} ({100*count/len(results):.1f}%)")

    # Clean and save
    for r in results:
        del r['tokens_per_step']

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f)

    print("Done!")


if __name__ == "__main__":
    main()
