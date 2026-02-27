#!/usr/bin/env python3
"""
Extract C1 Relevance Scorer Training Data from IAF top_positions

Per six_model.md: C1 outputs continuous relevance scores (0-1) per token,
NOT binary IO tags. This preserves the teacher's actual scattered attention
pattern instead of forcing premature collapse into contiguous spans.

Labels are normalized cumulative attention scores across all computation steps.
No thresholding, no IO conversion, no contiguity forcing.

Training uses MSE loss (regression), not CrossEntropy (classification).

Usage:
  python extract_c1_relevance.py --input data.json --output c1_relevance.json

  # Process from S3
  python extract_c1_relevance.py \
      --input s3://mycelium-data/iaf_extraction/instance1/iaf_v3_gpu0_valid.json \
      --output /tmp/c1_relevance_gpu0.json
"""

import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download file from S3."""
    cmd = ["aws", "s3", "cp", s3_path, local_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def load_iaf_records(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load IAF records from JSON file."""
    with open(path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    return data


def extract_problem_text(record: Dict) -> Tuple[str, int]:
    """
    Extract problem text and its token length from IAF record.

    Returns (problem_text, problem_token_length)
    """
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
            if input_ids:
                problem_len = len(input_ids)
            else:
                problem_len = len(problem_text) // 4

    return problem_text, problem_len


def extract_relevance_scores(record: Dict, problem_len: int,
                             top_k: int = 20,
                             normalize: str = 'max') -> List[float]:
    """
    Extract continuous relevance scores (0-1) for each problem token.

    Accumulates attention weights across all computation steps and heads,
    then normalizes to [0, 1].

    Args:
        record: IAF record with top_positions
        problem_len: Number of tokens in problem
        top_k: Consider top K positions per head (default: 20, generous)
        normalize: Normalization method:
            'max' - divide by max score (default, preserves relative importance)
            'sum' - divide by sum (makes it a probability distribution)
            'minmax' - min-max scaling to [0, 1]

    Returns:
        List of relevance scores (0-1) for each problem token
    """
    top_positions = record.get('top_positions', [])

    # Accumulate attention weights per position
    position_weights = defaultdict(float)

    for step_idx, step_data in enumerate(top_positions):
        if isinstance(step_data, dict):
            for head_name, positions in step_data.items():
                if isinstance(positions, list):
                    for pos_data in positions[:top_k]:
                        if isinstance(pos_data, dict):
                            pos = pos_data.get('pos', -1)
                            weight = pos_data.get('weight', 0)

                            # Only accumulate for problem tokens
                            if 0 <= pos < problem_len:
                                position_weights[pos] += weight

    # Create relevance scores array
    scores = [position_weights.get(i, 0.0) for i in range(problem_len)]

    # Normalize to [0, 1]
    if scores:
        max_score = max(scores)
        min_score = min(scores)
        sum_score = sum(scores)

        if normalize == 'max' and max_score > 0:
            scores = [s / max_score for s in scores]
        elif normalize == 'sum' and sum_score > 0:
            scores = [s / sum_score for s in scores]
        elif normalize == 'minmax' and max_score > min_score:
            scores = [(s - min_score) / (max_score - min_score) for s in scores]

    return scores


def process_record(record: Dict, top_k: int = 20,
                   normalize: str = 'max') -> Optional[Dict]:
    """
    Process a single IAF record into C1 relevance training data.

    Returns dict with:
        - text: problem text
        - relevance: continuous scores (0-1) per token
        - input_ids: tokenized problem (if available)
        - stats: summary statistics
    """
    try:
        problem_text, problem_len = extract_problem_text(record)

        if not problem_text or problem_len == 0:
            return None

        relevance = extract_relevance_scores(record, problem_len, top_k, normalize)

        if not relevance or max(relevance) == 0:
            return None

        # Compute stats
        nonzero = [r for r in relevance if r > 0]
        high_relevance = [r for r in relevance if r > 0.5]

        result = {
            'text': problem_text,
            'relevance': relevance,
            'stats': {
                'n_tokens': len(relevance),
                'n_nonzero': len(nonzero),
                'n_high': len(high_relevance),
                'mean': sum(relevance) / len(relevance) if relevance else 0,
                'max': max(relevance) if relevance else 0,
            }
        }

        # Include input_ids if available
        if 'input_ids' in record:
            result['input_ids'] = record['input_ids'][:problem_len]

        return result

    except Exception as e:
        print(f"Error processing record: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract C1 relevance scores from IAF (regression, not classification)')
    parser.add_argument('--input', required=True, help='Input path (S3 or local)')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--local', action='store_true', help='Input is local file')
    parser.add_argument('--limit', type=int, help='Limit number of records')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top K positions per head (default: 20)')
    parser.add_argument('--normalize', choices=['max', 'sum', 'minmax'], default='max',
                        help='Normalization method (default: max)')
    args = parser.parse_args()

    print("=" * 60)
    print("C1 RELEVANCE SCORER DATA EXTRACTION FROM IAF")
    print("(Continuous scores, not binary IO tags)")
    print("=" * 60)

    # Download if S3
    if args.input.startswith('s3://') and not args.local:
        local_input = '/tmp/iaf_input.json'
        print(f"Downloading {args.input}...")
        if not download_from_s3(args.input, local_input):
            print("ERROR: Failed to download from S3")
            sys.exit(1)
        input_path = local_input
    else:
        input_path = args.input

    # Load records
    print(f"Loading records from {input_path}...")
    records = load_iaf_records(input_path, args.limit)
    print(f"Loaded {len(records)} records")

    # Process
    print(f"Extracting relevance scores (top_k={args.top_k}, normalize={args.normalize})...")
    results = []
    for i, record in enumerate(records):
        result = process_record(record, args.top_k, args.normalize)
        if result:
            results.append(result)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(records)}, extracted {len(results)}")

    print(f"\nExtracted {len(results)} training samples from {len(records)} records")

    # Stats
    if results:
        total_tokens = sum(r['stats']['n_tokens'] for r in results)
        total_nonzero = sum(r['stats']['n_nonzero'] for r in results)
        total_high = sum(r['stats']['n_high'] for r in results)
        avg_mean = sum(r['stats']['mean'] for r in results) / len(results)

        print(f"\nStats:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Tokens with attention (>0): {total_nonzero:,} ({total_nonzero/total_tokens*100:.1f}%)")
        print(f"  High relevance tokens (>0.5): {total_high:,} ({total_high/total_tokens*100:.1f}%)")
        print(f"  Average mean relevance: {avg_mean:.3f}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f)

    print("Done!")


if __name__ == "__main__":
    main()
