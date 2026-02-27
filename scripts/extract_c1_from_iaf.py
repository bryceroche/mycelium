#!/usr/bin/env python3
"""
Extract C1 Segmenter Training Data from IAF top_positions

Maps teacher attention positions back to problem text tokens.
Tokens that receive high attention during computation steps = I (inside span).
Everything else = O (outside).

KEY INSIGHT: Using global union (any step attended = I) makes ~88% of tokens I.
Instead, use count-based threshold: only label I if attended by N+ steps.
This identifies consistently important tokens vs. one-off attention.

Designed to run via MapReduce on EC2 (handles 33GB Medusa data).

Usage:
  # Stricter extraction (recommended for training)
  python extract_c1_from_iaf.py --input data.json --output c1_data.json \
      --top-k 5 --weight-threshold 0.1 --min-step-count 2

  # Process a single shard from S3
  python extract_c1_from_iaf.py --input s3://mycelium-data/iaf_extraction/instance1/iaf_v3_gpu0_valid.json \
                                 --output /tmp/c1_data_gpu0.json
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

    IAF records may have:
    - 'question' or 'problem': the raw problem text
    - 'input_ids': tokenized input (problem only, before CoT)
    - 'problem_len' or 'input_len': length of problem in tokens

    Returns (problem_text, problem_token_length)
    """
    # Try different field names (IAF uses 'problem_text')
    problem_text = (record.get('problem_text') or record.get('question') or
                    record.get('problem') or record.get('input', ''))

    # Get problem token length (positions < this are problem tokens)
    problem_len = (record.get('input_len') or record.get('problem_len') or
                   record.get('n_input_tokens'))

    if problem_len is None:
        # Try input_tokens array
        input_tokens = record.get('input_tokens', [])
        if input_tokens:
            problem_len = len(input_tokens)
        else:
            # Try input_ids
            input_ids = record.get('input_ids', [])
            if input_ids:
                problem_len = len(input_ids)
            else:
                # Last resort: estimate from text
                # Rough estimate: ~4 chars per token for English
                problem_len = len(problem_text) // 4

    return problem_text, problem_len


def extract_attention_positions(record: Dict, problem_len: int,
                                  top_k: int = 5,
                                  weight_threshold: float = 0.1,
                                  min_step_count: int = 1) -> List[int]:
    """
    Extract problem token positions that received high attention.

    Args:
        record: IAF record with top_positions
        problem_len: Number of tokens in problem (positions >= this are CoT)
        top_k: Consider top K positions per head (default: 5, was 20)
        weight_threshold: Minimum attention weight to count (default: 0.1, was 0.01)
        min_step_count: Minimum computation steps that must attend to a token
                        for it to be labeled I. Higher = more selective.
                        1 = any step (original behavior)
                        2+ = consistently important tokens only

    Returns:
        List of problem token positions that were attended to by min_step_count+ steps
    """
    top_positions = record.get('top_positions', [])

    # Count how many steps attend to each position
    position_step_counts = defaultdict(int)

    for step_idx, step_data in enumerate(top_positions):
        # step_data is a dict like {"L22H4": [{"pos": 2, "weight": 0.98}, ...], ...}
        if isinstance(step_data, dict):
            # Track which positions this step attends to (avoid double-counting heads)
            step_positions = set()

            for head_name, positions in step_data.items():
                if isinstance(positions, list):
                    for pos_data in positions[:top_k]:
                        if isinstance(pos_data, dict):
                            pos = pos_data.get('pos', -1)
                            weight = pos_data.get('weight', 0)

                            # Only count problem tokens with sufficient attention weight
                            if 0 <= pos < problem_len and weight >= weight_threshold:
                                step_positions.add(pos)

            # Increment count for positions this step attended to
            for pos in step_positions:
                position_step_counts[pos] += 1

    # Filter to positions attended by min_step_count+ steps
    attended_positions = [
        pos for pos, count in position_step_counts.items()
        if count >= min_step_count
    ]

    return sorted(attended_positions)


def create_io_labels(problem_len: int, attended_positions: List[int]) -> List[int]:
    """
    Create IO labels for problem tokens.

    Args:
        problem_len: Number of tokens in problem
        attended_positions: Token positions that were attended to

    Returns:
        List of labels: 0 = O (outside), 1 = I (inside span)
    """
    labels = [0] * problem_len
    for pos in attended_positions:
        if 0 <= pos < problem_len:
            labels[pos] = 1

    return labels


def process_record(record: Dict,
                   top_k: int = 5,
                   weight_threshold: float = 0.1,
                   min_step_count: int = 1) -> Optional[Dict]:
    """
    Process a single IAF record into C1 training data.

    Returns dict with:
        - text: problem text
        - input_ids: tokenized problem (if available)
        - labels: IO labels (0=O, 1=I)
        - n_spans: number of contiguous I runs
    """
    try:
        problem_text, problem_len = extract_problem_text(record)

        if not problem_text or problem_len == 0:
            return None

        attended_positions = extract_attention_positions(
            record, problem_len, top_k, weight_threshold, min_step_count
        )

        if not attended_positions:
            return None

        labels = create_io_labels(problem_len, attended_positions)

        # Count spans (O→I transitions)
        n_spans = sum(1 for i in range(len(labels))
                      if labels[i] == 1 and (i == 0 or labels[i-1] == 0))

        result = {
            'text': problem_text,
            'labels': labels,
            'n_spans': n_spans,
            'n_attended': len(attended_positions),
        }

        # Include input_ids if available
        if 'input_ids' in record:
            result['input_ids'] = record['input_ids'][:problem_len]

        return result

    except Exception as e:
        print(f"Error processing record: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract C1 training data from IAF')
    parser.add_argument('--input', required=True, help='Input path (S3 or local)')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--local', action='store_true', help='Input is local file')
    parser.add_argument('--limit', type=int, help='Limit number of records')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top K positions per head (default: 5, was 20)')
    parser.add_argument('--weight-threshold', type=float, default=0.1,
                        help='Min attention weight (default: 0.1, was 0.01)')
    parser.add_argument('--min-step-count', type=int, default=2,
                        help='Min computation steps that must attend to a token (default: 2). '
                             'Higher = more selective. 1 = any step attends.')
    args = parser.parse_args()

    print("=" * 60)
    print("C1 TRAINING DATA EXTRACTION FROM IAF")
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
    print(f"Extracting C1 labels (top_k={args.top_k}, threshold={args.weight_threshold}, "
          f"min_steps={args.min_step_count})...")
    results = []
    for i, record in enumerate(records):
        result = process_record(record, args.top_k, args.weight_threshold, args.min_step_count)
        if result:
            results.append(result)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(records)}, extracted {len(results)}")

    print(f"\nExtracted {len(results)} training samples from {len(records)} records")

    # Stats
    if results:
        total_tokens = sum(len(r['labels']) for r in results)
        total_i = sum(sum(r['labels']) for r in results)
        total_spans = sum(r['n_spans'] for r in results)

        print(f"\nStats:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  I tokens: {total_i:,} ({total_i/total_tokens*100:.1f}%)")
        print(f"  O tokens: {total_tokens - total_i:,} ({(total_tokens-total_i)/total_tokens*100:.1f}%)")
        print(f"  Total spans: {total_spans:,}")
        print(f"  Spans per sample: {total_spans / len(results):.1f}")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f)

    print("Done!")


if __name__ == "__main__":
    main()
