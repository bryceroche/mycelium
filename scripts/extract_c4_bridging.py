#!/usr/bin/env python3
"""
C4 Data Extractor: Bridging Spans (Implicit Operations)

Extract low-IAF spans where the model is doing internal computation
without grounding to input tokens. These are bridging candidates.

Low IAF = model computing internally, not attending to problem text.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys


@dataclass
class BridgingSpan:
    problem_idx: int
    gen_start: int  # Start token in generation
    gen_end: int    # End token in generation
    span_text: str  # The generated text in this span
    avg_iaf: float  # Average IAF across span
    duration: int   # Number of tokens


def compute_avg_iaf(iaf_trace: Dict[str, float]) -> float:
    """Compute average IAF across all heads."""
    if not iaf_trace:
        return 1.0
    return sum(iaf_trace.values()) / len(iaf_trace)


def find_low_iaf_spans(
    iaf_traces: List[Dict[str, float]],
    threshold: float = 0.3,
    min_duration: int = 5
) -> List[Tuple[int, int, float]]:
    """
    Find sustained low-IAF regions.

    Returns list of (start, end, avg_iaf) tuples.
    """
    spans = []
    current_start = None
    current_iaf_sum = 0.0

    for i, trace in enumerate(iaf_traces):
        avg_iaf = compute_avg_iaf(trace)

        if avg_iaf < threshold:
            if current_start is None:
                current_start = i
                current_iaf_sum = avg_iaf
            else:
                current_iaf_sum += avg_iaf
        else:
            if current_start is not None:
                duration = i - current_start
                if duration >= min_duration:
                    avg = current_iaf_sum / duration
                    spans.append((current_start, i, avg))
                current_start = None
                current_iaf_sum = 0.0

    # Handle span at end
    if current_start is not None:
        duration = len(iaf_traces) - current_start
        if duration >= min_duration:
            avg = current_iaf_sum / duration
            spans.append((current_start, len(iaf_traces), avg))

    return spans


def extract_span_text(generated_cot: str, gen_start: int, gen_end: int, input_len: int) -> str:
    """
    Extract text for a generation span.

    This is approximate - we don't have exact token-to-char mapping,
    so we use a heuristic based on position ratios.
    """
    # Rough approximation: assume ~4 chars per token
    chars_per_token = len(generated_cot) / max(1, (gen_end - gen_start + input_len))
    start_char = int((gen_start) * chars_per_token)
    end_char = int((gen_end) * chars_per_token)

    # Clamp to valid range
    start_char = max(0, min(start_char, len(generated_cot)))
    end_char = max(start_char, min(end_char, len(generated_cot)))

    return generated_cot[start_char:end_char]


def extract_c4_labels(
    iaf_data: List[Dict],
    threshold: float = 0.3,
    min_duration: int = 5
) -> List[BridgingSpan]:
    """Extract C4 bridging span labels from IAF data."""
    all_spans = []

    for record in iaf_data:
        problem_idx = record.get('problem_idx', -1)
        iaf_traces = record.get('iaf_traces', [])
        generated_cot = record.get('generated_cot', '')
        input_len = record.get('input_len', 0)

        # Find low-IAF regions
        low_iaf_regions = find_low_iaf_spans(iaf_traces, threshold, min_duration)

        for start, end, avg_iaf in low_iaf_regions:
            span_text = extract_span_text(generated_cot, start, end, input_len)

            all_spans.append(BridgingSpan(
                problem_idx=problem_idx,
                gen_start=start,
                gen_end=end,
                span_text=span_text,
                avg_iaf=avg_iaf,
                duration=end - start
            ))

    return all_spans


def main():
    parser = argparse.ArgumentParser(description='Extract C4 bridging span labels from IAF data')
    parser.add_argument('input', help='Input IAF JSON file or S3 path')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.3, help='IAF threshold for low-IAF')
    parser.add_argument('--min-duration', type=int, default=5, help='Min tokens for bridging span')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    args = parser.parse_args()

    # Load data
    if args.input.startswith('s3://'):
        import subprocess
        result = subprocess.run(['aws', 's3', 'cp', args.input, '-'],
                              capture_output=True, text=True)
        data = json.loads(result.stdout)
    else:
        with open(args.input) as f:
            data = json.load(f)

    print(f"Loaded {len(data)} records", file=sys.stderr)

    # Extract labels
    spans = extract_c4_labels(data, args.threshold, args.min_duration)

    # Convert to JSON-serializable format
    output = [
        {
            'problem_idx': s.problem_idx,
            'gen_start': s.gen_start,
            'gen_end': s.gen_end,
            'span_text': s.span_text,
            'avg_iaf': s.avg_iaf,
            'duration': s.duration
        }
        for s in spans
    ]

    # Print stats
    if args.stats:
        print(f"\nFound {len(spans)} bridging spans across {len(data)} problems", file=sys.stderr)
        if spans:
            avg_duration = sum(s.duration for s in spans) / len(spans)
            avg_iaf = sum(s.avg_iaf for s in spans) / len(spans)
            print(f"  Avg duration: {avg_duration:.1f} tokens", file=sys.stderr)
            print(f"  Avg IAF: {avg_iaf:.3f}", file=sys.stderr)

            # Sample spans
            print("\nSample bridging spans:", file=sys.stderr)
            for s in spans[:3]:
                print(f"  [{s.gen_start}-{s.gen_end}] IAF={s.avg_iaf:.3f}: {s.span_text[:80]}...", file=sys.stderr)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} spans to {args.output}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
