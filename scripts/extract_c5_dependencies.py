#!/usr/bin/env python3
"""
C5 Data Extractor: Step Dependencies (DAG Wiring)

Extract dependency edges between computation steps.

Approach: Segment CoT using JSD-like peaks in IAF variance,
then detect value sharing between steps.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import sys


@dataclass
class DependencyEdge:
    problem_idx: int
    step_i: int      # Earlier step
    step_j: int      # Later step that depends on step_i
    confidence: float
    edge_type: str   # DEPENDS, INDEPENDENT


@dataclass
class ComputationStep:
    start_token: int
    end_token: int
    avg_iaf: float


def compute_iaf_stats(iaf_trace: Dict[str, float]) -> Tuple[float, float]:
    """Compute mean and variance of IAF across heads."""
    if not iaf_trace:
        return 1.0, 0.0
    values = list(iaf_trace.values())
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var


def segment_cot(
    iaf_traces: List[Dict[str, float]],
    window_size: int = 5
) -> List[ComputationStep]:
    """
    Segment CoT into computation steps using IAF transitions.

    Look for sharp drops in IAF (entering internal computation)
    and sharp rises (returning to input grounding).
    """
    if len(iaf_traces) < window_size * 2:
        return []

    # Compute smoothed IAF means
    means = [compute_iaf_stats(t)[0] for t in iaf_traces]

    # Find transitions (large changes in IAF)
    transitions = []
    for i in range(window_size, len(means) - window_size):
        left_avg = sum(means[i-window_size:i]) / window_size
        right_avg = sum(means[i:i+window_size]) / window_size
        delta = abs(right_avg - left_avg)

        if delta > 0.2:  # Significant transition
            transitions.append(i)

    # Convert transitions to steps
    steps = []
    prev = 0
    for t in transitions:
        if t - prev >= window_size:  # Minimum step size
            avg_iaf = sum(means[prev:t]) / (t - prev)
            steps.append(ComputationStep(
                start_token=prev,
                end_token=t,
                avg_iaf=avg_iaf
            ))
        prev = t

    # Add final step
    if len(means) - prev >= window_size:
        avg_iaf = sum(means[prev:]) / (len(means) - prev)
        steps.append(ComputationStep(
            start_token=prev,
            end_token=len(means),
            avg_iaf=avg_iaf
        ))

    return steps


def detect_dependencies(
    steps: List[ComputationStep],
    top_positions: List[Dict]
) -> List[Tuple[int, int, float]]:
    """
    Detect dependencies between steps based on attention patterns.

    A step j depends on step i if tokens in step j attend strongly
    to tokens in step i (cross-step attention).
    """
    dependencies = []

    for j, step_j in enumerate(steps):
        for i, step_i in enumerate(steps):
            if i >= j:
                continue  # Only consider earlier steps

            # Check if step_j attends to step_i's tokens
            cross_attention = 0.0
            count = 0

            for tok_idx in range(step_j.start_token, step_j.end_token):
                if tok_idx >= len(top_positions):
                    continue

                pos_dict = top_positions[tok_idx]
                for head, positions in pos_dict.items():
                    for p in positions:
                        pos = p.get('pos', -1)
                        weight = p.get('weight', 0)

                        # Check if attending to step_i's range
                        if step_i.start_token <= pos < step_i.end_token:
                            cross_attention += weight
                        count += 1

            # Normalize
            if count > 0:
                avg_cross = cross_attention / count
                if avg_cross > 0.05:  # Threshold for dependency
                    dependencies.append((i, j, avg_cross))

    return dependencies


def extract_c5_labels(iaf_data: List[Dict]) -> List[DependencyEdge]:
    """Extract C5 dependency labels from IAF data."""
    all_edges = []

    for record in iaf_data:
        problem_idx = record.get('problem_idx', -1)
        iaf_traces = record.get('iaf_traces', [])
        top_positions = record.get('top_positions', [])

        # Segment into steps
        steps = segment_cot(iaf_traces)

        if len(steps) < 2:
            continue

        # Detect dependencies
        deps = detect_dependencies(steps, top_positions)

        # Create dependency edges
        step_pairs_with_deps = set((i, j) for i, j, _ in deps)

        for j in range(len(steps)):
            for i in range(j):
                if (i, j) in step_pairs_with_deps:
                    conf = next(c for ii, jj, c in deps if ii == i and jj == j)
                    all_edges.append(DependencyEdge(
                        problem_idx=problem_idx,
                        step_i=i,
                        step_j=j,
                        confidence=conf,
                        edge_type="DEPENDS"
                    ))
                else:
                    all_edges.append(DependencyEdge(
                        problem_idx=problem_idx,
                        step_i=i,
                        step_j=j,
                        confidence=0.8,  # Default confidence for independence
                        edge_type="INDEPENDENT"
                    ))

    return all_edges


def main():
    parser = argparse.ArgumentParser(description='Extract C5 dependency labels from IAF data')
    parser.add_argument('input', help='Input IAF JSON file or S3 path')
    parser.add_argument('--output', '-o', help='Output JSON file')
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
    edges = extract_c5_labels(data)

    # Convert to JSON-serializable format
    output = [
        {
            'problem_idx': e.problem_idx,
            'step_i': e.step_i,
            'step_j': e.step_j,
            'edge_type': e.edge_type,
            'confidence': e.confidence
        }
        for e in edges
    ]

    # Print stats
    if args.stats:
        depends = sum(1 for e in edges if e.edge_type == "DEPENDS")
        independent = sum(1 for e in edges if e.edge_type == "INDEPENDENT")
        print(f"\nExtracted {len(edges)} edges:", file=sys.stderr)
        print(f"  DEPENDS: {depends} ({100*depends/max(1,len(edges)):.1f}%)", file=sys.stderr)
        print(f"  INDEPENDENT: {independent} ({100*independent/max(1,len(edges)):.1f}%)", file=sys.stderr)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} edges to {args.output}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
