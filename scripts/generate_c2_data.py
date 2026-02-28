#!/usr/bin/env python3
"""
Generate C2 Training Data with 15 Operational Labels

Collapses fine-grained Y operators into 15 pragmatic categories.
No more chasing perfect taxonomy - train C2 and refine from confusion matrix.

Labels:
    ADD, MUL, DIV, SQUARE, SQRT, CUBE, HIGH_POW, FRAC_POW,
    FACTORIAL, LOG, TRIG, MOD, RATIO, EQUATION, OTHER
"""

import json
import boto3
import numpy as np
from collections import Counter
from typing import Dict, Set, List, Tuple
import numpy as np

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'

# =============================================================================
# 15 OPERATIONAL LABELS
# =============================================================================

LABEL_MAP = {
    # Arithmetic
    'Add': 'ADD',

    'Mul': 'MUL',

    # Division/Inverse - collapse the junk drawer
    'inverse': 'DIV',
    'neg_pow': 'DIV',  # x^-2 etc is division-like

    # Powers - keep distinct
    'square': 'SQUARE',
    'sqrt': 'SQRT',
    'cube': 'CUBE',
    'high_pow': 'HIGH_POW',
    'pow_general': 'HIGH_POW',  # symbolic powers
    'frac_pow': 'FRAC_POW',
    'nth_root': 'FRAC_POW',  # roots are fractional powers
    'cbrt': 'FRAC_POW',
    'rsqrt': 'FRAC_POW',

    # Special functions
    'factorial': 'FACTORIAL',
    'binomial': 'FACTORIAL',  # related

    'log': 'LOG',
    'log_b': 'LOG',
    'exp': 'LOG',  # inverse of log

    'sin': 'TRIG',
    'cos': 'TRIG',
    'tan': 'TRIG',
    'cot': 'TRIG',
    'sec': 'TRIG',
    'csc': 'TRIG',
    'asin': 'TRIG',
    'acos': 'TRIG',
    'atan': 'TRIG',
    'sinh': 'TRIG',
    'cosh': 'TRIG',
    'tanh': 'TRIG',

    'Mod': 'MOD',

    # Comparison/Equation
    'Eq': 'EQUATION',
    'Ne': 'EQUATION',
    'Lt': 'EQUATION',
    'Le': 'EQUATION',
    'Gt': 'EQUATION',
    'Ge': 'EQUATION',
    'StrictLessThan': 'EQUATION',
    'StrictGreaterThan': 'EQUATION',

    # Calculus/Advanced (rare, collapse to OTHER)
    'Sum': 'OTHER',
    'Product': 'OTHER',
    'Integral': 'OTHER',
    'Derivative': 'OTHER',
    'Limit': 'OTHER',
    'Abs': 'OTHER',
    'floor': 'OTHER',
    'ceiling': 'OTHER',
}

# Priority order for multi-label steps
LABEL_PRIORITY = [
    'FACTORIAL', 'LOG', 'TRIG', 'MOD',  # Rare and distinctive
    'SQRT', 'CUBE', 'FRAC_POW', 'HIGH_POW', 'SQUARE',  # Power variants
    'EQUATION',
    'DIV', 'MUL', 'ADD',  # Common arithmetic last
    'OTHER',
]


def map_y_to_label(y_set: Set[str]) -> str:
    """
    Map a set of Y operators to a single operational label.
    Uses priority order to pick the most distinctive label.
    """
    if not y_set:
        return 'OTHER'

    # Map each operator
    labels = set()
    for op in y_set:
        if op in LABEL_MAP:
            labels.add(LABEL_MAP[op])
        # Skip unknown operators

    if not labels:
        return 'OTHER'

    # Pick highest priority label
    for label in LABEL_PRIORITY:
        if label in labels:
            return label

    return 'OTHER'


def load_data() -> Tuple[List[Dict], List[Set[str]]]:
    """Load steps and Y labels from S3."""
    print("Loading steps...")
    response = s3.get_object(Bucket=BUCKET, Key='ib_results_v2/aggregated_steps.json')
    steps_data = json.loads(response['Body'].read().decode('utf-8'))
    steps = steps_data['steps']
    print(f"  Loaded {len(steps)} steps")

    print("Loading Y labels...")
    response = s3.get_object(Bucket=BUCKET, Key='ib_y_labels/aggregated_y_labels.json')
    y_data = json.loads(response['Body'].read().decode('utf-8'))

    # Build Y by content hash
    y_by_hash = {}
    for result in y_data['results']:
        h = result.get('content_hash')
        y_ops = result.get('Y')
        if h and y_ops:
            y_by_hash[h] = set(y_ops)

    print(f"  Loaded {len(y_by_hash)} Y labels")

    return steps, y_by_hash


def generate_c2_data_per_step(steps: List[Dict], y_by_hash: Dict[str, Set[str]]) -> List[Dict]:
    """Generate C2 training data per-step with 15 operational labels."""
    c2_data = []
    label_counts = Counter()
    skipped = 0

    for step in steps:
        h = step.get('content_hash')
        text = step.get('text', '')

        if not h or h not in y_by_hash:
            skipped += 1
            continue

        y_set = y_by_hash[h]
        label = map_y_to_label(y_set)

        c2_data.append({
            'text': text,
            'label': label,
            'y_ops': list(y_set),  # Keep original for debugging
            'problem_idx': step.get('problem_idx'),
            'step_idx': step.get('step_idx'),
        })

        label_counts[label] += 1

    print(f"\nGenerated {len(c2_data)} training examples (per-step)")
    print(f"Skipped {skipped} steps (no Y label)")

    print(f"\nLabel distribution:")
    total = sum(label_counts.values())
    for label in LABEL_PRIORITY:
        count = label_counts.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:12}: {count:6} ({pct:5.1f}%) {bar}")

    return c2_data


def generate_c2_data_per_problem(steps: List[Dict], y_by_hash: Dict[str, Set[str]]) -> List[Dict]:
    """
    Generate C2 training data per-problem (multi-label + count).

    Each problem gets:
    - labels: set of all operation labels across its steps
    - count: number of operations (heartbeat from IAF)
    - text: problem text (first step or aggregated)
    """
    from collections import defaultdict

    # Group steps by problem
    problems = defaultdict(list)
    for step in steps:
        pid = step.get('problem_idx')
        if pid is not None:
            problems[pid].append(step)

    c2_data = []
    label_counts = Counter()

    for pid, prob_steps in problems.items():
        # Get problem text (usually step 0 has the problem statement)
        prob_steps.sort(key=lambda s: s.get('step_idx', 0))
        problem_text = prob_steps[0].get('text', '') if prob_steps else ''

        # Collect all labels across steps
        all_labels = set()
        op_count = 0

        for step in prob_steps:
            h = step.get('content_hash')
            if h and h in y_by_hash:
                y_set = y_by_hash[h]
                label = map_y_to_label(y_set)
                all_labels.add(label)
                op_count += 1

        if not all_labels:
            continue

        c2_data.append({
            'text': problem_text,
            'labels': list(all_labels),  # Multi-label
            'count': op_count,  # Heartbeat count for auxiliary head
            'problem_idx': pid,
        })

        for lbl in all_labels:
            label_counts[lbl] += 1

    print(f"\nGenerated {len(c2_data)} training examples (per-problem)")

    # Stats
    counts = [d['count'] for d in c2_data]
    print(f"Operation counts: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")

    label_per_problem = [len(d['labels']) for d in c2_data]
    print(f"Labels per problem: min={min(label_per_problem)}, max={max(label_per_problem)}, mean={np.mean(label_per_problem):.1f}")

    print(f"\nLabel distribution (problems containing each label):")
    total = len(c2_data)
    for label in LABEL_PRIORITY:
        count = label_counts.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:12}: {count:6} ({pct:5.1f}%) {bar}")

    return c2_data


def generate_c2_data(steps: List[Dict], y_by_hash: Dict[str, Set[str]]) -> Tuple[List[Dict], List[Dict]]:
    """Generate both per-step and per-problem C2 training data."""
    per_step = generate_c2_data_per_step(steps, y_by_hash)
    per_problem = generate_c2_data_per_problem(steps, y_by_hash)
    return per_step, per_problem


def save_data(per_step: List[Dict], per_problem: List[Dict]):
    """Save C2 training data (both per-step and per-problem)."""
    # Per-step data (single-label)
    step_output = {
        'n_examples': len(per_step),
        'labels': LABEL_PRIORITY,
        'label_map': LABEL_MAP,
        'examples': per_step,
    }

    step_path = '/tmp/c2_train_per_step.json'
    with open(step_path, 'w') as f:
        json.dump(step_output, f)
    print(f"\nSaved per-step to {step_path}")

    s3.upload_file(step_path, BUCKET, 'c2_training/c2_train_per_step.json')
    print(f"Uploaded to s3://{BUCKET}/c2_training/c2_train_per_step.json")

    # Per-problem data (multi-label + count)
    problem_output = {
        'n_examples': len(per_problem),
        'labels': LABEL_PRIORITY,
        'label_map': LABEL_MAP,
        'examples': per_problem,
    }

    problem_path = '/tmp/c2_train_per_problem.json'
    with open(problem_path, 'w') as f:
        json.dump(problem_output, f)
    print(f"\nSaved per-problem to {problem_path}")

    s3.upload_file(problem_path, BUCKET, 'c2_training/c2_train_per_problem.json')
    print(f"Uploaded to s3://{BUCKET}/c2_training/c2_train_per_problem.json")

    return step_output, problem_output


def main():
    print("=" * 70)
    print("C2 TRAINING DATA GENERATION")
    print("15 Operational Labels")
    print("=" * 70)

    steps, y_by_hash = load_data()
    per_step, per_problem = generate_c2_data(steps, y_by_hash)
    save_data(per_step, per_problem)

    print("\n" + "=" * 70)
    print("DONE - Ready for C2 training")
    print("=" * 70)
    print("\nPer-step: Single-label classification (step → operation)")
    print("Per-problem: Multi-label + count (problem → operations + heartbeat)")
    print("\nUse per-problem data for dual-head C2 with count regularization")


if __name__ == '__main__':
    main()
