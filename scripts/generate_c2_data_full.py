#!/usr/bin/env python3
"""
Generate C2 Training Data from ALL MATH problems (5850+)

Reads raw IAF chunks, extracts Y labels from CoT text.
Skip heartbeat count for now - just get multi-label ops per problem.

Uses Lambda MapReduce for parallel processing of 117 chunks.
"""

import json
import boto3
import re
import concurrent.futures
from collections import Counter
from typing import Set, List, Dict, Optional
import time

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

BUCKET = 'mycelium-data'
CHUNK_PREFIX = 'iaf_extraction/chunked/'
OUTPUT_KEY = 'c2_training/c2_train_full.json'

# 15 Operational Labels
LABELS = [
    "FACTORIAL", "LOG", "TRIG", "MOD",
    "SQRT", "CUBE", "FRAC_POW", "HIGH_POW", "SQUARE",
    "EQUATION", "DIV", "MUL", "ADD", "OTHER",
]

# Pattern-based Y extraction (faster than sympy for full CoT text)
# We look for operation indicators in the text
PATTERNS = {
    'FACTORIAL': [r'\d+!', r'factorial', r'permutation', r'combination', r'\\binom'],
    'LOG': [r'\\log', r'\\ln', r'\blog\b', r'\bln\b', r'logarithm'],
    'TRIG': [r'\\sin', r'\\cos', r'\\tan', r'\\cot', r'\\sec', r'\\csc',
             r'\bsin\b', r'\bcos\b', r'\btan\b', r'trigonometr'],
    'MOD': [r'\\mod', r'\\pmod', r'\bmod\b', r'modulo', r'remainder'],
    'SQRT': [r'\\sqrt', r'square root', r'√'],
    'CUBE': [r'\^3\b', r'\^{3}', r'cubed', r'cube root'],
    'FRAC_POW': [r'\^\{?\\frac', r'\^\{?\d+/\d+\}?'],
    'HIGH_POW': [r'\^[4-9]\b', r'\^{[4-9]}', r'\^\d{2,}', r'\^{1[0-9]}'],
    'SQUARE': [r'\^2\b', r'\^{2}', r'squared', r'square of'],
    'DIV': [r'\\frac', r'\\div', r'÷', r'divided by', r'ratio'],
    'MUL': [r'\\times', r'\\cdot', r'×', r'·', r'multiply', r'product'],
    'ADD': [r'\+', r'\\pm', r'sum', r'total', r'add', r'plus', r'minus', r'-'],
}


def extract_labels_from_text(text: str) -> Set[str]:
    """Extract operation labels from CoT text using patterns."""
    labels = set()
    text_lower = text.lower()

    for label, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                labels.add(label)
                break

    # Default to OTHER if nothing found
    if not labels:
        labels.add('OTHER')

    return labels


def process_chunk_locally(chunk_key: str) -> List[Dict]:
    """Process a single chunk locally."""
    response = s3.get_object(Bucket=BUCKET, Key=chunk_key)
    data = json.loads(response['Body'].read().decode('utf-8'))

    results = []
    for problem in data:
        cot = problem.get('generated_cot', '')
        problem_text = problem.get('problem_text', '')

        labels = extract_labels_from_text(cot)

        results.append({
            'problem_idx': problem.get('problem_idx'),
            'text': problem_text,
            'labels': list(labels),
            'level': problem.get('level'),
            'type': problem.get('type'),
        })

    return results


def list_chunks() -> List[str]:
    """List all IAF chunk files."""
    chunks = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=CHUNK_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                chunks.append(obj['Key'])
    return sorted(chunks)


def process_all_chunks(max_workers: int = 10):
    """Process all chunks in parallel."""
    chunks = list_chunks()
    print(f"Found {len(chunks)} chunks")

    all_results = []
    label_counts = Counter()

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk_locally, c): c for c in chunks}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            chunk_key = futures[future]
            try:
                results = future.result()
                all_results.extend(results)

                for r in results:
                    for lbl in r['labels']:
                        label_counts[lbl] += 1

                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    elapsed = time.time() - start_time
                    print(f"  {i+1}/{len(chunks)} chunks, {len(all_results)} problems ({elapsed:.0f}s)")

            except Exception as e:
                print(f"  ERROR {chunk_key}: {e}")

    return all_results, label_counts


def save_results(results: List[Dict], label_counts: Counter):
    """Save results to S3."""
    output = {
        'n_examples': len(results),
        'labels': LABELS,
        'label_distribution': dict(label_counts),
        'examples': results,
    }

    # Save locally
    local_path = '/tmp/c2_train_full.json'
    with open(local_path, 'w') as f:
        json.dump(output, f)
    print(f"\nSaved to {local_path}")

    # Upload to S3
    s3.upload_file(local_path, BUCKET, OUTPUT_KEY)
    print(f"Uploaded to s3://{BUCKET}/{OUTPUT_KEY}")

    return output


def main():
    print("=" * 70)
    print("C2 TRAINING DATA FROM ALL MATH PROBLEMS")
    print("=" * 70)

    results, label_counts = process_all_chunks(max_workers=10)

    print(f"\nTotal problems: {len(results)}")

    # Label distribution
    print(f"\nLabel distribution (problems containing each):")
    total = len(results)
    for label in LABELS:
        count = label_counts.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:12}: {count:5} ({pct:5.1f}%) {bar}")

    # Labels per problem
    labels_per_prob = [len(r['labels']) for r in results]
    import numpy as np
    print(f"\nLabels per problem: min={min(labels_per_prob)}, max={max(labels_per_prob)}, mean={np.mean(labels_per_prob):.1f}")

    save_results(results, label_counts)

    print("\n" + "=" * 70)
    print("DONE - Ready for C2 training")
    print("=" * 70)


if __name__ == '__main__':
    main()
