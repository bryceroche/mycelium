#!/usr/bin/env python3
"""
Run Y Extraction via Lambda MapReduce

Extracts sympy root operators (Y labels) from all CoT steps.
These Y labels are the target variable for true Information Bottleneck.

Usage:
    # Run full extraction via Lambda
    python run_y_extraction.py

    # Test locally on a sample
    python run_y_extraction.py --local --sample 100
"""

import json
import argparse
import boto3
import concurrent.futures
import time
from collections import Counter

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

BUCKET = 'mycelium-data'
STEPS_KEY = 'ib_results_v2/aggregated_steps.json'
OUTPUT_PREFIX = 'ib_y_labels/'
LAMBDA_FUNCTION = 'mycelium-sympy-y-extractor'

CHUNK_SIZE = 5000
MAX_WORKERS = 20


def load_steps():
    """Load aggregated steps from S3."""
    print(f"Loading steps from s3://{BUCKET}/{STEPS_KEY}...")
    response = s3.get_object(Bucket=BUCKET, Key=STEPS_KEY)
    data = json.loads(response['Body'].read().decode('utf-8'))
    print(f"  Loaded {data['n_steps']:,} steps")
    return data['steps']


def invoke_lambda(chunk_start: int) -> dict:
    """Invoke Lambda for a chunk."""
    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'key': STEPS_KEY,
                'chunk_start': chunk_start,
                'chunk_size': CHUNK_SIZE,
            })
        )
        result = json.loads(response['Payload'].read().decode('utf-8'))
        return {'chunk_start': chunk_start, 'success': True, 'result': result}
    except Exception as e:
        return {'chunk_start': chunk_start, 'success': False, 'error': str(e)}


def run_lambda_mapreduce(n_steps: int):
    """Run extraction via Lambda MapReduce."""
    # Calculate chunks
    chunks = list(range(0, n_steps, CHUNK_SIZE))
    print(f"\n[LAMBDA MAPREDUCE]")
    print(f"  Total steps: {n_steps:,}")
    print(f"  Chunk size: {CHUNK_SIZE:,}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Workers: {MAX_WORKERS}")

    start_time = time.time()
    results = []
    success_count = 0
    total_y_counts = Counter()
    total_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(invoke_lambda, cs): cs for cs in chunks}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)

            if result['success']:
                success_count += 1
                r = result['result']
                if 'y_distribution' in r:
                    for y, count in r['y_distribution'].items():
                        total_y_counts[y] += count
                total_failures += r.get('parse_failures', 0)
            else:
                print(f"  ERROR chunk {result['chunk_start']}: {result.get('error', 'unknown')}")

            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                elapsed = time.time() - start_time
                print(f"  {i+1}/{len(chunks)} chunks ({success_count} OK, {elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n  Complete: {success_count}/{len(chunks)} chunks in {elapsed:.1f}s")
    print(f"  Parse failures: {total_failures:,}")

    return total_y_counts


def run_local_test(steps: list, sample_size: int = 100):
    """Test Y extraction locally."""
    # Import the Lambda function locally
    import sys
    sys.path.insert(0, 'lambda/sympy_y_extractor')
    from lambda_function import extract_Y

    print(f"\n[LOCAL TEST]")
    print(f"  Sample size: {sample_size}")

    sample = steps[:sample_size]
    y_counts = Counter()  # Count individual operators
    failures = 0
    steps_with_y = 0

    for step in sample:
        y_set, expr = extract_Y(step['text'])
        if y_set:
            steps_with_y += 1
            for op in y_set:
                y_counts[op] += 1
        else:
            failures += 1

    print(f"\n  Parse failures: {failures}/{sample_size} ({100*failures/sample_size:.1f}%)")
    print(f"  Steps with Y: {steps_with_y}/{sample_size} ({100*steps_with_y/sample_size:.1f}%)")
    print(f"\n  Operator Distribution (operators can appear multiple times per step):")
    for op, count in y_counts.most_common(25):
        print(f"    {op:20} {count:5} ({100*count/sum(y_counts.values()):.1f}%)")

    return y_counts


def aggregate_results():
    """Aggregate all chunk results into a single file."""
    print("\n[AGGREGATE]")

    # List all chunk files
    paginator = s3.get_paginator('list_objects_v2')
    chunk_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json') and 'chunk' in obj['Key']:
                chunk_keys.append(obj['Key'])

    print(f"  Found {len(chunk_keys)} chunk files")

    # Aggregate
    all_results = []
    total_y_counts = Counter()

    for key in sorted(chunk_keys):
        response = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        all_results.extend(data['results'])
        for y, count in data.get('y_distribution', {}).items():
            total_y_counts[y] += count

    # Write aggregated file
    output = {
        'n_steps': len(all_results),
        'y_distribution': dict(total_y_counts),
        'results': all_results,
    }

    output_key = f'{OUTPUT_PREFIX}aggregated_y_labels.json'
    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output),
        ContentType='application/json'
    )
    print(f"  Wrote s3://{BUCKET}/{output_key}")

    return total_y_counts


def main():
    parser = argparse.ArgumentParser(description='Extract Y labels via Lambda MapReduce')
    parser.add_argument('--local', action='store_true', help='Run locally (test mode)')
    parser.add_argument('--sample', type=int, default=100, help='Sample size for local test')
    parser.add_argument('--aggregate-only', action='store_true', help='Only aggregate existing results')
    args = parser.parse_args()

    print("=" * 70)
    print("Y EXTRACTION: SymPy Root Operators")
    print("=" * 70)
    print("\nThis extracts the primary sympy operator from each CoT step.")
    print("Y = root of sympy AST: Add, Mul, Pow, sin, cos, etc.")
    print("This is the target variable for true Information Bottleneck.")

    if args.aggregate_only:
        y_counts = aggregate_results()
    elif args.local:
        steps = load_steps()
        y_counts = run_local_test(steps, args.sample)
    else:
        steps = load_steps()
        y_counts = run_lambda_mapreduce(len(steps))
        y_counts = aggregate_results()

    # Print final distribution
    print("\n" + "=" * 70)
    print("Y DISTRIBUTION (Top 25)")
    print("=" * 70)
    total = sum(y_counts.values())
    for y, count in y_counts.most_common(25):
        print(f"  {y:20} {count:6,} ({100*count/total:.1f}%)")

    print(f"\n  Total with Y: {total:,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
