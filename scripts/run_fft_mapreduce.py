#!/usr/bin/env python3
"""
FFT MapReduce Orchestrator

Invokes FFT Lambda on all IAF chunks in parallel and aggregates results.
"""

import boto3
import json
import concurrent.futures
from typing import List, Dict
import time

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

BUCKET = 'mycelium-data'
INPUT_PREFIX = 'iaf_extraction/chunked/'
FUNCTION_NAME = 'mycelium-fft-analyzer'
MAX_WORKERS = 50  # Lambda concurrent invocations


def list_chunks() -> List[str]:
    """List all IAF chunk keys."""
    chunks = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET, Prefix=INPUT_PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                chunks.append(key)

    return chunks


def invoke_lambda(key: str) -> Dict:
    """Invoke FFT Lambda on a single chunk."""
    try:
        response = lambda_client.invoke(
            FunctionName=FUNCTION_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps({'key': key})
        )

        result = json.loads(response['Payload'].read().decode('utf-8'))
        return {
            'key': key,
            'success': True,
            'result': result
        }
    except Exception as e:
        return {
            'key': key,
            'success': False,
            'error': str(e)
        }


def run_mapreduce(max_chunks: int = None):
    """Run FFT analysis on all chunks."""
    print("Listing chunks...")
    chunks = list_chunks()
    print(f"Found {len(chunks)} chunks")

    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"Processing first {max_chunks} chunks")

    # Invoke in parallel
    print(f"Invoking Lambda with {MAX_WORKERS} workers...")
    start_time = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(invoke_lambda, key): key for key in chunks}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(chunks)} chunks ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Aggregate results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed[:5]:
            print(f"  - {f['key']}: {f['error']}")

    # Aggregate statistics
    all_stats = []
    total_records = 0
    total_valid = 0

    for r in successful:
        result = r['result']
        if 'chunk_stats' in result:
            all_stats.append(result['chunk_stats'])
            total_records += result.get('n_records', 0)
            total_valid += result.get('n_valid', 0)

    # Compute global correlation
    correlations = [s['correlation_fft_cot'] for s in all_stats
                   if s.get('correlation_fft_cot') is not None]

    if correlations:
        import numpy as np
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        print(f"\n=== GLOBAL RESULTS ===")
        print(f"Total records: {total_records}")
        print(f"Valid FFT: {total_valid}")
        print(f"Mean correlation (FFT vs CoT ops): {mean_corr:.4f} +/- {std_corr:.4f}")
        print(f"Correlation range: [{min(correlations):.4f}, {max(correlations):.4f}]")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-chunks', type=int, default=None,
                       help='Max chunks to process (default: all)')
    parser.add_argument('--output', type=str, default='/tmp/fft_mapreduce_results.json')
    args = parser.parse_args()

    results = run_mapreduce(max_chunks=args.max_chunks)

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'n_chunks': len(results),
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")
