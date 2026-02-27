#!/usr/bin/env python3
"""
Invoke C1 Relevance Lambda on all golden_medusa chunks (parallel MapReduce)

Lists all chunks in S3, invokes Lambda for each in parallel, monitors progress.

Usage:
  # Dry run - show what would be invoked
  python invoke_c1_relevance_lambda.py --dry-run

  # Invoke all chunks (up to 50 concurrent)
  python invoke_c1_relevance_lambda.py --concurrency 50

  # Invoke specific chunks
  python invoke_c1_relevance_lambda.py --chunks chunk_000.json chunk_001.json
"""

import boto3
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Config
LAMBDA_FUNCTION = "mycelium-c1-relevance-extractor"
INPUT_BUCKET = "mycelium-data"
INPUT_PREFIX = "iaf_extraction/chunked/"
OUTPUT_PREFIX = "c1_relevance/chunks/"
REGION = "us-east-1"


def list_chunks(s3_client, bucket: str, prefix: str) -> List[str]:
    """List all chunk files in S3."""
    chunks = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json') and 'chunk' in key:
                chunks.append(key)

    return sorted(chunks)


def invoke_lambda(lambda_client, function_name: str, chunk_key: str) -> Dict:
    """Invoke Lambda for a single chunk."""
    payload = {
        'bucket': INPUT_BUCKET,
        'key': chunk_key
    }

    start_time = time.time()

    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps(payload)
        )

        result = json.loads(response['Payload'].read().decode('utf-8'))
        elapsed = time.time() - start_time

        if response['StatusCode'] == 200:
            body = result.get('body', {})
            return {
                'chunk': chunk_key,
                'status': 'success',
                'input_records': body.get('total_input_records', 0),
                'output_records': body.get('total_output_records', 0),
                'elapsed': elapsed
            }
        else:
            return {
                'chunk': chunk_key,
                'status': 'error',
                'error': result.get('errorMessage', 'Unknown error'),
                'elapsed': elapsed
            }

    except Exception as e:
        return {
            'chunk': chunk_key,
            'status': 'error',
            'error': str(e),
            'elapsed': time.time() - start_time
        }


def main():
    parser = argparse.ArgumentParser(description='Invoke C1 Relevance Lambda on chunks')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be invoked')
    parser.add_argument('--concurrency', type=int, default=50, help='Max concurrent invocations')
    parser.add_argument('--chunks', nargs='+', help='Specific chunks to process')
    parser.add_argument('--region', default=REGION, help='AWS region')
    args = parser.parse_args()

    print("=" * 60)
    print("C1 RELEVANCE EXTRACTION - LAMBDA MAPREDUCE")
    print("=" * 60)

    # Initialize clients
    s3 = boto3.client('s3', region_name=args.region)
    lambda_client = boto3.client('lambda', region_name=args.region)

    # List chunks
    print(f"\nListing chunks in s3://{INPUT_BUCKET}/{INPUT_PREFIX}...")
    all_chunks = list_chunks(s3, INPUT_BUCKET, INPUT_PREFIX)

    if args.chunks:
        # Filter to specific chunks
        chunks = [f"{INPUT_PREFIX}{c}" if not c.startswith(INPUT_PREFIX) else c
                  for c in args.chunks]
    else:
        chunks = all_chunks

    print(f"Found {len(all_chunks)} total chunks, processing {len(chunks)}")

    if args.dry_run:
        print("\n[DRY RUN] Would invoke Lambda for:")
        for chunk in chunks[:10]:
            print(f"  - {chunk}")
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 10} more")
        print(f"\nTotal: {len(chunks)} invocations @ ~10s each")
        print(f"With {args.concurrency} concurrency: ~{len(chunks) * 10 / args.concurrency:.0f}s")
        return

    # Invoke in parallel
    print(f"\nInvoking {len(chunks)} Lambdas (concurrency: {args.concurrency})...")
    print("-" * 60)

    start_time = time.time()
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(invoke_lambda, lambda_client, LAMBDA_FUNCTION, chunk): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Progress update
            status = "✓" if result['status'] == 'success' else "✗"
            chunk_name = result['chunk'].split('/')[-1]
            print(f"[{completed:3d}/{len(chunks)}] {status} {chunk_name} "
                  f"({result.get('output_records', 0)} records, {result['elapsed']:.1f}s)")

    # Summary
    elapsed = time.time() - start_time
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] != 'success']

    total_input = sum(r.get('input_records', 0) for r in successes)
    total_output = sum(r.get('output_records', 0) for r in successes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Chunks processed: {len(successes)}/{len(chunks)}")
    print(f"Total input records: {total_input:,}")
    print(f"Total output records: {total_output:,}")
    print(f"Output location: s3://{INPUT_BUCKET}/{OUTPUT_PREFIX}")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:5]:
            print(f"  - {f['chunk']}: {f.get('error', 'Unknown')}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    # Optionally merge results
    print("\nTo merge all results into single training file:")
    print(f"  python scripts/merge_c1_relevance.py --input s3://{INPUT_BUCKET}/{OUTPUT_PREFIX}")


if __name__ == "__main__":
    main()
