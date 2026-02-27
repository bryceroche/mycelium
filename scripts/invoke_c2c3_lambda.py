#!/usr/bin/env python3
"""
Invoke C2/C3 extraction Lambda on all IAF chunks (parallel MapReduce).

Lists all chunks in S3, invokes Lambda for each in parallel, monitors progress.

Usage:
  # Dry run - show what would be invoked
  python invoke_c2c3_lambda.py --dry-run

  # Invoke all chunks (up to 50 concurrent)
  python invoke_c2c3_lambda.py --concurrency 50

  # Invoke specific chunks
  python invoke_c2c3_lambda.py --limit 5
"""

import boto3
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Config
LAMBDA_FUNCTION = "mycelium-c2c3-extractor"
INPUT_BUCKET = "mycelium-data"
INPUT_PREFIX = "iaf_extraction/chunked/"
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
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        result = json.loads(response['Payload'].read().decode('utf-8'))
        elapsed = time.time() - start_time

        if response['StatusCode'] == 200 and result.get('statusCode') == 200:
            body = result.get('body', {})
            return {
                'chunk': chunk_key,
                'status': 'success',
                'c2_records': body.get('c2_records', 0),
                'c3_records': body.get('c3_records', 0),
                'operation_counts': body.get('operation_counts', {}),
                'elapsed': elapsed
            }
        else:
            return {
                'chunk': chunk_key,
                'status': 'error',
                'error': result.get('errorMessage', str(result)),
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
    parser = argparse.ArgumentParser(description='Invoke C2/C3 extraction Lambda on chunks')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be invoked')
    parser.add_argument('--concurrency', type=int, default=50, help='Max concurrent invocations')
    parser.add_argument('--limit', type=int, help='Limit number of chunks to process')
    parser.add_argument('--region', default=REGION, help='AWS region')
    args = parser.parse_args()

    print("=" * 60)
    print("C2/C3 EXTRACTION - LAMBDA MAPREDUCE")
    print("=" * 60)

    # Initialize clients
    s3 = boto3.client('s3', region_name=args.region)
    lambda_client = boto3.client('lambda', region_name=args.region)

    # List chunks
    print(f"\nListing chunks in s3://{INPUT_BUCKET}/{INPUT_PREFIX}...")
    chunks = list_chunks(s3, INPUT_BUCKET, INPUT_PREFIX)

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Limited to first {args.limit} chunks")

    print(f"Found {len(chunks)} chunks to process")

    if args.dry_run:
        print("\n[DRY RUN] Would invoke Lambda for:")
        for chunk in chunks[:10]:
            print(f"  - {chunk}")
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 10} more")
        print(f"\nEstimated time: ~{len(chunks) * 30 / args.concurrency:.0f}s")
        print(f"(assuming ~30s per chunk with {args.concurrency} concurrency)")
        return

    # Invoke in parallel
    print(f"\nInvoking {len(chunks)} Lambdas (concurrency: {args.concurrency})...")
    print("-" * 60)

    start_time = time.time()
    results = []
    completed = 0

    total_c2 = 0
    total_c3 = 0
    op_totals = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(invoke_lambda, lambda_client, LAMBDA_FUNCTION, chunk): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Accumulate stats
            if result['status'] == 'success':
                total_c2 += result.get('c2_records', 0)
                total_c3 += result.get('c3_records', 0)
                for op, count in result.get('operation_counts', {}).items():
                    op_totals[op] = op_totals.get(op, 0) + count

            # Progress update
            status = "✓" if result['status'] == 'success' else "✗"
            chunk_name = result['chunk'].split('/')[-1]
            c2 = result.get('c2_records', 0)
            c3 = result.get('c3_records', 0)
            print(f"[{completed:3d}/{len(chunks)}] {status} {chunk_name} "
                  f"(C2: {c2}, C3: {c3}, {result['elapsed']:.1f}s)")

    # Summary
    elapsed = time.time() - start_time
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] != 'success']

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Chunks processed: {len(successes)}/{len(chunks)}")
    print(f"Total C2 records: {total_c2:,}")
    print(f"Total C3 records: {total_c3:,}")
    print(f"\nOperation distribution:")
    for op, count in sorted(op_totals.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count:,}")

    print(f"\nC2 output: s3://{INPUT_BUCKET}/training/c2_multilabel/")
    print(f"C3 output: s3://{INPUT_BUCKET}/training/c3_fulltext/")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:5]:
            print(f"  - {f['chunk']}: {f.get('error', 'Unknown')}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    # Merge command
    print("\nTo merge all results:")
    print(f"  aws s3 cp s3://{INPUT_BUCKET}/training/c2_multilabel/ ./c2_multilabel/ --recursive")
    print(f"  cat c2_multilabel/*.json | jq -s 'add' > c2_training_data.json")


if __name__ == "__main__":
    main()
