#!/usr/bin/env python3
"""
Invoke IAF Chunker Lambda on all GPU files (parallel MapReduce)

Lists all valid.json files in S3, invokes Lambda for each in parallel.
8 files = 8 parallel Lambdas = ~15 minutes total (vs 1+ hour sequential)

Usage:
  # Dry run - show what would be invoked
  python invoke_chunker_lambda.py --dry-run

  # Invoke all GPU files in parallel
  python invoke_chunker_lambda.py

  # Invoke specific files
  python invoke_chunker_lambda.py --files iaf_v3_gpu0_valid.json iaf_v3_gpu1_valid.json
"""

import boto3
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Config
LAMBDA_FUNCTION = "mycelium-iaf-chunker"
INPUT_BUCKET = "mycelium-data"
INPUT_PREFIX = "iaf_extraction/"
REGION = "us-east-1"


def list_iaf_files(s3_client, bucket: str, prefix: str) -> List[str]:
    """List all valid.json IAF files in S3 (including nested instance dirs)."""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Only process valid.json files (not already chunked)
            if key.endswith('_valid.json') and '/chunked/' not in key:
                files.append(key)

    return sorted(files)


def invoke_lambda(lambda_client, function_name: str, bucket: str, key: str) -> Dict:
    """Invoke Lambda for a single file."""
    payload = {
        'bucket': bucket,
        'key': key
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

        if response['StatusCode'] == 200 and 'errorMessage' not in result:
            body = result.get('body', {})
            return {
                'file': key,
                'status': 'success',
                'chunks_created': body.get('chunks_created', 0),
                'input_records': body.get('input_records', 0),
                'total_size_mb': body.get('total_size_mb', 0),
                'elapsed': elapsed
            }
        else:
            return {
                'file': key,
                'status': 'error',
                'error': result.get('errorMessage', str(result)),
                'elapsed': elapsed
            }

    except Exception as e:
        return {
            'file': key,
            'status': 'error',
            'error': str(e),
            'elapsed': time.time() - start_time
        }


def main():
    parser = argparse.ArgumentParser(description='Invoke IAF Chunker Lambda on GPU files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be invoked')
    parser.add_argument('--files', nargs='+', help='Specific files to process')
    parser.add_argument('--region', default=REGION, help='AWS region')
    args = parser.parse_args()

    print("=" * 60)
    print("IAF CHUNKER - LAMBDA MAPREDUCE")
    print("=" * 60)

    # Initialize clients
    s3 = boto3.client('s3', region_name=args.region)
    lambda_client = boto3.client('lambda', region_name=args.region)

    # List files
    print(f"\nListing IAF files in s3://{INPUT_BUCKET}/{INPUT_PREFIX}...")
    all_files = list_iaf_files(s3, INPUT_BUCKET, INPUT_PREFIX)

    if args.files:
        # Filter to specific files
        files = [f"{INPUT_PREFIX}{f}" if not f.startswith(INPUT_PREFIX) else f
                 for f in args.files]
    else:
        files = all_files

    print(f"Found {len(all_files)} total files, processing {len(files)}")
    for f in files:
        print(f"  - {f}")

    if args.dry_run:
        print("\n[DRY RUN] Would invoke Lambda for each file")
        print(f"Total: {len(files)} parallel Lambda invocations")
        print(f"Estimated time: ~15 minutes (all running in parallel)")
        return

    # Invoke all in parallel (max 8 concurrent)
    print(f"\nInvoking {len(files)} Lambdas in parallel...")
    print("-" * 60)

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=len(files)) as executor:
        futures = {
            executor.submit(invoke_lambda, lambda_client, LAMBDA_FUNCTION, INPUT_BUCKET, f): f
            for f in files
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = "+" if result['status'] == 'success' else "X"
            filename = result['file'].split('/')[-1]
            if result['status'] == 'success':
                print(f"[{status}] {filename}: {result['chunks_created']} chunks, "
                      f"{result['input_records']} records, {result['elapsed']:.0f}s")
            else:
                print(f"[{status}] {filename}: ERROR - {result.get('error', 'Unknown')}")

    # Summary
    elapsed = time.time() - start_time
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] != 'success']

    total_chunks = sum(r.get('chunks_created', 0) for r in successes)
    total_records = sum(r.get('input_records', 0) for r in successes)
    total_size = sum(r.get('total_size_mb', 0) for r in successes)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.0f}s")
    print(f"Files processed: {len(successes)}/{len(files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total records: {total_records:,}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Output location: s3://{INPUT_BUCKET}/iaf_extraction/chunked/")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f['file']}: {f.get('error', 'Unknown')}")

    # Next steps
    print("\nChunks are ready for C1 relevance extraction:")
    print("  python scripts/invoke_c1_relevance_lambda.py")


if __name__ == "__main__":
    main()
