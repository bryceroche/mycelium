#!/usr/bin/env python3
"""
Invoke C4/C5/C6 extraction Lambda for all IAF chunks.

Uses parallel invocations for fast processing.
"""

import boto3
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')


def list_iaf_chunks(bucket: str, prefix: str) -> List[str]:
    """List all IAF chunk files in S3."""
    chunks = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json') and 'chunk' in key:
                chunks.append(key)

    return chunks


def invoke_extraction(
    bucket: str,
    key: str,
    output_prefix: str,
    function_name: str,
    dry_run: bool = False
) -> Dict:
    """Invoke Lambda for a single chunk."""
    payload = {
        'bucket': bucket,
        'key': key,
        'output_prefix': output_prefix
    }

    if dry_run:
        return {'status': 'dry_run', 'key': key}

    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        result = json.loads(response['Payload'].read().decode('utf-8'))
        return {'status': 'success', 'key': key, 'result': result}

    except Exception as e:
        return {'status': 'error', 'key': key, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Invoke C456 extraction Lambda')
    parser.add_argument('--bucket', default='mycelium-data', help='S3 bucket')
    parser.add_argument('--prefix', default='iaf_extraction/chunked/', help='IAF chunks prefix')
    parser.add_argument('--output-prefix', default='training_data/c456', help='Output prefix')
    parser.add_argument('--function-name', default='c456-extractor', help='Lambda function name')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers')
    parser.add_argument('--dry-run', action='store_true', help='List chunks without invoking')
    parser.add_argument('--limit', type=int, help='Limit number of chunks to process')
    args = parser.parse_args()

    # List chunks
    print(f"Listing chunks in s3://{args.bucket}/{args.prefix}")
    chunks = list_iaf_chunks(args.bucket, args.prefix)
    print(f"Found {len(chunks)} chunks")

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Processing first {args.limit} chunks")

    if args.dry_run:
        print("\nDry run - would process:")
        for chunk in chunks[:10]:
            print(f"  {chunk}")
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 10} more")
        return

    # Process in parallel
    print(f"\nInvoking Lambda with {args.workers} workers...")
    start_time = time.time()
    success = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                invoke_extraction,
                args.bucket,
                key,
                args.output_prefix,
                args.function_name,
                args.dry_run
            ): key
            for key in chunks
        }

        for future in as_completed(futures):
            result = future.result()
            key = futures[future]

            if result['status'] == 'success':
                success += 1
                body = result['result'].get('body', {})
                print(f"✓ {key.split('/')[-1]}: "
                      f"C4={body.get('c4_spans', 0)}, "
                      f"C5={body.get('c5_edges', 0)}, "
                      f"C6={body.get('c6_labels', 0)}")
            else:
                errors += 1
                print(f"✗ {key.split('/')[-1]}: {result.get('error', 'unknown error')}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Success: {success}, Errors: {errors}")


if __name__ == '__main__':
    main()
