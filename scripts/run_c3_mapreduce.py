#!/usr/bin/env python3
"""
Run C3 Training Data Extraction via Lambda MapReduce

Processes all IAF chunks in parallel using Lambda.
Each Lambda extracts expression training pairs from CoT.

Output: s3://mycelium-data/c3_training/c3_train_fulltext.jsonl
"""

import json
import boto3
import concurrent.futures
import time
from collections import Counter

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda', region_name='us-east-1')

BUCKET = 'mycelium-data'
CHUNK_PREFIX = 'iaf_extraction/chunked/'
OUTPUT_PREFIX = 'c3_training/chunks/'
LAMBDA_FUNCTION = 'mycelium-c3-extractor'


def list_chunks():
    """List all IAF chunk files."""
    chunks = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=CHUNK_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                chunks.append(obj['Key'])
    return sorted(chunks)


def invoke_lambda_async(chunk_key: str) -> dict:
    """Invoke Lambda asynchronously - don't wait for response."""
    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION,
            InvocationType='Event',  # Async
            Payload=json.dumps({'chunk_key': chunk_key}),
        )
        return {'chunk_key': chunk_key, 'success': True, 'status_code': response['StatusCode']}
    except Exception as e:
        return {'chunk_key': chunk_key, 'success': False, 'error': str(e)}


def aggregate_results():
    """Aggregate all chunk results into final JSONL file."""
    print("\n[AGGREGATE]")

    # List all chunk output files
    paginator = s3.get_paginator('list_objects_v2')
    chunk_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('_c3.json'):
                chunk_keys.append(obj['Key'])

    print(f"  Found {len(chunk_keys)} chunk outputs")

    # Aggregate
    all_pairs = []
    total_op_counts = Counter()

    for key in sorted(chunk_keys):
        response = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))

        all_pairs.extend(data['pairs'])
        for op, cnt in data.get('op_counts', {}).items():
            total_op_counts[op] += cnt

    print(f"  Total training pairs: {len(all_pairs)}")

    # Write as JSONL
    output_key = 'c3_training/c3_train_fulltext.jsonl'
    jsonl_content = '\n'.join(json.dumps(pair) for pair in all_pairs)

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=jsonl_content,
        ContentType='application/jsonl'
    )
    print(f"  Wrote s3://{BUCKET}/{output_key}")

    return total_op_counts, len(all_pairs)


def main():
    print("=" * 70)
    print("C3 TRAINING DATA EXTRACTION - LAMBDA MAPREDUCE (ASYNC)")
    print("=" * 70)

    chunks = list_chunks()
    print(f"\nFound {len(chunks)} IAF chunks")
    print(f"Launching {len(chunks)} async Lambda invocations...")

    # Launch all Lambdas asynchronously
    start_time = time.time()
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = list(executor.map(invoke_lambda_async, chunks))

    for r in futures:
        if r['success']:
            success_count += 1

    elapsed = time.time() - start_time
    print(f"  Launched {success_count}/{len(chunks)} in {elapsed:.1f}s")

    # Wait for results to appear in S3
    print("\nWaiting for Lambda results in S3...")
    expected_outputs = len(chunks)
    max_wait = 600  # 10 minutes max
    poll_interval = 10

    for _ in range(max_wait // poll_interval):
        time.sleep(poll_interval)
        # Count chunk outputs
        chunk_keys = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('_c3.json'):
                    chunk_keys.append(obj['Key'])

        elapsed = time.time() - start_time
        print(f"  {len(chunk_keys)}/{expected_outputs} chunks complete ({elapsed:.0f}s)")

        if len(chunk_keys) >= expected_outputs:
            break

    print(f"\n  Lambda processing complete")

    # Aggregate
    op_counts, n_pairs = aggregate_results()

    # Print distribution
    print("\n" + "=" * 70)
    print("OPERATION DISTRIBUTION")
    print("=" * 70)

    for op in sorted(op_counts.keys(), key=lambda x: -op_counts[x]):
        count = op_counts[op]
        pct = 100 * count / n_pairs if n_pairs > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {op:12}: {count:5} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 70)
    print("DONE - Ready for C3 training")
    print("=" * 70)
    print(f"\nData: s3://{BUCKET}/c3_training/c3_train_fulltext.jsonl")
    print(f"Total pairs: {n_pairs}")


if __name__ == '__main__':
    main()
