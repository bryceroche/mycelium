#!/usr/bin/env python3
"""
Run C2 Label Extraction via Lambda MapReduce

Processes all 117 IAF chunks in parallel using Lambda.
Each Lambda invocation processes one chunk (~50 problems).

Output: s3://mycelium-data/c2_training/c2_train_full.json
"""

import json
import boto3
import concurrent.futures
import time
from collections import Counter

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

BUCKET = 'mycelium-data'
CHUNK_PREFIX = 'iaf_extraction/chunked/'
OUTPUT_PREFIX = 'c2_training/chunks/'
LAMBDA_FUNCTION = 'mycelium-c2-label-extractor'

MAX_WORKERS = 50  # Lambda can handle high concurrency


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
            InvocationType='Event',  # Async - returns immediately
            Payload=json.dumps({'chunk_key': chunk_key}),
        )
        return {'chunk_key': chunk_key, 'success': True, 'status_code': response['StatusCode']}
    except Exception as e:
        return {'chunk_key': chunk_key, 'success': False, 'error': str(e)}


def aggregate_results():
    """Aggregate all chunk results into final training file."""
    print("\n[AGGREGATE]")

    # List all chunk output files
    paginator = s3.get_paginator('list_objects_v2')
    chunk_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('_c2.json'):
                chunk_keys.append(obj['Key'])

    print(f"  Found {len(chunk_keys)} chunk outputs")

    # Aggregate
    all_examples = []
    total_label_counts = Counter()

    for key in sorted(chunk_keys):
        response = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        all_examples.extend(data['examples'])
        for lbl, cnt in data.get('label_counts', {}).items():
            total_label_counts[lbl] += cnt

    print(f"  Total examples: {len(all_examples)}")

    # Labels
    LABELS = [
        "FACTORIAL", "LOG", "TRIG", "MOD",
        "SQRT", "CUBE", "FRAC_POW", "HIGH_POW", "SQUARE",
        "EQUATION", "DIV", "MUL", "ADD", "OTHER",
    ]

    # Save
    output = {
        'n_examples': len(all_examples),
        'labels': LABELS,
        'label_distribution': dict(total_label_counts),
        'examples': all_examples,
    }

    output_key = 'c2_training/c2_train_full.json'
    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output),
        ContentType='application/json'
    )
    print(f"  Wrote s3://{BUCKET}/{output_key}")

    return total_label_counts, len(all_examples)


def main():
    print("=" * 70)
    print("C2 LABEL EXTRACTION - LAMBDA MAPREDUCE (ASYNC)")
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
                if obj['Key'].endswith('_c2.json'):
                    chunk_keys.append(obj['Key'])

        elapsed = time.time() - start_time
        print(f"  {len(chunk_keys)}/{expected_outputs} chunks complete ({elapsed:.0f}s)")

        if len(chunk_keys) >= expected_outputs:
            break

    print(f"\n  Lambda processing complete")

    # Aggregate
    label_counts, n_examples = aggregate_results()

    # Print distribution
    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    for label in sorted(label_counts.keys(), key=lambda x: -label_counts[x]):
        count = label_counts[label]
        pct = 100 * count / n_examples if n_examples > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:12}: {count:5} ({pct:5.1f}%) {bar}")

    print("\n" + "=" * 70)
    print("DONE - Ready for C2 training")
    print("=" * 70)
    print(f"\nData: s3://{BUCKET}/c2_training/c2_train_full.json")
    print(f"Train: python train/train_c2.py --no-count-head")


if __name__ == '__main__':
    main()
