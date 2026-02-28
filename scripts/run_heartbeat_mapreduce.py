#!/usr/bin/env python3
"""
Run MATH Heartbeat Counting via Lambda MapReduce

Processes all IAF chunks in parallel using Lambda.
Each Lambda invocation processes one chunk (~50 problems).

Output: s3://mycelium-data/pulse_analysis/math_heartbeats.json
"""

import json
import boto3
import concurrent.futures
import time
from collections import defaultdict

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda', region_name='us-east-1')

BUCKET = 'mycelium-data'
CHUNK_PREFIX = 'iaf_extraction/chunked/'
OUTPUT_PREFIX = 'pulse_analysis/chunks/'
LAMBDA_FUNCTION = 'mycelium-heartbeat-counter'

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
    """Aggregate all chunk results into final output file."""
    print("\n[AGGREGATE]")

    # List all chunk output files
    paginator = s3.get_paginator('list_objects_v2')
    chunk_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('_heartbeats.json'):
                chunk_keys.append(obj['Key'])

    print(f"  Found {len(chunk_keys)} chunk outputs")

    # Aggregate
    all_results = []
    total_heartbeats = 0
    by_level = defaultdict(list)
    by_type = defaultdict(list)

    for key in sorted(chunk_keys):
        response = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))

        for r in data['results']:
            all_results.append(r)
            total_heartbeats += r['heartbeat_count']

            # Group by level and type
            level = r.get('level', 'Unknown')
            ptype = r.get('type', 'Unknown')
            by_level[level].append(r['heartbeat_count'])
            by_type[ptype].append(r['heartbeat_count'])

    n_problems = len(all_results)
    mean_heartbeats = total_heartbeats / n_problems if n_problems > 0 else 0

    print(f"  Total problems: {n_problems}")
    print(f"  Total heartbeats: {total_heartbeats}")
    print(f"  Mean heartbeats: {mean_heartbeats:.2f}")

    # Compute stats by level
    level_stats = {}
    for level, counts in sorted(by_level.items()):
        level_stats[level] = {
            'n': len(counts),
            'mean': round(sum(counts) / len(counts), 2),
            'min': min(counts),
            'max': max(counts),
        }

    # Compute stats by type
    type_stats = {}
    for ptype, counts in sorted(by_type.items()):
        type_stats[ptype] = {
            'n': len(counts),
            'mean': round(sum(counts) / len(counts), 2),
            'min': min(counts),
            'max': max(counts),
        }

    # Save aggregated output
    output = {
        'n_problems': n_problems,
        'total_heartbeats': total_heartbeats,
        'mean_heartbeats': round(mean_heartbeats, 2),
        'parameters': {
            'threshold': 0.5,
            'min_run_length': 5,
            'head': 'L22H4',
        },
        'by_level': level_stats,
        'by_type': type_stats,
        'results': all_results,
    }

    output_key = 'pulse_analysis/math_heartbeats.json'
    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output),
        ContentType='application/json'
    )
    print(f"  Wrote s3://{BUCKET}/{output_key}")

    return output


def main():
    print("=" * 70)
    print("MATH HEARTBEAT COUNTING - LAMBDA MAPREDUCE (ASYNC)")
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
                if obj['Key'].endswith('_heartbeats.json'):
                    chunk_keys.append(obj['Key'])

        elapsed = time.time() - start_time
        print(f"  {len(chunk_keys)}/{expected_outputs} chunks complete ({elapsed:.0f}s)")

        if len(chunk_keys) >= expected_outputs:
            break

    print(f"\n  Lambda processing complete")

    # Aggregate
    output = aggregate_results()

    # Print summary by level
    print("\n" + "=" * 70)
    print("HEARTBEATS BY LEVEL")
    print("=" * 70)

    for level, stats in sorted(output['by_level'].items()):
        bar = '#' * int(stats['mean'])
        print(f"  {level:10}: mean {stats['mean']:5.1f} (n={stats['n']:4}) {bar}")

    print("\n" + "=" * 70)
    print("HEARTBEATS BY TYPE")
    print("=" * 70)

    for ptype, stats in sorted(output['by_type'].items()):
        bar = '#' * int(stats['mean'])
        print(f"  {ptype:20}: mean {stats['mean']:5.1f} (n={stats['n']:4}) {bar}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nData: s3://{BUCKET}/pulse_analysis/math_heartbeats.json")


if __name__ == '__main__':
    main()
