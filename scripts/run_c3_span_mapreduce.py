#!/usr/bin/env python3
"""
C3 Span Reformatter MapReduce

Chunks the C3 training data and runs Lambda MapReduce to reformat
each (problem_text, template, expression) triple into span extraction format.

Usage:
    python scripts/run_c3_span_mapreduce.py --chunk      # Chunk data to S3
    python scripts/run_c3_span_mapreduce.py --run        # Run Lambda MapReduce
    python scripts/run_c3_span_mapreduce.py --collect    # Collect results
    python scripts/run_c3_span_mapreduce.py --all        # Do all steps
"""

import json
import boto3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

BUCKET = 'mycelium-data'
SOURCE_KEY = 'c3_training/c3_train_clean.jsonl'
CHUNK_PREFIX = 'c3_span_training/input_chunks/'
OUTPUT_PREFIX = 'c3_span_training/chunks/'
FINAL_OUTPUT_KEY = 'c3_span_training/c3_span_train.jsonl'
LAMBDA_FUNCTION = 'c3_span_reformatter'
CHUNK_SIZE = 500  # pairs per chunk


def chunk_data():
    """Download C3 training data, chunk it, and upload to S3."""
    print(f"Loading source data from s3://{BUCKET}/{SOURCE_KEY}...")

    response = s3.get_object(Bucket=BUCKET, Key=SOURCE_KEY)
    content = response['Body'].read().decode('utf-8')
    pairs = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    print(f"Loaded {len(pairs)} training pairs")

    # Chunk and upload
    n_chunks = (len(pairs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Creating {n_chunks} chunks of ~{CHUNK_SIZE} pairs each...")

    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, len(pairs))
        chunk = pairs[start:end]

        chunk_key = f"{CHUNK_PREFIX}chunk_{i:04d}.jsonl"
        chunk_content = '\n'.join(json.dumps(p) for p in chunk)

        s3.put_object(
            Bucket=BUCKET,
            Key=chunk_key,
            Body=chunk_content,
            ContentType='application/json'
        )

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{n_chunks} chunks...")

    print(f"Chunking complete: {n_chunks} chunks uploaded to s3://{BUCKET}/{CHUNK_PREFIX}")
    return n_chunks


def invoke_lambda(chunk_idx: int) -> dict:
    """Invoke Lambda for a single chunk."""
    chunk_key = f"{CHUNK_PREFIX}chunk_{chunk_idx:04d}.jsonl"

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION,
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'chunk_key': chunk_key,
                'chunk_idx': chunk_idx,
            })
        )

        result = json.loads(response['Payload'].read())
        return {'chunk_idx': chunk_idx, 'success': True, 'result': result}

    except Exception as e:
        return {'chunk_idx': chunk_idx, 'success': False, 'error': str(e)}


def run_mapreduce(n_chunks: int = None):
    """Run Lambda MapReduce on all chunks."""
    if n_chunks is None:
        # Count chunks in S3
        paginator = s3.get_paginator('list_objects_v2')
        n_chunks = 0
        for page in paginator.paginate(Bucket=BUCKET, Prefix=CHUNK_PREFIX):
            n_chunks += len(page.get('Contents', []))

    print(f"Running MapReduce on {n_chunks} chunks...")

    results = []
    total_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'all_found': 0,
        'partial_found': 0,
        'none_found': 0,
    }

    # Parallel invocation
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(invoke_lambda, i): i for i in range(n_chunks)}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result['success'] and 'stats' in result.get('result', {}):
                stats = result['result']['stats']
                total_stats['total'] += stats.get('total', 0)
                total_stats['success'] += stats.get('success', 0)
                total_stats['failed'] += stats.get('failed', 0)
                total_stats['all_found'] += stats.get('all_found', 0)
                total_stats['partial_found'] += stats.get('partial_found', 0)
                total_stats['none_found'] += stats.get('none_found', 0)

            completed = len(results)
            if completed % 10 == 0:
                print(f"  Completed {completed}/{n_chunks} chunks...")

    # Summary
    failed_chunks = [r for r in results if not r['success']]
    print(f"\nMapReduce complete:")
    print(f"  Chunks processed: {n_chunks}")
    print(f"  Failed chunks: {len(failed_chunks)}")
    print(f"\nTotal stats:")
    print(f"  Input pairs: {total_stats['total']}")
    print(f"  Reformatted: {total_stats['success']}")
    print(f"  Failed: {total_stats['failed']}")
    print(f"\nSpan finding:")
    print(f"  All operands found: {total_stats['all_found']}")
    print(f"  Partial found: {total_stats['partial_found']}")
    print(f"  None found: {total_stats['none_found']}")

    return results


def collect_results():
    """Collect all output chunks into a single file."""
    print(f"Collecting results from s3://{BUCKET}/{OUTPUT_PREFIX}...")

    # List all output chunks
    paginator = s3.get_paginator('list_objects_v2')
    output_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl'):
                output_keys.append(obj['Key'])

    print(f"Found {len(output_keys)} output chunks")

    # Combine all chunks
    all_pairs = []
    for key in sorted(output_keys):
        response = s3.get_object(Bucket=BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')
        pairs = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        all_pairs.extend(pairs)

    print(f"Total pairs: {len(all_pairs)}")

    # Upload combined file
    combined_content = '\n'.join(json.dumps(p) for p in all_pairs)
    s3.put_object(
        Bucket=BUCKET,
        Key=FINAL_OUTPUT_KEY,
        Body=combined_content,
        ContentType='application/json'
    )

    print(f"Combined output: s3://{BUCKET}/{FINAL_OUTPUT_KEY}")

    # Print stats
    all_found = sum(1 for p in all_pairs if p.get('all_found', False))
    print(f"\nFinal stats:")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  All operands found: {all_found} ({100*all_found/len(all_pairs):.1f}%)")

    return all_pairs


def main():
    parser = argparse.ArgumentParser(description='C3 Span Reformatter MapReduce')
    parser.add_argument('--chunk', action='store_true', help='Chunk data to S3')
    parser.add_argument('--run', action='store_true', help='Run Lambda MapReduce')
    parser.add_argument('--collect', action='store_true', help='Collect results')
    parser.add_argument('--all', action='store_true', help='Do all steps')
    args = parser.parse_args()

    if args.all or args.chunk:
        n_chunks = chunk_data()
    else:
        n_chunks = None

    if args.all or args.run:
        run_mapreduce(n_chunks)

    if args.all or args.collect:
        collect_results()

    if not (args.chunk or args.run or args.collect or args.all):
        parser.print_help()


if __name__ == '__main__':
    main()
