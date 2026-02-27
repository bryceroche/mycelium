#!/usr/bin/env python3
"""
Invoke C3 Extractor Lambda for all IAF chunks (MapReduce pattern)

Processes chunks in batches to stay within Lambda timeout limits.
Each Lambda invocation handles 1-5 chunks depending on size.
"""

import boto3
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Config
LAMBDA_NAME = "mycelium-c3-extractor"
INPUT_BUCKET = "mycelium-data"
CHUNKED_PREFIX = "iaf_extraction/chunked/"
MAX_CONCURRENT = 10
CHUNKS_PER_INVOKE = 1  # Process one chunk per invocation for memory safety


def list_chunks():
    """List all chunked IAF files in S3."""
    s3 = boto3.client('s3')
    chunks = []

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=INPUT_BUCKET, Prefix=CHUNKED_PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json') and 'chunk_' in key:
                chunks.append(key)

    return sorted(chunks)


def invoke_lambda(chunk_keys):
    """Invoke Lambda for a batch of chunks."""
    client = boto3.client('lambda')

    if len(chunk_keys) == 1:
        payload = {"bucket": INPUT_BUCKET, "key": chunk_keys[0]}
    else:
        payload = {"bucket": INPUT_BUCKET, "chunks": chunk_keys}

    response = client.invoke(
        FunctionName=LAMBDA_NAME,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )

    result = json.loads(response['Payload'].read())
    return result


def main():
    print("=" * 60)
    print("MYCELIUM: C3 EXTRACTOR LAMBDA INVOCATION")
    print("=" * 60)

    # List chunks
    print(f"\nListing chunks in s3://{INPUT_BUCKET}/{CHUNKED_PREFIX}...")
    chunks = list_chunks()
    print(f"Found {len(chunks)} chunks")

    if not chunks:
        print("No chunks found!")
        return

    # Optionally filter for testing
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        chunks = chunks[:n]
        print(f"Processing first {n} chunks (test mode)")

    # Batch chunks for Lambda invocations
    batches = []
    for i in range(0, len(chunks), CHUNKS_PER_INVOKE):
        batches.append(chunks[i:i + CHUNKS_PER_INVOKE])

    print(f"Created {len(batches)} Lambda invocation batches")

    # Process batches in parallel
    print(f"\nInvoking Lambda (max {MAX_CONCURRENT} concurrent)...")

    total_input = 0
    total_output = 0
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(invoke_lambda, batch): batch for batch in batches}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            batch = futures[future]
            try:
                result = future.result()
                if 'body' in result:
                    body = result['body']
                    total_input += body.get('total_input_records', 0)
                    total_output += body.get('total_output_pairs', 0)

                    for r in body.get('results', []):
                        if r.get('status') == 'error':
                            errors.append(f"{r['key']}: {r.get('error', 'unknown')}")
                else:
                    errors.append(f"{batch[0]}: {result}")
            except Exception as e:
                errors.append(f"{batch[0]}: {str(e)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Chunks processed: {len(batches)}")
    print(f"Total input records: {total_input}")
    print(f"Total output pairs: {total_output}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
