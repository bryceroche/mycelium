#!/usr/bin/env python3
"""
Merge C1 Relevance chunks into single training file

After Lambda MapReduce produces per-chunk relevance files, this script
merges them into a single training dataset.

Usage:
  # Merge from S3
  python merge_c1_relevance.py --input s3://mycelium-data/c1_relevance/chunks/

  # Merge from local
  python merge_c1_relevance.py --input /path/to/chunks/ --output training_data.json

  # Upload merged file to S3
  python merge_c1_relevance.py --input s3://mycelium-data/c1_relevance/chunks/ \
      --output s3://mycelium-data/c1_relevance/c1_all_relevance.json
"""

import json
import argparse
import boto3
from pathlib import Path
from typing import List, Dict
import sys


def list_s3_files(s3_client, bucket: str, prefix: str) -> List[str]:
    """List all JSON files in S3 prefix."""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                files.append(key)

    return sorted(files)


def load_from_s3(s3_client, bucket: str, key: str) -> List[Dict]:
    """Load JSON from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read().decode('utf-8'))


def save_to_s3(s3_client, bucket: str, key: str, data: List[Dict]):
    """Save JSON to S3."""
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data),
        ContentType='application/json'
    )


def parse_s3_path(path: str) -> tuple:
    """Parse s3://bucket/key into (bucket, key)."""
    path = path.replace('s3://', '')
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key


def main():
    parser = argparse.ArgumentParser(description='Merge C1 relevance chunks')
    parser.add_argument('--input', required=True, help='Input path (S3 or local)')
    parser.add_argument('--output', default=None, help='Output path (S3 or local)')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()

    print("=" * 60)
    print("MERGE C1 RELEVANCE CHUNKS")
    print("=" * 60)

    merged = []
    s3 = None

    # Load chunks
    if args.input.startswith('s3://'):
        s3 = boto3.client('s3', region_name=args.region)
        bucket, prefix = parse_s3_path(args.input)

        print(f"\nListing files in s3://{bucket}/{prefix}...")
        files = list_s3_files(s3, bucket, prefix)
        print(f"Found {len(files)} files")

        for i, key in enumerate(files):
            try:
                data = load_from_s3(s3, bucket, key)
                merged.extend(data)
                print(f"[{i+1:3d}/{len(files)}] {key.split('/')[-1]}: {len(data)} records")
            except Exception as e:
                print(f"[{i+1:3d}/{len(files)}] ERROR {key}: {e}")

    else:
        input_path = Path(args.input)
        files = sorted(input_path.glob('*.json'))
        print(f"\nFound {len(files)} files in {input_path}")

        for i, f in enumerate(files):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                merged.extend(data)
                print(f"[{i+1:3d}/{len(files)}] {f.name}: {len(data)} records")
            except Exception as e:
                print(f"[{i+1:3d}/{len(files)}] ERROR {f.name}: {e}")

    # Stats
    print("\n" + "=" * 60)
    print("MERGED DATASET STATS")
    print("=" * 60)
    print(f"Total samples: {len(merged):,}")

    if merged:
        total_tokens = sum(r['stats']['n_tokens'] for r in merged)
        total_nonzero = sum(r['stats']['n_nonzero'] for r in merged)
        total_high = sum(r['stats']['n_high'] for r in merged)
        avg_mean = sum(r['stats']['mean'] for r in merged) / len(merged)

        print(f"Total tokens: {total_tokens:,}")
        print(f"Tokens with attention: {total_nonzero:,} ({total_nonzero/total_tokens*100:.1f}%)")
        print(f"High relevance (>0.5): {total_high:,} ({total_high/total_tokens*100:.1f}%)")
        print(f"Average mean relevance: {avg_mean:.3f}")

    # Save
    if args.output:
        output_path = args.output
    elif args.input.startswith('s3://'):
        output_path = f"s3://{bucket}/c1_relevance/c1_all_relevance.json"
    else:
        output_path = str(Path(args.input).parent / "c1_all_relevance.json")

    print(f"\nSaving to {output_path}...")

    if output_path.startswith('s3://'):
        if s3 is None:
            s3 = boto3.client('s3', region_name=args.region)
        out_bucket, out_key = parse_s3_path(output_path)
        save_to_s3(s3, out_bucket, out_key, merged)
    else:
        with open(output_path, 'w') as f:
            json.dump(merged, f)

    print("Done!")
    print(f"\nOutput: {output_path}")
    print(f"Size: {len(json.dumps(merged)) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
