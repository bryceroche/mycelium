#!/usr/bin/env python3
"""
Map-Reduce Coordinator for Phase 2 Data Generation

Maps: Partitions MATH data → invokes Lambda functions in parallel
Reduces: Collects results → merges into final training dataset

Usage:
    python lambda/phase2_datagen/coordinator.py \
        --input s3://mycelium-data/math/math_train.json \
        --output s3://mycelium-data/phase2/phase2_train.json \
        --lambda-function phase2-datagen \
        --chunk-size 500 \
        --max-parallel 50

Requirements:
    - Lambda function deployed as 'phase2-datagen'
    - Lambda memory: 3GB (NOT 1GB!)
    - Lambda timeout: 15 minutes
    - IAM role with S3 read/write and Lambda invoke
"""

import os
import sys
import json
import time
import argparse
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import boto3
from botocore.config import Config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MapReduceConfig:
    input_bucket: str
    input_key: str
    output_bucket: str
    output_prefix: str
    lambda_function: str
    chunk_size: int = 500
    max_parallel: int = 50
    timeout: int = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Map phase: partition and invoke Lambdas
# ---------------------------------------------------------------------------

class Mapper:
    def __init__(self, config: MapReduceConfig):
        self.config = config
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client(
            'lambda',
            config=Config(
                retries={'max_attempts': 3},
                read_timeout=config.timeout + 60,
                connect_timeout=10,
            )
        )

    def load_input_data(self) -> List[Dict]:
        """Load input data from S3."""
        print(f"Loading input from s3://{self.config.input_bucket}/{self.config.input_key}")

        response = self.s3.get_object(
            Bucket=self.config.input_bucket,
            Key=self.config.input_key,
        )
        content = response["Body"].read().decode("utf-8")
        data = json.loads(content)

        print(f"Loaded {len(data)} problems")
        return data

    def partition(self, data: List[Dict]) -> List[Tuple[str, List[Dict]]]:
        """Partition data into chunks."""
        chunks = []
        for i in range(0, len(data), self.config.chunk_size):
            chunk_id = f"{i // self.config.chunk_size:04d}"
            chunk_data = data[i:i + self.config.chunk_size]
            chunks.append((chunk_id, chunk_data))

        print(f"Partitioned into {len(chunks)} chunks of ~{self.config.chunk_size} each")
        return chunks

    def invoke_lambda(self, chunk_id: str, chunk_data: List[Dict]) -> Dict:
        """Invoke Lambda function for one chunk."""
        payload = {
            "problems": chunk_data,
            "output_bucket": self.config.output_bucket,
            "output_prefix": f"{self.config.output_prefix}chunks/",
            "chunk_id": chunk_id,
        }

        try:
            response = self.lambda_client.invoke(
                FunctionName=self.config.lambda_function,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload).encode(),
            )

            result = json.loads(response['Payload'].read().decode())

            if response.get('FunctionError'):
                return {
                    "chunk_id": chunk_id,
                    "success": False,
                    "error": result.get("errorMessage", "Unknown error"),
                }

            body = result.get("body", result)
            return {
                "chunk_id": chunk_id,
                "success": True,
                "stats": body.get("stats", {}),
                "output_key": body.get("output_key"),
                "n_examples": body.get("n_examples", 0),
            }

        except Exception as e:
            return {
                "chunk_id": chunk_id,
                "success": False,
                "error": str(e),
            }

    def map_parallel(self, chunks: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        """Invoke Lambdas in parallel."""
        results = []
        total = len(chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
            futures = {
                executor.submit(self.invoke_lambda, chunk_id, chunk_data): chunk_id
                for chunk_id, chunk_data in chunks
            }

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)

                status = "✓" if result["success"] else "✗"
                print(f"[{completed}/{total}] Chunk {result['chunk_id']}: {status}", end="")
                if result["success"]:
                    print(f" ({result['n_examples']} examples)")
                else:
                    print(f" Error: {result.get('error', 'unknown')[:50]}")

        return results


# ---------------------------------------------------------------------------
# Reduce phase: merge results
# ---------------------------------------------------------------------------

class Reducer:
    def __init__(self, config: MapReduceConfig):
        self.config = config
        self.s3 = boto3.client('s3')

    def collect_results(self, map_results: List[Dict]) -> List[Dict]:
        """Collect all chunk results from S3."""
        all_examples = []

        successful = [r for r in map_results if r["success"]]
        print(f"\nCollecting results from {len(successful)} successful chunks...")

        for result in successful:
            output_key = result.get("output_key")
            if not output_key:
                continue

            try:
                response = self.s3.get_object(
                    Bucket=self.config.output_bucket,
                    Key=output_key,
                )
                content = response["Body"].read().decode("utf-8")
                chunk_examples = json.loads(content)
                all_examples.extend(chunk_examples)
            except Exception as e:
                print(f"  Warning: Failed to read {output_key}: {e}")

        print(f"Collected {len(all_examples)} total examples")
        return all_examples

    def merge_and_save(self, examples: List[Dict], output_key: str):
        """Merge examples and save to final output."""
        print(f"\nSaving {len(examples)} examples to s3://{self.config.output_bucket}/{output_key}")

        self.s3.put_object(
            Bucket=self.config.output_bucket,
            Key=output_key,
            Body=json.dumps(examples).encode("utf-8"),
            ContentType="application/json",
        )

        print("Done!")

    def compute_stats(self, examples: List[Dict]) -> Dict:
        """Compute statistics on generated data."""
        if not examples:
            return {"n_examples": 0}

        n_steps = [ex.get("n_steps", len(ex.get("span_groups", []))) for ex in examples]
        ops = [op for ex in examples for op in ex.get("gold_ops", [])]

        from collections import Counter
        op_counts = Counter(ops)

        return {
            "n_examples": len(examples),
            "avg_steps": sum(n_steps) / len(n_steps),
            "min_steps": min(n_steps),
            "max_steps": max(n_steps),
            "op_distribution": dict(op_counts.most_common(10)),
        }


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def run_map_reduce(config: MapReduceConfig):
    """Run full map-reduce pipeline."""
    start_time = time.time()

    # Map phase
    print("=" * 60)
    print("MAP PHASE")
    print("=" * 60)

    mapper = Mapper(config)
    input_data = mapper.load_input_data()
    chunks = mapper.partition(input_data)

    print(f"\nInvoking {len(chunks)} Lambda functions (max {config.max_parallel} parallel)...")
    map_results = mapper.map_parallel(chunks)

    # Stats
    successful = sum(1 for r in map_results if r["success"])
    failed = len(map_results) - successful
    print(f"\nMap complete: {successful} succeeded, {failed} failed")

    if failed > 0:
        print("\nFailed chunks:")
        for r in map_results:
            if not r["success"]:
                print(f"  {r['chunk_id']}: {r.get('error', 'unknown')[:80]}")

    # Reduce phase
    print("\n" + "=" * 60)
    print("REDUCE PHASE")
    print("=" * 60)

    reducer = Reducer(config)
    all_examples = reducer.collect_results(map_results)

    # Save final output
    output_key = config.output_prefix.rstrip('/') + '.json'
    if not output_key.endswith('.json'):
        output_key = f"{config.output_prefix}phase2_train.json"

    reducer.merge_and_save(all_examples, output_key)

    # Final stats
    stats = reducer.compute_stats(all_examples)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Examples generated: {stats['n_examples']}")
    if stats['n_examples'] > 0:
        print(f"Avg steps per example: {stats['avg_steps']:.1f}")
        print(f"Step range: {stats['min_steps']} - {stats['max_steps']}")
        print(f"Top operations: {stats['op_distribution']}")
    print(f"\nOutput: s3://{config.output_bucket}/{output_key}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_s3_path(path: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got: {path}")
    parts = path[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Data Generation via Lambda Map-Reduce")

    parser.add_argument("--input", type=str, required=True,
                        help="Input S3 path (s3://bucket/key.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output S3 path (s3://bucket/prefix/)")
    parser.add_argument("--lambda-function", type=str, default="phase2-datagen",
                        help="Lambda function name")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Problems per Lambda invocation")
    parser.add_argument("--max-parallel", type=int, default=50,
                        help="Max concurrent Lambda invocations")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Lambda timeout in seconds")

    args = parser.parse_args()

    # Parse paths
    input_bucket, input_key = parse_s3_path(args.input)
    output_bucket, output_prefix = parse_s3_path(args.output)

    config = MapReduceConfig(
        input_bucket=input_bucket,
        input_key=input_key,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        lambda_function=args.lambda_function,
        chunk_size=args.chunk_size,
        max_parallel=args.max_parallel,
        timeout=args.timeout,
    )

    run_map_reduce(config)


if __name__ == "__main__":
    main()
