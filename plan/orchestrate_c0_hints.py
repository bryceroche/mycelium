"""
Orchestrator: C0 Hint Vector Extraction

Fans out Lambda invocations to extract per-token hint vectors from IAF data.

Usage:
    python orchestrate_c0_hints.py --deploy  # First time: deploy Lambda
    python orchestrate_c0_hints.py           # Run extraction
"""

import json
import time
import argparse
import zipfile
import os
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np

s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

BUCKET = "mycelium-data"
LAMBDA_FUNCTION = "mycelium-c0-hints"
IAF_PREFIX = "iaf_extraction/chunked/"
OUTPUT_PREFIX = "c0_training_data/hint_vectors/"


def list_chunks(bucket: str, prefix: str) -> list[str]:
    """List all IAF chunks."""
    chunks = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and obj["Size"] > 0:
                if "/stats/" not in obj["Key"]:
                    chunks.append(obj["Key"])
    print(f"Found {len(chunks)} chunks in s3://{bucket}/{prefix}")
    return sorted(chunks)


def invoke_lambda(bucket, chunk_key, output_prefix):
    """Invoke Lambda for one chunk."""
    payload = {
        "bucket": bucket,
        "chunk_key": chunk_key,
        "output_prefix": output_prefix,
    }

    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    result = json.loads(response["Payload"].read().decode("utf-8"))
    if response.get("FunctionError"):
        raise RuntimeError(f"Lambda failed on {chunk_key}: {result}")
    return result


def map_phase(max_concurrent=50):
    """Run Lambda on all chunks in parallel."""
    chunks = list_chunks(BUCKET, IAF_PREFIX)
    if not chunks:
        return []

    print(f"\nMap phase: {len(chunks)} chunks, {max_concurrent} concurrent")
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_lambda, BUCKET, chunk, OUTPUT_PREFIX): chunk
            for chunk in chunks
        }

        completed = 0
        start = time.time()

        for future in as_completed(futures):
            chunk_key = futures[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                s = result.get("stats", {})
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed}/{len(chunks)}] {chunk_key.split('/')[-1]} | "
                    f"processed: {s.get('processed', 0)} | "
                    f"avg_entropy: {s.get('avg_entropy', 0):.3f} | "
                    f"avg_tension: {s.get('avg_tension', 0):.3f} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(map_results):
    """Compute global statistics across all chunks."""
    print("\n" + "=" * 60)
    print("REDUCE PHASE: Global Statistics")
    print("=" * 60)

    global_stats = {
        "total_problems": 0,
        "processed": 0,
        "skipped": 0,
        "total_entropy": 0.0,
        "total_tension": 0.0,
        "total_received": 0.0,
    }

    for result in map_results:
        s = result.get("stats", {})
        global_stats["total_problems"] += s.get("total", 0)
        global_stats["processed"] += s.get("processed", 0)
        global_stats["skipped"] += s.get("skipped", 0)
        global_stats["total_entropy"] += s.get("total_entropy", 0)
        global_stats["total_tension"] += s.get("total_tension", 0)
        global_stats["total_received"] += s.get("total_received", 0)

    if global_stats["processed"] > 0:
        global_stats["avg_entropy"] = global_stats["total_entropy"] / global_stats["processed"]
        global_stats["avg_tension"] = global_stats["total_tension"] / global_stats["processed"]
        global_stats["avg_received"] = global_stats["total_received"] / global_stats["processed"]

    print(f"\nTotal problems:  {global_stats['total_problems']}")
    print(f"Processed:       {global_stats['processed']}")
    print(f"Skipped:         {global_stats['skipped']}")
    print(f"Avg entropy:     {global_stats.get('avg_entropy', 0):.4f}")
    print(f"Avg tension:     {global_stats.get('avg_tension', 0):.4f}")
    print(f"Avg received:    {global_stats.get('avg_received', 0):.4f}")

    # Now load a sample of the data to compute detailed feature statistics
    print("\nLoading sample data for detailed statistics...")

    # Load first 10 chunks worth of data
    sample_data = []
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            if count >= 10:
                break

            response = s3.get_object(Bucket=BUCKET, Key=key)
            content = response["Body"].read().decode("utf-8")
            for line in content.strip().split("\n"):
                if line:
                    sample_data.append(json.loads(line))
            count += 1

        if count >= 10:
            break

    print(f"Loaded {len(sample_data)} sample records for analysis")

    if sample_data:
        # Compute per-feature statistics
        features = ["entropy", "received", "tension", "telegraph", "connectivity"]
        feature_stats = {}

        for feature in features:
            all_values = []
            for record in sample_data:
                values = record.get("hint_vectors", {}).get(feature, [])
                all_values.extend(values)

            if all_values:
                arr = np.array(all_values)
                feature_stats[feature] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p95": float(np.percentile(arr, 95)),
                    "skew": float((np.mean(arr) - np.median(arr)) / (np.std(arr) + 1e-10)),
                }

        print("\n--- Per-Feature Statistics (sample) ---")
        for feature, stats in feature_stats.items():
            print(f"\n{feature}:")
            print(f"  mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            print(f"  min={stats['min']:.4f}, max={stats['max']:.4f}")
            print(f"  p25={stats['p25']:.4f}, p50={stats['p50']:.4f}, p75={stats['p75']:.4f}")
            print(f"  skew={stats['skew']:.4f}")

        global_stats["feature_stats"] = feature_stats

        # Compute inter-feature correlations
        print("\n--- Inter-Feature Correlations ---")
        correlations = {}
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                # Compute correlation across all tokens in sample
                v1, v2 = [], []
                for record in sample_data:
                    vals1 = record.get("hint_vectors", {}).get(f1, [])
                    vals2 = record.get("hint_vectors", {}).get(f2, [])
                    min_len = min(len(vals1), len(vals2))
                    v1.extend(vals1[:min_len])
                    v2.extend(vals2[:min_len])

                if v1 and v2:
                    corr = np.corrcoef(v1, v2)[0, 1]
                    correlations[f"{f1}_vs_{f2}"] = float(corr)
                    print(f"  {f1} vs {f2}: {corr:.3f}")

        global_stats["correlations"] = correlations

    # Save global stats
    global_stats_clean = json.loads(json.dumps(global_stats, default=str))
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}global_stats.json",
        Body=json.dumps(global_stats_clean, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nSaved: s3://{BUCKET}/{OUTPUT_PREFIX}global_stats.json")

    return global_stats


def deploy_lambda():
    """Deploy or update the Lambda function."""
    zip_path = "lambda_c0_hints.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c0_hints.py", "lambda_function.py")

    print(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    try:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION, ZipFile=zip_bytes
        )
        print(f"Updated: {LAMBDA_FUNCTION}")
    except lambda_client.exceptions.ResourceNotFoundException:
        # Get role ARN
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="lambda-s3-execution-role")
        role_arn = role["Role"]["Arn"]

        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            MemorySize=3072,
            Timeout=900,
        )
        print(f"Created: {LAMBDA_FUNCTION}")

    # Update configuration
    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION,
        MemorySize=3072,
        Timeout=900,
    )
    print("Lambda configured: 3GB memory, 15min timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Deploy Lambda function")
    parser.add_argument("--max-concurrent", type=int, default=50)
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        print("Waiting 10s for Lambda to be ready...")
        time.sleep(10)

    map_results = map_phase(args.max_concurrent)
    reduce_phase(map_results)

    print("\n" + "=" * 60)
    print("C0 HINT EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: s3://{BUCKET}/{OUTPUT_PREFIX}")


if __name__ == "__main__":
    main()
