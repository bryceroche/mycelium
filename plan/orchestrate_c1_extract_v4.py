"""
MapReduce Orchestrator: C1 Training Data Extraction v4 (Shadow Reader)

Uses telegraph x alarm crossings (L22H3 x L22H4) instead of telegraph x telegraph.

Usage:
    python orchestrate_c1_extract_v4.py --deploy --max-concurrent 50

Output:
    s3://mycelium-data/c1_training_v4/
"""

import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np


s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

LAMBDA_FUNCTION = "mycelium-c1-extract-v4"
BUCKET = "mycelium-data"


def list_chunks(bucket: str, prefix: str) -> list[str]:
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
    payload = {
        "chunk_key": chunk_key,
        "output_prefix": output_prefix,
        "bucket": bucket,
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


def map_phase(bucket, iaf_prefix, output_prefix, max_concurrent):
    chunks = list_chunks(bucket, iaf_prefix)
    if not chunks:
        return []

    print(f"\nMap phase: {len(chunks)} chunks, {max_concurrent} concurrent")
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_lambda, bucket, chunk, output_prefix): chunk
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
                    f"gen: {s.get('generated', 0)} | "
                    f"co_trans: {s.get('co_transition_median', 0):.1f} | "
                    f"bound: {s.get('boundary_mean', 0):.1f} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix, map_results):
    print("\n" + "=" * 70)
    print("REDUCE: C1 Training Data v4 Statistics (Shadow Reader)")
    print("=" * 70)

    # Aggregate stats
    total = 0
    generated = 0
    skipped = 0
    crossing_counts = []
    o_ratios = []
    boundary_counts = []
    bp_depth_dist = {1: 0, 2: 0, 3: 0}
    zero_crossings = 0
    total_tokens = 0
    total_boundaries = 0

    for r in map_results:
        s = r.get("stats", {})
        total += s.get("total", 0)
        generated += s.get("generated", 0)
        skipped += s.get("skipped", 0)

        # Crossing stats (we have per-chunk summaries)
        if "crossing_mean" in s:
            n = s.get("generated", 1)
            crossing_counts.extend([s["crossing_mean"]] * n)  # approximate
            zero_crossings += s.get("zero_crossing_count", 0)

        if "o_ratio_mean" in s:
            o_ratios.append(s["o_ratio_mean"])

        if "boundary_mean" in s:
            boundary_counts.append(s["boundary_mean"])

        if "boundary_density" in s:
            # Approximate total
            pass

        for depth in [1, 2, 3]:
            bp_depth_dist[depth] += s.get("bp_depth_dist", {}).get(str(depth), 0)
            bp_depth_dist[depth] += s.get("bp_depth_dist", {}).get(depth, 0)

    # Print summary
    print(f"\nTotal problems:     {total:,}")
    print(f"Records generated:  {generated:,}")
    print(f"Skipped:            {skipped:,}")

    # For accurate stats, we need to read a sample of the actual output
    print("\n--- Loading sample for accurate statistics ---")
    sample_stats = get_sample_stats(bucket, output_prefix, n_chunks=10)

    print(f"\n--- Co-Transition Distribution (Telegraph + Alarm) ---")
    ct = sample_stats["co_transition_counts"]
    if ct:
        print(f"Mean:     {np.mean(ct):.2f}")
        print(f"Median:   {np.median(ct):.1f}")
        print(f"P25:      {np.percentile(ct, 25):.1f}")
        print(f"P75:      {np.percentile(ct, 75):.1f}")
        print(f"Max:      {max(ct)}")
        print(f"Zero co-transitions: {sum(1 for c in ct if c == 0)} ({sum(1 for c in ct if c == 0)/len(ct)*100:.1f}%)")

    print(f"\n--- Reading Ratio Distribution ---")
    rrs = sample_stats["reading_ratios"]
    if rrs:
        # Filter out 0.5 (neutral/no-data values)
        real_rrs = [r for r in rrs if r != 0.5 or ct[rrs.index(r)] > 0]
        if real_rrs:
            print(f"Mean:  {np.mean(real_rrs):.3f}")
            print(f"Std:   {np.std(real_rrs):.3f}")
            print(f"Interpretation: {'alarms fire during READING' if np.mean(real_rrs) > 0.5 else 'alarms fire during COMPUTING'}")

    print(f"\n--- Boundary Detection ---")
    bc = sample_stats["boundary_counts"]
    if bc:
        print(f"Mean boundaries/problem: {np.mean(bc):.1f}")
        total_tok = sample_stats["total_tokens"]
        total_bound = sum(bc)
        print(f"Boundary density:        {total_bound/max(1,total_tok)*100:.2f}% of tokens")

    print(f"\n--- BP Depth Distribution ---")
    gen = max(1, generated)
    for depth in [1, 2, 3]:
        count = bp_depth_dist[depth]
        pct = count / gen * 100
        bar = "█" * int(pct / 2)
        print(f"  {depth} rounds: {count:>6,} ({pct:5.1f}%) {bar}")

    # Upload global stats
    global_stats = {
        "total": total,
        "generated": generated,
        "skipped": skipped,
        "co_transition_stats": {
            "mean": float(np.mean(ct)) if ct else 0,
            "median": float(np.median(ct)) if ct else 0,
            "p25": float(np.percentile(ct, 25)) if ct else 0,
            "p75": float(np.percentile(ct, 75)) if ct else 0,
            "max": int(max(ct)) if ct else 0,
        },
        "reading_ratio_stats": {
            "mean": float(np.mean(rrs)) if rrs else 0.5,
            "std": float(np.std(rrs)) if rrs else 0,
        },
        "boundary_stats": {
            "mean": float(np.mean(bc)) if bc else 0,
            "density": total_bound/max(1,total_tok) if bc else 0,
        },
        "bp_depth_dist": bp_depth_dist,
    }

    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(global_stats, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")

    return global_stats


def get_sample_stats(bucket: str, prefix: str, n_chunks: int = 10) -> dict:
    """Load a sample of chunks and compute accurate stats."""
    paginator = s3.get_paginator("list_objects_v2")
    chunks = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and "/stats/" not in key and "global" not in key:
                chunks.append(key)
                if len(chunks) >= n_chunks:
                    break
        if len(chunks) >= n_chunks:
            break

    co_transition_counts = []
    reading_ratios = []
    boundary_counts = []
    total_tokens = 0

    for key in chunks:
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            for r in data.get("records", []):
                co_transition_counts.append(r.get("n_co_transitions", 0))
                reading_ratios.append(r.get("reading_ratio", 0.5))
                boundary_counts.append(r.get("n_boundaries", 0))
                total_tokens += r.get("n_input_tokens", 0)
        except Exception as e:
            print(f"  Warning: couldn't read {key}: {e}")

    return {
        "co_transition_counts": co_transition_counts,
        "reading_ratios": reading_ratios,
        "boundary_counts": boundary_counts,
        "total_tokens": total_tokens,
    }


def deploy_lambda(zip_path="lambda_c1_extract_v4.zip"):
    import zipfile
    import os

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c1_extract_v4.py", "lambda_function.py")

    print(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    # Get role and layer
    iam = boto3.client("iam")
    roles = iam.list_roles()["Roles"]
    role_arn = next(
        (r["Arn"] for r in roles if "lambda" in r["RoleName"].lower()),
        None
    )

    layers = lambda_client.list_layers()["Layers"]
    numpy_layer = next(
        (l["LatestMatchingVersion"]["LayerVersionArn"]
         for l in layers if "numpy" in l["LayerName"].lower()),
        None
    )

    try:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION, ZipFile=zip_bytes
        )
        print(f"Updated: {LAMBDA_FUNCTION}")
    except lambda_client.exceptions.ResourceNotFoundException:
        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            MemorySize=3072,
            Timeout=900,
            Layers=[numpy_layer] if numpy_layer else [],
        )
        print(f"Created: {LAMBDA_FUNCTION}")

    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION, MemorySize=3072, Timeout=900
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--output-prefix", default="c1_training_v4/")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        print("Waiting for Lambda to be ready...")
        time.sleep(10)

    results = map_phase(
        args.bucket, args.iaf_prefix, args.output_prefix, args.max_concurrent
    )

    reduce_phase(args.bucket, args.output_prefix, results)


if __name__ == "__main__":
    main()
