"""
MapReduce Orchestrator: IAF Knot Crossing Detection

Fans out Lambda invocations across 200MB IAF chunks in S3,
then reduces results into aggregate crossing statistics and
a knot classification dataset for C2 training.

Usage:
    python orchestrate_crossing_detection.py \
        --bucket mycelium-data \
        --iaf-prefix iaf/ \
        --output-prefix crossings/ \
        --max-concurrent 50
"""

import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3


s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

LAMBDA_FUNCTION = "mycelium-crossing-detect"


# ---------------------------------------------------------------------------
# Map phase: fan out Lambda invocations
# ---------------------------------------------------------------------------

def list_iaf_chunks(bucket: str, prefix: str) -> list[str]:
    """List all IAF chunk keys in S3."""
    chunks = []
    paginator = s3.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and obj["Size"] > 0:
                chunks.append(key)
    
    print(f"Found {len(chunks)} IAF chunks in s3://{bucket}/{prefix}")
    total_size_gb = sum(
        obj["Size"] for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ) / (1024 ** 3)
    print(f"Total size: {total_size_gb:.1f} GB")
    
    return sorted(chunks)


def invoke_lambda(bucket: str, chunk_key: str, output_prefix: str) -> dict:
    """Invoke one Lambda function for one chunk."""
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


def map_phase(
    bucket: str,
    iaf_prefix: str,
    output_prefix: str,
    max_concurrent: int = 50,
) -> list[dict]:
    """Fan out Lambda invocations across all chunks."""
    chunks = list_iaf_chunks(bucket, iaf_prefix)
    
    if not chunks:
        print("No chunks found!")
        return []
    
    print(f"\nStarting map phase: {len(chunks)} chunks, {max_concurrent} concurrent")
    
    results = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_lambda, bucket, chunk, output_prefix): chunk
            for chunk in chunks
        }
        
        completed = 0
        start_time = time.time()
        
        for future in as_completed(futures):
            chunk_key = futures[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                stats = result.get("stats", {})
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - completed) / rate if rate > 0 else 0
                
                print(
                    f"  [{completed}/{len(chunks)}] {chunk_key} | "
                    f"problems: {stats.get('total', 0)} | "
                    f"knotted: {stats.get('knotted', 0)} | "
                    f"max_cross: {stats.get('max_crossings', 0)} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nMap phase complete: {elapsed:.1f}s")
    print(f"  Succeeded: {len(results)}/{len(chunks)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print("\nFailed chunks:")
        for chunk, error in failed:
            print(f"  {chunk}: {error}")
    
    return results


# ---------------------------------------------------------------------------
# Reduce phase: aggregate statistics and build datasets
# ---------------------------------------------------------------------------

def reduce_phase(
    bucket: str,
    output_prefix: str,
    map_results: list[dict],
) -> dict:
    """
    Aggregate crossing detection results across all chunks.
    
    Produces:
    1. Global statistics (crossing number distribution, signature frequencies)
    2. Knot classification dataset (problem_id → knot_signature) for C2 training
    3. Belief propagation depth predictor dataset (signature → recommended rounds)
    """
    print("\n" + "=" * 60)
    print("REDUCE PHASE")
    print("=" * 60)
    
    # --- Aggregate stats ---
    global_stats = {
        "total_problems": 0,
        "unknotted": 0,
        "knotted": 0,
        "max_crossings": 0,
        "crossing_type_counts": {"over": 0, "under": 0},
        "signature_counts": defaultdict(int),
        "crossing_number_histogram": defaultdict(int),
        "unknotting_number_histogram": defaultdict(int),
        "steps_vs_crossings": [],  # for correlation analysis
    }
    
    # Collect all per-problem results for dataset generation
    all_problems = []
    
    for map_result in map_results:
        chunk_stats = map_result.get("stats", {})
        
        global_stats["total_problems"] += chunk_stats.get("total", 0)
        global_stats["unknotted"] += chunk_stats.get("unknotted", 0)
        global_stats["knotted"] += chunk_stats.get("knotted", 0)
        global_stats["max_crossings"] = max(
            global_stats["max_crossings"],
            chunk_stats.get("max_crossings", 0),
        )
        
        for ctype, count in chunk_stats.get("crossing_type_counts", {}).items():
            global_stats["crossing_type_counts"][ctype] += count
        
        for sig, count in chunk_stats.get("signature_counts", {}).items():
            global_stats["signature_counts"][sig] += count
        
        # Download full results for dataset building
        output_key = map_result.get("output_key")
        if output_key:
            try:
                resp = s3.get_object(Bucket=bucket, Key=output_key)
                chunk_results = json.loads(resp["Body"].read().decode("utf-8"))
                for problem in chunk_results.get("results", []):
                    all_problems.append(problem)
                    
                    cn = problem.get("crossing_number", 0)
                    un = problem.get("unknotting_number", 0)
                    global_stats["crossing_number_histogram"][cn] += 1
                    global_stats["unknotting_number_histogram"][un] += 1
                    global_stats["steps_vs_crossings"].append({
                        "n_steps": problem.get("n_steps", 0),
                        "n_crossings": cn,
                    })
            except Exception as e:
                print(f"  Warning: couldn't download {output_key}: {e}")
    
    # --- Print summary ---
    print(f"\nTotal problems analyzed: {global_stats['total_problems']}")
    print(f"Unknotted (linear):     {global_stats['unknotted']} "
          f"({global_stats['unknotted']/max(1,global_stats['total_problems'])*100:.1f}%)")
    print(f"Knotted (coupled):      {global_stats['knotted']} "
          f"({global_stats['knotted']/max(1,global_stats['total_problems'])*100:.1f}%)")
    print(f"Max crossing number:    {global_stats['max_crossings']}")
    
    print(f"\nCrossing types:")
    for ctype, count in global_stats["crossing_type_counts"].items():
        print(f"  {ctype}: {count}")
    
    print(f"\nCrossing number distribution:")
    for cn in sorted(global_stats["crossing_number_histogram"].keys()):
        count = global_stats["crossing_number_histogram"][cn]
        pct = count / max(1, global_stats["total_problems"]) * 100
        bar = "█" * int(pct)
        print(f"  {cn:2d}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print(f"\nTop knot signatures:")
    sig_sorted = sorted(
        global_stats["signature_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for sig, count in sig_sorted[:20]:
        label = sig if sig else "(unknot)"
        pct = count / max(1, global_stats["total_problems"]) * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    # --- Steps vs crossings correlation ---
    if global_stats["steps_vs_crossings"]:
        steps = [p["n_steps"] for p in global_stats["steps_vs_crossings"]]
        crosses = [p["n_crossings"] for p in global_stats["steps_vs_crossings"]]
        if len(steps) > 1:
            import numpy as np
            corr = np.corrcoef(steps, crosses)[0, 1]
            print(f"\nSteps ↔ Crossings correlation: {corr:.3f}")
    
    # --- Build knot classification dataset for C2 ---
    knot_dataset = _build_knot_classification_dataset(all_problems, global_stats)
    
    # --- Build belief prop depth dataset ---
    depth_dataset = _build_depth_prediction_dataset(all_problems)
    
    # --- Upload everything ---
    _upload_reduce_results(
        bucket, output_prefix, global_stats, knot_dataset, depth_dataset
    )
    
    return global_stats


def _build_knot_classification_dataset(
    all_problems: list[dict],
    global_stats: dict,
) -> list[dict]:
    """
    Build training data for knot signature prediction head on C2.
    
    Maps problem_id → knot_signature (categorical label).
    Only include signatures with >= 10 examples (learnable).
    """
    # Filter to learnable signatures
    min_count = 10
    learnable_sigs = {
        sig for sig, count in global_stats["signature_counts"].items()
        if count >= min_count
    }
    
    # Add "RARE" bucket for infrequent signatures
    dataset = []
    for problem in all_problems:
        sig = problem.get("knot_signature", "")
        dataset.append({
            "problem_id": problem["problem_id"],
            "knot_signature": sig if sig in learnable_sigs else "RARE",
            "crossing_number": problem.get("crossing_number", 0),
            "unknotting_number": problem.get("unknotting_number", 0),
            "n_steps": problem.get("n_steps", 0),
        })
    
    n_classes = len(learnable_sigs) + 1  # +1 for RARE
    print(f"\nKnot classification dataset:")
    print(f"  Examples: {len(dataset)}")
    print(f"  Classes:  {n_classes} ({len(learnable_sigs)} specific + RARE)")
    
    return dataset


def _build_depth_prediction_dataset(all_problems: list[dict]) -> list[dict]:
    """
    Build training data for belief propagation depth prediction.
    
    Maps problem features → recommended number of BP rounds.
    
    Heuristic-free mapping:
      unknotting_number 0 → 1 round  (no crossings to resolve)
      unknotting_number 1 → 2 rounds (one correction pass)
      unknotting_number 2+ → 3 rounds (multiple corrections)
    
    This mapping itself will be validated/replaced by actual BP
    convergence traces once Phase 2 training produces them.
    The unknotting number is the LEARNED invariant — the round
    mapping just needs to be calibrated against empirical convergence.
    """
    dataset = []
    for problem in all_problems:
        un = problem.get("unknotting_number", 0)
        
        # Initial round estimate from unknotting number
        # Will be refined with actual BP convergence data
        if un == 0:
            recommended_rounds = 1
        elif un == 1:
            recommended_rounds = 2
        else:
            recommended_rounds = 3
        
        dataset.append({
            "problem_id": problem["problem_id"],
            "unknotting_number": un,
            "crossing_number": problem.get("crossing_number", 0),
            "knot_signature": problem.get("knot_signature", ""),
            "n_steps": problem.get("n_steps", 0),
            "recommended_rounds": recommended_rounds,
        })
    
    # Distribution
    round_counts = defaultdict(int)
    for d in dataset:
        round_counts[d["recommended_rounds"]] += 1
    
    print(f"\nBP depth prediction dataset:")
    print(f"  Examples: {len(dataset)}")
    for rounds in sorted(round_counts):
        pct = round_counts[rounds] / max(1, len(dataset)) * 100
        print(f"  {rounds} rounds: {round_counts[rounds]} ({pct:.1f}%)")
    
    return dataset


def _upload_reduce_results(
    bucket: str,
    output_prefix: str,
    global_stats: dict,
    knot_dataset: list[dict],
    depth_dataset: list[dict],
):
    """Upload all reduce outputs to S3."""
    
    # Global stats (convert defaultdicts)
    stats_clean = json.loads(json.dumps(global_stats, default=str))
    
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(stats_clean, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")
    
    # Knot classification dataset
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}knot_classification_dataset.json",
        Body=json.dumps(knot_dataset).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded: s3://{bucket}/{output_prefix}knot_classification_dataset.json")
    
    # BP depth dataset
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}bp_depth_dataset.json",
        Body=json.dumps(depth_dataset).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded: s3://{bucket}/{output_prefix}bp_depth_dataset.json")
    
    # Signature → example mapping (for qualitative analysis)
    sig_examples = defaultdict(list)
    for problem in knot_dataset[:5000]:  # cap for size
        sig = problem["knot_signature"]
        if len(sig_examples[sig]) < 5:  # 5 examples per sig
            sig_examples[sig].append(problem["problem_id"])
    
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}signature_examples.json",
        Body=json.dumps(dict(sig_examples), indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded: s3://{bucket}/{output_prefix}signature_examples.json")


# ---------------------------------------------------------------------------
# Deploy Lambda
# ---------------------------------------------------------------------------

def deploy_lambda(zip_path: str = "lambda_crossing_detect.zip"):
    """
    Package and deploy the crossing detection Lambda.
    
    Requirements:
    - Lambda runtime: python3.11
    - Memory: 3072 MB (3 GB)
    - Timeout: 900s (15 min — large chunks with many problems)
    - Layers: numpy (via AWS-provided or custom layer)
    """
    import zipfile
    import os
    
    # Package
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("knot_crossing.py", "lambda_function.py")
    
    print(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")
    
    # Deploy (update if exists, create if not)
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    
    try:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION,
            ZipFile=zip_bytes,
        )
        print(f"Updated Lambda: {LAMBDA_FUNCTION}")
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function
        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION,
            Runtime="python3.11",
            Role="arn:aws:iam::873408158081:role/lambda-s3-execution-role",
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            MemorySize=3072,
            Timeout=900,
            Environment={
                "Variables": {
                    "BUCKET": "mycelium-data",
                }
            },
        )
        print(f"Created Lambda: {LAMBDA_FUNCTION}")
    
    # Update config
    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION,
        MemorySize=3072,
        Timeout=900,
    )
    print(f"Config: 3GB memory, 900s timeout")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IAF Knot Crossing Detection MapReduce")
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--output-prefix", default="crossings/")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--deploy", action="store_true", help="Deploy Lambda first")
    parser.add_argument("--reduce-only", action="store_true", help="Skip map, just reduce")
    args = parser.parse_args()
    
    if args.deploy:
        deploy_lambda()
        print("Waiting 10s for deployment to propagate...")
        time.sleep(10)
    
    if not args.reduce_only:
        # Map phase
        map_results = map_phase(
            bucket=args.bucket,
            iaf_prefix=args.iaf_prefix,
            output_prefix=args.output_prefix,
            max_concurrent=args.max_concurrent,
        )
    else:
        # Reconstruct map results from existing stats files
        print("Reduce-only mode: reading existing results...")
        map_results = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=args.bucket,
            Prefix=f"{args.output_prefix}stats/",
        ):
            for obj in page.get("Contents", []):
                resp = s3.get_object(Bucket=args.bucket, Key=obj["Key"])
                stats = json.loads(resp["Body"].read().decode("utf-8"))
                chunk_name = obj["Key"].split("/")[-1]
                map_results.append({
                    "stats": stats,
                    "output_key": f"{args.output_prefix}{chunk_name}",
                })
        print(f"Found {len(map_results)} existing chunk results")
    
    # Reduce phase
    global_stats = reduce_phase(
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        map_results=map_results,
    )
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nOutputs in s3://{args.bucket}/{args.output_prefix}:")
    print(f"  global_stats.json                  — aggregate statistics")
    print(f"  knot_classification_dataset.json    — C2 training data")
    print(f"  bp_depth_dataset.json               — adaptive inference depth")
    print(f"  signature_examples.json             — qualitative examples per knot type")
    print(f"  *.json                              — per-chunk crossing results")


if __name__ == "__main__":
    main()
