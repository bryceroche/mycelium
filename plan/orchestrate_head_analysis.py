"""
MapReduce Orchestrator: Attention Head Telegraph Analysis

Fans out Lambda invocations to analyze the 10 pre-selected attention heads,
then reduces results into global statistics, rankings, and matrices.

Usage:
    python orchestrate_head_analysis.py --deploy --max-concurrent 50

Output:
    s3://mycelium-data/head_analysis/
    ├── per_head_stats.json
    ├── sync_matrix.json
    ├── correlation_matrix.json
    ├── layer_summary.json
    ├── global_summary.json
    ├── example_waveforms.json
    └── per_chunk_stats/*.json
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

LAMBDA_FUNCTION = "mycelium-head-analysis"
BUCKET = "mycelium-data"

HEAD_KEYS = [
    "L5H19", "L14H0", "L22H3", "L22H4", "L23H11",
    "L23H23", "L24H4", "L24H6", "L24H16", "L25H1"
]

# Head to layer mapping
HEAD_LAYERS = {
    "L5H19": 5, "L14H0": 14, "L22H3": 22, "L22H4": 22,
    "L23H11": 23, "L23H23": 23, "L24H4": 24, "L24H6": 24,
    "L24H16": 24, "L25H1": 25
}


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
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(chunks)}] {chunk_key.split('/')[-1]} | "
                      f"processed: {result.get('n_processed', 0)} | ETA: {eta:.0f}s")
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix):
    """
    Load all chunk stats and reduce into global statistics.
    """
    print("\n" + "=" * 60)
    print("REDUCE: Head Analysis Statistics")
    print("=" * 60)

    # Load all chunk stats
    chunk_stats_prefix = f"{output_prefix}per_chunk_stats/"
    chunks = list_chunks(bucket, chunk_stats_prefix)

    all_head_stats = {head: defaultdict(list) for head in HEAD_KEYS}
    all_correlations = defaultdict(list)
    all_sync_scores = defaultdict(list)
    all_waveforms = []
    total_problems = 0

    for chunk_key in chunks:
        resp = s3.get_object(Bucket=bucket, Key=chunk_key)
        data = json.loads(resp["Body"].read().decode("utf-8"))

        total_problems += data.get("n_processed", 0)

        # Aggregate head stats (weighted by chunk size)
        for head in HEAD_KEYS:
            chunk_head = data.get("head_stats", {}).get(head, {})
            for metric, vals in chunk_head.items():
                if isinstance(vals, dict) and "mean" in vals:
                    all_head_stats[head][metric].append(vals["mean"])

        # Aggregate correlations
        for key, vals in data.get("correlations", {}).items():
            if isinstance(vals, dict) and "mean" in vals:
                all_correlations[key].append(vals["mean"])

        # Aggregate sync scores
        for key, vals in data.get("sync_scores", {}).items():
            if isinstance(vals, dict) and "mean" in vals:
                all_sync_scores[key].append(vals["mean"])

        # Collect waveforms
        all_waveforms.extend(data.get("example_waveforms", []))

    print(f"\nTotal problems analyzed: {total_problems}")

    # --- Compute global per-head stats ---
    global_head_stats = {}
    for head in HEAD_KEYS:
        stats = all_head_stats[head]
        global_head_stats[head] = {
            metric: float(np.mean(vals)) if vals else 0.0
            for metric, vals in stats.items()
        }
        global_head_stats[head]["layer"] = HEAD_LAYERS[head]

    # --- Compute telegraph quality score ---
    # New formula: bimodality × contrast × state_balance
    # state_balance = 1 - |frac_high - 0.5| × 2 (penalizes skewed heads)
    for head in HEAD_KEYS:
        s = global_head_stats[head]
        bimodal = s.get("bimodal_coef", 0)
        contrast = s.get("contrast_ratio", 1)
        frac_high = s.get("frac_high", 0.5)

        # State balance: 1.0 when 50/50, 0.0 when all one state
        state_balance = 1.0 - abs(frac_high - 0.5) * 2
        state_balance = max(0.01, state_balance)  # floor to avoid zero

        # Normalize contrast (log scale, capped at 10x)
        contrast_norm = min(np.log(max(contrast, 1)) / np.log(10), 1.0)

        # Telegraph quality: balanced time-in-state is critical
        # telegraph_score = bimodality × contrast × balance
        quality = bimodal * contrast_norm * state_balance

        global_head_stats[head]["telegraph_quality"] = float(quality)
        global_head_stats[head]["state_balance"] = float(state_balance)
        global_head_stats[head]["frac_high"] = float(frac_high)

        # Also compute "rare event" score for heads that are mostly in one state
        # but have strong signal when they do transition
        rare_event_score = contrast_norm * (1 - state_balance) * bimodal
        global_head_stats[head]["rare_event_score"] = float(rare_event_score)

    # --- Rank heads by telegraph quality ---
    ranked_heads = sorted(HEAD_KEYS, key=lambda h: global_head_stats[h]["telegraph_quality"], reverse=True)

    # --- Build correlation matrix (10x10) ---
    corr_matrix = np.zeros((len(HEAD_KEYS), len(HEAD_KEYS)))
    for i, h1 in enumerate(HEAD_KEYS):
        for j, h2 in enumerate(HEAD_KEYS):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                key = f"{h1}_{h2}"
                vals = all_correlations.get(key, [])
                corr_matrix[i, j] = np.mean(vals) if vals else 0.0
                corr_matrix[j, i] = corr_matrix[i, j]

    # --- Build sync matrix (10x10) ---
    sync_matrix = np.zeros((len(HEAD_KEYS), len(HEAD_KEYS)))
    for i, h1 in enumerate(HEAD_KEYS):
        for j, h2 in enumerate(HEAD_KEYS):
            if i == j:
                sync_matrix[i, j] = 1.0
            elif i < j:
                key = f"{h1}_{h2}"
                vals = all_sync_scores.get(key, [])
                sync_matrix[i, j] = np.mean(vals) if vals else 0.0
                sync_matrix[j, i] = sync_matrix[i, j]

    # --- Layer summary ---
    layer_summary = defaultdict(list)
    for head in HEAD_KEYS:
        layer = HEAD_LAYERS[head]
        layer_summary[layer].append({
            "head": head,
            "telegraph_quality": global_head_stats[head]["telegraph_quality"],
            "contrast_ratio": global_head_stats[head].get("contrast_ratio", 0),
            "bimodal_coef": global_head_stats[head].get("bimodal_coef", 0),
        })

    # --- Find synchronized clusters ---
    sync_threshold = 0.3
    clusters = []
    used = set()
    for i, h1 in enumerate(HEAD_KEYS):
        if h1 in used:
            continue
        cluster = [h1]
        for j, h2 in enumerate(HEAD_KEYS):
            if i != j and h2 not in used and sync_matrix[i, j] > sync_threshold:
                cluster.append(h2)
        if len(cluster) > 1:
            clusters.append(cluster)
            used.update(cluster)

    # --- Select example waveforms for top 3 and bottom 3 heads ---
    top_heads = ranked_heads[:3]
    bottom_heads = ranked_heads[-3:]
    selected_waveforms = {"top_heads": {}, "bottom_heads": {}}

    # Get 5 examples for each group
    for head in top_heads:
        examples = []
        for wf in all_waveforms[:50]:  # Search first 50
            if head in wf.get("signals", {}):
                examples.append({
                    "problem_id": wf["problem_id"],
                    "signal": wf["signals"][head][:200]  # First 200 steps
                })
                if len(examples) >= 5:
                    break
        selected_waveforms["top_heads"][head] = examples

    for head in bottom_heads:
        examples = []
        for wf in all_waveforms[:50]:
            if head in wf.get("signals", {}):
                examples.append({
                    "problem_id": wf["problem_id"],
                    "signal": wf["signals"][head][:200]
                })
                if len(examples) >= 5:
                    break
        selected_waveforms["bottom_heads"][head] = examples

    # --- Rank by rare event score too ---
    rare_event_ranked = sorted(HEAD_KEYS, key=lambda h: global_head_stats[h].get("rare_event_score", 0), reverse=True)

    # --- Print summary ---
    print("\n--- Head Rankings (BALANCED telegraph quality) ---")
    print("    Head     | Quality | Balance | Frac_Hi | Low_Mean | High_Mean | Contrast")
    print("    " + "-" * 75)
    for i, head in enumerate(ranked_heads):
        s = global_head_stats[head]
        frac_hi_pct = s.get('frac_high', 0.5) * 100
        low_mean = s.get('low_mean', 0)
        high_mean = s.get('high_mean', 0)
        print(f"  {i+1:2d}. {head:8s} | {s['telegraph_quality']:.3f}   | {s.get('state_balance', 0):.2f}    | "
              f"{frac_hi_pct:4.0f}%   | {low_mean:7.3f}  | {high_mean:8.3f}  | {s.get('contrast_ratio', 0):6.1f}")

    print("\n--- Rare Event Detectors (high contrast, low balance) ---")
    for head in rare_event_ranked[:3]:
        s = global_head_stats[head]
        frac_hi_pct = s.get('frac_high', 0.5) * 100
        print(f"  {head:8s} | rare_score: {s.get('rare_event_score', 0):.3f} | "
              f"frac_high: {frac_hi_pct:.0f}% | contrast: {s.get('contrast_ratio', 0):.1f}")

    print("\n--- Layer Distribution ---")
    for layer in sorted(layer_summary.keys()):
        heads = layer_summary[layer]
        avg_quality = np.mean([h["telegraph_quality"] for h in heads])
        print(f"  Layer {layer}: {len(heads)} heads, avg quality: {avg_quality:.3f}")

    print("\n--- Synchronized Clusters ---")
    if clusters:
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {', '.join(cluster)}")
    else:
        print("  No strong clusters found (sync threshold: 0.3)")

    print("\n--- Recommended Heads for C1 ---")
    recommended = ranked_heads[:4]
    print(f"  Primary: {', '.join(recommended[:2])}")
    print(f"  Secondary: {', '.join(recommended[2:4])}")

    # --- Upload results ---
    # Per-head stats
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}per_head_stats.json",
        Body=json.dumps(global_head_stats, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Correlation matrix
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}correlation_matrix.json",
        Body=json.dumps({
            "head_order": HEAD_KEYS,
            "matrix": corr_matrix.tolist()
        }, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Sync matrix
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}sync_matrix.json",
        Body=json.dumps({
            "head_order": HEAD_KEYS,
            "matrix": sync_matrix.tolist()
        }, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Layer summary
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}layer_summary.json",
        Body=json.dumps(dict(layer_summary), indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Example waveforms
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}example_waveforms.json",
        Body=json.dumps(selected_waveforms, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    # Global summary
    global_summary = {
        "total_problems": total_problems,
        "n_heads": len(HEAD_KEYS),
        "ranked_heads": [
            {
                "rank": i + 1,
                "head": head,
                "layer": HEAD_LAYERS[head],
                "telegraph_quality": global_head_stats[head]["telegraph_quality"],
                "state_balance": global_head_stats[head].get("state_balance", 0),
                "frac_high": global_head_stats[head].get("frac_high", 0.5),
                "contrast_ratio": global_head_stats[head].get("contrast_ratio", 0),
                "bimodal_coef": global_head_stats[head].get("bimodal_coef", 0),
                "rare_event_score": global_head_stats[head].get("rare_event_score", 0),
            }
            for i, head in enumerate(ranked_heads)
        ],
        "heartbeat_heads": ranked_heads[:2],  # Best for phase detection
        "secondary_heartbeat": ranked_heads[2:4],
        "rare_event_heads": rare_event_ranked[:2],  # Best for major boundary detection
        "synchronized_clusters": clusters,
        "noise_heads": ranked_heads[-2:] if len(ranked_heads) >= 2 else [],
        "head_roles": {
            "heartbeat_clock": ranked_heads[0] if ranked_heads else None,
            "secondary_clock": ranked_heads[1] if len(ranked_heads) > 1 else None,
            "rare_alarm": rare_event_ranked[0] if rare_event_ranked else None,
        },
        "layer_quality": {
            layer: float(np.mean([h["telegraph_quality"] for h in heads]))
            for layer, heads in layer_summary.items()
        },
    }

    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_summary.json",
        Body=json.dumps(global_summary, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"\n--- Output Files ---")
    for f in ["per_head_stats.json", "correlation_matrix.json", "sync_matrix.json",
              "layer_summary.json", "example_waveforms.json", "global_summary.json"]:
        print(f"  s3://{bucket}/{output_prefix}{f}")

    return global_summary


def deploy_lambda(zip_path="lambda_head_analysis.zip"):
    import zipfile
    import os

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_head_analysis.py", "lambda_function.py")

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
        roles = iam.list_roles()["Roles"]
        role_arn = next(
            (r["Arn"] for r in roles if "lambda" in r["RoleName"].lower()),
            None
        )
        if not role_arn:
            raise RuntimeError("No Lambda execution role found")

        # Get numpy layer
        layers = lambda_client.list_layers()["Layers"]
        numpy_layer = next(
            (l["LatestMatchingVersion"]["LayerVersionArn"]
             for l in layers if "numpy" in l["LayerName"].lower()),
            None
        )

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
    parser.add_argument("--output-prefix", default="head_analysis/")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--reduce-only", action="store_true",
                        help="Skip map phase, just reduce existing chunk stats")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        print("Waiting for Lambda to be ready...")
        time.sleep(10)

    if not args.reduce_only:
        map_phase(
            args.bucket, args.iaf_prefix,
            f"{args.output_prefix}per_chunk_stats/",
            args.max_concurrent,
        )

    reduce_phase(args.bucket, args.output_prefix)


if __name__ == "__main__":
    main()
