"""
MapReduce Orchestrator: C1 Training Data Generation

Fans out Lambda invocations to generate C1 multi-task training data
from IAF extractions, optionally using pre-computed crossing results.

Usage:
    # With pre-computed crossings from v2 run:
    python orchestrate_c1_datagen.py \
        --bucket mycelium-data \
        --iaf-prefix iaf_extraction/chunked/ \
        --crossings-prefix crossings_v2/ \
        --output-prefix c1_training/ \
        --max-concurrent 50

    # Without pre-computed crossings (compute inline):
    python orchestrate_c1_datagen.py \
        --bucket mycelium-data \
        --iaf-prefix iaf_extraction/chunked/ \
        --output-prefix c1_training/ \
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

LAMBDA_FUNCTION = "mycelium-c1-datagen"


def list_chunks(bucket: str, prefix: str) -> list[str]:
    chunks = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and obj["Size"] > 0:
                # Skip stats subdirectory
                if "/stats/" not in obj["Key"]:
                    chunks.append(obj["Key"])
    print(f"Found {len(chunks)} chunks in s3://{bucket}/{prefix}")
    return sorted(chunks)


def invoke_lambda(bucket, chunk_key, output_prefix, crossings_prefix=None):
    payload = {
        "chunk_key": chunk_key,
        "output_prefix": output_prefix,
        "bucket": bucket,
    }
    if crossings_prefix:
        payload["crossings_prefix"] = crossings_prefix

    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )
    result = json.loads(response["Payload"].read().decode("utf-8"))
    if response.get("FunctionError"):
        raise RuntimeError(f"Lambda failed on {chunk_key}: {result}")
    return result


def map_phase(bucket, iaf_prefix, output_prefix, crossings_prefix, max_concurrent):
    chunks = list_chunks(bucket, iaf_prefix)
    if not chunks:
        return []

    print(f"\nMap phase: {len(chunks)} chunks, {max_concurrent} concurrent")
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(
                invoke_lambda, bucket, chunk, output_prefix, crossings_prefix
            ): chunk
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
                    f"avg_trans: {s.get('avg_transitions', 0):.1f} | "
                    f"avg_spans: {s.get('avg_spans', 0):.1f} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix, map_results):
    print("\n" + "=" * 60)
    print("REDUCE: C1 Training Data Statistics")
    print("=" * 60)

    global_stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "bio_label_dist": {"B-COMP": 0, "I-COMP": 0, "O": 0},
        "knot_class_dist": defaultdict(int),
        "bp_depth_dist": {1: 0, 2: 0, 3: 0},
        "avg_steps": 0.0,
        "avg_spans": 0.0,
    }

    total_steps = 0
    total_spans = 0

    for r in map_results:
        s = r.get("stats", {})
        global_stats["total"] += s.get("total", 0)
        global_stats["generated"] += s.get("generated", 0)
        global_stats["skipped"] += s.get("skipped", 0)

        for label, count in s.get("bio_label_dist", {}).items():
            global_stats["bio_label_dist"][label] = (
                global_stats["bio_label_dist"].get(label, 0) + count
            )

        for cls, count in s.get("knot_class_dist", {}).items():
            global_stats["knot_class_dist"][int(cls)] += count

        for depth, count in s.get("bp_depth_dist", {}).items():
            global_stats["bp_depth_dist"][int(depth)] = (
                global_stats["bp_depth_dist"].get(int(depth), 0) + count
            )

        n = s.get("generated", 0)
        total_steps += s.get("avg_steps", 0) * n
        total_spans += s.get("avg_spans", 0) * n

    gen = max(1, global_stats["generated"])
    global_stats["avg_steps"] = total_steps / gen
    global_stats["avg_spans"] = total_spans / gen

    # Print summary
    print(f"\nTotal problems:     {global_stats['total']}")
    print(f"Records generated:  {global_stats['generated']}")
    print(f"Skipped:            {global_stats['skipped']}")

    print(f"\n--- Segmentation Statistics ---")
    print(f"Avg steps/problem:  {global_stats['avg_steps']:.1f}")
    print(f"Avg spans/problem:  {global_stats['avg_spans']:.1f}")

    print(f"\n--- BIO Label Distribution ---")
    bio = global_stats["bio_label_dist"]
    total_tokens = sum(bio.values())
    for label in ["B-COMP", "I-COMP", "O"]:
        count = bio.get(label, 0)
        pct = count / max(1, total_tokens) * 100
        print(f"  {label:8s}: {count:>10,} ({pct:5.1f}%)")

    print(f"\n--- BP Depth Distribution ---")
    for depth in [1, 2, 3]:
        count = global_stats["bp_depth_dist"].get(depth, 0)
        pct = count / gen * 100
        bar = "█" * int(pct / 2)
        print(f"  {depth} rounds: {count:>6,} ({pct:5.1f}%) {bar}")

    print(f"\n--- Knot Class Distribution ---")
    from lambda_c1_datagen import KNOT_CLASSES, RARE_CLASS
    idx_to_sig = {v: k for k, v in KNOT_CLASSES.items()}
    idx_to_sig[RARE_CLASS] = "RARE"

    kcd = sorted(global_stats["knot_class_dist"].items(), key=lambda x: -x[1])
    for cls_idx, count in kcd[:15]:
        sig = idx_to_sig.get(cls_idx, f"class_{cls_idx}")
        if sig == "":
            sig = "(unknot)"
        pct = count / gen * 100
        print(f"  {sig:12s} (cls {cls_idx:2d}): {count:>6,} ({pct:5.1f}%)")

    # Upload global stats
    stats_clean = json.loads(json.dumps(global_stats, default=str))
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(stats_clean, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")

    return global_stats


def deploy_lambda(zip_path="lambda_c1_datagen.zip"):
    import zipfile, os

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c1_datagen.py", "lambda_function.py")

    print(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    try:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION, ZipFile=zip_bytes
        )
        print(f"Updated: {LAMBDA_FUNCTION}")
    except lambda_client.exceptions.ResourceNotFoundException:
        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION,
            Runtime="python3.11",
            Role="arn:aws:iam::role/mycelium-lambda-role",
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            MemorySize=3072,
            Timeout=900,
            Environment={"Variables": {"BUCKET": "mycelium-data"}},
        )
        print(f"Created: {LAMBDA_FUNCTION}")

    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION, MemorySize=3072, Timeout=900
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--crossings-prefix", default=None,
                        help="Pre-computed crossings (e.g. crossings_v2/)")
    parser.add_argument("--output-prefix", default="c1_training/")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        time.sleep(10)

    results = map_phase(
        args.bucket, args.iaf_prefix, args.output_prefix,
        args.crossings_prefix, args.max_concurrent,
    )

    reduce_phase(args.bucket, args.output_prefix, results)


if __name__ == "__main__":
    main()
