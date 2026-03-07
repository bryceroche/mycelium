"""
MapReduce Orchestrator: C1 Training Data Extraction (Validated Heads)

Uses L22H3 (heartbeat), L23H11 (crossings), L22H4 (alarms) to extract
C1 multi-task training data.

Usage:
    python orchestrate_c1_extract_v3.py --deploy --max-concurrent 50

Output:
    s3://mycelium-data/c1_training_v3/
"""

import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3


s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

LAMBDA_FUNCTION = "mycelium-c1-extract-v3"
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
                    f"trans: {s.get('avg_transitions', 0):.1f} | "
                    f"cross: {s.get('avg_crossings', 0):.1f} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix, map_results):
    print("\n" + "=" * 70)
    print("REDUCE: C1 Training Data Statistics (Validated Heads)")
    print("=" * 70)

    global_stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "bio_label_dist": {"B-COMP": 0, "I-COMP": 0, "O": 0},
        "knot_class_dist": defaultdict(int),
        "bp_depth_dist": {1: 0, 2: 0, 3: 0},
        "avg_transitions": 0.0,
        "avg_crossings": 0.0,
        "avg_spans": 0.0,
        "avg_alarms": 0.0,
    }

    total_trans = 0
    total_cross = 0
    total_spans = 0
    total_alarms = 0

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
        total_trans += s.get("avg_transitions", 0) * n
        total_cross += s.get("avg_crossings", 0) * n
        total_spans += s.get("avg_spans", 0) * n
        total_alarms += s.get("avg_alarms", 0) * n

    gen = max(1, global_stats["generated"])
    global_stats["avg_transitions"] = total_trans / gen
    global_stats["avg_crossings"] = total_cross / gen
    global_stats["avg_spans"] = total_spans / gen
    global_stats["avg_alarms"] = total_alarms / gen

    # Print summary
    print(f"\nTotal problems:     {global_stats['total']:,}")
    print(f"Records generated:  {global_stats['generated']:,}")
    print(f"Skipped:            {global_stats['skipped']:,}")

    print(f"\n--- Heartbeat Statistics (L22H3) ---")
    print(f"Avg transitions/problem:  {global_stats['avg_transitions']:.1f}")
    print(f"Avg spans/problem:        {global_stats['avg_spans']:.1f}")

    print(f"\n--- Crossing Statistics (L22H3 × L23H11) ---")
    print(f"Avg crossings/problem:    {global_stats['avg_crossings']:.1f}")

    print(f"\n--- Alarm Statistics (L22H4) ---")
    print(f"Avg alarm events/problem: {global_stats['avg_alarms']:.1f}")

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

    print(f"\n--- Knot Class Distribution (top 15) ---")
    # 31 classes: 0=unknot, 1-2=1cross, 3-6=2cross, 7-14=3cross, 15-30=4cross
    KNOT_SIGS = [
        "",  # 0: unknot
        "U", "O",  # 1-2: 1 crossing
        "UU", "UO", "OU", "OO",  # 3-6: 2 crossings
        "UUU", "UUO", "UOU", "UOO", "OUU", "OUO", "OOU", "OOO",  # 7-14: 3 crossings
        "UUUU", "UUUO", "UUOU", "UUOO", "UOUU", "UOUO", "UOOU", "UOOO",  # 15-22: 4 cross
        "OUUU", "OUUO", "OUOU", "OUOO", "OOUU", "OOUO", "OOOU", "OOOO",  # 23-30: 4 cross
    ]
    kcd = sorted(global_stats["knot_class_dist"].items(), key=lambda x: -x[1])
    for cls_idx, count in kcd[:15]:
        sig = KNOT_SIGS[cls_idx] if cls_idx < len(KNOT_SIGS) else f"class_{cls_idx}"
        if sig == "":
            sig = "(unknot)"
        pct = count / gen * 100
        print(f"  {sig:12s} (cls {cls_idx:2d}): {count:>6,} ({pct:5.1f}%)")

    # Upload global stats
    global_stats["knot_class_dist"] = dict(global_stats["knot_class_dist"])
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(global_stats, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")

    return global_stats


def deploy_lambda(zip_path="lambda_c1_extract_v3.zip"):
    import zipfile
    import os

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c1_extract_v3.py", "lambda_function.py")

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
    parser.add_argument("--output-prefix", default="c1_training_v3/")
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
