"""
MapReduce Orchestrator: C2/C3 Extraction with Sonnet

Fans out Lambda invocations that use Claude Sonnet for CoT parsing.
API key passed via event payload.

Usage:
    python orchestrate_c2c3_sonnet.py \
        --bucket mycelium-data \
        --iaf-prefix iaf_extraction/chunked/ \
        --output-prefix c2c3_sonnet_labels/ \
        --max-concurrent 30

    # Deploy Lambda first:
    python orchestrate_c2c3_sonnet.py --deploy
"""

import json
import time
import argparse
import zipfile
import os
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

LAMBDA_FUNCTION = "mycelium-c2c3-sonnet"

# Load API key
API_KEY_PATH = Path(__file__).parent.parent / "secrets" / "anthropic_key.txt"
if API_KEY_PATH.exists():
    ANTHROPIC_API_KEY = API_KEY_PATH.read_text().strip()
else:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


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


def invoke_lambda(bucket, chunk_key, output_prefix, api_key):
    payload = {
        "chunk_key": chunk_key,
        "output_prefix": output_prefix,
        "bucket": bucket,
        "api_key": api_key,
    }

    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )
    result = json.loads(response["Payload"].read().decode("utf-8"))
    if response.get("FunctionError"):
        raise RuntimeError(f"Lambda failed: {result}")
    return result


def map_phase(bucket, iaf_prefix, output_prefix, max_concurrent, api_key):
    chunks = list_chunks(bucket, iaf_prefix)
    if not chunks:
        return []

    print(f"\nMap phase: {len(chunks)} chunks, {max_concurrent} concurrent")
    results = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_lambda, bucket, chunk, output_prefix, api_key): chunk
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
                    f"success: {s.get('parse_success', 0)} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix, map_results):
    """Aggregate stats."""
    print("\n" + "=" * 60)
    print("REDUCE: C2/C3 Sonnet Extraction Statistics")
    print("=" * 60)

    stats_prefix = f"{output_prefix}stats/"
    paginator = s3.get_paginator("list_objects_v2")

    global_stats = {
        "total_problems": 0,
        "parse_success": 0,
        "parse_failed": 0,
        "label_dist": Counter(),
    }

    for page in paginator.paginate(Bucket=bucket, Prefix=stats_prefix):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                chunk_stats = json.loads(resp["Body"].read().decode("utf-8"))

                global_stats["total_problems"] += chunk_stats.get("generated", 0)
                global_stats["parse_success"] += chunk_stats.get("parse_success", 0)
                global_stats["parse_failed"] += chunk_stats.get("parse_failed", 0)

                for lbl, cnt in chunk_stats.get("label_dist", {}).items():
                    global_stats["label_dist"][lbl] += cnt

            except Exception as e:
                print(f"  Warning: Failed to read {obj['Key']}: {e}")

    total = max(1, global_stats["total_problems"])
    print(f"\nTotal problems: {global_stats['total_problems']}")
    print(f"Parse success:  {global_stats['parse_success']} ({100*global_stats['parse_success']/total:.1f}%)")
    print(f"Parse failed:   {global_stats['parse_failed']}")

    print(f"\nC2 Label Distribution:")
    for lbl, cnt in global_stats["label_dist"].most_common():
        pct = 100 * cnt / total
        print(f"  {lbl:12s}: {cnt:5} ({pct:5.1f}%)")

    # Save global stats
    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(dict(global_stats), default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")


def deploy_lambda(zip_path="lambda_c2c3_sonnet.zip"):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c2c3_sonnet.py", "lambda_function.py")

    print(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    # Get or create IAM role
    iam = boto3.client("iam")
    role_name = "mycelium-lambda-role"
    try:
        role = iam.get_role(RoleName=role_name)
        role_arn = role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
        )
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
        )
        role_arn = role["Role"]["Arn"]
        print(f"Created IAM role: {role_arn}")
        time.sleep(10)

    try:
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION,
            ZipFile=zip_bytes
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
        )
        print(f"Created: {LAMBDA_FUNCTION}")

    # Wait for function to be ready
    print("Waiting for Lambda to be ready...")
    time.sleep(5)

    try:
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION,
            MemorySize=3072,
            Timeout=900
        )
        print("Configured: 3GB memory, 15min timeout")
    except Exception as e:
        print(f"Config update pending: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--output-prefix", default="c2c3_sonnet_labels/")
    parser.add_argument("--max-concurrent", type=int, default=30)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ERROR: No Anthropic API key found!")
        print("Set ANTHROPIC_API_KEY env var or create secrets/anthropic_key.txt")
        return

    if args.deploy:
        deploy_lambda()
        print("\nWaiting 10s for deployment to propagate...")
        time.sleep(10)

    # Optionally limit chunks for testing
    results = map_phase(
        args.bucket, args.iaf_prefix, args.output_prefix, args.max_concurrent, ANTHROPIC_API_KEY
    )

    reduce_phase(args.bucket, args.output_prefix, results)


if __name__ == "__main__":
    main()
