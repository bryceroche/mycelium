"""
MapReduce Orchestrator: C2/C3 Training Label Extraction

Fans out Lambda invocations to extract C2/C3 per-segment training labels
from IAF data (CoT parsing + alignment).

Usage:
    python orchestrate_c2c3_extract.py \
        --bucket mycelium-data \
        --iaf-prefix iaf_extraction/chunked/ \
        --output-prefix c2c3_training_data/ \
        --max-concurrent 50

    # Deploy Lambda first:
    python orchestrate_c2c3_extract.py --deploy
"""

import json
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3


s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

LAMBDA_FUNCTION = "mycelium-c2c3-extract"


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
                    f"parse: {s.get('parse_rate', 0):.1%} | "
                    f"align: {s.get('align_rate', 0):.1%} | "
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def reduce_phase(bucket, output_prefix, map_results):
    """Aggregate stats and generate quality report."""
    print("\n" + "=" * 60)
    print("REDUCE: C2/C3 Training Data Statistics")
    print("=" * 60)

    # Collect all chunk stats
    stats_prefix = f"{output_prefix}stats/"
    paginator = s3.get_paginator("list_objects_v2")

    global_stats = {
        "total_problems": 0,
        "total_cot_steps": 0,
        "parsed_steps": 0,
        "aligned_steps": 0,
        "total_segments": 0,
        "no_op_segments": 0,
        "op_segments": 0,
        "multi_op_segments": 0,
        "operation_dist": defaultdict(int),
        "total_operands": 0,
        "operands_found": 0,
        "operands_back_ref": 0,
        "unparsed_examples": [],
    }

    for page in paginator.paginate(Bucket=bucket, Prefix=stats_prefix):
        for obj in page.get("Contents", []):
            try:
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                chunk_stats = json.loads(resp["Body"].read().decode("utf-8"))

                global_stats["total_problems"] += chunk_stats.get("generated", 0)
                global_stats["total_cot_steps"] += chunk_stats.get("total_steps", 0)
                global_stats["parsed_steps"] += chunk_stats.get("parsed_steps", 0)
                global_stats["aligned_steps"] += chunk_stats.get("aligned_steps", 0)
                global_stats["total_segments"] += chunk_stats.get("total_segments", 0)
                global_stats["no_op_segments"] += chunk_stats.get("no_op_segments", 0)
                global_stats["op_segments"] += chunk_stats.get("op_segments", 0)
                global_stats["multi_op_segments"] += chunk_stats.get("multi_op_segments", 0)
                global_stats["total_operands"] += chunk_stats.get("total_operands", 0)
                global_stats["operands_found"] += chunk_stats.get("operands_found", 0)
                global_stats["operands_back_ref"] += chunk_stats.get("operands_back_ref", 0)

                for op, count in chunk_stats.get("operation_dist", {}).items():
                    global_stats["operation_dist"][op] += count

                # Collect examples (up to 50 total)
                for ex in chunk_stats.get("unparsed_examples", []):
                    if len(global_stats["unparsed_examples"]) < 50:
                        global_stats["unparsed_examples"].append(ex)

            except Exception as e:
                print(f"  Warning: Failed to read stats {obj['Key']}: {e}")

    # Compute rates
    total_steps = max(1, global_stats["total_cot_steps"])
    total_segs = max(1, global_stats["total_segments"])
    total_ops = max(1, global_stats["total_operands"])

    global_stats["parse_rate"] = global_stats["parsed_steps"] / total_steps
    global_stats["align_rate"] = global_stats["aligned_steps"] / total_steps
    global_stats["no_op_fraction"] = global_stats["no_op_segments"] / total_segs
    global_stats["operand_found_rate"] = global_stats["operands_found"] / total_ops
    global_stats["back_ref_rate"] = global_stats["operands_back_ref"] / total_ops

    # Print summary
    print(f"\nTotal problems:      {global_stats['total_problems']}")
    print(f"Total CoT steps:     {global_stats['total_cot_steps']}")
    print(f"Mean steps/problem:  {total_steps / max(1, global_stats['total_problems']):.1f}")

    print(f"\n--- Parsing Quality ---")
    print(f"Clean parse rate:    {global_stats['parse_rate']:.1%}")
    print(f"  Parsed:            {global_stats['parsed_steps']}")
    print(f"  Unparsed:          {total_steps - global_stats['parsed_steps']}")

    print(f"\n--- Alignment Quality ---")
    print(f"Alignment rate:      {global_stats['align_rate']:.1%}")
    print(f"  Aligned:           {global_stats['aligned_steps']}")
    print(f"  Unaligned:         {total_steps - global_stats['aligned_steps']}")

    print(f"\n--- Segment Distribution ---")
    print(f"Total segments:      {global_stats['total_segments']}")
    print(f"  NO_OP segments:    {global_stats['no_op_segments']} ({global_stats['no_op_fraction']:.1%})")
    print(f"  With operation:    {global_stats['op_segments']} ({global_stats['op_segments']/total_segs:.1%})")
    print(f"  Multi-op windows:  {global_stats['multi_op_segments']}")

    print(f"\n--- C3 Operand Stats ---")
    print(f"Total operands:      {global_stats['total_operands']}")
    print(f"  Found in window:   {global_stats['operands_found']} ({global_stats['operand_found_rate']:.1%})")
    print(f"  Back-references:   {global_stats['operands_back_ref']} ({global_stats['back_ref_rate']:.1%})")

    print(f"\n--- Operation Distribution (per-segment) ---")
    op_dist = dict(global_stats["operation_dist"])
    sorted_ops = sorted(op_dist.items(), key=lambda x: -x[1])
    for op, count in sorted_ops[:15]:
        pct = count / total_segs * 100
        bar = "█" * int(pct / 2)
        print(f"  {op:12s}: {count:>6,} ({pct:5.1f}%) {bar}")

    # Upload global stats
    stats_clean = {
        "total_problems": global_stats["total_problems"],
        "total_cot_steps": global_stats["total_cot_steps"],
        "mean_steps_per_problem": total_steps / max(1, global_stats["total_problems"]),
        "total_segments": global_stats["total_segments"],
        "segments_breakdown": {
            "NO_OP": global_stats["no_op_segments"],
            "single_op": global_stats["op_segments"] - global_stats["multi_op_segments"],
            "multi_op": global_stats["multi_op_segments"],
            "no_op_fraction": global_stats["no_op_fraction"],
        },
        "parsing": {
            "clean_parse_rate": global_stats["parse_rate"],
            "operation_distribution": op_dist,
            "unparsed_examples": global_stats["unparsed_examples"][:20],
        },
        "alignment": {
            "aligned_rate": global_stats["align_rate"],
        },
        "c3_stats": {
            "total_operands": global_stats["total_operands"],
            "operand_found_in_window_rate": global_stats["operand_found_rate"],
            "back_reference_rate": global_stats["back_ref_rate"],
        },
    }

    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}global_stats.json",
        Body=json.dumps(stats_clean, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nUploaded: s3://{bucket}/{output_prefix}global_stats.json")

    # Generate quality report with worked examples
    generate_quality_report(bucket, output_prefix)

    return global_stats


def generate_quality_report(bucket, output_prefix):
    """Generate detailed quality report with worked examples."""
    print("\nGenerating quality report...")

    # Sample a few JSONL files for worked examples
    examples = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl") and len(examples) < 50:
                try:
                    resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                    content = resp["Body"].read().decode("utf-8")
                    for line in content.split("\n")[:5]:  # First 5 from each chunk
                        if line.strip() and len(examples) < 50:
                            record = json.loads(line)
                            # Include only essential fields for review
                            examples.append({
                                "problem_id": record.get("problem_id"),
                                "problem_text": record.get("problem_text", "")[:500],
                                "cot_text": record.get("cot_text", "")[:500],
                                "n_steps": record.get("n_cot_steps"),
                                "parse_quality": record.get("parse_quality"),
                                "c2_labels": record.get("per_window_labels", {}).get("c2_labels"),
                                "c3_labels": record.get("per_window_labels", {}).get("c3_labels"),
                                "sample_steps": record.get("parsed_steps", [])[:3],
                            })
                except Exception as e:
                    continue

    quality_report = {
        "description": "Worked examples for human review of C2/C3 extraction quality",
        "n_examples": len(examples),
        "examples": examples,
    }

    s3.put_object(
        Bucket=bucket,
        Key=f"{output_prefix}quality_report.json",
        Body=json.dumps(quality_report, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Uploaded: s3://{bucket}/{output_prefix}quality_report.json")


def deploy_lambda(zip_path="lambda_c2c3_extract.zip"):
    import zipfile
    import os

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_c2c3_extract.py", "lambda_function.py")

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
        # Create role with basic Lambda execution + S3 access
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
            Description="Lambda role for mycelium data processing"
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
        time.sleep(10)  # Wait for role propagation

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
            Environment={"Variables": {"BUCKET": "mycelium-data"}},
        )
        print(f"Created: {LAMBDA_FUNCTION}")

    # Ensure config is correct
    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION,
        MemorySize=3072,
        Timeout=900
    )
    print(f"Configured: 3GB memory, 15min timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--output-prefix", default="c2c3_training_data/")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--deploy", action="store_true", help="Deploy Lambda function first")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        time.sleep(10)  # Wait for deployment

    results = map_phase(
        args.bucket, args.iaf_prefix, args.output_prefix, args.max_concurrent
    )

    reduce_phase(args.bucket, args.output_prefix, results)


if __name__ == "__main__":
    main()
