"""
Orchestrate Lambda MapReduce: Extract Per-Step X Vectors

Invokes lambda_x_vectors for each IAF chunk to extract aligned (X, Y) pairs.

Usage:
    python orchestrate_x_vectors.py --deploy   # Deploy/update Lambda
    python orchestrate_x_vectors.py --run      # Run extraction
    python orchestrate_x_vectors.py --status   # Check progress
    python orchestrate_x_vectors.py --aggregate # Combine results into matrices
"""

import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np

s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

BUCKET = "mycelium-data"
FUNCTION_NAME = "mycelium-x-vectors"
IAF_PREFIX = "iaf_extraction/chunked/"
OUTPUT_PREFIX = "ib_ready/chunks/"


def get_iaf_chunks():
    """Get all IAF chunk keys in sorted order."""
    paginator = s3.get_paginator("list_objects_v2")
    chunks = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=IAF_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/stats/" not in obj["Key"]:
                chunks.append(obj["Key"])

    # Sort to ensure consistent chunk_index mapping
    chunks.sort()
    return chunks


def deploy_lambda():
    """Deploy or update the Lambda function."""
    import zipfile
    import io

    # Read the Lambda code
    with open("lambda_x_vectors.py", "r") as f:
        code = f.read()

    # Create zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", code.replace(
            "def lambda_handler",
            "# Entry point\ndef lambda_handler"
        ))

    zip_buffer.seek(0)
    zip_bytes = zip_buffer.read()

    try:
        # Try to update existing function
        response = lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_bytes,
        )
        print(f"Updated Lambda function: {FUNCTION_NAME}")

    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function
        response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime="python3.11",
            Role="arn:aws:iam::873408158081:role/mycelium-lambda-role",
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=900,
            MemorySize=3008,
        )
        print(f"Created Lambda function: {FUNCTION_NAME}")

    # Wait for function to be active
    print("Waiting for function to be active...")
    time.sleep(10)

    return response


def invoke_chunk(chunk_key: str, chunk_index: int):
    """Invoke Lambda for one chunk."""
    payload = {
        "chunk_key": chunk_key,
        "chunk_index": chunk_index,
        "output_prefix": OUTPUT_PREFIX,
    }

    try:
        response = lambda_client.invoke(
            FunctionName=FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )

        result = json.loads(response["Payload"].read().decode("utf-8"))

        if response.get("FunctionError"):
            return {"chunk_key": chunk_key, "status": "error", "error": result}

        return {"chunk_key": chunk_key, "status": "success", "result": result}

    except Exception as e:
        return {"chunk_key": chunk_key, "status": "error", "error": str(e)}


def run_extraction(max_concurrent: int = 10):
    """Run X vector extraction on all chunks."""
    chunks = get_iaf_chunks()
    print(f"Found {len(chunks)} IAF chunks")

    results = []
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(invoke_chunk, chunk_key, idx): (idx, chunk_key)
            for idx, chunk_key in enumerate(chunks)
        }

        for future in as_completed(futures):
            idx, chunk_key = futures[future]
            result = future.result()
            results.append(result)

            if result["status"] == "success":
                success += 1
                stats = result.get("result", {}).get("stats", {})
                aligned = stats.get("aligned_steps", 0)
                total = stats.get("total_steps", 0)
                print(f"[{success + failed}/{len(chunks)}] {chunk_key}: {aligned}/{total} aligned")
            else:
                failed += 1
                print(f"[{success + failed}/{len(chunks)}] {chunk_key}: FAILED - {result.get('error', 'unknown')}")

    print(f"\nComplete: {success} success, {failed} failed")

    # Save run summary
    summary = {
        "total_chunks": len(chunks),
        "success": success,
        "failed": failed,
        "results": results,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}run_summary.json",
        Body=json.dumps(summary, indent=2, default=str).encode("utf-8"),
    )

    return results


def check_status():
    """Check extraction progress."""
    paginator = s3.get_paginator("list_objects_v2")

    chunks_done = 0
    stats_files = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{OUTPUT_PREFIX}stats/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("_stats.json"):
                chunks_done += 1
                stats_files.append(obj["Key"])

    print(f"Chunks processed: {chunks_done}")

    if stats_files:
        # Sample a few stats
        total_aligned = 0
        total_steps = 0

        for key in stats_files[:10]:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            stats = json.loads(resp["Body"].read().decode("utf-8"))
            total_aligned += stats.get("aligned_steps", 0)
            total_steps += stats.get("total_steps", 0)

        if total_steps > 0:
            rate = total_aligned / total_steps * 100
            print(f"Alignment rate (sample): {rate:.1f}% ({total_aligned}/{total_steps})")


def aggregate_results():
    """Combine all chunk results into final X, Y matrices."""
    paginator = s3.get_paginator("list_objects_v2")

    print("Collecting aligned results...")

    all_x = []
    all_y = []
    step_map = []

    # Y label stats
    y_stats = {
        "step_type": {},
        "complexity_change": {},
        "output_type": {},
        "step_position": {},
        "reference_distance": {},
    }

    chunk_count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=OUTPUT_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl") and "/stats/" not in obj["Key"]:
                resp = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
                content = resp["Body"].read().decode("utf-8")

                for line in content.strip().split("\n"):
                    if not line.strip():
                        continue

                    record = json.loads(line)

                    if record.get("alignment_status") != "success":
                        continue

                    x_vector = record.get("x_vector", [])
                    y_vector = record.get("y_vector", [])
                    y_labels = record.get("y_labels", {})

                    if len(x_vector) == 20 and len(y_vector) == 7:
                        all_x.append(x_vector)
                        all_y.append(y_vector)
                        step_map.append({
                            "problem_id": record.get("problem_id"),
                            "step_idx": record.get("step_idx"),
                        })

                        # Collect Y stats
                        for key in y_stats:
                            val = str(y_labels.get(key, "unknown"))
                            y_stats[key][val] = y_stats[key].get(val, 0) + 1

                chunk_count += 1
                if chunk_count % 20 == 0:
                    print(f"  Processed {chunk_count} chunks, {len(all_x)} aligned steps...")

    print(f"\nTotal aligned steps: {len(all_x)}")

    if not all_x:
        print("No aligned data found!")
        return

    # Convert to numpy arrays
    X_matrix = np.array(all_x, dtype=np.float32)
    Y_matrix = np.array(all_y, dtype=np.int32)

    print(f"X_matrix shape: {X_matrix.shape}")
    print(f"Y_matrix shape: {Y_matrix.shape}")

    # Compute X feature statistics
    feature_names = []
    for head in ['L22H3', 'L22H4', 'L23H11', 'L23H23', 'L24H6']:
        for feat in ['reading', 'slope', 'entropy', 'focus']:
            feature_names.append(f"{head}_{feat}")

    feature_stats = {}
    for i, name in enumerate(feature_names):
        col = X_matrix[:, i]
        feature_stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    # Save matrices
    print("Saving matrices to S3...")

    # X matrix
    import io
    x_buffer = io.BytesIO()
    np.save(x_buffer, X_matrix)
    x_buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key="ib_ready/X_matrix.npy", Body=x_buffer.read())

    # Y matrix
    y_buffer = io.BytesIO()
    np.save(y_buffer, Y_matrix)
    y_buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key="ib_ready/Y_matrix.npy", Body=y_buffer.read())

    # Step map
    s3.put_object(
        Bucket=BUCKET,
        Key="ib_ready/step_map.json",
        Body=json.dumps(step_map).encode("utf-8"),
    )

    # Feature stats
    s3.put_object(
        Bucket=BUCKET,
        Key="ib_ready/feature_stats.json",
        Body=json.dumps(feature_stats, indent=2).encode("utf-8"),
    )

    # Y stats
    s3.put_object(
        Bucket=BUCKET,
        Key="ib_ready/y_stats.json",
        Body=json.dumps(y_stats, indent=2).encode("utf-8"),
    )

    # Summary
    summary = {
        "n_aligned_steps": len(all_x),
        "x_shape": list(X_matrix.shape),
        "y_shape": list(Y_matrix.shape),
        "feature_names": feature_names,
        "y_dimensions": [
            "step_type", "complexity_change", "n_operands", "has_dependency",
            "output_type", "step_position", "reference_distance"
        ],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="ib_ready/summary.json",
        Body=json.dumps(summary, indent=2).encode("utf-8"),
    )

    print(f"\nSaved to s3://{BUCKET}/ib_ready/")
    print(f"  X_matrix.npy: {X_matrix.shape}")
    print(f"  Y_matrix.npy: {Y_matrix.shape}")

    # Print feature stats
    print("\nX Feature Statistics:")
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for name in feature_names[:10]:
        s = feature_stats[name]
        print(f"{name:<20} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
    print("...")

    # Print Y distribution
    print("\nY Distribution:")
    for dim, counts in y_stats.items():
        print(f"  {dim}:")
        for val, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            pct = count / len(all_x) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Deploy Lambda")
    parser.add_argument("--run", action="store_true", help="Run extraction")
    parser.add_argument("--status", action="store_true", help="Check status")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results")
    parser.add_argument("--concurrent", type=int, default=10, help="Max concurrent Lambdas")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
    elif args.run:
        run_extraction(args.concurrent)
    elif args.status:
        check_status()
    elif args.aggregate:
        aggregate_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
