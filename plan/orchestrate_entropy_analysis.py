"""
Orchestrator: Per-Token Entropy Analysis + C1-A Error Correlation

1. Map Phase: Extract per-token entropy maps from all IAF chunks via Lambda
2. Reduce Phase: Merge results, cross-reference with C1-A errors, compute correlations

Usage:
    python orchestrate_entropy_analysis.py --deploy  # First time: deploy Lambda
    python orchestrate_entropy_analysis.py           # Run analysis
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

s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

BUCKET = "mycelium-data"
LAMBDA_FUNCTION = "mycelium-entropy-extract"
IAF_PREFIX = "iaf_extraction/chunked/"
OUTPUT_PREFIX = "analysis/c0_entropy/"


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
            executor.submit(invoke_lambda, BUCKET, chunk, f"{OUTPUT_PREFIX}maps/"): chunk
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
                    f"ETA: {eta:.0f}s"
                )
            except Exception as e:
                failed.append((chunk_key, str(e)))
                print(f"  [{completed}/{len(chunks)}] FAILED {chunk_key}: {e}")

    print(f"\nMap complete: {time.time()-start:.1f}s, {len(results)} ok, {len(failed)} failed")
    return results


def load_entropy_maps():
    """Load all entropy map results from S3."""
    print("\nLoading entropy maps from S3...")
    maps_prefix = f"{OUTPUT_PREFIX}maps/"

    all_maps = {}
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=maps_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            response = s3.get_object(Bucket=BUCKET, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))

            for record in data:
                problem_id = str(record.get("problem_id", ""))
                all_maps[problem_id] = record

    print(f"Loaded {len(all_maps)} entropy maps")
    return all_maps


def load_c1a_training_data():
    """Load C1-A training data with boundary_probs."""
    print("Loading C1-A training data...")
    key = "c1_training_v6/merged_training.jsonl"

    response = s3.get_object(Bucket=BUCKET, Key=key)
    content = response["Body"].read().decode("utf-8")

    records = {}
    for line in content.strip().split("\n"):
        if line:
            record = json.loads(line)
            problem_id = record.get("problem_idx", "")
            records[problem_id] = record

    print(f"Loaded {len(records)} C1-A training records")
    return records


def load_c1a_errors():
    """Load C1-A error analysis."""
    print("Loading C1-A error analysis...")
    key = "analysis/c1a_error_analysis.json"

    response = s3.get_object(Bucket=BUCKET, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    # Index by problem text (since error analysis uses text, not idx)
    errors = {}
    for example in data.get("false_negatives", {}).get("examples", []):
        text = example.get("problem_text", "")
        errors[text] = example

    print(f"Loaded {len(errors)} C1-A error examples")
    return data, errors


def compute_correlation(x, y):
    """Compute Pearson correlation between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov / (std_x * std_y)


def reduce_phase(map_results):
    """
    Analyze entropy maps and correlate with C1-A errors.
    """
    print("\n" + "=" * 60)
    print("REDUCE PHASE: Entropy Analysis + C1-A Correlation")
    print("=" * 60)

    # Load all data
    entropy_maps = load_entropy_maps()
    c1a_data = load_c1a_training_data()
    error_data, error_by_text = load_c1a_errors()

    # Build problem_text -> problem_id mapping for error correlation
    text_to_id = {r.get("problem_text", ""): pid for pid, r in c1a_data.items()}

    # === Analysis A: Per-Token Entropy vs Boundary Correlation ===
    print("\n--- A. Per-Token Entropy vs Boundary Correlation ---")

    entropy_at_boundary = []
    entropy_at_non_boundary = []
    problem_correlations = []

    for problem_id, c1a_record in c1a_data.items():
        if problem_id not in entropy_maps:
            continue

        entropy_data = entropy_maps[problem_id]
        entropy_map = entropy_data.get("entropy_map_weighted", [])
        bio_labels = c1a_record.get("bio_labels", [])

        if not entropy_map or not bio_labels:
            continue

        # Align lengths
        min_len = min(len(entropy_map), len(bio_labels))

        for i in range(min_len):
            is_boundary = bio_labels[i] in ("B-COMP", "I-COMP")
            if is_boundary:
                entropy_at_boundary.append(entropy_map[i])
            else:
                entropy_at_non_boundary.append(entropy_map[i])

        # Per-problem correlation
        boundary_binary = [1 if l in ("B-COMP", "I-COMP") else 0 for l in bio_labels[:min_len]]
        if sum(boundary_binary) > 0:  # Has at least one boundary
            corr = compute_correlation(entropy_map[:min_len], boundary_binary)
            problem_correlations.append(corr)

    mean_entropy_boundary = sum(entropy_at_boundary) / len(entropy_at_boundary) if entropy_at_boundary else 0
    mean_entropy_non_boundary = sum(entropy_at_non_boundary) / len(entropy_at_non_boundary) if entropy_at_non_boundary else 0
    avg_problem_corr = sum(problem_correlations) / len(problem_correlations) if problem_correlations else 0

    print(f"  Mean entropy at boundary tokens:     {mean_entropy_boundary:.4f}")
    print(f"  Mean entropy at non-boundary tokens: {mean_entropy_non_boundary:.4f}")
    print(f"  Ratio (boundary/non-boundary):       {mean_entropy_boundary/max(mean_entropy_non_boundary, 1e-10):.2f}x")
    print(f"  Avg per-problem correlation:         {avg_problem_corr:.4f}")

    # === Analysis B: C1-A Error Overlap with High-Entropy Regions ===
    print("\n--- B. C1-A Error Overlap with High-Entropy Regions ---")

    total_missed_windows = 0
    missed_in_high_entropy = 0
    total_false_positives = 0
    fp_in_high_entropy = 0

    high_entropy_problems = []
    low_entropy_problems = []

    for problem_id, entropy_data in entropy_maps.items():
        c1a_record = c1a_data.get(problem_id, {})
        problem_text = c1a_record.get("problem_text", "")

        # Check if this problem is in error set
        error_info = error_by_text.get(problem_text)

        mean_entropy = entropy_data.get("mean_entropy", 0)
        high_regions = entropy_data.get("high_entropy_regions", [])

        # Track for examples
        if mean_entropy > 0:
            high_entropy_problems.append((mean_entropy, problem_id, entropy_data, c1a_record, error_info))

        if error_info:
            missed_windows = error_info.get("missed_windows", [])
            n_input = entropy_data.get("n_input_tokens", 0)

            for mw in missed_windows:
                window_idx = mw.get("window_idx", 0)
                # Convert window index to token position (W=16, S=8)
                window_start = window_idx * 8
                window_end = min(window_start + 16, n_input)

                total_missed_windows += 1

                # Check overlap with high-entropy regions
                for region_start, region_end, _ in high_regions:
                    if window_start < region_end and window_end > region_start:
                        missed_in_high_entropy += 1
                        break

    # Sort by entropy for examples
    high_entropy_problems.sort(key=lambda x: -x[0])
    low_entropy_problems = sorted(high_entropy_problems, key=lambda x: x[0])

    fn_overlap_rate = missed_in_high_entropy / max(total_missed_windows, 1)
    print(f"  Total C1-A missed windows:              {total_missed_windows}")
    print(f"  Missed windows in high-entropy regions: {missed_in_high_entropy} ({fn_overlap_rate*100:.1f}%)")

    # === Analysis C: Problem-Level Statistics ===
    print("\n--- C. Problem-Level Statistics ---")

    all_mean_entropies = [e.get("mean_entropy", 0) for e in entropy_maps.values()]
    all_high_fractions = [e.get("high_entropy_fraction", 0) for e in entropy_maps.values()]

    print(f"  Total problems with entropy maps: {len(entropy_maps)}")
    print(f"  Mean entropy (dataset-wide):      {sum(all_mean_entropies)/len(all_mean_entropies):.4f}")
    print(f"  Entropy std (across problems):    {math.sqrt(sum((e - sum(all_mean_entropies)/len(all_mean_entropies))**2 for e in all_mean_entropies)/len(all_mean_entropies)):.4f}")
    print(f"  Mean high-entropy fraction:       {sum(all_high_fractions)/len(all_high_fractions)*100:.1f}%")

    # === Analysis D: Viability Assessment ===
    print("\n--- D. C0 Viability Assessment ---")

    viability = {
        "entropy_boundary_ratio": mean_entropy_boundary / max(mean_entropy_non_boundary, 1e-10),
        "avg_problem_correlation": avg_problem_corr,
        "false_negative_entropy_overlap": fn_overlap_rate,
        "total_missed_windows": total_missed_windows,
        "missed_in_high_entropy": missed_in_high_entropy,
    }

    # Decision logic
    if fn_overlap_rate > 0.5:
        recommendation = "proceed_with_c0"
        rationale = f"{fn_overlap_rate*100:.0f}% of C1-A's false negatives fall in high-entropy regions. Entropy-guided text expansion is likely to help."
    elif fn_overlap_rate > 0.3:
        recommendation = "marginal"
        rationale = f"{fn_overlap_rate*100:.0f}% overlap is borderline. C0 might help marginally."
    else:
        recommendation = "skip_c0"
        rationale = f"Only {fn_overlap_rate*100:.0f}% of false negatives are in high-entropy regions. The bottleneck is likely elsewhere."

    viability["recommendation"] = recommendation
    viability["rationale"] = rationale

    print(f"  False negative overlap rate: {fn_overlap_rate*100:.1f}%")
    print(f"  Recommendation: {recommendation}")
    print(f"  Rationale: {rationale}")

    # === Save Results ===
    print("\n--- Saving Results ---")

    # Summary statistics
    summary = {
        "total_problems": len(entropy_maps),
        "mean_entropy_boundary": mean_entropy_boundary,
        "mean_entropy_non_boundary": mean_entropy_non_boundary,
        "entropy_ratio": mean_entropy_boundary / max(mean_entropy_non_boundary, 1e-10),
        "avg_problem_correlation": avg_problem_corr,
        "total_missed_windows": total_missed_windows,
        "missed_in_high_entropy": missed_in_high_entropy,
        "fn_overlap_rate": fn_overlap_rate,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}summary_statistics.json",
        Body=json.dumps(summary, indent=2).encode("utf-8"),
    )
    print(f"  Saved: s3://{BUCKET}/{OUTPUT_PREFIX}summary_statistics.json")

    # Viability assessment
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}c0_viability.json",
        Body=json.dumps(viability, indent=2).encode("utf-8"),
    )
    print(f"  Saved: s3://{BUCKET}/{OUTPUT_PREFIX}c0_viability.json")

    # High entropy examples (top 20)
    examples_high = []
    for mean_e, pid, e_data, c1a_rec, err in high_entropy_problems[:20]:
        examples_high.append({
            "problem_id": pid,
            "problem_text": c1a_rec.get("problem_text", "")[:500],
            "mean_entropy": mean_e,
            "high_entropy_regions": e_data.get("high_entropy_regions", []),
            "n_missed_windows": err.get("n_missed", 0) if err else 0,
        })

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}examples_high_entropy.json",
        Body=json.dumps(examples_high, indent=2).encode("utf-8"),
    )
    print(f"  Saved: s3://{BUCKET}/{OUTPUT_PREFIX}examples_high_entropy.json")

    # Low entropy examples (bottom 20)
    examples_low = []
    for mean_e, pid, e_data, c1a_rec, err in low_entropy_problems[:20]:
        examples_low.append({
            "problem_id": pid,
            "problem_text": c1a_rec.get("problem_text", "")[:500],
            "mean_entropy": mean_e,
            "high_entropy_regions": e_data.get("high_entropy_regions", []),
            "n_missed_windows": err.get("n_missed", 0) if err else 0,
        })

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}examples_low_entropy.json",
        Body=json.dumps(examples_low, indent=2).encode("utf-8"),
    )
    print(f"  Saved: s3://{BUCKET}/{OUTPUT_PREFIX}examples_low_entropy.json")

    return viability


def deploy_lambda():
    """Deploy or update the Lambda function."""
    zip_path = "lambda_entropy_extract.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("lambda_entropy_extract.py", "lambda_function.py")

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
    parser.add_argument("--skip-map", action="store_true", help="Skip map phase, only run reduce")
    args = parser.parse_args()

    if args.deploy:
        deploy_lambda()
        print("Waiting 10s for Lambda to be ready...")
        time.sleep(10)

    if not args.skip_map:
        map_results = map_phase(args.max_concurrent)
    else:
        map_results = []

    viability = reduce_phase(map_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Recommendation: {viability['recommendation']}")
    print(f"Rationale: {viability['rationale']}")


if __name__ == "__main__":
    main()
