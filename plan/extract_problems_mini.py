"""Extract just problem_text and cot_text from a few IAF chunks for batch testing."""

import json
import boto3
from botocore.config import Config

s3_config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=s3_config)
BUCKET = "mycelium-data"

def extract_sample():
    """Extract problems from first 10 chunks."""
    problems = []

    paginator = s3.get_paginator("list_objects_v2")
    chunk_keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix="iaf_extraction/chunked/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/stats/" not in obj["Key"]:
                chunk_keys.append(obj["Key"])

    # All chunks (or first 10 for testing)
    chunk_keys = sorted(chunk_keys)
    print(f"Extracting from {len(chunk_keys)} chunks...")

    for i, chunk_key in enumerate(chunk_keys):
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=chunk_key)
            chunk_data = json.loads(resp["Body"].read().decode("utf-8"))
            chunk_problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

            for p in chunk_problems:
                problems.append({
                    "problem_id": f"chunk{i}_{p.get('problem_idx', len(problems))}",
                    "problem_text": p.get("problem_text", ""),
                    "cot_text": p.get("generated_cot", ""),
                })

            print(f"  {i+1}/{len(chunk_keys)}: {len(problems)} problems")

        except Exception as e:
            print(f"  Failed {chunk_key}: {e}")

    # Save to file
    output_file = "/tmp/problems_all.json"
    with open(output_file, "w") as f:
        json.dump(problems, f)

    print(f"\nSaved {len(problems)} problems to {output_file}")
    return problems

if __name__ == "__main__":
    extract_sample()
