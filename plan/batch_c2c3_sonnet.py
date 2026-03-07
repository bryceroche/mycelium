"""
Anthropic Batch API for C2/C3 Extraction

Submits all problems as a batch request - async processing, no rate limits, 50% cheaper.
Results available within 24 hours.

Usage:
    # Submit batch
    python batch_c2c3_sonnet.py --submit

    # Check status
    python batch_c2c3_sonnet.py --status

    # Download results when complete
    python batch_c2c3_sonnet.py --download
"""

import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime

import boto3
from botocore.config import Config
import anthropic

# S3 client with increased timeout
s3_config = Config(
    read_timeout=120,
    connect_timeout=30,
    retries={'max_attempts': 3}
)

# Load API key
API_KEY_PATH = Path(__file__).parent.parent / "secrets" / "anthropic_key.txt"
if API_KEY_PATH.exists():
    ANTHROPIC_API_KEY = API_KEY_PATH.read_text().strip()
else:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

s3 = boto3.client("s3", config=s3_config)
BUCKET = "mycelium-data"

EXTRACTION_PROMPT = """Given this math problem and its chain-of-thought solution, parse each reasoning step.

PROBLEM:
{problem_text}

CHAIN-OF-THOUGHT SOLUTION:
{cot_text}

For each reasoning step, identify:
1. step_type: One of [simplify, factor, expand, substitute, evaluate, apply_theorem, solve_equation, modular_arithmetic, count, compare, trigonometric, logarithm, power, sqrt, fraction, other]
2. step_text: The exact text of this step (brief excerpt)
3. operands: List of numeric values or variable names used
4. result: The computed result (if any)
5. references_step: Index of previous step this depends on, or null

Also identify:
- problem_operations: Which operation types appear in the PROBLEM
  Choose from: FACTORIAL, LOG, TRIG, MOD, SQRT, CUBE, FRAC_POW, HIGH_POW, SQUARE, EQUATION, DIV, MUL, ADD, OTHER

Return JSON only:
{{
  "problem_operations": ["SQRT", "ADD"],
  "steps": [
    {{"step_idx": 0, "step_type": "substitute", "step_text": "Let x = 5", "operands": ["5", "x"], "result": null, "references_step": null}}
  ]
}}"""


def get_failed_problem_ids():
    """Get set of problem IDs that failed Sonnet parsing."""
    failed_ids = set()
    success_ids = set()

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="c2c3_sonnet_labels/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl") and "/stats/" not in obj["Key"]:
                resp = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
                content = resp["Body"].read().decode("utf-8")

                for line in content.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        pid = record.get("problem_id")
                        if record.get("parse_success", False):
                            success_ids.add(pid)
                        else:
                            failed_ids.add(pid)
                    except json.JSONDecodeError:
                        continue

    print(f"Found {len(success_ids)} successful, {len(failed_ids)} failed")
    return failed_ids, success_ids


def get_failed_problems():
    """Get failed problems with full data from IAF chunks."""
    failed_ids, success_ids = get_failed_problem_ids()

    # Get full problem data from IAF chunks
    problems = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix="iaf_extraction/chunked/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/stats/" not in obj["Key"]:
                resp = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
                chunk_data = json.loads(resp["Body"].read().decode("utf-8"))
                chunk_problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

                for p in chunk_problems:
                    pid = str(p.get("problem_idx", ""))
                    if pid in failed_ids:
                        problems.append({
                            "problem_id": pid,
                            "problem_text": p.get("problem_text", ""),
                            "cot_text": p.get("generated_cot", ""),
                            "level": p.get("level", ""),
                            "type": p.get("type", ""),
                        })

    print(f"Retrieved {len(problems)} failed problems with full data")
    return problems


def get_all_problems():
    """Get all problems from IAF chunks."""
    problems = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="iaf_extraction/chunked/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/stats/" not in obj["Key"]:
                resp = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
                chunk_data = json.loads(resp["Body"].read().decode("utf-8"))
                chunk_problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

                for p in chunk_problems:
                    problems.append({
                        "problem_id": str(p.get("problem_idx", len(problems))),
                        "problem_text": p.get("problem_text", ""),
                        "cot_text": p.get("generated_cot", ""),
                        "level": p.get("level", ""),
                        "type": p.get("type", ""),
                        "n_tokens": p.get("input_len", 0),
                        "source_chunk": obj["Key"],
                    })

    print(f"Found {len(problems)} total problems")
    return problems


def create_batch_requests(problems):
    """Create batch request format for Anthropic API."""
    requests = []

    for p in problems:
        if not p.get("problem_text") or not p.get("cot_text"):
            continue

        prompt = EXTRACTION_PROMPT.format(
            problem_text=p["problem_text"][:2000],
            cot_text=p["cot_text"][:4000],
        )

        requests.append({
            "custom_id": p["problem_id"],
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            }
        })

    return requests


def submit_batch(requests):
    """Submit batch to Anthropic API."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Write requests to JSONL file
    batch_file = "/tmp/c2c3_batch_requests.jsonl"
    with open(batch_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Created batch file with {len(requests)} requests")

    # Upload to Anthropic
    with open(batch_file, "rb") as f:
        batch = client.beta.messages.batches.create(
            requests=[json.loads(line) for line in open(batch_file)]
        )

    print(f"Submitted batch: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch ID
    batch_info = {
        "batch_id": batch.id,
        "created_at": datetime.now().isoformat(),
        "n_requests": len(requests),
        "status": batch.processing_status,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_sonnet_labels/batch_info.json",
        Body=json.dumps(batch_info, indent=2).encode("utf-8"),
    )

    return batch


def check_status():
    """Check batch status."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Get batch ID
    try:
        resp = s3.get_object(Bucket=BUCKET, Key="c2c3_sonnet_labels/batch_info.json")
        batch_info = json.loads(resp["Body"].read().decode("utf-8"))
        batch_id = batch_info["batch_id"]
    except Exception as e:
        print(f"No batch found: {e}")
        return None

    batch = client.beta.messages.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Requests: {batch.request_counts}")

    return batch


def download_results():
    """Download and process batch results."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Get batch ID
    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_sonnet_labels/batch_info.json")
    batch_info = json.loads(resp["Body"].read().decode("utf-8"))
    batch_id = batch_info["batch_id"]

    batch = client.beta.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch not complete: {batch.processing_status}")
        return

    # Get results
    results = []
    for result in client.beta.messages.batches.results(batch_id):
        results.append(result)

    print(f"Downloaded {len(results)} results")

    # Process results
    processed = []
    success = 0
    failed = 0

    for r in results:
        problem_id = r.custom_id

        if r.result.type == "succeeded":
            content = r.result.message.content[0].text

            # Parse JSON
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()

                parsed = json.loads(json_str)
                parsed["parse_success"] = True
                parsed["problem_id"] = problem_id
                processed.append(parsed)
                success += 1
            except Exception as e:
                processed.append({
                    "problem_id": problem_id,
                    "parse_success": False,
                    "error": str(e),
                })
                failed += 1
        else:
            processed.append({
                "problem_id": problem_id,
                "parse_success": False,
                "error": str(r.result),
            })
            failed += 1

    print(f"Success: {success}, Failed: {failed}")

    # Upload results
    jsonl_content = "\n".join(json.dumps(p) for p in processed)
    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_sonnet_labels/batch_results.jsonl",
        Body=jsonl_content.encode("utf-8"),
    )

    print(f"Uploaded results to s3://{BUCKET}/c2c3_sonnet_labels/batch_results.jsonl")

    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit batch")
    parser.add_argument("--status", action="store_true", help="Check batch status")
    parser.add_argument("--download", action="store_true", help="Download results")
    parser.add_argument("--failed-only", action="store_true", help="Only process failed problems")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ERROR: No API key found")
        return

    if args.submit:
        if args.failed_only:
            problems = get_failed_problems()
        else:
            problems = get_all_problems()

        requests = create_batch_requests(problems)
        print(f"Created {len(requests)} batch requests")

        submit_batch(requests)

    elif args.status:
        check_status()

    elif args.download:
        download_results()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
