"""
Anthropic Batch API: 7-Dimensional Structural Labels for IB Annealing

Extracts per-step structural properties:
1. step_type (12 categories)
2. complexity_change (reduces/neutral/increases)
3. n_operands (integer)
4. has_dependency (boolean)
5. output_type (5 categories)
6. step_position (early/middle/late)
7. reference_distance (none/local/medium/distant)

These become Y vectors for IB annealing: X=teacher attention, Y=structure → clusters → C2 labels

Usage:
    python batch_c2c3_v2.py --submit
    python batch_c2c3_v2.py --status
    python batch_c2c3_v2.py --download
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import boto3
from botocore.config import Config
import anthropic

# S3 with increased timeout
s3_config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=s3_config)
BUCKET = "mycelium-data"

# Load API key
API_KEY_PATH = Path(__file__).parent.parent / "secrets" / "anthropic_key.txt"
ANTHROPIC_API_KEY = API_KEY_PATH.read_text().strip() if API_KEY_PATH.exists() else os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """You are a mathematical reasoning analyzer. You will receive a math problem and its chain-of-thought solution. Parse each reasoning step and extract structural properties.

For each step, extract ALL of the following:

1. step_type (exactly one):
   - evaluate: compute a numerical result from an expression
   - simplify: reduce an expression to simpler form
   - substitute: replace a variable or expression with another
   - solve_equation: isolate a variable or find a solution
   - factor: decompose into factors
   - expand: distribute or multiply out expressions
   - apply_theorem: invoke a named theorem, formula, or identity
   - count: enumerate or compute combinations/permutations
   - compare: compare values, test conditions, check divisibility
   - convert: change representation (decimal↔fraction, degrees↔radians, etc.)
   - setup: define variables, establish equations from problem text
   - other: none of the above

2. complexity_change (exactly one):
   - reduces: output is simpler than input (fewer terms, resolved variable)
   - neutral: roughly same complexity (substitution, conversion)
   - increases: output is more complex (expanding, introducing cases)

3. n_operands: integer, how many distinct mathematical entities are inputs to this step (numbers, variables, expressions). Count each unique entity once.

4. has_dependency: boolean, does this step use a result computed in a previous step? true if it references an intermediate value, false if it only uses values from the original problem.

5. output_type (exactly one):
   - number: produces a concrete numerical value
   - expression: produces a symbolic expression
   - equation: produces an equation or inequality
   - boolean: produces true/false or yes/no
   - set: produces a set or list of values

6. step_position (exactly one, based on where this step falls in the solution):
   - early: first 25% of steps (setup, initial reading)
   - middle: middle 50% of steps (core computation)
   - late: last 25% of steps (final evaluation, answer extraction)

7. reference_distance (exactly one, how far back does this step reference?):
   - none: does not reference any previous step (uses only original problem values)
   - local: references the immediately previous step (step N-1)
   - medium: references 2-3 steps back
   - distant: references 4+ steps back or combines results from multiple earlier steps

Also extract:
- operands: list of the mathematical entities used as input (as strings)
- result: what this step produces (as string)
- text_reference: quote the specific phrase from the original problem text that this step addresses. Use EXACT words. Set to null if purely computational.

Return ONLY valid JSON, no markdown, no explanation:
{
    "steps": [
        {
            "step_idx": 0,
            "raw_cot_text": "the exact text of this CoT step",
            "step_type": "...",
            "complexity_change": "...",
            "n_operands": int,
            "has_dependency": bool,
            "output_type": "...",
            "step_position": "...",
            "reference_distance": "...",
            "operands": ["...", "..."],
            "result": "...",
            "text_reference": "..." or null
        }
    ]
}"""


def get_all_problems():
    """Get all problems from IAF chunks - streaming with minimal memory."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    problems = []
    paginator = s3.get_paginator("list_objects_v2")

    print("Loading IAF chunks...")

    # First, collect all chunk keys
    chunk_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="iaf_extraction/chunked/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and "/stats/" not in obj["Key"]:
                chunk_keys.append(obj["Key"])

    print(f"Found {len(chunk_keys)} chunks to load")

    def load_chunk(chunk_key):
        """Load one chunk with retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=chunk_key)
                chunk_data = json.loads(resp["Body"].read().decode("utf-8"))
                chunk_problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

                # Extract only needed fields
                results = []
                for p in chunk_problems:
                    results.append({
                        "problem_id": f"{chunk_key.split('/')[-1].replace('.json', '')}_{p.get('problem_idx', len(results))}",
                        "problem_text": p.get("problem_text", ""),
                        "cot_text": p.get("generated_cot", ""),
                    })
                return results

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  Failed: {chunk_key}: {e}")
                    return []
        return []

    # Load chunks in parallel (8 concurrent)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_chunk, key): key for key in chunk_keys}
        done = 0

        for future in as_completed(futures):
            results = future.result()
            problems.extend(results)
            done += 1
            if done % 20 == 0:
                print(f"  Loaded {done}/{len(chunk_keys)} chunks, {len(problems)} problems...")

    print(f"Loaded {len(problems)} total problems from {len(chunk_keys)} chunks")
    return problems


def create_batch_requests(problems):
    """Create batch requests with 6-dim extraction prompt."""
    requests = []

    for p in problems:
        if not p.get("problem_text") or not p.get("cot_text"):
            continue

        user_content = f"Problem: {p['problem_text'][:3000]}\n\nSolution (Chain of Thought):\n{p['cot_text'][:5000]}"

        requests.append({
            "custom_id": p["problem_id"],
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}]
            }
        })

    return requests


def submit_batch(requests):
    """Submit batch to Anthropic API."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"Submitting batch with {len(requests)} requests...")

    batch = client.beta.messages.batches.create(requests=requests)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save batch info
    batch_info = {
        "batch_id": batch.id,
        "created_at": datetime.now().isoformat(),
        "n_requests": len(requests),
        "status": batch.processing_status,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_training_data_v2/batch_info.json",
        Body=json.dumps(batch_info, indent=2).encode("utf-8"),
    )

    return batch


def check_status():
    """Check batch status."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_data_v2/batch_info.json")
        batch_info = json.loads(resp["Body"].read().decode("utf-8"))
        batch_id = batch_info["batch_id"]
    except Exception as e:
        print(f"No batch found: {e}")
        return None

    batch = client.beta.messages.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Request counts: {batch.request_counts}")

    return batch


def validate_step_position(step_idx, total_steps):
    """Deterministic step_position calculation - don't trust LLM on simple arithmetic."""
    if total_steps == 0:
        return "middle"
    if step_idx < total_steps * 0.25:
        return "early"
    elif step_idx < total_steps * 0.75:
        return "middle"
    else:
        return "late"


def validate_reference_distance(has_dependency, ref_distance):
    """Ensure reference_distance is consistent with has_dependency."""
    if not has_dependency:
        return "none"
    if ref_distance not in ["none", "local", "medium", "distant"]:
        return "local"
    return ref_distance


def download_results():
    """Download and process batch results."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_data_v2/batch_info.json")
    batch_info = json.loads(resp["Body"].read().decode("utf-8"))
    batch_id = batch_info["batch_id"]

    batch = client.beta.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch not complete: {batch.processing_status}")
        return

    # Download results
    print("Downloading results...")
    results = list(client.beta.messages.batches.results(batch_id))
    print(f"Downloaded {len(results)} results")

    # Save raw responses
    raw_responses = []
    parsed_steps = []
    stats = {
        "total": len(results),
        "success": 0,
        "failed": 0,
        "step_type": Counter(),
        "complexity_change": Counter(),
        "output_type": Counter(),
        "step_position": Counter(),
        "reference_distance": Counter(),
        "has_dependency": Counter(),
        "n_operands": Counter(),
    }

    for r in results:
        problem_id = r.custom_id

        raw_responses.append({
            "problem_id": problem_id,
            "result_type": r.result.type,
            "response": r.result.message.content[0].text if r.result.type == "succeeded" else str(r.result),
        })

        if r.result.type == "succeeded":
            content = r.result.message.content[0].text

            try:
                # Parse JSON
                if content.strip().startswith("{"):
                    parsed = json.loads(content)
                else:
                    # Try to extract JSON
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        raise ValueError("No JSON found")

                steps = parsed.get("steps", [])
                total_steps = len(steps)

                # Validate and collect stats
                for step in steps:
                    step["problem_id"] = problem_id

                    # Validate step_type (dim 0)
                    step_type = step.get("step_type", "other")
                    if step_type not in ["evaluate", "simplify", "substitute", "solve_equation", "factor", "expand", "apply_theorem", "count", "compare", "convert", "setup", "other"]:
                        step_type = "other"
                    step["step_type"] = step_type
                    stats["step_type"][step_type] += 1

                    # Validate complexity_change (dim 1)
                    complexity = step.get("complexity_change", "neutral")
                    if complexity not in ["reduces", "neutral", "increases"]:
                        complexity = "neutral"
                    step["complexity_change"] = complexity
                    stats["complexity_change"][complexity] += 1

                    # Validate n_operands (dim 2)
                    n_ops = step.get("n_operands", 1)
                    n_ops = max(1, min(4, int(n_ops))) if isinstance(n_ops, (int, float)) else 1
                    step["n_operands"] = n_ops
                    stats["n_operands"][str(n_ops)] += 1

                    # Validate has_dependency (dim 3)
                    has_dep = bool(step.get("has_dependency", False))
                    step["has_dependency"] = has_dep
                    stats["has_dependency"][str(has_dep)] += 1

                    # Validate output_type (dim 4)
                    output = step.get("output_type", "expression")
                    if output not in ["number", "expression", "equation", "boolean", "set"]:
                        output = "expression"
                    step["output_type"] = output
                    stats["output_type"][output] += 1

                    # Compute step_position deterministically (dim 5)
                    step_idx = step.get("step_idx", 0)
                    step_pos = validate_step_position(step_idx, total_steps)
                    step["step_position"] = step_pos
                    stats["step_position"][step_pos] += 1

                    # Validate reference_distance with consistency check (dim 6)
                    ref_dist = step.get("reference_distance", "none")
                    ref_dist = validate_reference_distance(has_dep, ref_dist)
                    step["reference_distance"] = ref_dist
                    stats["reference_distance"][ref_dist] += 1

                    parsed_steps.append(step)

                stats["success"] += 1

            except Exception as e:
                stats["failed"] += 1

        else:
            stats["failed"] += 1

    print(f"Parse success: {stats['success']}/{stats['total']}")

    # Save results
    print("Uploading results to S3...")

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_training_data_v2/raw_sonnet_responses.jsonl",
        Body="\n".join(json.dumps(r) for r in raw_responses).encode("utf-8"),
    )

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_training_data_v2/parsed_steps.jsonl",
        Body="\n".join(json.dumps(s) for s in parsed_steps).encode("utf-8"),
    )

    # Convert Counter to dict for JSON
    stats_dict = {
        "total": stats["total"],
        "success": stats["success"],
        "failed": stats["failed"],
        "n_steps": len(parsed_steps),
        "step_type": dict(stats["step_type"]),
        "complexity_change": dict(stats["complexity_change"]),
        "n_operands": dict(stats["n_operands"]),
        "has_dependency": dict(stats["has_dependency"]),
        "output_type": dict(stats["output_type"]),
        "step_position": dict(stats["step_position"]),
        "reference_distance": dict(stats["reference_distance"]),
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="c2c3_training_data_v2/stats.json",
        Body=json.dumps(stats_dict, indent=2).encode("utf-8"),
    )

    print(f"\nResults saved to s3://{BUCKET}/c2c3_training_data_v2/")
    print(f"\nStats:")
    print(f"  Total problems: {stats['total']}")
    print(f"  Parse success: {stats['success']}")
    print(f"  Total steps: {len(parsed_steps)}")
    print(f"\n  Step types: {dict(stats['step_type'].most_common(5))}")
    print(f"  Step positions: {dict(stats['step_position'])}")
    print(f"  Reference distances: {dict(stats['reference_distance'])}")

    return parsed_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ERROR: No API key")
        return

    if args.submit:
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
