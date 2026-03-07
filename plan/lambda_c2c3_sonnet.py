"""
Lambda function: C2/C3 Extraction using Claude Sonnet

Uses Claude Sonnet API to parse CoT text and extract operation labels.
API key passed via environment variable or event payload.

Memory: 3GB
Timeout: 15 minutes (900s)
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

# Try to import anthropic - will fail if not in Lambda layer
try:
    import anthropic
except ImportError:
    anthropic = None

s3 = boto3.client("s3")

BUCKET = "mycelium-data"
WINDOW_SIZE = 16
STRIDE = 8

# Step types for CoT parsing
STEP_TO_C2 = {
    "simplify": "ADD",
    "factor": "MUL",
    "expand": "MUL",
    "substitute": "EQUATION",
    "evaluate": "ADD",
    "apply_theorem": "OTHER",
    "solve_equation": "EQUATION",
    "modular_arithmetic": "MOD",
    "count": "FACTORIAL",
    "compare": "OTHER",
    "trigonometric": "TRIG",
    "logarithm": "LOG",
    "power": "HIGH_POW",
    "sqrt": "SQRT",
    "fraction": "DIV",
    "other": "OTHER",
}

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


def parse_cot_with_sonnet(problem_text: str, cot_text: str, api_key: str, max_retries: int = 3) -> dict:
    """Parse CoT with Claude Sonnet with retry logic."""
    import urllib.request
    import ssl
    import time
    import random

    prompt = EXTRACTION_PROMPT.format(
        problem_text=problem_text[:2000],
        cot_text=cot_text[:4000],
    )

    for attempt in range(max_retries):
        try:
            data = json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": api_key,
                    "Anthropic-Version": "2023-06-01",
                },
            )

            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["content"][0]["text"]

            # Parse JSON response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            result = json.loads(json_str)
            result["parse_success"] = True
            return result

        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                # Rate limited - exponential backoff with jitter
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + random.uniform(0, 0.5))
                continue
            raise

    raise Exception("Max retries exceeded")


def build_segments(problem_text: str, parsed_cot: dict, n_tokens: int) -> list:
    """Build per-window segments from parsed CoT."""
    windows = []
    for w_start in range(0, n_tokens, STRIDE):
        w_end = min(w_start + WINDOW_SIZE, n_tokens)
        if w_end - w_start < WINDOW_SIZE // 2:
            break
        windows.append({"start": w_start, "end": w_end})

    if not windows:
        return []

    c2_labels = parsed_cot.get("problem_operations", ["OTHER"])

    # Find operand positions
    steps = parsed_cot.get("steps", [])
    operand_positions = []
    for step in steps:
        for op in step.get("operands", []):
            op_str = str(op)
            for match in re.finditer(re.escape(op_str), problem_text):
                operand_positions.append({
                    "value": op_str,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "step_type": step.get("step_type", "other"),
                })

    # Map to windows
    char_to_token = n_tokens / max(1, len(problem_text))
    operand_to_window = {}

    for op in operand_positions:
        start_tok = int(op["start_char"] * char_to_token)
        end_tok = int(op["end_char"] * char_to_token) + 1

        best_window = None
        best_overlap = 0
        for w_idx, w in enumerate(windows):
            overlap = max(0, min(w["end"], end_tok) - max(w["start"], start_tok))
            if overlap > best_overlap:
                best_overlap = overlap
                best_window = w_idx

        if best_window is not None:
            if best_window not in operand_to_window:
                operand_to_window[best_window] = []
            operand_to_window[best_window].append({
                "value": op["value"],
                "start": start_tok - windows[best_window]["start"],
                "end": end_tok - windows[best_window]["start"],
                "step_type": op["step_type"],
            })

    # Build segments
    segments = []
    for w_idx, w in enumerate(windows):
        ops = operand_to_window.get(w_idx, [])
        c2_label = "NO_OP" if not ops else STEP_TO_C2.get(ops[0].get("step_type", "other"), "OTHER")

        segments.append({
            "window_idx": w_idx,
            "window_start": w["start"],
            "window_end": w["end"],
            "c2_label": c2_label,
            "c2_problem_labels": c2_labels,
            "c3_operands": ops,
            "n_operands": len(ops),
        })

    return segments


def process_problem(problem: dict, api_key: str) -> dict:
    """Process one problem."""
    problem_text = problem.get("problem_text", "")
    cot_text = problem.get("generated_cot", "")
    problem_id = str(problem.get("problem_idx", "unknown"))
    n_tokens = problem.get("input_len", problem.get("num_tokens", 0))

    if not problem_text or not cot_text:
        return None

    if n_tokens == 0:
        n_tokens = len(problem_text) // 4 + 1

    try:
        parsed_cot = parse_cot_with_sonnet(problem_text, cot_text, api_key)
    except Exception as e:
        return {
            "problem_id": problem_id,
            "parse_success": False,
            "error": str(e),
        }

    segments = build_segments(problem_text, parsed_cot, n_tokens)
    n_no_op = sum(1 for seg in segments if seg["c2_label"] == "NO_OP")

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "n_input_tokens": n_tokens,
        "level": problem.get("level", "unknown"),
        "type": problem.get("type", "unknown"),
        "c2_labels": parsed_cot.get("problem_operations", []),
        "n_cot_steps": len(parsed_cot.get("steps", [])),
        "parse_success": parsed_cot.get("parse_success", False),
        "n_windows": len(segments),
        "segments": segments,
        "stats": {"n_windows_with_ops": len(segments) - n_no_op, "n_windows_no_op": n_no_op},
        "cot_text": cot_text[:1000],
        "parsed_steps": parsed_cot.get("steps", []),
    }


def lambda_handler(event, context):
    """
    Process one IAF chunk with Sonnet.

    Event:
        chunk_key: S3 key for IAF chunk
        output_prefix: S3 prefix for output
        api_key: Anthropic API key (or from env)
        bucket: S3 bucket (default: mycelium-data)
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c2c3_sonnet_labels/")
    api_key = event.get("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))

    if not api_key:
        return {"statusCode": 500, "error": "No API key provided"}

    # Download chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    # Process problems in parallel (2 concurrent to respect rate limits)
    records = []
    stats = {"total": 0, "generated": 0, "parse_success": 0, "parse_failed": 0, "label_dist": {}}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_problem, p, api_key): i for i, p in enumerate(problems)}

        for future in as_completed(futures):
            stats["total"] += 1
            try:
                result = future.result()
                if result is None:
                    continue
                records.append(result)
                stats["generated"] += 1

                if result.get("parse_success"):
                    stats["parse_success"] += 1
                else:
                    stats["parse_failed"] += 1

                for lbl in result.get("c2_labels", []):
                    stats["label_dist"][lbl] = stats["label_dist"].get(lbl, 0) + 1

            except Exception as e:
                stats["parse_failed"] += 1
                print(f"Error: {e}")

    # Upload results
    chunk_name = chunk_key.split("/")[-1].replace(".json", ".jsonl")
    output_key = f"{output_prefix}{chunk_name}"

    jsonl_content = "\n".join(json.dumps(r, default=str) for r in records)

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    # Stats
    stats_key = f"{output_prefix}stats/{chunk_name.replace('.jsonl', '.json')}"
    s3.put_object(
        Bucket=bucket,
        Key=stats_key,
        Body=json.dumps(stats, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "chunk_key": chunk_key,
        "output_key": output_key,
        "stats": stats,
    }
