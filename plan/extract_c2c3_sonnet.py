"""
C2/C3 Training Label Extraction using Claude Sonnet

Uses Claude Sonnet to parse CoT text and extract:
- Step types (operation categories)
- Mathematical entities (operands, expressions)
- Alignment to problem text regions

This runs locally/VM (not Lambda) because we need direct Anthropic API access.
One-time cost: ~$5-10 for 5,591 problems.

Usage:
    python extract_c2c3_sonnet.py \
        --bucket mycelium-data \
        --iaf-prefix iaf_extraction/chunked/ \
        --output-prefix c2c3_sonnet_labels/ \
        --max-concurrent 10
"""

import json
import re
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import anthropic


# Load API key
API_KEY_PATH = Path(__file__).parent.parent / "secrets" / "anthropic_key.txt"
if API_KEY_PATH.exists():
    ANTHROPIC_API_KEY = API_KEY_PATH.read_text().strip()
else:
    import os
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not ANTHROPIC_API_KEY:
    raise ValueError("No Anthropic API key found!")

# Initialize clients
s3 = boto3.client("s3")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Window parameters matching C1-A
WINDOW_SIZE = 16
STRIDE = 8

# The 14 coarse C2 labels
C2_LABELS = [
    "FACTORIAL", "LOG", "TRIG", "MOD", "SQRT", "CUBE", "FRAC_POW",
    "HIGH_POW", "SQUARE", "EQUATION", "DIV", "MUL", "ADD", "OTHER"
]

# Step types for CoT parsing (expanded from C2 labels)
STEP_TYPES = [
    "simplify",      # Algebraic simplification
    "factor",        # Factoring expressions
    "expand",        # Expanding products
    "substitute",    # Substituting values
    "evaluate",      # Computing numeric results
    "apply_theorem", # Using a mathematical theorem
    "solve_equation",# Solving equations
    "modular_arithmetic", # Mod operations
    "count",         # Combinatorics/counting
    "compare",       # Comparisons
    "trigonometric", # Trig operations
    "logarithm",     # Log operations
    "power",         # Exponents
    "sqrt",          # Square roots
    "fraction",      # Fraction operations
    "other",         # Catch-all
]

# Map step types to C2 labels
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


# ---------------------------------------------------------------------------
# Sonnet CoT Parsing
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Given this math problem and its chain-of-thought solution, parse each reasoning step.

PROBLEM:
{problem_text}

CHAIN-OF-THOUGHT SOLUTION:
{cot_text}

For each reasoning step, identify:
1. step_type: One of [simplify, factor, expand, substitute, evaluate, apply_theorem, solve_equation, modular_arithmetic, count, compare, trigonometric, logarithm, power, sqrt, fraction, other]
2. step_text: The exact text of this step (brief excerpt)
3. operands: List of numeric values or variable names used in this step
4. result: The computed result of this step (if any)
5. references_step: Index of previous step this depends on (0-indexed), or null

Also identify:
- problem_operations: Which operation types appear in the PROBLEM itself (not solution)
  Choose from: FACTORIAL, LOG, TRIG, MOD, SQRT, CUBE, FRAC_POW, HIGH_POW, SQUARE, EQUATION, DIV, MUL, ADD, OTHER

Return JSON only, no explanation:
{{
  "problem_operations": ["SQRT", "ADD"],
  "steps": [
    {{
      "step_idx": 0,
      "step_type": "substitute",
      "step_text": "Let x = 5",
      "operands": ["5", "x"],
      "result": null,
      "references_step": null
    }},
    ...
  ]
}}"""


def parse_cot_with_sonnet(problem_text: str, cot_text: str) -> dict:
    """
    Use Claude Sonnet to parse CoT into structured steps.
    """
    prompt = EXTRACTION_PROMPT.format(
        problem_text=problem_text[:2000],
        cot_text=cot_text[:4000],
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response
        content = response.content[0].text

        # Try to parse JSON
        # Handle code blocks
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()

        result = json.loads(json_str)
        result["parse_success"] = True
        result["model"] = "claude-sonnet-4-20250514"

        return result

    except json.JSONDecodeError as e:
        return {
            "parse_success": False,
            "error": f"JSON parse error: {str(e)}",
            "raw_response": content[:500] if 'content' in dir() else "",
            "problem_operations": [],
            "steps": [],
        }
    except Exception as e:
        return {
            "parse_success": False,
            "error": str(e),
            "problem_operations": [],
            "steps": [],
        }


# ---------------------------------------------------------------------------
# Per-Segment Label Building
# ---------------------------------------------------------------------------

def build_per_segment_labels(
    problem_text: str,
    parsed_cot: dict,
    n_tokens: int,
    W: int = WINDOW_SIZE,
    S: int = STRIDE,
) -> list[dict]:
    """
    Build per-window segment labels from Sonnet's parsed output.
    """
    # Build window grid
    windows = []
    for w_start in range(0, n_tokens, S):
        w_end = min(w_start + W, n_tokens)
        if w_end - w_start < W // 2:
            break
        windows.append({"start": w_start, "end": w_end})

    if not windows:
        return []

    # Get problem-level C2 labels
    c2_labels = parsed_cot.get("problem_operations", [])
    if not c2_labels:
        c2_labels = ["OTHER"]

    # Extract operands from steps and map to windows
    steps = parsed_cot.get("steps", [])

    # Find operand positions in problem text
    operand_positions = []
    for step in steps:
        for op in step.get("operands", []):
            op_str = str(op)
            for match in re.finditer(re.escape(op_str), problem_text):
                operand_positions.append({
                    "value": op_str,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "step_idx": step.get("step_idx", 0),
                    "step_type": step.get("step_type", "other"),
                })

    # Estimate char-to-token ratio
    text_len = max(1, len(problem_text))
    char_to_token = n_tokens / text_len

    # Map operands to windows
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
        ops_in_window = operand_to_window.get(w_idx, [])

        if len(ops_in_window) == 0:
            c2_label = "NO_OP"
        else:
            # Map step type to C2 label
            step_type = ops_in_window[0].get("step_type", "other")
            c2_label = STEP_TO_C2.get(step_type, "OTHER")

        segment = {
            "window_idx": w_idx,
            "window_start": w["start"],
            "window_end": w["end"],
            "c2_label": c2_label,
            "c2_problem_labels": c2_labels,
            "c3_operands": ops_in_window,
            "n_operands": len(ops_in_window),
        }

        segments.append(segment)

    return segments


# ---------------------------------------------------------------------------
# Process Single Problem
# ---------------------------------------------------------------------------

def process_problem(problem: dict) -> dict:
    """
    Process one problem: parse CoT with Sonnet, build segment labels.
    """
    problem_text = problem.get("problem_text", "")
    cot_text = problem.get("generated_cot", "")
    problem_id = str(problem.get("problem_idx", "unknown"))
    n_tokens = problem.get("input_len", problem.get("num_tokens", 0))

    if not problem_text or not cot_text:
        return None

    if n_tokens == 0:
        n_tokens = len(problem_text) // 4 + 1

    # Parse CoT with Sonnet
    parsed_cot = parse_cot_with_sonnet(problem_text, cot_text)

    # Build segments
    segments = build_per_segment_labels(problem_text, parsed_cot, n_tokens)

    # Stats
    n_no_op = sum(1 for seg in segments if seg["c2_label"] == "NO_OP")
    n_with_op = len(segments) - n_no_op

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
        "stats": {
            "n_windows_with_ops": n_with_op,
            "n_windows_no_op": n_no_op,
        },
        "cot_text": cot_text[:1000],
        "parsed_steps": parsed_cot.get("steps", []),
    }


# ---------------------------------------------------------------------------
# Chunk Processing
# ---------------------------------------------------------------------------

def process_chunk(bucket: str, chunk_key: str, output_prefix: str) -> dict:
    """
    Process one IAF chunk.
    """
    # Download chunk
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    records = []
    stats = {
        "total": 0,
        "generated": 0,
        "parse_success": 0,
        "parse_failed": 0,
        "label_dist": {},
    }

    for problem in problems:
        stats["total"] += 1

        try:
            result = process_problem(problem)
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
            print(f"  Error processing problem: {e}")
            continue

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
        "chunk_key": chunk_key,
        "output_key": output_key,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_chunks(bucket: str, prefix: str) -> list[str]:
    """List IAF chunks."""
    chunks = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json") and obj["Size"] > 0:
                if "/stats/" not in obj["Key"]:
                    chunks.append(obj["Key"])
    print(f"Found {len(chunks)} chunks in s3://{bucket}/{prefix}")
    return sorted(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="mycelium-data")
    parser.add_argument("--iaf-prefix", default="iaf_extraction/chunked/")
    parser.add_argument("--output-prefix", default="c2c3_sonnet_labels/")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to process")
    parser.add_argument("--test", action="store_true", help="Test with one problem")
    args = parser.parse_args()

    if args.test:
        # Test with one problem
        print("Testing Sonnet extraction...")
        test_problem = {
            "problem_text": r"Find the value of $\sqrt{16} + 3^2$.",
            "generated_cot": r"We need to evaluate $\sqrt{16} + 3^2$. First, $\sqrt{16} = 4$. Then, $3^2 = 9$. Finally, $4 + 9 = 13$.",
            "problem_idx": 0,
            "input_len": 20,
        }
        result = process_problem(test_problem)
        print(json.dumps(result, indent=2))
        return

    chunks = list_chunks(args.bucket, args.iaf_prefix)
    if args.limit:
        chunks = chunks[:args.limit]

    print(f"\nProcessing {len(chunks)} chunks with {args.max_concurrent} workers")
    print("This uses Sonnet API - estimated cost: ~$0.01 per problem")

    results = []
    failed = []
    start = time.time()

    # Sequential processing to respect rate limits
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] Processing {chunk.split('/')[-1]}...")
        try:
            result = process_chunk(args.bucket, chunk, args.output_prefix)
            results.append(result)
            s = result["stats"]
            print(f"    gen: {s['generated']} | success: {s['parse_success']} | failed: {s['parse_failed']}")
        except Exception as e:
            failed.append((chunk, str(e)))
            print(f"    FAILED: {e}")

        # Rate limiting
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"\nComplete: {elapsed:.1f}s, {len(results)} ok, {len(failed)} failed")

    # Summary stats
    total_gen = sum(r["stats"]["generated"] for r in results)
    total_success = sum(r["stats"]["parse_success"] for r in results)
    print(f"Total problems: {total_gen}, Parse success: {total_success} ({100*total_success/max(1,total_gen):.1f}%)")


if __name__ == "__main__":
    main()
