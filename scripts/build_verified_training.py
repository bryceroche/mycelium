#!/usr/bin/env python3
"""
Build Verified Training Data for Canonicalizer

Extracts training examples where the target telegram is EXECUTION-VERIFIED
against the teacher's intermediate results.

Input: parsed_steps.jsonl (Sonnet-extracted CoT with operands/results)
Output: verified_training.jsonl (only steps that execute correctly)

Key insight: Every training target must produce the correct intermediate result
when executed by SymPy. No format conversion - actual problem solving.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.oracle import execute_telegram, parse_telegram_expr, compare_answers

# Step type → verb mapping
STEP_TYPE_TO_VERB = {
    "setup": "GIVEN",
    "evaluate": "EVAL",
    "solve_equation": "SOLVE",
    "simplify": "SIMPLIFY",
    "substitute": "SUBS",
    "apply_theorem": "APPLY",
    "expand": "EXPAND",
    "factor": "SIMPLIFY",
    "compute": "EVAL",
    "other": "EVAL",
}

# Step type → scaffold mapping
STEP_TYPE_TO_SCAFFOLD = {
    "setup": "SETUP",
    "evaluate": "COMPUTE",
    "solve_equation": "SOLVE",
    "simplify": "SIMPLIFY",
    "substitute": "SUBSTITUTE",
    "apply_theorem": "THEOREM",
    "expand": "EXPAND",
    "factor": "SIMPLIFY",
    "compute": "COMPUTE",
    "other": "OTHER",
}


@dataclass
class VerifiedExample:
    """A single verified training example."""
    problem_id: str
    step_idx: int
    scaffold: str
    telegram: str
    expected_result: str
    actual_result: str
    raw_cot: str


def parse_operands(operands_str) -> List[str]:
    """Parse operands from string or list representation."""
    # Already a list
    if isinstance(operands_str, list):
        return operands_str

    if not operands_str or not isinstance(operands_str, str):
        return []

    try:
        # Try literal eval first
        result = ast.literal_eval(operands_str)
        if isinstance(result, list):
            return result
        return [str(result)]
    except:
        # Fallback: split by comma, handling brackets
        cleaned = operands_str.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if not cleaned:
            return []
        # Simple split - won't handle nested structures perfectly
        return [s.strip().strip("'\"") for s in cleaned.split(",") if s.strip()]


def normalize_expr(expr: str) -> str:
    """Normalize expression for SymPy parsing."""
    return (expr
        .replace("\\(", "").replace("\\)", "")
        .replace("\\[", "").replace("\\]", "")
        .replace("\\frac", "frac")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("^", "**")
        .strip())


def build_telegram(step_type: str, operands: List[str], result: str) -> Optional[str]:
    """Build a telegram from step components."""
    verb = STEP_TYPE_TO_VERB.get(step_type, "EVAL")

    if not operands:
        return None

    # For SETUP/GIVEN, use the first operand or result
    if verb == "GIVEN":
        expr = normalize_expr(operands[0]) if operands else normalize_expr(result)
        return f"GIVEN {expr}"

    # For SOLVE, format as equation
    if verb == "SOLVE":
        if len(operands) >= 2:
            lhs = normalize_expr(operands[0])
            rhs = normalize_expr(operands[1])
            return f"SOLVE {lhs} = {rhs}"
        elif operands:
            return f"SOLVE {normalize_expr(operands[0])}"

    # For SUBS, need old/new values
    if verb == "SUBS" and len(operands) >= 2:
        expr = normalize_expr(operands[0])
        # This is tricky - substitution needs context
        return f"EVAL {expr}"  # Fallback to EVAL

    # For most operations, use the first operand
    expr = normalize_expr(operands[0])
    return f"{verb} {expr}"


def verify_telegram(telegram: str, expected_result: str, previous_results: List) -> tuple:
    """
    Execute telegram and verify against expected result.

    Returns (success: bool, actual_result: str)
    """
    try:
        exec_result = execute_telegram(telegram, previous_results, timeout=5)

        if not exec_result["success"]:
            return False, exec_result.get("error", "execution failed")

        actual = exec_result["result"]
        if actual is None:
            return False, "null result"

        # Try to compare with expected
        try:
            # Parse expected result
            expected_expr = parse_telegram_expr(normalize_expr(expected_result))
            if expected_expr is None:
                # Can't parse expected - check string match
                return str(actual).strip() == expected_result.strip(), str(actual)

            # Compare symbolically
            if compare_answers(actual, normalize_expr(expected_result), timeout=3):
                return True, str(actual)
        except:
            pass

        # Fallback: string comparison
        actual_str = str(actual).strip()
        expected_str = expected_result.strip()
        return actual_str == expected_str, actual_str

    except Exception as e:
        return False, str(e)


def process_problem(steps: List[Dict]) -> List[VerifiedExample]:
    """Process all steps for a single problem, returning verified examples."""
    verified = []
    previous_results = []

    for step in steps:
        step_idx = int(step.get("step_idx", 0))
        step_type = step.get("step_type", "other")
        operands_str = step.get("operands", "[]")
        result = step.get("result", "")
        raw_cot = step.get("raw_cot_text", "")
        problem_id = step.get("problem_id", "unknown")

        # Skip steps with prose results (not mathematical)
        if any(word in result.lower() for word in
               ["equation", "setup", "common", "expand", "simplif", "substitut"]):
            continue

        # Parse operands
        operands = parse_operands(operands_str)

        # Build telegram
        telegram = build_telegram(step_type, operands, result)
        if not telegram:
            continue

        # Verify execution
        success, actual = verify_telegram(telegram, result, previous_results)

        if success:
            scaffold = STEP_TYPE_TO_SCAFFOLD.get(step_type, "OTHER")
            verified.append(VerifiedExample(
                problem_id=problem_id,
                step_idx=step_idx,
                scaffold=scaffold,
                telegram=telegram,
                expected_result=result,
                actual_result=actual,
                raw_cot=raw_cot[:200]
            ))

            # Add to previous results for chaining
            try:
                prev_result = parse_telegram_expr(actual)
                previous_results.append(prev_result)
            except:
                previous_results.append(actual)

    return verified


def format_training_example(example: VerifiedExample, problem_text: str = "") -> Dict:
    """Format a verified example for training."""
    # Match the canonicalizer training format
    prompt = f"Segment: {example.raw_cot}\nType: {example.scaffold}\nWrite the SymPy expression:"

    # Extract just the expression from the telegram
    parts = example.telegram.split(None, 1)
    target = parts[1] if len(parts) > 1 else example.telegram

    return {
        "input": prompt,
        "target": target,
        "problem_id": example.problem_id,
        "step_idx": example.step_idx,
        "scaffold": example.scaffold,
        "verified": True,
        "expected_result": example.expected_result,
        "actual_result": example.actual_result,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://mycelium-data/c2c3_training_data_v2/parsed_steps.jsonl")
    parser.add_argument("--output", default="verified_training.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of steps to process")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Building Verified Training Data ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()

    # Load steps
    print("Loading parsed steps...")
    if args.input.startswith("s3://"):
        result = subprocess.run(
            ["aws", "s3", "cp", args.input, "-"],
            capture_output=True
        )
        lines = result.stdout.decode().strip().split("\n")
    else:
        with open(args.input) as f:
            lines = f.readlines()

    if args.limit:
        lines = lines[:args.limit]

    print(f"Loaded {len(lines)} steps")

    # Group by problem
    problems = {}
    for line in lines:
        if not line.strip():
            continue
        step = json.loads(line)
        pid = step.get("problem_id", "unknown")
        if pid not in problems:
            problems[pid] = []
        problems[pid].append(step)

    print(f"Found {len(problems)} problems")
    print()

    # Process each problem
    print("Verifying steps...")
    all_verified = []
    stats = {"total": 0, "verified": 0, "by_scaffold": {}}

    for i, (pid, steps) in enumerate(problems.items()):
        # Sort by step_idx
        steps = sorted(steps, key=lambda x: int(x.get("step_idx", 0)))

        verified = process_problem(steps)
        all_verified.extend(verified)

        stats["total"] += len(steps)
        stats["verified"] += len(verified)

        for ex in verified:
            stats["by_scaffold"][ex.scaffold] = stats["by_scaffold"].get(ex.scaffold, 0) + 1

        if (i + 1) % 100 == 0:
            rate = stats["verified"] / max(stats["total"], 1)
            print(f"  {i+1}/{len(problems)} problems, {stats['verified']}/{stats['total']} verified ({rate:.1%})")

    print()
    print(f"Total verified: {stats['verified']}/{stats['total']} ({stats['verified']/max(stats['total'],1):.1%})")
    print("By scaffold:")
    for scaffold, count in sorted(stats["by_scaffold"].items(), key=lambda x: -x[1]):
        print(f"  {scaffold}: {count}")

    # Format and save
    print()
    print(f"Saving to {args.output}...")

    with open(args.output, "w") as f:
        for ex in all_verified:
            example = format_training_example(ex)
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(all_verified)} verified examples")

    # Upload to S3
    s3_path = "s3://mycelium-data/verified_training/verified_training.jsonl"
    subprocess.run(["aws", "s3", "cp", args.output, s3_path], capture_output=True)
    print(f"Uploaded to {s3_path}")

    # Show sample
    if args.verbose and all_verified:
        print()
        print("=== Sample Verified Examples ===")
        for ex in all_verified[:5]:
            print(f"[{ex.scaffold}] {ex.telegram}")
            print(f"  Expected: {ex.expected_result}")
            print(f"  Actual: {ex.actual_result}")
            print()


if __name__ == "__main__":
    main()
