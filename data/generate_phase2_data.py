#!/usr/bin/env python3
"""
Phase 2 Data Generation: Extract training data from teacher CoT traces

Takes: MATH dataset problems with teacher chain-of-thought solutions
Outputs: (span_groups, gold_ops, gold_adjacency, gold_answer) tuples

Process:
  1. Parse CoT into computation steps
  2. For each step, identify operation and extract operands
  3. Build adjacency matrix from step dependencies
  4. Verify with SymPy that execution produces correct answer

The gold_adjacency comes from CoT step ordering — if step 2 uses the
result of step 1, there's an edge 1→2. This is implicit in teacher CoT,
we just extract it as an explicit adjacency matrix.

Usage:
    python data/generate_phase2_data.py \\
        --input s3://mycelium-data/math/math_train.json \\
        --output s3://mycelium-data/phase2/phase2_train.json \\
        --verify
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.c5_sympy_executor import C5_SymPyExecutor, OP_LABELS, N_OPS
import torch

# ---------------------------------------------------------------------------
# Operation pattern matchers
# ---------------------------------------------------------------------------

# Patterns to identify operations in CoT text
# These are learned patterns from IB template discovery, not hand-coded heuristics
OP_PATTERNS = {
    "ADD": [
        r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)",
        r"sum of",
        r"total of",
        r"combined",
        r"altogether",
        r"plus",
    ],
    "SUB": [
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
        r"difference",
        r"subtract",
        r"minus",
        r"less than",
        r"remaining",
    ],
    "MUL": [
        r"(\d+(?:\.\d+)?)\s*[×*]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*\\times\s*(\d+(?:\.\d+)?)",
        r"product of",
        r"times",
        r"multiply",
    ],
    "DIV": [
        r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*÷\s*(\d+(?:\.\d+)?)",
        r"divided by",
        r"quotient",
        r"ratio of",
    ],
    "POW": [
        r"(\d+(?:\.\d+)?)\s*\^\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*\*\*\s*(\d+(?:\.\d+)?)",
        r"squared",
        r"cubed",
        r"to the power",
    ],
    "SQRT": [
        r"\\sqrt\{",
        r"sqrt\(",
        r"square root",
        r"√",
    ],
    "MOD": [
        r"mod\s+\d+",
        r"remainder",
        r"modulo",
    ],
    "PERCENT_OF": [
        r"(\d+(?:\.\d+)?)\s*%\s*of",
        r"percent of",
        r"percentage of",
    ],
    "PERCENT_CHANGE": [
        r"percent change",
        r"percent increase",
        r"percent decrease",
        r"% change",
    ],
    "GCD": [
        r"gcd\(",
        r"greatest common divisor",
        r"highest common factor",
    ],
    "LCM": [
        r"lcm\(",
        r"least common multiple",
    ],
    "COMB": [
        r"\\binom\{",
        r"C\(\d+,\s*\d+\)",
        r"choose",
        r"combination",
    ],
    "SOLVE_LINEAR": [
        r"solve for",
        r"=\s*\d+",
        r"equation",
    ],
}


# ---------------------------------------------------------------------------
# Step parser
# ---------------------------------------------------------------------------

@dataclass
class ComputationStep:
    """A single computation step extracted from CoT."""
    text: str                       # raw text of this step
    operation: str                  # detected operation label
    operands: List[float]           # extracted operand values
    result: Optional[float]         # computed result (if available)
    references: List[int]           # indices of steps this depends on
    confidence: float = 1.0         # detection confidence


def parse_cot_into_steps(cot_text: str) -> List[str]:
    """
    Split CoT text into individual computation steps.

    Strategies:
      1. Split by newlines
      2. Split by sentence boundaries
      3. Split by "Step N:" markers
      4. Split by "Therefore", "So", "Thus" markers
    """
    # Try to find explicit step markers
    step_pattern = r"(?:Step\s*\d+[:.]\s*|(?:Therefore|So|Thus|Hence)[,:]?\s*)"

    # Split by step markers or newlines
    parts = re.split(step_pattern, cot_text, flags=re.IGNORECASE)

    # Filter empty parts and strip
    steps = [s.strip() for s in parts if s and s.strip()]

    # If no steps found, split by sentences
    if len(steps) <= 1:
        sentences = re.split(r'[.!?]+', cot_text)
        steps = [s.strip() for s in sentences if s and s.strip() and any(c.isdigit() for c in s)]

    return steps


def detect_operation(step_text: str) -> Tuple[str, float]:
    """
    Detect which operation is being performed in this step.

    Returns: (operation_label, confidence)
    """
    step_lower = step_text.lower()

    scores = defaultdict(float)

    for op, patterns in OP_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, step_text, re.IGNORECASE):
                scores[op] += 1.0
            elif pattern in step_lower:
                scores[op] += 0.5

    if not scores:
        return "ADD", 0.1  # default fallback

    best_op = max(scores.keys(), key=lambda k: scores[k])
    confidence = min(1.0, scores[best_op] / 3.0)

    return best_op, confidence


def extract_operands(step_text: str) -> List[float]:
    """
    Extract numeric operands from step text.

    Looks for:
      - Integers: 42, 1000
      - Decimals: 3.14, 0.5
      - Fractions: 1/2, 3/4 (converted to decimal)
      - Scientific notation: 1e6, 2.5e-3
    """
    operands = []

    # Find all numbers
    number_pattern = r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?'
    matches = re.findall(number_pattern, step_text, re.IGNORECASE)

    for m in matches:
        try:
            operands.append(float(m))
        except ValueError:
            pass

    # Also look for fractions
    frac_pattern = r'(\d+)\s*/\s*(\d+)'
    frac_matches = re.findall(frac_pattern, step_text)
    for num, denom in frac_matches:
        if int(denom) != 0:
            operands.append(int(num) / int(denom))

    return operands


def find_step_references(step_text: str, previous_results: List[float]) -> List[int]:
    """
    Find which previous steps this step references.

    A step references a previous step if:
      - It mentions that step's result value
      - It says "from step N" or "using the result above"
    """
    references = []
    step_lower = step_text.lower()

    # Check for explicit references
    explicit_refs = re.findall(r'(?:step|result)\s*(\d+)', step_lower)
    for ref in explicit_refs:
        idx = int(ref) - 1  # convert to 0-indexed
        if 0 <= idx < len(previous_results):
            references.append(idx)

    # Check for value references
    for i, result in enumerate(previous_results):
        if result is not None:
            # Check if this result value appears in the step text
            result_str = str(int(result) if result == int(result) else result)
            if result_str in step_text:
                references.append(i)

    return list(set(references))  # dedupe


def parse_cot_to_steps(cot_text: str) -> List[ComputationStep]:
    """
    Full CoT parsing: text → list of ComputationStep.
    """
    raw_steps = parse_cot_into_steps(cot_text)
    steps = []
    results = []

    for i, step_text in enumerate(raw_steps):
        op, conf = detect_operation(step_text)
        operands = extract_operands(step_text)
        refs = find_step_references(step_text, results)

        # Try to compute result (for reference detection in later steps)
        result = None
        if operands:
            # Simple heuristic: result is often the last number in the step
            nums_in_step = re.findall(r'-?\d+(?:\.\d+)?', step_text)
            if nums_in_step:
                try:
                    result = float(nums_in_step[-1])
                except ValueError:
                    pass

        steps.append(ComputationStep(
            text=step_text,
            operation=op,
            operands=operands,
            result=result,
            references=refs,
            confidence=conf,
        ))
        results.append(result)

    return steps


def steps_to_adjacency(steps: List[ComputationStep]) -> List[List[int]]:
    """
    Convert step references to adjacency matrix.

    adjacency[i][j] = 1 means step i's result feeds into step j.
    """
    n = len(steps)
    adj = [[0] * n for _ in range(n)]

    for j, step in enumerate(steps):
        for i in step.references:
            if 0 <= i < n and i < j:  # enforce topological order
                adj[i][j] = 1

    return adj


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class Phase2Example:
    """Training example for Phase 2."""
    problem_text: str
    span_groups: List[str]          # text per step
    gold_ops: List[str]             # operation per step
    gold_adjacency: List[List[int]] # adjacency matrix
    gold_answer: float
    cot_text: str                   # original CoT for reference
    verified: bool = False          # SymPy verification passed


def process_example(
    problem: Dict,
    executor: C5_SymPyExecutor,
    verify: bool = True,
) -> Optional[Phase2Example]:
    """
    Process a single MATH example into Phase 2 training data.
    """
    problem_text = problem.get("problem", problem.get("question", ""))
    cot_text = problem.get("solution", problem.get("cot", ""))
    answer = problem.get("answer", problem.get("gold_answer"))

    # Parse answer
    if isinstance(answer, str):
        # Try to extract numeric answer
        nums = re.findall(r'-?\d+(?:\.\d+)?', answer)
        if nums:
            try:
                answer = float(nums[-1])
            except ValueError:
                return None
        else:
            return None

    # Parse CoT into steps
    steps = parse_cot_to_steps(cot_text)

    if len(steps) == 0:
        return None

    # Build training example
    span_groups = [s.text for s in steps]
    gold_ops = [s.operation for s in steps]
    gold_adjacency = steps_to_adjacency(steps)

    # Verify with SymPy
    verified = False
    if verify:
        args = [[s.operands[i] if i < len(s.operands) else None for i in range(3)] for s in steps]
        adj_tensor = torch.tensor(gold_adjacency, dtype=torch.float)

        result = executor.execute(gold_ops, args, adj_tensor)
        if result.success and result.result is not None:
            if abs(result.result - answer) < 1e-6 or (answer != 0 and abs((result.result - answer) / answer) < 0.01):
                verified = True

    return Phase2Example(
        problem_text=problem_text,
        span_groups=span_groups,
        gold_ops=gold_ops,
        gold_adjacency=gold_adjacency,
        gold_answer=answer,
        cot_text=cot_text,
        verified=verified,
    )


def generate_phase2_data(
    input_data: List[Dict],
    verify: bool = True,
    min_steps: int = 1,
    max_steps: int = 10,
) -> List[Dict]:
    """
    Generate Phase 2 training data from MATH dataset.
    """
    executor = C5_SymPyExecutor()
    examples = []
    stats = {
        "total": 0,
        "parsed": 0,
        "verified": 0,
        "skipped_no_steps": 0,
        "skipped_too_many_steps": 0,
    }

    for problem in input_data:
        stats["total"] += 1

        ex = process_example(problem, executor, verify)

        if ex is None:
            stats["skipped_no_steps"] += 1
            continue

        if len(ex.span_groups) < min_steps:
            stats["skipped_no_steps"] += 1
            continue

        if len(ex.span_groups) > max_steps:
            stats["skipped_too_many_steps"] += 1
            continue

        stats["parsed"] += 1
        if ex.verified:
            stats["verified"] += 1

        examples.append(asdict(ex))

    print(f"\nData generation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return examples


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def load_data(path: str) -> List[Dict]:
    """Load data from local file or S3."""
    if path.startswith("s3://"):
        import boto3
        import io

        path_parts = path[5:].split("/", 1)
        bucket, key = path_parts[0], path_parts[1]

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        return json.loads(content)
    else:
        with open(path) as f:
            return json.load(f)


def save_data(data: List[Dict], path: str):
    """Save data to local file or S3."""
    content = json.dumps(data, indent=2)

    if path.startswith("s3://"):
        import boto3

        path_parts = path[5:].split("/", 1)
        bucket, key = path_parts[0], path_parts[1]

        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
        print(f"Saved {len(data)} examples to {path}")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Saved {len(data)} examples to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 2 training data")

    parser.add_argument("--input", type=str, required=True,
                        help="Input MATH data path (local or s3://)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for Phase 2 data")
    parser.add_argument("--verify", action="store_true",
                        help="Verify examples with SymPy")
    parser.add_argument("--min-steps", type=int, default=1,
                        help="Minimum number of steps")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum number of steps")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of input examples to process")

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input}...")
    input_data = load_data(args.input)
    print(f"Loaded {len(input_data)} examples")

    if args.limit:
        input_data = input_data[:args.limit]
        print(f"Limited to {len(input_data)} examples")

    # Generate Phase 2 data
    print("\nGenerating Phase 2 training data...")
    output_data = generate_phase2_data(
        input_data,
        verify=args.verify,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    # Save output
    save_data(output_data, args.output)

    # Summary
    print(f"\nGenerated {len(output_data)} Phase 2 training examples")
    if args.verify:
        verified_count = sum(1 for ex in output_data if ex.get("verified", False))
        print(f"  Verified: {verified_count} ({100*verified_count/len(output_data):.1f}%)")


if __name__ == "__main__":
    main()
