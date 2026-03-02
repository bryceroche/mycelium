"""
Lambda function for Phase 2 data generation.

Processes MATH problems → Phase 2 training examples.
Designed for map-reduce: each invocation handles a chunk of problems.

Memory: 3GB (set in Lambda config, not 1GB!)
Timeout: 15 minutes
"""

import json
import re
import boto3
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Operation pattern matchers (same as data/generate_phase2_data.py)
# ---------------------------------------------------------------------------

OP_LABELS = [
    "ADD", "SUB", "MUL", "DIV",
    "POW", "SQRT", "MOD",
    "PERCENT_OF", "PERCENT_CHANGE",
    "RATIO", "PROPORTION",
    "SOLVE_LINEAR", "SOLVE_QUADRATIC", "SOLVE_SYSTEM",
    "GCD", "LCM", "COMB", "PERM",
    "ABS", "NEG", "RECIPROCAL",
    "SUM", "PRODUCT", "MEAN",
    "MIN", "MAX",
    "FLOOR", "CEIL", "ROUND",
]

OP_PATTERNS = {
    "ADD": [r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)", r"sum of", r"total", r"plus"],
    "SUB": [r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", r"difference", r"subtract", r"minus"],
    "MUL": [r"(\d+(?:\.\d+)?)\s*[×*]\s*(\d+(?:\.\d+)?)", r"product", r"times", r"multiply"],
    "DIV": [r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", r"divided by", r"quotient", r"ratio"],
    "POW": [r"\^", r"\*\*", r"squared", r"cubed", r"power"],
    "SQRT": [r"\\sqrt", r"sqrt\(", r"square root", r"√"],
    "MOD": [r"mod\s+\d+", r"remainder", r"modulo"],
    "PERCENT_OF": [r"(\d+(?:\.\d+)?)\s*%\s*of", r"percent of"],
    "PERCENT_CHANGE": [r"percent change", r"percent increase", r"percent decrease"],
    "GCD": [r"gcd", r"greatest common divisor"],
    "LCM": [r"lcm", r"least common multiple"],
    "COMB": [r"\\binom", r"choose", r"combination"],
    "SOLVE_LINEAR": [r"solve for", r"=\s*\d+", r"equation"],
}


# ---------------------------------------------------------------------------
# Step parsing (lightweight version for Lambda)
# ---------------------------------------------------------------------------

def parse_cot_into_steps(cot_text: str) -> List[str]:
    """Split CoT into computation steps."""
    step_pattern = r"(?:Step\s*\d+[:.]\s*|(?:Therefore|So|Thus|Hence)[,:]?\s*)"
    parts = re.split(step_pattern, cot_text, flags=re.IGNORECASE)
    steps = [s.strip() for s in parts if s and s.strip()]

    if len(steps) <= 1:
        sentences = re.split(r'[.!?]+', cot_text)
        steps = [s.strip() for s in sentences if s and s.strip() and any(c.isdigit() for c in s)]

    return steps


def detect_operation(step_text: str) -> Tuple[str, float]:
    """Detect operation in step."""
    step_lower = step_text.lower()
    scores = defaultdict(float)

    for op, patterns in OP_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, step_text, re.IGNORECASE):
                scores[op] += 1.0
            elif pattern in step_lower:
                scores[op] += 0.5

    if not scores:
        return "ADD", 0.1

    best_op = max(scores.keys(), key=lambda k: scores[k])
    return best_op, min(1.0, scores[best_op] / 3.0)


def extract_operands(step_text: str) -> List[float]:
    """Extract numeric operands from step."""
    operands = []
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', step_text, re.IGNORECASE)
    for m in matches:
        try:
            operands.append(float(m))
        except ValueError:
            pass
    return operands


def find_step_references(step_text: str, previous_results: List[float]) -> List[int]:
    """Find which previous steps this step references."""
    references = []
    step_lower = step_text.lower()

    explicit_refs = re.findall(r'(?:step|result)\s*(\d+)', step_lower)
    for ref in explicit_refs:
        idx = int(ref) - 1
        if 0 <= idx < len(previous_results):
            references.append(idx)

    for i, result in enumerate(previous_results):
        if result is not None:
            result_str = str(int(result) if result == int(result) else result)
            if result_str in step_text:
                references.append(i)

    return list(set(references))


def process_single_problem(problem: Dict) -> Optional[Dict]:
    """Process one MATH problem into Phase 2 format."""
    problem_text = problem.get("problem", problem.get("question", ""))
    cot_text = problem.get("solution", problem.get("cot", ""))
    answer = problem.get("answer", problem.get("gold_answer"))

    # Parse answer
    if isinstance(answer, str):
        nums = re.findall(r'-?\d+(?:\.\d+)?', answer)
        if nums:
            try:
                answer = float(nums[-1])
            except ValueError:
                return None
        else:
            return None

    if answer is None:
        return None

    # Parse CoT
    raw_steps = parse_cot_into_steps(cot_text)
    if not raw_steps:
        return None

    span_groups = []
    gold_ops = []
    results = []

    for step_text in raw_steps:
        op, conf = detect_operation(step_text)
        operands = extract_operands(step_text)

        # Estimate result (last number in step)
        nums = re.findall(r'-?\d+(?:\.\d+)?', step_text)
        result = float(nums[-1]) if nums else None

        span_groups.append(step_text)
        gold_ops.append(op)
        results.append(result)

    # Build adjacency matrix
    n = len(span_groups)
    adjacency = [[0] * n for _ in range(n)]

    for j, step_text in enumerate(span_groups):
        refs = find_step_references(step_text, results[:j])
        for i in refs:
            if 0 <= i < j:
                adjacency[i][j] = 1

    return {
        "problem_text": problem_text,
        "span_groups": span_groups,
        "gold_ops": gold_ops,
        "gold_adjacency": adjacency,
        "gold_answer": answer,
        "cot_text": cot_text,
        "n_steps": len(span_groups),
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """
    Lambda entry point.

    Event format:
    {
        "input_bucket": "mycelium-data",
        "input_key": "math/chunk_001.json",
        "output_bucket": "mycelium-data",
        "output_prefix": "phase2/processed/",
        "chunk_id": "001"
    }

    Or for direct data:
    {
        "problems": [...],
        "output_bucket": "mycelium-data",
        "output_prefix": "phase2/processed/",
        "chunk_id": "001"
    }
    """
    try:
        # Get input problems
        if "problems" in event:
            problems = event["problems"]
        else:
            input_bucket = event["input_bucket"]
            input_key = event["input_key"]

            response = s3.get_object(Bucket=input_bucket, Key=input_key)
            content = response["Body"].read().decode("utf-8")
            problems = json.loads(content)

        # Process each problem
        results = []
        stats = {"total": 0, "success": 0, "failed": 0}

        for problem in problems:
            stats["total"] += 1
            try:
                result = process_single_problem(problem)
                if result:
                    results.append(result)
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                stats["failed"] += 1

        # Write output to S3
        output_bucket = event.get("output_bucket", "mycelium-data")
        output_prefix = event.get("output_prefix", "phase2/processed/")
        chunk_id = event.get("chunk_id", "unknown")

        output_key = f"{output_prefix}chunk_{chunk_id}.json"

        s3.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=json.dumps(results).encode("utf-8"),
            ContentType="application/json",
        )

        return {
            "statusCode": 200,
            "body": {
                "chunk_id": chunk_id,
                "stats": stats,
                "output_key": output_key,
                "n_examples": len(results),
            }
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }


# Local testing
if __name__ == "__main__":
    # Test with sample problem
    test_problem = {
        "problem": "What is 2 + 3?",
        "solution": "Step 1: Add 2 and 3. 2 + 3 = 5. Therefore, the answer is 5.",
        "answer": "5"
    }

    result = process_single_problem(test_problem)
    print(json.dumps(result, indent=2))
