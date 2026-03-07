"""
Error Attribution Diagnostic

Test with ground truth at each level to find WHERE the pipeline breaks.

Levels:
1. Segment recall - does segmentation find the right steps?
2. Operand accuracy - does extraction find the right operands?
3. Template coverage - does the right template exist?
4. Step execution - does the template produce the right result?
5. Chain correct - does chaining produce the final answer?

The FIRST level that fails is the bottleneck.
"""

import io
import json
import re
from collections import defaultdict
import sympy
from sympy import Symbol, Integer, Rational, sqrt, pi
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

from latex_to_sympy import latex_to_sympy, extract_operands_from_latex
from factor_graph_v2 import TEMPLATE_REGISTRY, Templates


# Step type to template mapping
STEP_TYPE_TO_TEMPLATES = {
    "arithmetic": ["arithmetic_add", "arithmetic_sub", "arithmetic_mul", "arithmetic_div"],
    "add": ["arithmetic_add"],
    "subtract": ["arithmetic_sub"],
    "multiply": ["arithmetic_mul"],
    "divide": ["arithmetic_div"],
    "power": ["arithmetic_pow"],
    "sqrt": ["compute_sqrt"],
    "solve": ["solve_equation"],
    "substitute": ["substitute"],
    "simplify": ["simplify", "evaluate"],
    "evaluate": ["evaluate", "identity"],
    "factor": ["factor"],
    "expand": ["expand"],
    "apply_theorem": ["identity", "evaluate"],  # No direct template
    "setup": ["identity"],
    "conclusion": ["identity", "evaluate"],
}


def load_sonnet_ground_truth(max_problems=50):
    """Load Sonnet-extracted ground truth steps."""
    print("Loading Sonnet ground truth...")

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="c2c3_sonnet_labels/instance1_iaf_v3")
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".jsonl")]

    problems = []
    for key in files[:5]:  # Load from first few files
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            content = resp["Body"].read().decode("utf-8")

            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                d = json.loads(line)

                if d.get("parsed_steps") and d.get("segments"):
                    problems.append(d)

                if len(problems) >= max_problems:
                    break

        except Exception as e:
            continue

        if len(problems) >= max_problems:
            break

    print(f"  Loaded {len(problems)} problems with ground truth")
    return problems


def parse_operand(op_str):
    """Convert operand string to SymPy object."""
    if not op_str:
        return None

    # Try LaTeX converter first
    result = latex_to_sympy(str(op_str))
    if result is not None:
        return result

    # Try sympify
    try:
        return sympy.sympify(str(op_str))
    except:
        pass

    # Return as symbol
    return Symbol(str(op_str))


def operands_match(expected, extracted):
    """Check if extracted operands match expected."""
    if not expected or not extracted:
        return False, "empty"

    # Convert to sets of values
    expected_set = set()
    for op in expected:
        parsed = parse_operand(op)
        if parsed is not None:
            try:
                # Normalize to float for comparison
                expected_set.add(float(sympy.N(parsed)))
            except:
                expected_set.add(str(parsed))

    extracted_set = set()
    for op in extracted:
        try:
            extracted_set.add(float(sympy.N(op)))
        except:
            extracted_set.add(str(op))

    if not expected_set or not extracted_set:
        return False, "parse_failed"

    # Check overlap
    overlap = expected_set & extracted_set
    if len(overlap) >= min(len(expected_set), 2):  # At least 2 or all expected
        return True, "match"

    return False, f"mismatch: expected {expected_set}, got {extracted_set}"


def find_compatible_templates(step_type, operands):
    """Find templates that could handle this step type."""
    step_type_lower = step_type.lower() if step_type else ""

    candidates = []

    # Direct mapping
    for key, templates in STEP_TYPE_TO_TEMPLATES.items():
        if key in step_type_lower:
            candidates.extend(templates)

    # Fallback based on operand types
    if not candidates:
        parsed = [parse_operand(op) for op in operands if op]
        parsed = [p for p in parsed if p is not None]

        has_numbers = any(hasattr(p, 'is_number') and p.is_number for p in parsed)

        if has_numbers and len(parsed) >= 2:
            candidates = ["arithmetic_mul", "arithmetic_add", "arithmetic_div", "arithmetic_sub"]
        elif has_numbers:
            candidates = ["evaluate", "identity"]
        else:
            candidates = ["simplify", "evaluate", "identity"]

    return list(set(candidates))


def try_execute_step(step_type, operands, expected_result=None):
    """Try to execute a step with given operands."""
    templates = find_compatible_templates(step_type, operands)

    # Parse operands
    parsed_ops = []
    for op in operands:
        p = parse_operand(op)
        if p is not None:
            parsed_ops.append(p)

    if not parsed_ops:
        return False, None, "no_valid_operands"

    # Try each template
    for template_name in templates:
        if template_name not in TEMPLATE_REGISTRY:
            continue

        meta = TEMPLATE_REGISTRY[template_name]
        fn = meta["fn"]
        n_args = meta["n_args"]

        if len(parsed_ops) < n_args:
            continue

        try:
            result = fn(*parsed_ops[:n_args])
            if result is not None:
                # Check against expected if provided
                if expected_result:
                    expected_parsed = parse_operand(expected_result)
                    if expected_parsed is not None:
                        try:
                            diff = sympy.simplify(result - expected_parsed)
                            if diff == 0:
                                return True, result, "exact_match"
                        except:
                            pass

                return True, result, "executed"

        except Exception as e:
            continue

    return False, None, "no_template_executed"


def attribute_problem_errors(problem):
    """
    Attribute errors for a single problem.
    Returns breakdown of where pipeline fails.
    """
    result = {
        "problem_id": problem.get("problem_id"),
        "n_parsed_steps": 0,
        "n_segments": 0,

        # Level 1: Segment coverage
        "segment_coverage": 0.0,

        # Level 2: Operand extraction - aligned
        "operand_matches": 0,
        "operand_total": 0,
        "operand_accuracy": 0.0,

        # Level 2b: Operand extraction - anywhere
        "operand_exists_anywhere": 0,
        "operand_anywhere_rate": 0.0,

        # Level 3: Template coverage
        "template_exists": 0,
        "template_total": 0,
        "template_coverage": 0.0,

        # Level 4: Step execution
        "step_executes": 0,
        "step_total": 0,
        "step_execution_rate": 0.0,

        # Level 5: Chain correctness
        "chain_produces_result": False,
        "chain_result": None,

        # First failure
        "first_failure": None,
        "failure_details": [],
    }

    parsed_steps = problem.get("parsed_steps", [])
    segments = problem.get("segments", [])

    result["n_parsed_steps"] = len(parsed_steps)
    result["n_segments"] = len(segments)

    if not parsed_steps:
        result["first_failure"] = "no_parsed_steps"
        return result

    # Level 1: Do we have segments for each step?
    result["segment_coverage"] = min(len(segments), len(parsed_steps)) / len(parsed_steps)

    # Collect ALL operands from ALL segments
    all_segment_operands = set()
    for seg in segments:
        for op in seg.get("c3_operands", []):
            if isinstance(op, dict) and "value" in op:
                parsed = parse_operand(op["value"])
                if parsed is not None:
                    try:
                        all_segment_operands.add(float(sympy.N(parsed)))
                    except:
                        all_segment_operands.add(str(parsed))

    # Analyze each step
    chain_results = []

    for i, step in enumerate(parsed_steps):
        step_type = step.get("step_type", "unknown")
        expected_operands = step.get("operands", [])
        expected_result = step.get("result")

        result["step_total"] += 1

        # Level 2: Check operand extraction
        # Get corresponding segment if available
        if i < len(segments):
            segment = segments[i]

            # Segments have Sonnet-extracted operands in c3_operands
            segment_operands = segment.get("c3_operands", [])
            extracted = []
            for op in segment_operands:
                if isinstance(op, dict) and "value" in op:
                    parsed = parse_operand(op["value"])
                    if parsed is not None:
                        extracted.append(parsed)

            if expected_operands:
                result["operand_total"] += 1
                match, reason = operands_match(expected_operands, extracted)
                if match:
                    result["operand_matches"] += 1
                else:
                    if result["first_failure"] is None:
                        result["first_failure"] = "operand_extraction"
                        result["failure_details"].append({
                            "step": i,
                            "type": "operand",
                            "expected": expected_operands,
                            "extracted": [str(e)[:20] for e in extracted[:5]],
                            "reason": reason,
                        })
        else:
            # No segment for this step
            extracted = []
            if expected_operands:
                result["operand_total"] += 1

        # Level 2b: Check if operands exist ANYWHERE in segments
        if expected_operands:
            expected_set = set()
            for op in expected_operands:
                parsed = parse_operand(op)
                if parsed is not None:
                    try:
                        expected_set.add(float(sympy.N(parsed)))
                    except:
                        expected_set.add(str(parsed))

            found_count = len(expected_set & all_segment_operands)
            if found_count >= min(len(expected_set), 2):  # At least 2 or all
                result["operand_exists_anywhere"] += 1

        # Level 3: Check template exists
        templates = find_compatible_templates(step_type, expected_operands)
        result["template_total"] += 1
        if templates:
            result["template_exists"] += 1
        else:
            if result["first_failure"] is None:
                result["first_failure"] = "no_template"
                result["failure_details"].append({
                    "step": i,
                    "type": "template",
                    "step_type": step_type,
                    "operands": expected_operands,
                })

        # Level 4: Try to execute with ground truth operands
        success, step_result, reason = try_execute_step(
            step_type, expected_operands, expected_result
        )

        if success:
            result["step_executes"] += 1
            chain_results.append(step_result)
        else:
            if result["first_failure"] is None:
                result["first_failure"] = "step_execution"
                result["failure_details"].append({
                    "step": i,
                    "type": "execution",
                    "step_type": step_type,
                    "operands": expected_operands,
                    "reason": reason,
                })

    # Compute rates
    if result["operand_total"] > 0:
        result["operand_accuracy"] = result["operand_matches"] / result["operand_total"]
        result["operand_anywhere_rate"] = result["operand_exists_anywhere"] / result["operand_total"]

    if result["template_total"] > 0:
        result["template_coverage"] = result["template_exists"] / result["template_total"]

    if result["step_total"] > 0:
        result["step_execution_rate"] = result["step_executes"] / result["step_total"]

    # Level 5: Chain result
    if chain_results:
        result["chain_produces_result"] = True
        result["chain_result"] = str(chain_results[-1])

    if result["first_failure"] is None:
        result["first_failure"] = "none" if result["chain_produces_result"] else "chain_empty"

    return result


def main():
    print("="*60)
    print("ERROR ATTRIBUTION DIAGNOSTIC")
    print("="*60)
    print("\nTesting with ground truth at each level to find")
    print("WHERE the pipeline breaks.")

    # Load ground truth
    problems = load_sonnet_ground_truth(max_problems=50)

    # Attribute errors
    print(f"\nAnalyzing {len(problems)} problems...")

    all_results = []
    failure_counts = defaultdict(int)

    for i, problem in enumerate(problems):
        result = attribute_problem_errors(problem)
        all_results.append(result)
        failure_counts[result["first_failure"]] += 1

    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)

    # Level metrics
    avg_segment_coverage = sum(r["segment_coverage"] for r in all_results) / len(all_results)
    avg_operand_accuracy = sum(r["operand_accuracy"] for r in all_results) / len(all_results)
    avg_operand_anywhere = sum(r["operand_anywhere_rate"] for r in all_results) / len(all_results)
    avg_template_coverage = sum(r["template_coverage"] for r in all_results) / len(all_results)
    avg_step_execution = sum(r["step_execution_rate"] for r in all_results) / len(all_results)
    chain_success_rate = sum(1 for r in all_results if r["chain_produces_result"]) / len(all_results)

    print(f"\nLevel 1 - Segment Coverage:       {avg_segment_coverage*100:5.1f}%")
    print(f"Level 2a - Operand Accuracy:      {avg_operand_accuracy*100:5.1f}%  (aligned)")
    print(f"Level 2b - Operand Exists Anywhere: {avg_operand_anywhere*100:5.1f}%  (unaligned)")
    print(f"Level 3 - Template Coverage:      {avg_template_coverage*100:5.1f}%")
    print(f"Level 4 - Step Execution Rate:    {avg_step_execution*100:5.1f}%")
    print(f"Level 5 - Chain Produces Result:  {chain_success_rate*100:5.1f}%")

    # First failure breakdown
    print(f"\nFIRST FAILURE POINT:")
    for failure, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_results) * 100
        print(f"  {failure:25s}: {count:3d} ({pct:5.1f}%)")

    # Detailed failure examples
    print(f"\nFAILURE EXAMPLES:")

    for failure_type in ["operand_extraction", "no_template", "step_execution"]:
        examples = [r for r in all_results if r["first_failure"] == failure_type][:3]
        if examples:
            print(f"\n  {failure_type.upper()}:")
            for ex in examples:
                if ex["failure_details"]:
                    detail = ex["failure_details"][0]
                    print(f"    Problem {ex['problem_id']}, Step {detail.get('step', '?')}:")
                    if "expected" in detail:
                        print(f"      Expected: {detail['expected']}")
                        print(f"      Got: {detail.get('extracted', 'N/A')}")
                    if "step_type" in detail:
                        print(f"      Step type: {detail['step_type']}")
                        print(f"      Operands: {detail.get('operands', [])}")
                    if "reason" in detail:
                        print(f"      Reason: {detail['reason']}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Find the bottleneck
    if avg_segment_coverage < 0.5:
        bottleneck = "SEGMENTATION"
        print(f"\n✗ BOTTLENECK: {bottleneck}")
        print(f"  Only {avg_segment_coverage*100:.0f}% of CoT steps have segments")
        print("  Fix: Improve C1-A boundary detection")

    elif avg_operand_accuracy < 0.5:
        bottleneck = "OPERAND EXTRACTION"
        print(f"\n✗ BOTTLENECK: {bottleneck}")
        print(f"  Only {avg_operand_accuracy*100:.0f}% of operands extracted correctly")
        print("  Fix: Improve C3 or text-based extraction")

    elif avg_template_coverage < 0.7:
        bottleneck = "TEMPLATE LIBRARY"
        print(f"\n✗ BOTTLENECK: {bottleneck}")
        print(f"  Only {avg_template_coverage*100:.0f}% of steps have matching templates")
        print("  Fix: Add templates for missing operation types")

    elif avg_step_execution < 0.5:
        bottleneck = "STEP EXECUTION"
        print(f"\n✗ BOTTLENECK: {bottleneck}")
        print(f"  Only {avg_step_execution*100:.0f}% of steps execute successfully")
        print("  Fix: Improve SymPy parsing or template implementations")

    elif chain_success_rate < 0.5:
        bottleneck = "CHAIN ASSEMBLY"
        print(f"\n✗ BOTTLENECK: {bottleneck}")
        print(f"  Only {chain_success_rate*100:.0f}% of chains produce results")
        print("  Fix: Improve result chaining and DAG assembly")

    else:
        bottleneck = "ANSWER MATCHING"
        print(f"\n~ BOTTLENECK: {bottleneck}")
        print("  Most steps execute, chains produce results")
        print("  Fix: Check answer format matching")

    # Save results
    print("\nSaving results...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/error_attribution.json",
        Body=json.dumps({
            "n_problems": len(problems),
            "avg_segment_coverage": avg_segment_coverage,
            "avg_operand_accuracy": avg_operand_accuracy,
            "avg_template_coverage": avg_template_coverage,
            "avg_step_execution": avg_step_execution,
            "chain_success_rate": chain_success_rate,
            "failure_counts": dict(failure_counts),
            "bottleneck": bottleneck,
        }, indent=2).encode("utf-8")
    )


if __name__ == "__main__":
    main()
