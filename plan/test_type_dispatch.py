"""
Test type-based dispatch baseline on 50 problems.
Compare to old keyword heuristics baseline (42% produce result, ~0% correct).

This is a LOCAL test - doesn't need GPU or C1-A/C3 models.
Uses existing windows and ground truth to test the factor graph v2 logic.
"""

import io
import json
import re
import sympy
from sympy import Symbol, Integer, Float, Rational, Eq, sqrt
import boto3
from collections import defaultdict

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Import from factor_graph_v2
from factor_graph_v2 import (
    run_pipeline, extract_operands_from_text,
    TEMPLATE_REGISTRY, evaluate_answer, print_trace
)


def load_problems_with_text(n=50):
    """Load N problems with text segments and answers from math500_72b_final."""
    print(f"Loading {n} problems with text and answers...")

    # List problem files in math500_72b_final
    paginator = s3.get_paginator("list_objects_v2")
    problem_keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix="math500_72b_final/problem_"):
        for obj in page.get("Contents", []):
            problem_keys.append(obj["Key"])
            if len(problem_keys) >= n:
                break
        if len(problem_keys) >= n:
            break

    problems = []
    for key in problem_keys[:n]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            p = json.loads(resp["Body"].read().decode("utf-8"))

            # Extract text (generated chain-of-thought) and answer
            text = p.get("generated_cot", "")
            answer = p.get("gold_answer", "")

            if text and answer:
                p["_text"] = text
                p["_answer"] = str(answer)
                problems.append(p)

        except Exception as e:
            continue

    print(f"  Loaded {len(problems)} problems with answers")
    return problems


def segment_text(text):
    """
    Segment text into reasoning steps.
    Simple heuristic: split on sentence boundaries.
    """
    # Clean LaTeX
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)

    # Split on sentence boundaries
    segments = re.split(r'(?<=[.!?])\s+', text)

    windows = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) > 5:  # Skip very short segments
            windows.append({"text": seg})

    return windows


def run_evaluation(problems):
    """Run type-based dispatch on all problems."""
    results = {
        "total": len(problems),
        "produce_result": 0,
        "correct": 0,
        "segments_with_result": 0,
        "total_segments": 0,
        "templates_used": defaultdict(int),
        "eval_status": defaultdict(int),  # Track evaluation statuses
        "errors": [],
        "example_traces": [],
        "correct_examples": [],
        "incorrect_examples": [],
    }

    for i, problem in enumerate(problems):
        prob_id = problem.get("idx", f"problem_{i}")
        text = problem.get("_text", "")
        answer = problem.get("_answer", "")

        if not text:
            results["errors"].append({"problem_id": prob_id, "error": "no_text"})
            continue

        # Segment text
        windows = segment_text(text)

        if not windows:
            results["errors"].append({"problem_id": prob_id, "error": "no_windows"})
            continue

        results["total_segments"] += len(windows)

        # Run pipeline
        try:
            final_result, segments = run_pipeline(windows)

            # Track templates used and successful segments
            for seg in segments:
                if seg.selected_template:
                    results["templates_used"][seg.selected_template] += 1
                if seg.success and seg.result is not None:
                    results["segments_with_result"] += 1

            if final_result is not None:
                results["produce_result"] += 1

                # Check correctness
                if answer:
                    eval_result = evaluate_answer(final_result, answer)
                    results["eval_status"][eval_result.get("status", "unknown")] += 1

                    if eval_result.get("correct"):
                        results["correct"] += 1
                        if len(results["correct_examples"]) < 10:
                            results["correct_examples"].append({
                                "problem_id": prob_id,
                                "result": str(final_result),
                                "expected": answer,
                            })
                    else:
                        if len(results["incorrect_examples"]) < 20:
                            results["incorrect_examples"].append({
                                "problem_id": prob_id,
                                "result": str(final_result),
                                "expected": answer,
                                "status": eval_result.get("status"),
                            })

            # Save example trace (first 5)
            if len(results["example_traces"]) < 5:
                results["example_traces"].append({
                    "problem_id": prob_id,
                    "n_windows": len(windows),
                    "final_result": str(final_result) if final_result else None,
                    "expected": answer,
                    "segments": [
                        {
                            "text": seg.text[:100],
                            "operands": [str(op)[:20] for op in seg.operands[:3]],
                            "template": seg.selected_template,
                            "result": str(seg.result)[:50] if seg.result else None,
                        }
                        for seg in segments
                    ]
                })

        except Exception as e:
            results["errors"].append({
                "problem_id": prob_id,
                "error": str(e)
            })

    return results


def main():
    print("="*60)
    print("TYPE-BASED DISPATCH BASELINE EVALUATION")
    print("="*60)

    # Load problems (all 209)
    problems = load_problems_with_text(209)

    # Run evaluation
    print("\nRunning type-based dispatch pipeline...")
    results = run_evaluation(problems)

    # Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    pct_produce = results["produce_result"] / results["total"] * 100
    pct_correct = results["correct"] / results["total"] * 100
    seg_success_rate = (results["segments_with_result"] / results["total_segments"] * 100
                        if results["total_segments"] > 0 else 0)

    print(f"\nProblems evaluated: {results['total']}")
    print(f"Produce result: {results['produce_result']} ({pct_produce:.1f}%)")
    print(f"Correct answers: {results['correct']} ({pct_correct:.1f}%)")
    print(f"\nSegment statistics:")
    print(f"  Total segments: {results['total_segments']}")
    print(f"  Segments with result: {results['segments_with_result']} ({seg_success_rate:.1f}%)")

    print(f"\nTemplates used (top 10):")
    for template, count in sorted(results["templates_used"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {template}: {count}")

    print(f"\nEvaluation status breakdown:")
    for status, count in sorted(results["eval_status"].items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    print(f"\nErrors: {len(results['errors'])}")
    for err in results["errors"][:3]:
        print(f"  {err['problem_id']}: {err['error']}")

    # Example traces
    print("\n" + "="*60)
    print("EXAMPLE TRACES")
    print("="*60)
    for trace in results["example_traces"][:3]:
        print(f"\nProblem {trace['problem_id']}: {trace['n_windows']} windows")
        print(f"  Final result: {trace['final_result']}")
        for seg in trace["segments"][:3]:
            template = seg['template'] or 'None'
            print(f"    {template:20s} → {seg['result']}")
            print(f"      text: {seg['text'][:60]}...")

    # Correct examples
    if results["correct_examples"]:
        print("\n" + "="*60)
        print("CORRECT EXAMPLES")
        print("="*60)
        for ex in results["correct_examples"][:3]:
            print(f"  Problem {ex['problem_id']}: {ex['result']} = {ex['expected']}")

    # Incorrect examples with analysis
    if results["incorrect_examples"]:
        print("\n" + "="*60)
        print("INCORRECT EXAMPLES (with failure analysis)")
        print("="*60)
        for ex in results["incorrect_examples"][:10]:
            print(f"\n  Problem {ex['problem_id']}:")
            print(f"    Got:      {ex['result']}")
            print(f"    Expected: {ex['expected']}")
            print(f"    Status:   {ex.get('status', 'unknown')}")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Old greedy (keyword heuristics):  42% produce result, ~0% correct")
    print(f"New greedy (type-based dispatch): {pct_produce:.0f}% produce result, {pct_correct:.1f}% correct")

    if pct_produce > 42:
        print(f"\n✓ IMPROVEMENT: +{pct_produce - 42:.0f}% in result production")
    elif pct_produce < 42:
        print(f"\n✗ REGRESSION: -{42 - pct_produce:.0f}% in result production")
    else:
        print("\n= SAME result production rate")

    if pct_correct > 1:
        print(f"✓ CORRECTNESS: {pct_correct:.1f}% correct answers")

    # Save results to S3
    print("\nSaving results to S3...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/type_dispatch_v2_results.json",
        Body=json.dumps({
            "produce_result_pct": pct_produce,
            "correct_pct": pct_correct,
            "segment_success_rate": seg_success_rate,
            "total_problems": results["total"],
            "templates_used": dict(results["templates_used"]),
            "example_traces": results["example_traces"],
            "correct_examples": results["correct_examples"],
            "incorrect_examples": results["incorrect_examples"],
        }, indent=2, default=str).encode("utf-8")
    )
    print(f"  Saved to s3://{BUCKET}/factor_graph_eval/type_dispatch_v2_results.json")


if __name__ == "__main__":
    main()
