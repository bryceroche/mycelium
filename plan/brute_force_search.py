"""
Brute Force Template Search

The honest test: does searching over template combinations find correct
answers that greedy selection misses?

Greedy:       Pick first template that executes → 3.8% correct
Search:       Try ALL combinations, find consistent end-to-end chain

With 4 segments × 3 templates = 81 combinations. Totally enumerable.
"""

import io
import json
import re
from itertools import product
from collections import defaultdict
import sympy
from sympy import Symbol, Integer, Float, Rational, sqrt, pi, E
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Import our tools
from latex_to_sympy import latex_to_sympy, extract_operands_from_latex
from factor_graph_v2 import (
    TEMPLATE_REGISTRY, Templates, classify_operands,
    get_candidate_templates, try_template
)


def segment_text(text):
    """Split text into reasoning segments."""
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)

    segments = re.split(r'(?<=[.!?])\s+', text)
    return [{"text": s.strip()} for s in segments if len(s.strip()) > 10]


def get_all_candidates(segment_text):
    """Get ALL template candidates for a segment (not just first)."""
    operands = extract_operands_from_latex(segment_text)
    info = classify_operands(operands)

    candidates = []

    # Try ALL templates, keep those that could potentially work
    for name, meta in TEMPLATE_REGISTRY.items():
        n_args = meta["n_args"]
        requires = meta["requires"]

        # Check basic compatibility
        can_try = False

        if requires == "numeric":
            if info["numeric_count"] >= n_args:
                can_try = True
            elif info["numeric_count"] >= 1:  # Might get operand from backref
                can_try = True

        elif requires == "expression":
            if info["has_expression"] or info["has_symbol"]:
                can_try = True

        elif requires == "equation":
            if info["has_equation"] or info["has_expression"]:
                can_try = True

        elif requires == "any":
            can_try = True

        if can_try:
            candidates.append(name)

    # Always include identity and evaluate as fallbacks
    for fallback in ["identity", "evaluate"]:
        if fallback not in candidates:
            candidates.append(fallback)

    return candidates, operands


def execute_template(template_name, operands, previous_result=None):
    """Try to execute a template. Returns (success, result)."""
    if template_name not in TEMPLATE_REGISTRY:
        return False, None

    meta = TEMPLATE_REGISTRY[template_name]
    fn = meta["fn"]
    n_args = meta["n_args"]
    requires = meta["requires"]

    # Separate operand types
    numbers = [op for op in operands if hasattr(op, 'is_number') and op.is_number]
    symbols = [op for op in operands if isinstance(op, Symbol)]
    expressions = [op for op in operands
                   if hasattr(op, 'free_symbols') and op.free_symbols and not isinstance(op, Symbol)]

    # Build argument list based on template requirements
    args = []

    if requires == "numeric":
        if previous_result is not None and hasattr(previous_result, 'is_number'):
            args = [previous_result] + numbers[:n_args-1]
        else:
            args = numbers[:n_args]

    elif requires == "expression":
        if n_args == 1:
            if expressions:
                args = [expressions[0]]
            elif previous_result is not None:
                args = [previous_result]
        elif n_args == 3:  # substitute
            if expressions and symbols and numbers:
                args = [expressions[0], symbols[0], numbers[0]]
            elif previous_result is not None and symbols and numbers:
                args = [previous_result, symbols[0], numbers[0]]

    elif requires == "equation":
        from sympy import Eq
        equations = [op for op in operands if isinstance(op, Eq)]
        if equations and symbols:
            args = [equations[0], symbols[0]]
        elif expressions and symbols:
            args = [expressions[0], symbols[0]]

    else:  # "any"
        if n_args == 1:
            if numbers:
                args = [numbers[0]]
            elif previous_result is not None:
                args = [previous_result]
            elif expressions:
                args = [expressions[0]]

    if len(args) < n_args:
        return False, None

    try:
        result = fn(*args[:n_args])
        if result is not None:
            return True, result
        return False, None
    except Exception:
        return False, None


def search_template_combinations(segments, max_candidates=5, max_combinations=1000):
    """
    Brute force search over all template combinations.
    Returns best configuration that executes end-to-end.
    """
    # Get candidates for each segment
    segment_candidates = []
    segment_operands = []

    for seg in segments:
        candidates, operands = get_all_candidates(seg["text"])
        # Limit candidates to avoid explosion
        segment_candidates.append(candidates[:max_candidates])
        segment_operands.append(operands)

    # Count total combinations
    n_combos = 1
    for cands in segment_candidates:
        n_combos *= len(cands)

    if n_combos > max_combinations:
        # Too many - use greedy fallback
        return None, None, {"status": "too_many_combinations", "n_combos": n_combos}

    best_result = None
    best_chain = None
    best_score = -1
    n_tried = 0
    n_executed = 0

    # Enumerate all combinations
    for combo in product(*segment_candidates):
        n_tried += 1

        results = []
        previous = None
        failed = False
        score = 0

        for i, (template_name, operands) in enumerate(zip(combo, segment_operands)):
            success, result = execute_template(template_name, operands, previous)

            if not success:
                failed = True
                break

            results.append(result)
            previous = result
            score += 1

            # Bonus for meaningful operations (not just identity)
            if template_name not in ["identity", "evaluate"]:
                score += 0.5

        if not failed:
            n_executed += 1
            if score > best_score:
                best_score = score
                best_result = results[-1] if results else None
                best_chain = list(zip(combo, results))

    stats = {
        "n_segments": len(segments),
        "n_combinations": n_combos,
        "n_tried": n_tried,
        "n_executed": n_executed,
        "best_score": best_score,
    }

    return best_result, best_chain, stats


def evaluate_answer(result, ground_truth):
    """Compare result to ground truth."""
    if result is None:
        return {"correct": False, "status": "no_result"}

    try:
        # Parse ground truth
        gt = ground_truth.strip()
        gt = re.sub(r'^\$+|\$+$', '', gt)

        expected = latex_to_sympy(gt)
        if expected is None:
            return {"correct": False, "status": "gt_parse_error"}

        # Handle tuple/set
        if isinstance(expected, (tuple, set)):
            if isinstance(result, (tuple, set)):
                if set(expected) == set(result) if isinstance(expected, set) else expected == result:
                    return {"correct": True, "status": "exact"}
            return {"correct": False, "status": "type_mismatch"}

        # Compare expressions
        try:
            diff = sympy.simplify(result - expected)
            if diff == 0:
                return {"correct": True, "status": "exact"}
        except:
            pass

        # Numeric comparison
        try:
            r_float = float(sympy.N(result))
            e_float = float(sympy.N(expected))
            if abs(r_float - e_float) < 1e-6:
                return {"correct": True, "status": "numeric"}
        except:
            pass

        return {"correct": False, "status": "wrong",
                "result": str(result), "expected": str(expected)}

    except Exception as e:
        return {"correct": False, "status": "error", "error": str(e)}


def load_problems(n=209):
    """Load MATH problems with answers."""
    print(f"Loading {n} problems...")

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

            text = p.get("generated_cot", "")
            answer = p.get("gold_answer", "")

            if text and answer:
                problems.append({
                    "idx": p.get("idx"),
                    "text": text,
                    "answer": answer,
                    "category": p.get("category", "unknown"),
                })
        except:
            continue

    print(f"  Loaded {len(problems)} problems")
    return problems


def main():
    print("="*60)
    print("BRUTE FORCE TEMPLATE SEARCH")
    print("="*60)
    print("\nHypothesis: Searching over template combinations finds")
    print("correct answers that greedy selection misses.")
    print("\nGreedy baseline: 3.8% correct")
    print("Target: >10% correct (2.6x improvement)")

    # Load problems
    problems = load_problems(209)

    # Run search
    print(f"\nRunning brute force search on {len(problems)} problems...")

    results = {
        "total": len(problems),
        "produce_result": 0,
        "correct": 0,
        "greedy_would_match": 0,
        "search_found_better": 0,
        "too_many_combos": 0,
        "search_stats": [],
        "correct_examples": [],
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    for i, problem in enumerate(problems):
        segments = segment_text(problem["text"])

        if not segments:
            continue

        # Limit to first 6 segments to keep search tractable
        segments = segments[:6]

        # Run search
        result, chain, stats = search_template_combinations(
            segments,
            max_candidates=4,
            max_combinations=2000
        )

        if stats.get("status") == "too_many_combinations":
            results["too_many_combos"] += 1
            # Fall back to greedy for this problem
            from factor_graph_v2 import run_pipeline
            result, _ = run_pipeline(segments)

        results["search_stats"].append(stats)

        if result is not None:
            results["produce_result"] += 1

            # Check correctness
            eval_result = evaluate_answer(result, problem["answer"])

            cat = problem["category"]
            results["by_category"][cat]["total"] += 1

            if eval_result.get("correct"):
                results["correct"] += 1
                results["by_category"][cat]["correct"] += 1

                if len(results["correct_examples"]) < 10:
                    results["correct_examples"].append({
                        "idx": problem["idx"],
                        "result": str(result),
                        "expected": problem["answer"],
                        "chain": [(t, str(r)[:30]) for t, r in chain] if chain else None,
                        "n_combos": stats.get("n_combinations", 0),
                    })

        if (i + 1) % 50 == 0:
            pct = results["correct"] / (i + 1) * 100
            print(f"  [{i+1}/{len(problems)}] Correct: {results['correct']} ({pct:.1f}%)")

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    pct_produce = results["produce_result"] / results["total"] * 100
    pct_correct = results["correct"] / results["total"] * 100

    print(f"\nProblems: {results['total']}")
    print(f"Produce result: {results['produce_result']} ({pct_produce:.1f}%)")
    print(f"Correct answers: {results['correct']} ({pct_correct:.1f}%)")
    print(f"Too many combos (fallback): {results['too_many_combos']}")

    # Search statistics
    executed_counts = [s.get("n_executed", 0) for s in results["search_stats"] if "n_executed" in s]
    combo_counts = [s.get("n_combinations", 0) for s in results["search_stats"] if "n_combinations" in s]

    if executed_counts:
        print(f"\nSearch statistics:")
        print(f"  Avg combinations tried: {sum(combo_counts)/len(combo_counts):.0f}")
        print(f"  Avg valid chains found: {sum(executed_counts)/len(executed_counts):.1f}")

    # By category
    print(f"\nBy category:")
    for cat, stats in sorted(results["by_category"].items(), key=lambda x: -x[1]["correct"]):
        total = stats["total"]
        correct = stats["correct"]
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {cat:25s}: {correct:2d}/{total:3d} ({pct:5.1f}%)")

    # Correct examples
    if results["correct_examples"]:
        print(f"\nCorrect examples:")
        for ex in results["correct_examples"][:5]:
            print(f"\n  Problem {ex['idx']}:")
            print(f"    Result: {ex['result']}")
            print(f"    Expected: {ex['expected']}")
            print(f"    Combos searched: {ex['n_combos']}")
            if ex["chain"]:
                print(f"    Chain: {' → '.join(t for t, r in ex['chain'][:3])}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    improvement = pct_correct / 3.8 if pct_correct > 0 else 0

    print(f"\nGreedy baseline: 3.8% correct")
    print(f"Brute force search: {pct_correct:.1f}% correct")
    print(f"Improvement: {improvement:.2f}x")

    if pct_correct > 10:
        print(f"\n✓ SUCCESS: Search finds correct answers greedy misses!")
        print("  The factor graph approach is validated.")
        print("  Next: Implement smarter search (beam search, ODE dynamics)")
    elif pct_correct > 5:
        print(f"\n~ PARTIAL: Some improvement, but modest")
        print("  May need better templates or operand extraction")
    else:
        print(f"\n✗ NO IMPROVEMENT: Search doesn't help")
        print("  The templates or operands are fundamentally broken")

    # Save results
    print("\nSaving results...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/brute_force_search_results.json",
        Body=json.dumps({
            "pct_correct": pct_correct,
            "pct_produce": pct_produce,
            "total": results["total"],
            "correct": results["correct"],
            "correct_examples": results["correct_examples"],
            "by_category": {k: dict(v) for k, v in results["by_category"].items()},
        }, indent=2, default=str).encode("utf-8")
    )


if __name__ == "__main__":
    main()
