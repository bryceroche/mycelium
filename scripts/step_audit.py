#!/usr/bin/env python3
"""Step-level audit tool: Trace exactly which steps fail and why.

Usage:
    python scripts/step_audit.py --problems 3 --seed 123
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from mycelium.solver import Solver, StepResult
from mycelium.step_signatures import StepSignatureDB
from mycelium.client import get_client


def format_step_audit(step: StepResult, indent: int = 0) -> list[str]:
    """Format a single step's audit trail."""
    prefix = "  " * indent
    lines = []

    # Step header with execution method
    method_icon = {
        "dsl": "⚙️",
        "llm": "🤖",
        "routing": "🔀",
        "decomposition": "📦",
        "formula": "📐",
    }.get(step.execution_method, "❓")

    status = "✅" if step.success else "❌"

    lines.append(f"{prefix}{status} {method_icon} [{step.execution_method:12}] {step.step_id}")
    lines.append(f"{prefix}   Task: {step.task[:80]}...")

    # Result (truncated)
    result_preview = str(step.result)[:60].replace('\n', ' ')
    lines.append(f"{prefix}   Result: {result_preview}")

    # DSL details if executed
    if step.dsl_executed or step.execution_method == "dsl":
        lines.append(f"{prefix}   DSL: script='{step.dsl_script[:50]}' conf={step.dsl_confidence:.2f}")

    # Signature info
    if step.signature_matched:
        sig = step.signature_matched
        sig_name = sig.step_type or sig.description[:40] if sig.description else "unknown"
        lines.append(f"{prefix}   Sig: '{sig_name}' (sim={step.signature_similarity:.2f})")
        if step.was_injected:
            lines.append(f"{prefix}   💉 INJECTED")

    # Flag suspicious results
    try:
        num_result = float(step.result)
        if num_result == 1.0 and step.dsl_executed:
            lines.append(f"{prefix}   ⚠️  SUSPICIOUS: DSL returned exactly 1.0")
        elif num_result == 0.0 and step.dsl_executed:
            lines.append(f"{prefix}   ⚠️  SUSPICIOUS: DSL returned exactly 0.0")
    except (ValueError, TypeError):
        pass

    # Recurse into sub-steps
    if step.sub_step_results:
        lines.append(f"{prefix}   Sub-steps ({len(step.sub_step_results)}):")
        for sub in step.sub_step_results:
            lines.extend(format_step_audit(sub, indent + 2))

    return lines


def print_problem_audit(problem: dict, result, ground_truth: str):
    """Print full audit for a single problem."""
    print("\n" + "=" * 80)
    print(f"PROBLEM: {problem.get('problem', '')[:100]}...")
    print(f"GROUND TRUTH: {ground_truth}")
    print(f"PREDICTED: {result.answer}")
    print(f"SUCCESS: {'✅ CORRECT' if result.success else '❌ WRONG'}")
    print(f"TIME: {result.elapsed_ms:.0f}ms")
    print("-" * 80)

    # Print step-by-step audit
    print("\nSTEP-BY-STEP EXECUTION:")
    for step in result.step_results:
        for line in format_step_audit(step):
            print(line)
        print()

    # Summary stats
    total = len(result.step_results)
    dsl_steps = sum(1 for s in result.step_results if s.execution_method == "dsl")
    llm_steps = sum(1 for s in result.step_results if s.execution_method == "llm")
    injected = sum(1 for s in result.step_results if s.was_injected)

    print("-" * 80)
    print(f"STATS: {total} steps | {dsl_steps} DSL | {llm_steps} LLM | {injected} injected")

    # Identify likely failure point
    if not result.success:
        print("\n🔍 FAILURE ANALYSIS:")
        # Look for suspicious DSL results
        for step in result.step_results:
            try:
                num = float(step.result)
                if step.dsl_executed and num in (0.0, 1.0):
                    print(f"   ⚠️  {step.step_id}: DSL returned {num} - likely bad param mapping")
            except (ValueError, TypeError):
                pass

        # Check for failed steps
        for step in result.step_results:
            if not step.success:
                print(f"   ❌ {step.step_id}: Step marked as failed")


async def main():
    parser = argparse.ArgumentParser(description="Step-level audit tool")
    parser.add_argument("--problems", type=int, default=3, help="Number of problems")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--level", type=int, default=5, help="MATH level")
    args = parser.parse_args()

    # Load problems from MATH dataset (with caching)
    cache_file = Path(__file__).parent / f"math_L{args.level}_cache.json"

    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        import json
        with open(cache_file) as f:
            all_problems = json.load(f)
    else:
        print(f"Loading {args.problems} MATH Level {args.level} problems (seed={args.seed})...")
        subsets = ["algebra", "counting_and_probability", "geometry",
                   "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

        all_problems = []
        level_str = f"Level {args.level}"
        for subset in subsets:
            try:
                dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
                for p in dataset:
                    if p.get("level") == level_str:
                        all_problems.append({"problem": p["problem"], "solution": p["solution"], "level": p["level"]})
            except Exception as e:
                print(f"  Warning: Could not load {subset}: {e}")

        # Cache for next time
        import json
        with open(cache_file, "w") as f:
            json.dump(all_problems, f)
        print(f"  Cached {len(all_problems)} problems to {cache_file}")

    print(f"  Found {len(all_problems)} Level {args.level} problems")
    filtered = all_problems

    # Sample with seed
    import random
    random.seed(args.seed)
    problems = random.sample(filtered, min(args.problems, len(filtered)))

    # Initialize solver
    step_db = StepSignatureDB()
    solver = Solver(
        step_db=step_db,
        match_mode="auto",
        injection_mode="all",
        use_hints=True,
    )

    # Run each problem with full audit
    correct = 0
    for i, problem in enumerate(problems):
        print(f"\n{'='*80}")
        print(f"PROBLEM {i+1}/{len(problems)}")

        # Extract answer from solution
        solution = problem.get("solution", "")
        if "boxed{" in solution:
            ground_truth = solution.split("boxed{")[-1].split("}")[0]
        else:
            ground_truth = solution

        result = await solver.solve(
            problem=problem["problem"],
            ground_truth=ground_truth,
        )

        # Check correctness (basic string match)
        pred_clean = result.answer.strip().lower()
        gt_clean = ground_truth.strip().lower()
        is_correct = pred_clean == gt_clean or pred_clean in gt_clean or gt_clean in pred_clean
        result.success = is_correct

        if is_correct:
            correct += 1

        print_problem_audit(problem, result, ground_truth)

    # Final summary
    print("\n" + "=" * 80)
    print(f"FINAL: {correct}/{len(problems)} correct ({100*correct/len(problems):.0f}%)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
