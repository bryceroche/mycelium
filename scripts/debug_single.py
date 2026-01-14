#!/usr/bin/env python3
"""Debug a single problem with detailed step output."""

import asyncio
import sys
import json
import argparse
import logging
sys.path.insert(0, 'src')

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

from mycelium.solver import Solver
from mycelium.step_signatures import StepSignatureDB
from datasets import load_dataset


async def debug_problem(problem_idx: int = 0, level: int = 5, db_path: str = "mycelium.db", seed: int = 42):
    """Run a single problem with debug output."""
    import random
    random.seed(seed)

    # Load dataset (same as pipeline_runner.py)
    subsets = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

    all_problems = []
    level_str = f"Level {level}"

    for subset in subsets:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
            for p in ds:
                if p["level"] == level_str:
                    all_problems.append({"problem": p["problem"], "solution": p["solution"], "level": p["level"], "type": subset})
        except Exception as e:
            print(f"Warning: Could not load {subset}: {e}")

    # Shuffle with seed for reproducibility
    random.shuffle(all_problems)
    problems = all_problems

    if problem_idx >= len(problems):
        print(f"Only {len(problems)} L{level} problems available")
        return

    problem = problems[problem_idx]

    print("=" * 80)
    print(f"PROBLEM {problem_idx} (seed={seed}, type={problem['type']}):")
    print("=" * 80)
    print(problem["problem"][:500])
    print("\n" + "=" * 80)
    print(f"EXPECTED ANSWER: {problem['solution'].split('boxed{')[-1].split('}')[0] if 'boxed' in problem['solution'] else 'see solution'}")
    print("=" * 80)

    # Initialize solver (match pipeline_runner.py settings)
    step_db = StepSignatureDB(db_path=db_path)
    solver = Solver(
        step_db=step_db,
        match_mode="auto",
        injection_mode="all",
        use_hints=True,
    )

    # Solve
    print("\nSOLVING...")
    result = await solver.solve(problem["problem"])

    print("\n" + "=" * 80)
    print("STEP RESULTS:")
    print("=" * 80)

    for i, step in enumerate(result.step_results):
        print(f"\n--- Step {i+1}: {step.step_id} ---")
        print(f"Task: {step.task}")
        print(f"Result: {step.result[:200]}..." if len(step.result) > 200 else f"Result: {step.result}")
        print(f"Injected: {step.was_injected}")
        if step.signature_matched:
            sig = step.signature_matched
            print(f"Matched Signature: {sig.signature_id[:30]}...")
            if sig.dsl_script:
                try:
                    dsl = json.loads(sig.dsl_script)
                    print(f"  DSL Type: {dsl.get('type')}")
                    print(f"  DSL Script: {dsl.get('script', '')[:50]}")
                except:
                    pass
        if step.numeric_value is not None:
            print(f"Numeric Value: {step.numeric_value}")

    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(f"Predicted: {result.answer}")
    print(f"Success: {result.success}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Problem index within level")
    parser.add_argument("--level", type=int, default=5, help="Difficulty level (1-5)")
    parser.add_argument("--db", type=str, default="mycelium.db", help="Database path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for problem selection")
    args = parser.parse_args()

    asyncio.run(debug_problem(args.idx, args.level, args.db, args.seed))
