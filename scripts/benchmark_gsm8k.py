#!/usr/bin/env python3
"""Benchmark the dual-signal solver on GSM8K.

Run on VM with:
    python scripts/benchmark_gsm8k.py --num-problems 2000

Requires: datasets, torch, transformers, sentence-transformers
"""

import argparse
import json
import re
import time
from pathlib import Path

def extract_answer_from_solution(solution: str) -> float:
    """Extract numeric answer from GSM8K solution string."""
    # GSM8K answers are formatted as "#### <number>"
    match = re.search(r'####\s*([\d,]+\.?\d*)', solution)
    if match:
        return float(match.group(1).replace(',', ''))

    # Fallback: find last number
    numbers = re.findall(r'[\d,]+\.?\d*', solution)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark GSM8K")
    parser.add_argument("--num-problems", type=int, default=2000, help="Number of problems to test")
    parser.add_argument("--output", type=str, default="gsm8k_benchmark_results.json", help="Output file")
    parser.add_argument("--mock", action="store_true", help="Use mock model (no GPU)")
    parser.add_argument("--templates", type=str, default=None, help="Path to custom templates JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("GSM8K Benchmark - Dual Signal Solver")
    print("=" * 60)

    # Load GSM8K dataset
    print("\n1. Loading GSM8K dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        print(f"   Loaded {len(dataset)} problems")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("   Install with: pip install datasets")
        return

    # Initialize solver
    print("\n2. Initializing solver...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from mycelium.dual_signal_solver import DualSignalSolver

        solver = DualSignalSolver(mock_model=args.mock, templates_path=args.templates)
        print("   Solver ready")
    except Exception as e:
        print(f"   Error initializing solver: {e}")
        return

    # Run benchmark
    print(f"\n3. Running benchmark on {args.num_problems} problems...")
    print("-" * 60)

    correct = 0
    total = 0
    errors = 0
    results = []

    start_time = time.time()

    for i, example in enumerate(dataset):
        if i >= args.num_problems:
            break

        problem = example["question"]
        expected = extract_answer_from_solution(example["answer"])

        try:
            result = solver.solve(problem)
            predicted = result.answer

            # Check if correct (within 0.01 tolerance)
            is_correct = abs(predicted - expected) < 0.01
            if is_correct:
                correct += 1

            total += 1

            results.append({
                "problem_id": i,
                "problem": problem[:100] + "..." if len(problem) > 100 else problem,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "spans": result.spans_detected,
                "ops": [{"type": op.dsl_expr, "value": op.value, "entity": op.entity}
                       for op in result.operations],
            })

            # Progress update every 100 problems
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                acc = correct / total * 100
                print(f"   [{i+1:4d}/{args.num_problems}] Accuracy: {acc:.1f}% | Rate: {rate:.1f} prob/sec")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   Error on problem {i}: {e}")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total problems: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} problems/sec)")

    # Save results
    output_data = {
        "config": {
            "num_problems": args.num_problems,
            "mock_model": args.mock,
        },
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total * 100 if total > 0 else 0,
            "errors": errors,
            "time_seconds": elapsed,
            "problems_per_second": total / elapsed if elapsed > 0 else 0,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
