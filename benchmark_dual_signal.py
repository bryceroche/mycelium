#!/usr/bin/env python3
"""GSM8K Benchmark for Dual Signal Solver with Indicator Token Attention"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset
from mycelium.dual_signal_solver import DualSignalSolver

def extract_answer(answer_str: str) -> float:
    """Extract numeric answer from GSM8K answer string."""
    if "####" in answer_str:
        answer_str = answer_str.split("####")[-1].strip()
    try:
        return float(answer_str.replace(",", "").replace("$", ""))
    except ValueError:
        return None

def benchmark_gsm8k(num_problems: int = 1000, show_progress: int = 100):
    """Benchmark the dual signal solver on GSM8K."""
    print("=" * 70)
    print(f"GSM8K BENCHMARK - DUAL SIGNAL SOLVER ({num_problems} problems)")
    print("=" * 70)

    print("\n[1] Loading GSM8K test set...")
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    print(f"    Total problems: {len(ds)}")
    print(f"    Evaluating: {num_problems}")

    print("\n[2] Initializing solver...")
    solver = DualSignalSolver()
    # Trigger pipeline initialization
    solver._ensure_pipeline()
    pipeline = solver._pipeline
    print(f"    Pipeline initialized")
    if hasattr(pipeline, 'store') and pipeline.store:
        print(f"    Templates: {len(pipeline.store.templates)}")

    print("\n[3] Running evaluation...")
    correct = 0
    total = 0
    errors = 0

    for i in range(min(num_problems, len(ds))):
        problem = ds[i]
        question = problem['question']
        expected = extract_answer(problem['answer'])

        if expected is None:
            continue

        total += 1

        try:
            result = solver.solve(question)
            answer = result.answer if result else None

            # Check if correct (within tolerance)
            is_correct = False
            if answer is not None and expected is not None:
                if abs(expected) < 1e-6:
                    is_correct = abs(answer - expected) < 1e-6
                else:
                    is_correct = abs(answer - expected) / abs(expected) < 0.01

            if is_correct:
                correct += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    ERROR at {i+1}: {str(e)[:50]}")

        if (i + 1) % show_progress == 0:
            pct = 100 * correct / total if total > 0 else 0
            print(f"    Progress: {i+1}/{num_problems} | Accuracy: {correct}/{total} = {pct:.1f}%")

    accuracy = 100 * correct / total if total > 0 else 0
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {correct}/{total} = {accuracy:.1f}%")
    print(f"Errors: {errors}")
    print("=" * 70)

    return accuracy, correct, total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=1000)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()
    benchmark_gsm8k(args.num_problems, args.progress)
