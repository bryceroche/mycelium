#!/usr/bin/env python3
"""
GSM8K Benchmark for Dual-Signal Solver

Uses the 207 specialized templates (deduplicated from 17k spans) to solve
GSM8K problems and measure accuracy.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from mycelium.dual_signal_solver import DualSignalSolver


def extract_answer(answer_str: str) -> float:
    """Extract numeric answer from GSM8K answer string."""
    # GSM8K format: "... #### 42"
    if "####" in answer_str:
        answer_str = answer_str.split("####")[-1].strip()
    # Remove commas and convert
    try:
        return float(answer_str.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def benchmark_gsm8k(num_problems: int = 100, show_examples: int = 5):
    """Benchmark the dual-signal solver on GSM8K."""
    print("=" * 70)
    print("GSM8K BENCHMARK - DUAL-SIGNAL SOLVER (207 TEMPLATES)")
    print("=" * 70)

    # Initialize solver
    print("\n[1] Initializing solver...")
    solver = DualSignalSolver()
    pipeline = solver._ensure_pipeline()

    # Check templates loaded
    template_count = len(pipeline.store.templates)
    print(f"    Templates loaded: {template_count}")

    # Load GSM8K
    print("\n[2] Loading GSM8K test set...")
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    print(f"    Total problems: {len(ds)}")
    print(f"    Evaluating: {num_problems}")

    # Evaluate
    print("\n[3] Running evaluation...")
    correct = 0
    total = 0
    examples = []

    for i in range(min(num_problems, len(ds))):
        problem = ds[i]
        question = problem['question']
        expected = extract_answer(problem['answer'])

        if expected is None:
            continue

        total += 1

        try:
            result = solver.solve(question)
            predicted = result.answer if result else None

            # Check if correct (within tolerance for floats)
            is_correct = False
            if predicted is not None and expected is not None:
                if abs(expected) < 1e-6:
                    is_correct = abs(predicted - expected) < 1e-6
                else:
                    is_correct = abs(predicted - expected) / abs(expected) < 0.01

            if is_correct:
                correct += 1
                status = "CORRECT"
            else:
                status = f"WRONG (got {predicted})"

            # Collect examples
            if len(examples) < show_examples:
                examples.append({
                    'idx': i + 1,
                    'question': question[:100] + "...",
                    'expected': expected,
                    'predicted': predicted,
                    'correct': is_correct,
                    'spans': result.spans_detected if result else 0,
                })

            # Progress
            if (i + 1) % 20 == 0 or i < 5:
                print(f"    {i+1:3d}. {status} (expected={expected})")

        except Exception as e:
            print(f"    {i+1:3d}. ERROR: {e}")

    # Results
    accuracy = 100 * correct / total if total > 0 else 0
    print("\n" + "=" * 70)
    print(f"RESULTS: {correct}/{total} = {accuracy:.1f}%")
    print("=" * 70)

    # Show examples
    if examples:
        print("\nExample outputs:")
        print("-" * 70)
        for ex in examples:
            status = "✓" if ex['correct'] else "✗"
            print(f"{status} Problem {ex['idx']}: {ex['question']}")
            print(f"  Expected: {ex['expected']}, Got: {ex['predicted']}, Spans: {ex['spans']}")
            print()

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark dual-signal solver on GSM8K")
    parser.add_argument("--num-problems", type=int, default=50, help="Number of problems")
    parser.add_argument("--show-examples", type=int, default=5, help="Examples to show")
    args = parser.parse_args()

    benchmark_gsm8k(args.num_problems, args.show_examples)
