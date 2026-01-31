"""
Test GSM8K with the v3 flat prototype architecture.

Usage:
    uv run python scripts/test_gsm8k_v3.py [--num N] [--verbose]
"""

import argparse
import logging
import re
import sys
from datasets import load_dataset

from mycelium.solver import Solver, SolveContext
from mycelium.answer_norm import answers_equivalent, answers_equivalent_llm


def extract_answer(answer_str: str) -> str:
    """Extract numeric answer from GSM8K answer string."""
    match = re.search(r'####\s*([0-9,.-]+)', answer_str)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'[0-9,]+\.?[0-9]*', answer_str)
    if numbers:
        return numbers[-1].replace(',', '')
    return "0"


def main():
    parser = argparse.ArgumentParser(description="Test GSM8K with v3 architecture")
    parser.add_argument("--num", type=int, default=10, help="Number of problems to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--start", type=int, default=0, help="Starting problem index")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM for semantic answer comparison")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k["train"]

    print(f"\n{'='*70}")
    print(f"Testing v3 architecture on {args.num} GSM8K problems")
    print(f"{'='*70}")

    solver = Solver(model="gpt-4o-mini")

    # Show signature stats
    sig_count = solver.step_db.count_signatures()
    func_names = solver.step_db.get_all_func_names()
    print(f"Signatures: {sig_count} | Functions: {len(func_names)}")
    print(f"{'='*70}\n")

    correct = 0
    wrong = 0
    errors = 0

    for i in range(args.start, args.start + args.num):
        item = train_data[i]
        question = item["question"]
        expected = extract_answer(item["answer"])

        print(f"\n[{i+1-args.start}/{args.num}] {question[:70]}...")
        print(f"Expected: {expected}")

        try:
            context = SolveContext(max_depth=3)
            answer = solver.solve_with_trend(question, context)

            print(f"Got: {answer}")

            # Check answer equivalence
            is_correct = False
            if answer is not None:
                if args.llm_judge:
                    is_correct = answers_equivalent_llm(str(answer), expected)
                else:
                    is_correct = answers_equivalent(str(answer), expected, tolerance=0.01)

            if is_correct:
                print("✓ CORRECT")
                correct += 1
            else:
                print(f"✗ WRONG (expected {expected})")
                wrong += 1

        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors += 1
            if args.verbose:
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    total = correct + wrong + errors
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Wrong:   {wrong}/{total}")
    print(f"Errors:  {errors}/{total}")
    print(f"{'='*70}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
