"""
Test the new RecursiveDecomposer on GSM8K problems.

Usage:
    uv run python scripts/test_recursive_decomposer.py [--num N]
"""

import argparse
import logging
import re
from datasets import load_dataset

from mycelium.recursive_decomposer import RecursiveDecomposer, get_proposals, clear_proposals


def extract_answer(answer_str: str) -> str:
    """Extract numeric answer from GSM8K answer string."""
    match = re.search(r'####\s*([0-9,.-]+)', answer_str)
    if match:
        return match.group(1).replace(',', '')
    return "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=20, help="Number of problems")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k["train"]

    print(f"\n{'='*70}")
    print(f"Testing RecursiveDecomposer on {args.num} GSM8K problems")
    print(f"{'='*70}\n")

    # Clear proposals from previous runs
    clear_proposals()

    decomposer = RecursiveDecomposer()

    correct = 0
    wrong = 0
    errors = 0

    for i in range(args.start, args.start + args.num):
        item = train_data[i]
        question = item["question"]
        expected = extract_answer(item["answer"])

        print(f"[{i+1-args.start}/{args.num}] {question[:60]}...")
        print(f"  Expected: {expected}")

        try:
            # Solve directly - decomposer handles everything internally
            result = decomposer.solve(question)
            print(f"  Result: {result}")

            # Compare
            try:
                if abs(float(result) - float(expected)) < 0.01:
                    print(f"  ✓ CORRECT")
                    correct += 1
                else:
                    print(f"  ✗ WRONG (expected {expected}, got {result})")
                    wrong += 1
            except:
                print(f"  ✗ WRONG (comparison failed)")
                wrong += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            if args.verbose:
                import traceback
                traceback.print_exc()

        print()

    print(f"{'='*70}")
    print(f"Results: {correct}/{args.num} correct ({100*correct/args.num:.1f}%)")
    print(f"Wrong: {wrong}, Errors: {errors}")
    print(f"{'='*70}")

    # Show proposals for review
    proposals = get_proposals()
    if proposals:
        print(f"\n{'='*70}")
        print(f"SIGNATURE PROPOSALS FOR REVIEW ({len(proposals)})")
        print(f"{'='*70}")
        for i, p in enumerate(proposals, 1):
            print(f"\n{i}. Template: \"{p.template[:60]}...\"" if len(p.template) > 60 else f"\n{i}. Template: \"{p.template}\"")
            print(f"   Function: {p.func}")
            print(f"   Similarity to nearest: {p.similarity_to_nearest:.3f}")
    else:
        print(f"\n{'='*70}")
        print("No new signature proposals (all templates matched existing)")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
