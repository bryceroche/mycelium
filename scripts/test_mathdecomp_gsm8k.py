"""
Test mathdecomp on GSM8K problems.

Usage:
    uv run python scripts/test_mathdecomp_gsm8k.py
"""

import re
from datasets import load_dataset
from mycelium.mathdecomp import decompose_with_api
from mycelium.mathdecomp.executor import trace_execution


def extract_answer(answer_str: str) -> float:
    """Extract numeric answer from GSM8K answer string."""
    match = re.search(r'####\s*([0-9,.-]+)', answer_str)
    if match:
        return float(match.group(1).replace(',', ''))
    numbers = re.findall(r'[0-9,]+\.?[0-9]*', answer_str)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return 0.0


def main():
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    train_data = gsm8k["train"]

    NUM_PROBLEMS = 20

    print(f"\n{'='*70}")
    print(f"Testing mathdecomp on {NUM_PROBLEMS} GSM8K problems")
    print(f"{'='*70}")

    correct = 0
    failed_verification = 0
    wrong_answer = 0

    for i in range(NUM_PROBLEMS):
        item = train_data[i]
        question = item["question"]
        expected = extract_answer(item["answer"])

        print(f"\n[{i+1}/{NUM_PROBLEMS}] {question[:60]}...")
        print(f"Expected answer: {expected}")

        decomp = decompose_with_api(question, expected_answer=expected)

        if decomp.verified:
            print(f"✓ CORRECT - {len(decomp.steps)} steps")
            correct += 1
        elif decomp.error and "Wrong answer" in decomp.error:
            print(f"✗ WRONG ANSWER: got {decomp.answer_value}, expected {expected}")
            wrong_answer += 1
            # Show the trace anyway
            if decomp.steps:
                print(trace_execution(decomp))
        else:
            print(f"✗ VERIFICATION FAILED: {decomp.error}")
            failed_verification += 1
            if decomp.steps:
                print(f"  Steps: {[s.semantic for s in decomp.steps]}")

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Correct:              {correct}/{NUM_PROBLEMS} ({100*correct/NUM_PROBLEMS:.1f}%)")
    print(f"Wrong answer:         {wrong_answer}/{NUM_PROBLEMS}")
    print(f"Verification failed:  {failed_verification}/{NUM_PROBLEMS}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
