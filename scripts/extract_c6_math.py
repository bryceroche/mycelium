#!/usr/bin/env python3
"""
Extract C6 Answer Type Training Data from MATH dataset.

MATH has actual answer type variety (unlike GSM8K which is 100% integers).

Answer types:
- integer: whole numbers
- fraction: \frac{a}{b}
- radical: \sqrt{...}
- decimal: numbers with decimal points
- expression: algebraic expressions with variables
- other: sets, coordinates, text, etc.

Usage:
  python extract_c6_math.py --output c6_math_train.json
"""

import json
import argparse
import re
from collections import Counter
from datasets import load_dataset


def extract_boxed(solution: str) -> str:
    """Extract final answer from \boxed{...}."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if matches:
        return matches[-1].strip()
    return None


def classify_answer(ans: str) -> str:
    """Classify answer into type categories."""
    if ans is None:
        return None

    ans = ans.strip()

    # Integer: just digits, possibly negative
    if re.match(r"^-?\d+$", ans):
        return "integer"

    # Decimal: digits with decimal point
    if re.match(r"^-?\d+\.\d+$", ans):
        return "decimal"

    # Fraction: contains \frac
    if "frac" in ans:
        return "fraction"

    # Radical: contains \sqrt
    if "sqrt" in ans:
        return "radical"

    # Expression: contains variables or exponents
    if re.search(r"[a-z]", ans.lower()) and re.search(r"[\^+-]", ans):
        return "expression"

    # Single letter (multiple choice style)
    if ans in "ABCDEFGHIJ":
        return "letter"

    # Everything else
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Limit examples")
    args = parser.parse_args()

    print("=" * 60)
    print("C6 ANSWER TYPE EXTRACTION FROM MATH")
    print("=" * 60)

    # Load MATH dataset
    print("\nLoading MATH dataset...")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    print(f"Loaded {len(ds)} problems")

    # Extract and classify
    results = []
    type_counts = Counter()

    for i, ex in enumerate(ds):
        if args.limit and i >= args.limit:
            break

        problem = ex["problem"]
        solution = ex["solution"]
        answer = extract_boxed(solution)
        answer_type = classify_answer(answer)

        if answer_type is None:
            continue

        type_counts[answer_type] += 1

        results.append({
            "text": problem,
            "label": answer_type,
            "answer": answer,
            "math_type": ex.get("type", ""),
            "level": ex.get("level", ""),
        })

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1} problems, extracted {len(results)}")

    print(f"\nExtracted {len(results)} examples")

    # Stats
    print("\nAnswer type distribution:")
    for t, count in type_counts.most_common():
        pct = 100 * count / len(results)
        print(f"  {t:12s}: {count:5d} ({pct:5.1f}%)")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f)

    print("Done!")


if __name__ == "__main__":
    main()
