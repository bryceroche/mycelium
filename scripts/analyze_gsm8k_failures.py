"""Read a JSONL from diagnose_gsm8k_failures.py and categorize failures.

Heuristic categorization:
- NO_ANSWER:        parsed is None or no #### in gen_text
- STEP_COUNT_MISMATCH:  gen has wrong number of #### segments vs K
- WRONG_NUMBERS:    gen uses numbers not in the question (semantic confab/halluc)
- ARITH_ERROR:      structure right, numbers from question, but arithmetic off
- SEMANTIC_CONFAB:  gen uses wrong nouns (different objects/people from question)
- WRONG_FINAL:      intermediate steps look right but final answer parsing fails

Outputs: counts per category + 3 samples per category.
"""
import json
import sys
import re
from collections import defaultdict


def extract_numbers(text: str) -> set:
    """Extract all integers in a string, dedupe."""
    # Handle digit-spaced "1 6" notation by removing spaces inside digit groups first
    # Pattern: digits possibly separated by single spaces
    cleaned = re.sub(r'(?<=\d) (?=\d)', '', text)
    return set(int(n) for n in re.findall(r'\d+', cleaned))


def categorize(ex: dict) -> str:
    if ex["ok"]:
        return "CORRECT"
    K = ex["k"]
    gen = ex["gen_text"]
    question = ex["question"]
    parsed = ex["parsed"]

    n_hash_gen = gen.count("####")
    n_hash_gold = K
    q_numbers = extract_numbers(question)
    gen_numbers = extract_numbers(gen)

    if parsed is None:
        return "NO_ANSWER"
    if n_hash_gen == 0:
        return "NO_HASH_SEGMENTS"
    if n_hash_gen < n_hash_gold:
        return "TRUNCATED"
    if n_hash_gen > n_hash_gold + 1:
        return "TOO_MANY_STEPS"

    # extra numbers introduced that weren't in question OR derivable
    # heuristic: most "real" GSM8K problems use 2-5 input numbers
    intro_numbers = gen_numbers - q_numbers
    # remove plausibly-derived numbers: if all gen numbers can be explained by
    # arithmetic of question numbers, treat as arith. Otherwise semantic confab.
    suspicious = [n for n in intro_numbers if n > 100 and n not in {sum(q_numbers), max(q_numbers) * 2}]
    if len(suspicious) >= 2:
        return "POSSIBLE_HALLUC_NUMBERS"

    # Check for obvious semantic confab: gen mentions a noun not in question
    # crude: gen text length is much longer than gold, or contains unusual words
    # Skip — too heuristic. Default to ARITH_ERROR for the rest.
    return "ARITH_ERROR"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v60_take2_diag.jsonl"
    examples = [json.loads(line) for line in open(path)]
    print(f"Loaded {len(examples)} examples from {path}\n")

    cat_counts = defaultdict(int)
    cat_samples = defaultdict(list)
    for ex in examples:
        cat = categorize(ex)
        cat_counts[cat] += 1
        if len(cat_samples[cat]) < 3:
            cat_samples[cat].append(ex)

    print("=== category counts ===")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = n / len(examples) * 100
        print(f"  {cat:25s} {n:3d}  ({pct:.1f}%)")

    print("\n=== samples per category ===")
    for cat in sorted(cat_counts, key=lambda x: -cat_counts[x]):
        print(f"\n--- {cat} ({cat_counts[cat]}) ---")
        for ex in cat_samples[cat][:3]:
            print(f"  K={ex['k']} gold={ex['gold_answer']} parsed={ex['parsed']}")
            print(f"  Q: {ex['question'][:100]}")
            print(f"  GEN: {ex['gen_text'][:300]}")
            print()


if __name__ == "__main__":
    main()
