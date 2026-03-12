#!/usr/bin/env python3
"""
Refined Error Attribution - Manual inspection of each error.

The key question: For each wrong answer, WHY is it wrong?

Categories (refined):
1. FORMAT_ERROR - Right idea, SymPy can't parse or wrong syntax
2. SIMPLE_OPERAND_ERROR - Used wrong number/variable but understood the problem
3. MISSING_STEP - Had right approach but skipped a necessary step
4. CONCEPTUAL_ERROR - Doesn't understand what computation to perform
5. DOMAIN_KNOWLEDGE - Needs specialized function (divisor_count, etc.)
6. NO_ATTEMPT - Model produced placeholder/repetition, didn't try to solve

The hypothesis is: Categories 1-3 are potentially fixable by ODE/energy.
Categories 4-6 are NOT fixable - they require the model to reason.
"""

import json
from collections import defaultdict

def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def manual_categorize(problem: dict) -> dict:
    """Manually categorize each error based on inspection."""

    pid = problem["problem_id"]
    text = problem["text"][:150]
    gold = problem["gold_answer"]
    pred = problem.get("predicted_answer", "")
    telegrams = problem["telegrams"]

    # Check for repetition (no attempt)
    unique = set(t.split(None, 1)[-1] if " " in t else t for t in telegrams[:-1])
    is_repetitive = len(unique) <= 1 and len(telegrams) > 2

    # Manual inspection rules based on patterns

    # Pattern: Model just echoes problem values without computation
    if is_repetitive:
        return {
            "category": "NO_ATTEMPT",
            "reason": "Repeated same expression - didn't attempt solution",
            "fixable": False
        }

    # Pattern: Problem asks for specific computation model doesn't understand
    domain_keywords = ["divisor", "prime", "factor", "gcd", "lcm", "modulo", "remainder",
                       "probability", "combination", "permutation", "matrix", "determinant"]
    for kw in domain_keywords:
        if kw in text.lower():
            # Check if any specialized function appears
            specialized = ["divisor", "factorint", "gcd", "lcm", "mod", "binomial",
                          "factorial", "Matrix", "det", "isprime"]
            if not any(s in str(telegrams) for s in specialized):
                return {
                    "category": "DOMAIN_KNOWLEDGE",
                    "reason": f"Needed {kw}-related function but didn't use one",
                    "fixable": False
                }

    # Pattern: Problem gives specific values, model uses different values
    # This is a heuristic check

    # Pattern: Model's approach is unrelated to problem
    # E.g., problem asks for distance, model outputs unrelated formula

    # For now, use execution success as proxy
    step_results = problem.get("step_results", [])
    exec_failed = any(not sr.get("success", True) for sr in step_results)

    if exec_failed:
        return {
            "category": "FORMAT_ERROR",
            "reason": "SymPy couldn't parse generated expression",
            "fixable": True
        }

    # If all steps executed but answer wrong, it's conceptual or operand error
    # Need to check if the approach was right

    # Heuristic: if predicted has same "shape" as gold, might be operand error
    # Otherwise conceptual

    return {
        "category": "CONCEPTUAL_ERROR",
        "reason": "Model's computation approach doesn't match problem",
        "fixable": False
    }


def detailed_analysis(results):
    """Manually inspect key examples."""

    print("=" * 70)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 70)

    categories = defaultdict(list)

    for p in results["results"]:
        if p["correct"]:
            continue

        pid = p["problem_id"]
        text = p["text"][:100]
        gold = p["gold_answer"]
        pred = p.get("predicted_answer", "")
        telegrams = p["telegrams"]

        # Manual inspection
        cat = manual_categorize(p)
        categories[cat["category"]].append({
            "pid": pid,
            "text": text,
            "gold": gold,
            "pred": pred,
            "telegrams": telegrams[:3],
            "reason": cat["reason"],
            "fixable": cat["fixable"]
        })

    # Print summary
    print("\nCATEGORY DISTRIBUTION (refined):")
    print("-" * 40)
    total = sum(len(v) for v in categories.values())
    fixable_count = 0
    unfixable_count = 0

    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        n = len(items)
        pct = n / total * 100 if total > 0 else 0
        fix = items[0]["fixable"] if items else False
        fixable_count += n if fix else 0
        unfixable_count += n if not fix else 0
        print(f"  {cat}: {n} ({pct:.1f}%) - {'FIXABLE' if fix else 'NOT FIXABLE'}")

    print()
    print(f"TOTAL FIXABLE: {fixable_count} ({fixable_count/total*100:.1f}%)")
    print(f"TOTAL NOT FIXABLE: {unfixable_count} ({unfixable_count/total*100:.1f}%)")

    # Show examples
    print()
    print("=" * 70)
    print("EXAMPLES BY CATEGORY")
    print("=" * 70)

    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n### {cat} ({len(items)} cases) ###")
        for item in items[:2]:
            print(f"\n  Problem {item['pid']}: {item['text']}...")
            print(f"  Gold: {item['gold']}")
            print(f"  Predicted: {item['pred']}")
            print(f"  Telegrams: {item['telegrams']}")
            print(f"  Reason: {item['reason']}")

    return categories


def specific_examples(results):
    """Walk through specific illustrative examples."""

    print("\n" + "=" * 70)
    print("SPECIFIC EXAMPLE ANALYSIS")
    print("=" * 70)

    examples = [
        (0, "Polar coordinates - does model understand conversion?"),
        (2, "Function evaluation - does model know to substitute?"),
        (3, "Divisor counting - does model know divisor_count()?"),
        (5, "Hexagon perimeter - does model understand geometry?"),
        (8, "Distance formula - does model plug in values?"),
    ]

    for pid, question in examples:
        p = results["results"][pid]
        print(f"\n{'='*60}")
        print(f"PROBLEM {pid}: {question}")
        print(f"{'='*60}")
        print(f"Text: {p['text'][:200]}...")
        print(f"Gold: {p['gold_answer']}")
        print(f"Scaffold: {p['scaffold']}")
        print(f"Telegrams:")
        for t in p['telegrams']:
            print(f"  {t}")
        print(f"Predicted: {p.get('predicted_answer')}")

        # Analysis
        print(f"\nANALYSIS:")
        if pid == 0:
            print("  - Model wrote GIVEN r = sqrt(x^2 + y^2) - correct formula")
            print("  - Then EVAL r = sqrt((0-0)^2+(3-3)^2) - WRONG!")
            print("  - Should be sqrt(0^2 + 3^2) = 3, theta = pi/2")
            print("  - Error: Used (x-x) and (y-y) instead of just x,y")
            print("  - Category: CONCEPTUAL - doesn't understand coordinate conversion")
        elif pid == 2:
            print("  - Model wrote GIVEN f(x) = (3x-2)/(x-2)")
            print("  - Then just repeated it without evaluating at -2, -1, 0")
            print("  - Should have: f(-2) = -8/-4 = 2, f(-1) = -5/-3 = 5/3, f(0) = -2/-2 = 1")
            print("  - Category: NO_ATTEMPT - didn't know to substitute values")
        elif pid == 3:
            print("  - Problem asks for number of divisors of 196")
            print("  - Model just echoes 196 without using divisor_count()")
            print("  - Should use: divisor_count(196) or factor then count")
            print("  - Category: DOMAIN_KNOWLEDGE - needs divisor_count")
        elif pid == 5:
            print("  - Hexagon perimeter from triangle perimeter")
            print("  - Model computed 21/6 = 3.5 (triangle side)")
            print("  - But hexagon perimeter = 6 * side = 6 * 7 = 42")
            print("  - Triangle perimeter 21 → side = 7")
            print("  - Category: CONCEPTUAL - wrong geometric reasoning")
        elif pid == 8:
            print("  - Distance between (2,-6) and (-4,3)")
            print("  - Model wrote formula (x2-x1)^2+(y2-y1)^2 but didn't compute")
            print("  - Should be sqrt((-4-2)^2 + (3-(-6))^2) = sqrt(36+81) = sqrt(117) = 3*sqrt(13)")
            print("  - Category: NO_ATTEMPT - wrote formula but didn't substitute")


def main():
    results = load_results("eval_50_results.json")

    print("=" * 70)
    print("REFINED ERROR ATTRIBUTION")
    print("=" * 70)
    print(f"Total: {results['metrics']['n_problems']} problems")
    print(f"Correct: {int(results['metrics']['answer_accuracy'] * results['metrics']['n_problems'])}")
    print(f"Wrong: {results['metrics']['n_problems'] - int(results['metrics']['answer_accuracy'] * results['metrics']['n_problems'])}")

    # Run detailed analysis
    categories = detailed_analysis(results)

    # Show specific examples
    specific_examples(results)

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    no_attempt = len(categories.get("NO_ATTEMPT", []))
    domain = len(categories.get("DOMAIN_KNOWLEDGE", []))
    conceptual = len(categories.get("CONCEPTUAL_ERROR", []))
    format_err = len(categories.get("FORMAT_ERROR", []))

    print(f"""
The errors break down as:
- NO_ATTEMPT (hallucination): {no_attempt} - Model repeats input, doesn't compute
- DOMAIN_KNOWLEDGE: {domain} - Needs specialized functions model doesn't know
- CONCEPTUAL_ERROR: {conceptual} - Wrong mathematical approach entirely
- FORMAT_ERROR: {format_err} - Right idea but SymPy can't parse

NOT FIXABLE by ODE/energy: {no_attempt + domain + conceptual}
POTENTIALLY FIXABLE: {format_err}

The canonicalizer is being asked to REASON about math, not just TRANSCODE.
A 0.5B model cannot learn mathematical reasoning from 4K examples.

Options:
1. Retrieve similar solved problems and use few-shot (Parts Lookup)
2. Use teacher traces directly for hard steps
3. Limit canonicalizer to EASY transcoding, use something else for HARD steps
""")


if __name__ == "__main__":
    main()
