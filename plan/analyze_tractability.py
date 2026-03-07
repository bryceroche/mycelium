"""
Analyze what fraction of MATH problems are template-tractable.

A problem is template-tractable if:
1. Answer is evaluable (not text like "Evelyn", not a proof)
2. CoT steps map to template operations
3. Structure is linear or simple DAG (not fluid algebraic manipulation)

This gives us a realistic ceiling for the pipeline architecture.
"""

import json
import re
import boto3
from collections import defaultdict

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Answer type patterns
NUMERIC_PATTERNS = [
    r'^-?\d+$',                          # Integer
    r'^-?\d+\.\d+$',                      # Decimal
    r'^-?\d+/\d+$',                       # Fraction
    r'^\\frac\{-?\d+\}\{-?\d+\}$',        # LaTeX fraction
    r'^-?\d+\\frac\{-?\d+\}\{-?\d+\}$',   # Mixed number
]

EXPRESSION_PATTERNS = [
    r'^-?\d*\\?sqrt\{[^}]+\}$',           # sqrt(n) or n*sqrt(m)
    r'^\d+\\sqrt\{\d+\}$',                # 3\sqrt{13}
    r'^\\sqrt\{\d+\}$',                   # \sqrt{13}
    r'^[0-9a-z\+\-\*\/\^\(\)\s]+$',       # Simple algebraic
]

TUPLE_PATTERNS = [
    r'^\\left\s*\([^)]+\)\s*\\right\s*\)$',  # \left( ... \right)
    r'^\([^)]+,[^)]+\)$',                     # (a, b)
]

SET_PATTERNS = [
    r'^\{[^}]+\}$',                       # {a, b, c}
    r'^\\left\s*\{[^}]+\\right\s*\}$',    # \left\{ ... \right\}
]

TEXT_INDICATORS = [
    r'\\text\{',                          # \text{something}
    r'^[A-Z][a-z]+$',                     # Single capitalized word (name)
    r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday',
    r'January|February|March|April|May|June|July|August|September|October|November|December',
    r'yes|no|true|false',
    r'impossible|infinite|undefined',
]

# Template-mappable operation indicators in CoT
TEMPLATE_OPERATIONS = {
    'arithmetic': [
        r'\d+\s*[\+\-\*\/]\s*\d+',         # a + b, a - b, etc.
        r'multiply|times|product',
        r'divide|quotient|ratio',
        r'add|sum|plus|total',
        r'subtract|minus|difference',
    ],
    'power_root': [
        r'\^2|\^3|\^\d+',                  # x^n
        r'square|squared|cube|cubed',
        r'sqrt|square root',
        r'\\sqrt',
    ],
    'solve': [
        r'solve for|solving for',
        r'find [a-z](?:\s|$)',
        r'let [a-z]\s*=',
        r'[a-z]\s*=\s*\d',
    ],
    'simplify': [
        r'simplif',
        r'factor',
        r'expand',
        r'reduce',
        r'cancel',
    ],
    'substitute': [
        r'substitut',
        r'plug in|plugging in',
        r'replace .* with',
    ],
    'evaluate': [
        r'evaluat',
        r'compute|computing',
        r'calculat',
    ],
}

# Non-template operations (fluid reasoning)
FLUID_INDICATORS = [
    r'by symmetry',
    r'without loss of generality',
    r'WLOG',
    r'by induction',
    r'by contradiction',
    r'therefore|hence|thus we have',
    r'it follows that',
    r'this implies',
    r'observe that|note that|notice that',
    r'consider the case',
    r'if .* then .* else',
    r'geometric',
    r'parallel|perpendicular|angle|triangle|circle',
    r'probability|expected value|variance',
    r'permutation|combination|choose',
    r'modulo|mod \d+|\\pmod',
    r'gcd|lcm|prime|factor',
    r'matrix|determinant|eigenvalue',
    r'integral|derivative|limit',
    r'series|sequence|sum.*n=',
]


def classify_answer(answer: str) -> dict:
    """Classify answer type and tractability."""
    answer = answer.strip()

    result = {
        "raw": answer,
        "type": "unknown",
        "evaluable": False,
        "reason": None,
    }

    # Check for text indicators first
    for pattern in TEXT_INDICATORS:
        if re.search(pattern, answer, re.IGNORECASE):
            result["type"] = "text"
            result["evaluable"] = False
            result["reason"] = "contains_text"
            return result

    # Check numeric patterns
    for pattern in NUMERIC_PATTERNS:
        if re.match(pattern, answer):
            result["type"] = "numeric"
            result["evaluable"] = True
            return result

    # Check expression patterns
    for pattern in EXPRESSION_PATTERNS:
        if re.match(pattern, answer, re.IGNORECASE):
            result["type"] = "expression"
            result["evaluable"] = True
            return result

    # Check tuple patterns
    for pattern in TUPLE_PATTERNS:
        if re.match(pattern, answer):
            result["type"] = "tuple"
            result["evaluable"] = True  # Could be evaluable
            return result

    # Check set patterns
    for pattern in SET_PATTERNS:
        if re.match(pattern, answer):
            result["type"] = "set"
            result["evaluable"] = True  # Could be evaluable
            return result

    # Check if it's a degree/angle
    if re.search(r'^\d+\\?°|\\circ|degrees?$', answer, re.IGNORECASE):
        result["type"] = "angle"
        result["evaluable"] = True
        return result

    # Check if it's a simple variable or expression
    if re.match(r'^[a-z]$', answer):
        result["type"] = "variable"
        result["evaluable"] = False
        result["reason"] = "symbolic_only"
        return result

    # Complex LaTeX - might be evaluable
    if '\\' in answer:
        result["type"] = "complex_latex"
        # Check for common evaluable patterns
        if re.search(r'\\frac|\\sqrt|\\pi', answer):
            result["evaluable"] = True
        else:
            result["evaluable"] = False
            result["reason"] = "complex_latex"
        return result

    # Fallback: try to see if it looks numeric-ish
    if re.match(r'^[\d\+\-\*\/\^\(\)\.\s]+$', answer):
        result["type"] = "numeric_expr"
        result["evaluable"] = True
        return result

    result["reason"] = "unrecognized_format"
    return result


def analyze_cot_tractability(cot: str) -> dict:
    """Analyze if CoT steps are template-mappable."""
    result = {
        "n_sentences": 0,
        "template_matches": defaultdict(int),
        "fluid_matches": [],
        "template_coverage": 0.0,
        "has_fluid_reasoning": False,
    }

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', cot)
    result["n_sentences"] = len(sentences)

    template_sentences = 0

    for sent in sentences:
        sent_lower = sent.lower()

        # Check for template operations
        has_template = False
        for op_type, patterns in TEMPLATE_OPERATIONS.items():
            for pattern in patterns:
                if re.search(pattern, sent_lower):
                    result["template_matches"][op_type] += 1
                    has_template = True
                    break

        if has_template:
            template_sentences += 1

        # Check for fluid reasoning
        for pattern in FLUID_INDICATORS:
            if re.search(pattern, sent_lower):
                result["fluid_matches"].append(pattern)
                result["has_fluid_reasoning"] = True

    if result["n_sentences"] > 0:
        result["template_coverage"] = template_sentences / result["n_sentences"]

    return result


def analyze_problem(problem: dict) -> dict:
    """Full tractability analysis for a problem."""
    answer = problem.get("gold_answer", "")
    cot = problem.get("generated_cot", "")
    category = problem.get("category", "unknown")
    level = problem.get("level", "unknown")

    answer_analysis = classify_answer(answer)
    cot_analysis = analyze_cot_tractability(cot)

    # Determine tractability
    tractable = (
        answer_analysis["evaluable"] and
        cot_analysis["template_coverage"] >= 0.3 and
        not cot_analysis["has_fluid_reasoning"]
    )

    # Partial tractability (might work with improvements)
    partial = (
        answer_analysis["evaluable"] and
        (cot_analysis["template_coverage"] >= 0.2 or not cot_analysis["has_fluid_reasoning"])
    )

    return {
        "idx": problem.get("idx"),
        "category": category,
        "level": level,
        "answer": answer_analysis,
        "cot": cot_analysis,
        "tractable": tractable,
        "partial_tractable": partial,
    }


def main():
    print("="*60)
    print("MATH PROBLEM TRACTABILITY ANALYSIS")
    print("="*60)

    # Load all problems from math500_72b_final
    print("\nLoading problems...")
    paginator = s3.get_paginator("list_objects_v2")
    problem_keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix="math500_72b_final/problem_"):
        for obj in page.get("Contents", []):
            problem_keys.append(obj["Key"])

    print(f"  Found {len(problem_keys)} problems")

    # Analyze all
    results = []
    for i, key in enumerate(problem_keys):
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            problem = json.loads(resp["Body"].read().decode("utf-8"))
            analysis = analyze_problem(problem)
            results.append(analysis)
        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"  Analyzed {i+1}/{len(problem_keys)}...")

    print(f"\nAnalyzed {len(results)} problems")

    # Summary statistics
    print("\n" + "="*60)
    print("ANSWER TYPE DISTRIBUTION")
    print("="*60)

    answer_types = defaultdict(int)
    for r in results:
        answer_types[r["answer"]["type"]] += 1

    for atype, count in sorted(answer_types.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {atype:20s}: {count:4d} ({pct:5.1f}%)")

    # Evaluable answers
    evaluable = sum(1 for r in results if r["answer"]["evaluable"])
    print(f"\nEvaluable answers: {evaluable}/{len(results)} ({evaluable/len(results)*100:.1f}%)")

    # Template coverage distribution
    print("\n" + "="*60)
    print("COT TEMPLATE COVERAGE")
    print("="*60)

    coverage_buckets = defaultdict(int)
    for r in results:
        cov = r["cot"]["template_coverage"]
        if cov >= 0.5:
            coverage_buckets["high (≥50%)"] += 1
        elif cov >= 0.3:
            coverage_buckets["medium (30-50%)"] += 1
        elif cov >= 0.1:
            coverage_buckets["low (10-30%)"] += 1
        else:
            coverage_buckets["minimal (<10%)"] += 1

    for bucket, count in sorted(coverage_buckets.items()):
        pct = count / len(results) * 100
        print(f"  {bucket:20s}: {count:4d} ({pct:5.1f}%)")

    # Fluid reasoning
    has_fluid = sum(1 for r in results if r["cot"]["has_fluid_reasoning"])
    print(f"\nProblems with fluid reasoning: {has_fluid}/{len(results)} ({has_fluid/len(results)*100:.1f}%)")

    # Most common fluid patterns
    fluid_counts = defaultdict(int)
    for r in results:
        for pattern in r["cot"]["fluid_matches"]:
            fluid_counts[pattern] += 1

    print("\nTop fluid reasoning indicators:")
    for pattern, count in sorted(fluid_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")

    # Tractability
    print("\n" + "="*60)
    print("TRACTABILITY ASSESSMENT")
    print("="*60)

    tractable = sum(1 for r in results if r["tractable"])
    partial = sum(1 for r in results if r["partial_tractable"])

    print(f"\nFully tractable:     {tractable:4d} ({tractable/len(results)*100:.1f}%)")
    print(f"Partially tractable: {partial:4d} ({partial/len(results)*100:.1f}%)")
    print(f"Not tractable:       {len(results)-partial:4d} ({(len(results)-partial)/len(results)*100:.1f}%)")

    # By category
    print("\n" + "="*60)
    print("TRACTABILITY BY CATEGORY")
    print("="*60)

    by_category = defaultdict(lambda: {"total": 0, "tractable": 0, "partial": 0})
    for r in results:
        cat = r["category"]
        by_category[cat]["total"] += 1
        if r["tractable"]:
            by_category[cat]["tractable"] += 1
        if r["partial_tractable"]:
            by_category[cat]["partial"] += 1

    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["tractable"]/max(x[1]["total"],1)):
        total = stats["total"]
        tract = stats["tractable"]
        part = stats["partial"]
        print(f"  {cat:25s}: {tract:3d}/{total:3d} tractable ({tract/total*100:5.1f}%), "
              f"{part:3d} partial ({part/total*100:5.1f}%)")

    # By level
    print("\n" + "="*60)
    print("TRACTABILITY BY LEVEL")
    print("="*60)

    by_level = defaultdict(lambda: {"total": 0, "tractable": 0, "partial": 0})
    for r in results:
        lvl = str(r["level"])
        by_level[lvl]["total"] += 1
        if r["tractable"]:
            by_level[lvl]["tractable"] += 1
        if r["partial_tractable"]:
            by_level[lvl]["partial"] += 1

    for lvl, stats in sorted(by_level.items()):
        total = stats["total"]
        tract = stats["tractable"]
        part = stats["partial"]
        print(f"  Level {lvl}: {tract:3d}/{total:3d} tractable ({tract/total*100:5.1f}%), "
              f"{part:3d} partial ({part/total*100:5.1f}%)")

    # Example tractable problems
    print("\n" + "="*60)
    print("EXAMPLE TRACTABLE PROBLEMS")
    print("="*60)

    tractable_examples = [r for r in results if r["tractable"]][:5]
    for ex in tractable_examples:
        print(f"\n  Problem {ex['idx']} ({ex['category']}, Level {ex['level']}):")
        print(f"    Answer: {ex['answer']['raw'][:50]} (type: {ex['answer']['type']})")
        print(f"    Template coverage: {ex['cot']['template_coverage']*100:.0f}%")
        print(f"    Templates: {dict(ex['cot']['template_matches'])}")

    # Example non-tractable problems
    print("\n" + "="*60)
    print("EXAMPLE NON-TRACTABLE PROBLEMS")
    print("="*60)

    nontractable_examples = [r for r in results if not r["partial_tractable"]][:5]
    for ex in nontractable_examples:
        print(f"\n  Problem {ex['idx']} ({ex['category']}, Level {ex['level']}):")
        print(f"    Answer: {ex['answer']['raw'][:50]} (type: {ex['answer']['type']})")
        print(f"    Reason: {ex['answer'].get('reason', 'fluid_reasoning')}")
        if ex['cot']['fluid_matches']:
            print(f"    Fluid indicators: {ex['cot']['fluid_matches'][:3]}")

    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    tract_pct = tractable / len(results) * 100
    part_pct = partial / len(results) * 100

    if tract_pct >= 40:
        print(f"\n✓ VIABLE: {tract_pct:.0f}% tractable is a solid foundation.")
        print("  Proceed with LaTeX parsing fixes + result chaining.")
    elif part_pct >= 40:
        print(f"\n~ MARGINAL: {tract_pct:.0f}% tractable, {part_pct:.0f}% partial.")
        print("  Worth improving, but expect ~30-40% ceiling on MATH.")
    else:
        print(f"\n✗ CHALLENGING: Only {tract_pct:.0f}% tractable.")
        print("  Consider restricting to tractable subset or different architecture.")

    # Save results
    print("\nSaving analysis to S3...")
    summary = {
        "total_problems": len(results),
        "tractable": tractable,
        "partial_tractable": partial,
        "evaluable_answers": evaluable,
        "answer_type_distribution": dict(answer_types),
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "by_level": {k: dict(v) for k, v in by_level.items()},
    }

    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/tractability_analysis.json",
        Body=json.dumps(summary, indent=2).encode("utf-8")
    )
    print(f"  Saved to s3://{BUCKET}/factor_graph_eval/tractability_analysis.json")


if __name__ == "__main__":
    main()
