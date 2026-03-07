"""
Test Expression Extraction → SymPy Parsing → Templates

Hypothesis: The gap is PARSING, not ML. Extract mathematical expression
STRINGS from segments, parse with SymPy, feed to templates.

Current (broken):
    C3 extracts ["x", "3", "="] → template(x, 3) → ???

Should be:
    Extract "2*x + 3 = 7" → SymPy parses → Eq(2*x+3, 7) → solve → x=2
"""

import re
import json
import io
import sympy
from sympy import Symbol, Eq, solve, simplify, expand, factor, sqrt, Rational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"


def extract_math_expressions(text):
    """
    Extract mathematical expression STRINGS from text.
    Returns list of (expr_string, expr_type) tuples.
    """
    expressions = []

    # Pattern 1: LaTeX display math \[ ... \] or $$ ... $$
    for match in re.finditer(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', text, re.DOTALL):
        expr = match.group(1) or match.group(2)
        if expr:
            expressions.append((expr.strip(), "display"))

    # Pattern 2: Inline LaTeX $ ... $
    for match in re.finditer(r'\$([^$]+)\$', text):
        expr = match.group(1).strip()
        if len(expr) > 1:  # Skip single characters
            expressions.append((expr, "inline"))

    # Pattern 3: Equations with = sign (not in LaTeX)
    for match in re.finditer(r'([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)\s*=\s*([a-zA-Z0-9\s\+\-\*\/\^\(\)]+)', text):
        lhs, rhs = match.groups()
        if lhs.strip() and rhs.strip():
            expressions.append((f"{lhs.strip()} = {rhs.strip()}", "equation"))

    return expressions


def latex_to_sympy_expr(latex_str):
    """Convert LaTeX string to SymPy expression."""
    s = latex_str

    # Clean LaTeX
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('\\div', '/')

    # Fractions
    for _ in range(5):
        new_s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        if new_s == s:
            break
        s = new_s

    # Square roots
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)

    # Powers
    s = re.sub(r'\^{([^{}]+)}', r'**(\1)', s)
    s = re.sub(r'\^(\d+)', r'**\1', s)

    # Greek letters
    s = s.replace('\\pi', 'pi')
    s = s.replace('\\theta', 'theta')
    s = s.replace('\\alpha', 'alpha')

    # Text removal
    s = re.sub(r'\\text\{[^}]*\}', '', s)

    # Clean up
    s = re.sub(r'\s+', ' ', s).strip()

    # Handle implicit multiplication: 2x → 2*x
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

    return s


def parse_math_expression(expr_str, expr_type="unknown"):
    """
    Parse a math expression string into SymPy object(s).
    Returns (parsed_object, object_type) or (None, error).
    """
    try:
        # Convert LaTeX to parseable form
        s = latex_to_sympy_expr(expr_str)

        # Check if it's an equation
        if '=' in s and expr_type != "inequality":
            parts = s.split('=')
            if len(parts) == 2:
                lhs = parse_expr(parts[0].strip(),
                               transformations=standard_transformations + (implicit_multiplication,))
                rhs = parse_expr(parts[1].strip(),
                               transformations=standard_transformations + (implicit_multiplication,))
                return Eq(lhs, rhs), "equation"

        # Otherwise parse as expression
        parsed = parse_expr(s, transformations=standard_transformations + (implicit_multiplication,))
        return parsed, "expression"

    except Exception as e:
        return None, f"parse_error: {e}"


def apply_template_to_parsed(parsed_obj, obj_type):
    """
    Apply appropriate template to parsed mathematical object.
    Returns (success, result, template_used).
    """
    if parsed_obj is None:
        return False, None, "no_object"

    if obj_type == "equation":
        # Try to solve
        eq = parsed_obj
        free_syms = list(eq.free_symbols)

        if free_syms:
            try:
                solutions = solve(eq, free_syms[0])
                if solutions:
                    return True, solutions[0], "solve"
            except:
                pass

        # Try simplify
        try:
            result = simplify(eq)
            return True, result, "simplify"
        except:
            pass

    else:  # expression
        # Try simplify
        try:
            result = simplify(parsed_obj)
            if result != parsed_obj:
                return True, result, "simplify"
        except:
            pass

        # Try evaluate if it's numeric
        try:
            if parsed_obj.is_number:
                return True, parsed_obj, "evaluate"
        except:
            pass

        # Return as-is
        return True, parsed_obj, "identity"

    return False, None, "no_template"


def process_problem(problem):
    """
    Process a single problem:
    1. Extract math expressions from CoT
    2. Parse with SymPy
    3. Apply templates
    4. Check if final answer matches
    """
    cot = problem.get("generated_cot", "")
    expected = problem.get("gold_answer", "")

    result = {
        "problem_id": problem.get("idx"),
        "expressions_found": 0,
        "expressions_parsed": 0,
        "templates_applied": 0,
        "results": [],
        "final_result": None,
        "expected": expected,
        "correct": False,
    }

    # Extract expressions
    expressions = extract_math_expressions(cot)
    result["expressions_found"] = len(expressions)

    # Process each expression
    for expr_str, expr_type in expressions:
        parsed, parse_result = parse_math_expression(expr_str, expr_type)

        if parsed is not None:
            result["expressions_parsed"] += 1

            success, template_result, template_name = apply_template_to_parsed(parsed, parse_result)

            if success:
                result["templates_applied"] += 1
                result["results"].append({
                    "expr": expr_str[:50],
                    "template": template_name,
                    "result": str(template_result)[:50],
                })
                result["final_result"] = template_result

    # Check correctness
    if result["final_result"] is not None and expected:
        try:
            # Parse expected answer
            expected_parsed, _ = parse_math_expression(expected)
            if expected_parsed is not None:
                diff = simplify(result["final_result"] - expected_parsed)
                if diff == 0:
                    result["correct"] = True
        except:
            pass

    return result


def load_problems(n=20):
    """Load MATH problems."""
    print(f"Loading {n} problems...")

    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix="math500_72b_final/problem_"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= n:
                break
        if len(keys) >= n:
            break

    problems = []
    for key in keys[:n]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            p = json.loads(resp["Body"].read().decode("utf-8"))
            if p.get("generated_cot") and p.get("gold_answer"):
                problems.append(p)
        except:
            continue

    print(f"  Loaded {len(problems)} problems")
    return problems


def main():
    print("="*60)
    print("EXPRESSION EXTRACTION + SYMPY PARSING TEST")
    print("="*60)
    print("\nHypothesis: Extract math expression STRINGS, parse with SymPy,")
    print("feed properly-typed objects to templates.")

    # Load problems
    problems = load_problems(50)

    # Process
    print(f"\nProcessing {len(problems)} problems...")

    results = []
    correct = 0

    for i, problem in enumerate(problems):
        result = process_problem(problem)
        results.append(result)
        if result["correct"]:
            correct += 1

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    total = len(results)
    avg_found = sum(r["expressions_found"] for r in results) / total
    avg_parsed = sum(r["expressions_parsed"] for r in results) / total
    avg_applied = sum(r["templates_applied"] for r in results) / total
    pct_correct = correct / total * 100

    print(f"\nProblems: {total}")
    print(f"Avg expressions found per problem: {avg_found:.1f}")
    print(f"Avg expressions parsed: {avg_parsed:.1f}")
    print(f"Avg templates applied: {avg_applied:.1f}")
    print(f"\nCorrect answers: {correct} ({pct_correct:.1f}%)")

    # Correct examples
    correct_examples = [r for r in results if r["correct"]][:5]
    if correct_examples:
        print(f"\nCorrect examples:")
        for ex in correct_examples:
            print(f"\n  Problem {ex['problem_id']}:")
            print(f"    Expected: {ex['expected']}")
            print(f"    Got: {ex['final_result']}")
            if ex['results']:
                print(f"    Steps: {len(ex['results'])}")
                for step in ex['results'][-2:]:
                    print(f"      {step['template']}: {step['expr'][:30]}... → {step['result']}")

    # Failure analysis
    failed = [r for r in results if not r["correct"] and r["expressions_found"] > 0][:5]
    if failed:
        print(f"\nFailure examples:")
        for ex in failed:
            print(f"\n  Problem {ex['problem_id']}:")
            print(f"    Expected: {ex['expected']}")
            print(f"    Got: {ex['final_result']}")
            print(f"    Expressions found: {ex['expressions_found']}")
            print(f"    Parsed: {ex['expressions_parsed']}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    print(f"\nGreedy baseline (token extraction): 3.8% correct")
    print(f"Expression extraction + SymPy: {pct_correct:.1f}% correct")

    if pct_correct > 10:
        print(f"\n✓ SUCCESS: Expression extraction works!")
        print("  The gap was PARSING, not ML.")
        print("  C3 should become span extractor for math expressions.")
    elif pct_correct > 5:
        print(f"\n~ PARTIAL: Some improvement")
        print("  Expression extraction helps but needs refinement.")
    else:
        print(f"\n✗ NOT ENOUGH: Expression extraction alone isn't sufficient")
        print("  May need more sophisticated math parsing.")

    # Save
    print("\nSaving results...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/expression_extraction_test.json",
        Body=json.dumps({
            "pct_correct": pct_correct,
            "avg_expressions_found": avg_found,
            "avg_expressions_parsed": avg_parsed,
            "correct_examples": [r for r in results if r["correct"]][:10],
        }, indent=2, default=str).encode("utf-8")
    )


if __name__ == "__main__":
    main()
