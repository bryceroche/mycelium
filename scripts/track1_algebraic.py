#!/usr/bin/env python3
"""
Track 1 Combined: High-precision patterns first, brute force fallback.

Order:
1. Function composition (100% precision)
2. Substitution (85% precision)
3. Equation solving with sum/avg/product detection
4. Brute force eval (with filtering)
"""

import json
import re
import math
from tqdm import tqdm
import argparse
from functools import reduce
import operator

from sympy.parsing.latex import parse_latex
from sympy import N, ceiling, floor, binomial, Symbol, solve, Eq, Piecewise
from sympy import simplify


def clean_latex(expr):
    expr = re.sub(r'\\left\s*', '', expr)
    expr = re.sub(r'\\right\s*', '', expr)
    expr = re.sub(r'\\times', '*', expr)
    expr = re.sub(r'\\cdot', '*', expr)
    expr = re.sub(r'(\d)\(', r'\1*(', expr)
    expr = re.sub(r'\\text\{[^}]*\}', '', expr)
    expr = re.sub(r'\\textbf\{[^}]*\}', '', expr)
    expr = re.sub(r'\$', '', expr)
    expr = re.sub(r'(\d),(\d{3})', r'\1\2', expr)
    return expr.strip()


def is_simple_assignment(expr_str):
    """Check if this is just a simple assignment like x=2."""
    expr_str = expr_str.strip()
    return bool(re.match(r'^[a-zA-Z]\s*=\s*-?\d+\.?\d*$', expr_str))


def extract_all_latex(text):
    """Extract all LaTeX expressions."""
    expressions = []
    for m in re.finditer(r'\$([^\$]+)\$', text):
        expressions.append(m.group(1))
    for m in re.finditer(r'\\\((.+?)\\\)', text, re.DOTALL):
        expressions.append(m.group(1))
    for m in re.finditer(r'\\\[(.+?)\\\]', text, re.DOTALL):
        expressions.append(m.group(1))
    seen = set()
    result = []
    for e in expressions:
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            result.append(e)
    return result


def wants_sum(text):
    return bool(re.search(r'sum\s+of.*(?:values|solutions|roots)', text, re.I))

def wants_product(text):
    return bool(re.search(r'product\s+of.*(?:values|solutions|roots)', text, re.I))

def wants_average(text):
    return bool(re.search(r'(?:average|mean)\s+of.*(?:values|solutions)', text, re.I))


# ============== HIGH-PRECISION PATTERNS ==============

def extract_function_definitions(text):
    """Extract f(x) = expr definitions."""
    definitions = {}
    patterns = [
        r'\$([a-zA-Z])\s*\(\s*([a-zA-Z])\s*\)\s*=\s*([^$]+)\$',
        r'([a-zA-Z])\s*\(\s*([a-zA-Z])\s*\)\s*=\s*([^,\.\n]+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            func_name = match.group(1)
            var_name = match.group(2)
            expr_str = match.group(3).strip().rstrip('.')
            if func_name not in definitions:
                definitions[func_name] = (var_name, expr_str)
    return definitions


def extract_composition_target(text):
    """Extract g(f(2)) type targets."""
    patterns = [
        r'(?:find|what is|compute|evaluate)\s+\$?([a-zA-Z])\s*\(\s*([a-zA-Z])\s*\(\s*(-?\d+\.?\d*)\s*\)\s*\)\$?',
        r'\$([a-zA-Z])\s*\(\s*([a-zA-Z])\s*\(\s*(-?\d+\.?\d*)\s*\)\s*\)\$?\s*[=\?]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1), match.group(2), float(match.group(3))
    return None, None, None


def try_composition(text):
    """Try function composition."""
    try:
        definitions = extract_function_definitions(text)
        outer, inner, value = extract_composition_target(text)
        if not outer or not inner:
            return None, None
        if outer not in definitions or inner not in definitions:
            return None, None

        inner_var, inner_expr = definitions[inner]
        outer_var, outer_expr = definitions[outer]

        x = Symbol(inner_var)
        inner_parsed = parse_latex(clean_latex(inner_expr))
        inner_result = float(N(inner_parsed.subs(x, value)))

        y = Symbol(outer_var)
        outer_parsed = parse_latex(clean_latex(outer_expr))
        result = float(N(outer_parsed.subs(y, inner_result)))

        if math.isnan(result) or math.isinf(result):
            return None, None
        return result, 'composition'
    except:
        return None, None


def extract_assignments(text):
    """Extract variable assignments."""
    assignments = {}
    patterns = [
        r'\$([a-zA-Z])\s*=\s*(-?\d+\.?\d*)\$',
        r'(?<![a-zA-Z])([a-zA-Z])\s*=\s*(-?\d+\.?\d*)(?![a-zA-Z])',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            var, val = match.group(1), match.group(2)
            try:
                assignments[var] = float(val)
            except:
                pass
    return assignments


def extract_target_expression(text):
    """Find the expression we're asked to evaluate."""
    patterns = [
        r'(?:find|compute|calculate|evaluate|what is|determine)\s+(?:the\s+)?(?:value\s+of\s+)?\$([^$]+)\$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            expr = match.group(1)
            if len(expr) > 1 and re.search(r'[\+\-\*/\^]|frac|sqrt', expr):
                return expr
    return None


def try_substitution(text):
    """Try variable substitution."""
    try:
        assignments = extract_assignments(text)
        if not assignments:
            return None, None

        target = extract_target_expression(text)
        if target:
            parsed = parse_latex(clean_latex(target))
            for var, val in assignments.items():
                parsed = parsed.subs(Symbol(var), val)
            result = float(N(parsed))
            if not math.isnan(result) and not math.isinf(result):
                return result, 'substitution'

        # Try all expressions
        for expr in extract_all_latex(text):
            if len(expr) > 3 and not is_simple_assignment(expr) and '=' not in expr:
                if re.search(r'[\+\-\*/\^]|frac|sqrt', expr):
                    try:
                        parsed = parse_latex(clean_latex(expr))
                        for var, val in assignments.items():
                            parsed = parsed.subs(Symbol(var), val)
                        result = float(N(parsed))
                        if not math.isnan(result) and not math.isinf(result):
                            return result, 'substitution'
                    except:
                        pass
        return None, None
    except:
        return None, None


def try_equation_solve(text):
    """Try solving equations."""
    try:
        equations = []
        for match in re.finditer(r'\$([^$]*=[^$]*)\$', text):
            eq = match.group(1)
            if not is_simple_assignment(eq):
                equations.append(eq)

        for eq_str in equations:
            cleaned = clean_latex(eq_str)
            if '=' not in cleaned:
                continue

            parts = cleaned.split('=')
            if len(parts) != 2:
                continue

            lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
            if not lhs_str or not rhs_str:
                continue

            try:
                lhs = parse_latex(lhs_str)
                rhs = parse_latex(rhs_str)
                free = lhs.free_symbols | rhs.free_symbols

                if len(free) == 1:
                    var = list(free)[0]
                    eq = Eq(lhs, rhs)
                    solutions = solve(eq, var)

                    if solutions:
                        if wants_sum(text):
                            return sum(float(N(s)) for s in solutions), 'solve_sum'
                        elif wants_average(text):
                            return sum(float(N(s)) for s in solutions) / len(solutions), 'solve_avg'
                        elif wants_product(text):
                            return reduce(operator.mul, [float(N(s)) for s in solutions], 1), 'solve_prod'
                        else:
                            result = float(N(solutions[0]))
                            if not math.isnan(result) and not math.isinf(result):
                                return result, 'solve'
            except:
                pass

        return None, None
    except:
        return None, None


def try_direct_eval(text):
    """Try direct evaluation - brute force but filtered."""
    try:
        expressions = extract_all_latex(text)

        for expr in expressions:
            # Skip simple assignments
            if is_simple_assignment(expr):
                continue

            # Skip if has variables
            cleaned = clean_latex(expr)
            if not cleaned or len(cleaned) < 2:
                continue

            # Skip single letter
            if re.match(r'^[a-zA-Z]$', cleaned):
                continue

            # Skip if has free variables
            try:
                parsed = parse_latex(cleaned)
                if parsed.free_symbols:
                    continue

                result = float(N(parsed))
                if math.isnan(result) or math.isinf(result):
                    continue

                return result, 'eval'
            except:
                pass

        return None, None
    except:
        return None, None


def extract_gold(answer_str):
    if not answer_str:
        return None
    answer_str = str(answer_str)
    match = re.search(r'\\boxed\{([^}]+)\}', answer_str)
    if match:
        answer_str = match.group(1)
    try:
        result = float(N(parse_latex(clean_latex(answer_str))))
        if not math.isnan(result) and not math.isinf(result):
            return result
    except:
        pass
    frac = re.search(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', answer_str)
    if frac:
        n, d = float(frac.group(1)), float(frac.group(2))
        if d != 0:
            return n / d
    num = re.search(r'-?\d+\.?\d*', answer_str)
    return float(num.group()) if num else None


def answers_match(pred, gold, tol=0.01):
    if pred is None or gold is None:
        return False
    if abs(pred - gold) < 1e-9:
        return True
    if gold != 0 and abs(pred - gold) / abs(gold) < tol:
        return True
    return abs(pred - gold) < tol


def solve_combined(text):
    """Combined solver: high-precision first, brute force fallback."""

    # 1. Function composition (100% precision)
    result, method = try_composition(text)
    if result is not None:
        return result, method

    # 2. Substitution (85% precision)
    result, method = try_substitution(text)
    if result is not None:
        return result, method

    # 3. Equation solving (60% precision)
    result, method = try_equation_solve(text)
    if result is not None:
        return result, method

    # 4. Direct evaluation (filtered brute force)
    result, method = try_direct_eval(text)
    if result is not None:
        return result, method

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading {args.test_file}...")
    problems = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    problems = problems[:args.limit]
    print(f"Loaded {len(problems)} problems")

    total = 0
    stats = {}
    correct_stats = {}
    no_parse = 0
    correct = 0

    print("\nEvaluating combined solver...")

    for prob in tqdm(problems):
        question = prob.get('question', prob.get('problem', ''))
        gold = extract_gold(prob.get('answer', ''))

        if gold is None:
            continue

        total += 1
        pred, method = solve_combined(question)

        if pred is not None and method:
            stats[method] = stats.get(method, 0) + 1
            if answers_match(pred, gold):
                correct += 1
                correct_stats[method] = correct_stats.get(method, 0) + 1
        else:
            no_parse += 1

    total_attempted = sum(stats.values())

    print(f"\n{'='*60}")
    print(f"COMBINED SOLVER RESULTS")
    print(f"{'='*60}")
    print(f"Total: {total}")

    for method in sorted(stats.keys()):
        c = correct_stats.get(method, 0)
        t = stats[method]
        prec = 100*c/t if t > 0 else 0
        print(f"  {method}: {c}/{t} ({prec:.1f}% precision)")

    print(f"\nNO PARSE: {no_parse} ({100*no_parse/total:.1f}%)")
    print(f"\n*** COVERAGE: {total_attempted}/{total} = {100*total_attempted/total:.1f}% ***")
    print(f"*** ACCURACY: {correct}/{total} = {100*correct/total:.1f}% ***")


if __name__ == "__main__":
    main()
