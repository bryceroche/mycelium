"""
Test the RecursiveDecomposer on MATH dataset (Levels 1-3).

Usage:
    uv run python scripts/test_math_levels.py [--num N] [--levels 1,2,3] [--subject algebra]
"""

import argparse
import logging
import re
from datasets import load_dataset

from mycelium.recursive_decomposer import RecursiveDecomposer


def extract_answer(solution: str) -> str:
    """Extract answer from MATH \boxed{} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    if match:
        ans = match.group(1)
        # Clean up LaTeX
        ans = ans.replace('\\frac', '').replace('{', '').replace('}', ' ').strip()
        ans = ans.replace('\\', '').replace('$', '')
        return ans
    return "N/A"


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = str(ans).strip().lower()
    # Remove trailing .0
    if ans.endswith('.0'):
        ans = ans[:-2]
    # Handle fractions like "10 11" -> try as fraction
    return ans


def answers_match(got: str, expected: str) -> bool:
    """Check if answers match (numeric, string, or symbolic)."""
    got_norm = normalize_answer(got)
    exp_norm = normalize_answer(expected)

    # Direct match
    if got_norm == exp_norm:
        return True

    # Numeric comparison
    try:
        got_float = float(got_norm)
        exp_float = float(exp_norm)
        if abs(got_float - exp_float) < 0.01:
            return True
    except:
        pass

    # Complex number comparison (handle i vs j notation)
    try:
        def parse_complex(s):
            s = str(s).strip().lower()
            s = s.replace('i', 'j').replace(' ', '')  # Normalize to Python j
            # Handle forms like "6+9j", "(6+9j)", "-11+27j"
            s = s.strip('()')
            return complex(s)

        got_complex = parse_complex(got)
        exp_complex = parse_complex(expected)
        if abs(got_complex - exp_complex) < 0.01:
            return True
    except:
        pass

    # Coordinate/tuple comparison (e.g., "(9,11)" vs "9, 11")
    try:
        import re
        def parse_coords(s):
            s = str(s).replace('(', '').replace(')', '').replace(' ', '')
            parts = re.split(r'[,;]', s)
            return tuple(float(p) for p in parts if p)

        got_coords = parse_coords(got)
        exp_coords = parse_coords(expected)
        if len(got_coords) == len(exp_coords):
            if all(abs(g - e) < 0.01 for g, e in zip(got_coords, exp_coords)):
                return True
    except:
        pass

    # Symbolic comparison using SymPy
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr

        # Normalize LaTeX to Python syntax
        def latex_to_sympy(s):
            s = s.replace('^', '**').replace('{', '(').replace('}', ')')
            s = s.replace('\\cdot', '*').replace('\\times', '*')
            s = s.replace('\\frac', '').replace('\\left', '').replace('\\right', '')
            # Handle implicit multiplication: 3x -> 3*x, x( -> x*(
            import re
            s = re.sub(r'(\d)([a-z])', r'\1*\2', s)
            s = re.sub(r'([a-z])\(', r'\1*(', s)  # x( -> x*(
            s = re.sub(r'\)([a-z])', r')*\1', s)  # )x -> )*x
            s = re.sub(r'\)\(', r')*(', s)  # )( -> )*(
            return s

        got_sympy = latex_to_sympy(str(got))
        exp_sympy = latex_to_sympy(str(expected))

        # Try to parse and compare
        got_expr = parse_expr(got_sympy)
        exp_expr = parse_expr(exp_sympy)

        # Check if expressions are equivalent
        if simplify(got_expr - exp_expr) == 0:
            return True
    except:
        pass

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=20, help="Number of problems")
    parser.add_argument("--levels", type=str, default="1,2", help="Comma-separated levels")
    parser.add_argument("--subject", type=str, default="algebra",
                       choices=['algebra', 'prealgebra', 'counting_and_probability',
                               'number_theory', 'geometry'])
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    levels = [f"Level {l}" for l in args.levels.split(',')]

    print(f"Loading MATH {args.subject} dataset...")
    ds = load_dataset('EleutherAI/hendrycks_math', args.subject, split='test')

    # Filter by level and exclude diagram problems
    problems = [x for x in ds if x['level'] in levels and '[asy]' not in x['problem']]

    print(f"\n{'='*70}")
    print(f"Testing RecursiveDecomposer on MATH {args.subject} {args.levels}")
    print(f"Found {len(problems)} problems without diagrams")
    print(f"{'='*70}\n")

    decomposer = RecursiveDecomposer()

    correct = 0
    wrong = 0
    errors = 0

    # Track failure patterns
    failures = []

    for i in range(args.start, min(args.start + args.num, len(problems))):
        item = problems[i]
        question = item["problem"]
        expected = extract_answer(item["solution"])
        level = item["level"]

        print(f"[{i+1-args.start}/{args.num}] [{level}] {question[:60]}...")
        print(f"  Expected: {expected}")

        try:
            result = decomposer.solve(question)
            print(f"  Result: {result}")

            if answers_match(str(result), expected):
                print(f"  ✓ CORRECT")
                correct += 1
            else:
                print(f"  ✗ WRONG")
                wrong += 1
                failures.append({
                    'idx': i,
                    'level': level,
                    'question': question,
                    'expected': expected,
                    'got': result
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            failures.append({
                'idx': i,
                'level': level,
                'question': question,
                'expected': expected,
                'got': f"ERROR: {e}"
            })
            if args.verbose:
                import traceback
                traceback.print_exc()

        print()

    print(f"{'='*70}")
    print(f"Results: {correct}/{args.num} correct ({100*correct/args.num:.1f}%)")
    print(f"Wrong: {wrong}, Errors: {errors}")
    print(f"{'='*70}")

    if failures and len(failures) <= 10:
        print(f"\n{'='*70}")
        print("FAILURES FOR ANALYSIS")
        print(f"{'='*70}")
        for f in failures:
            print(f"\n[{f['level']}] {f['question'][:100]}...")
            print(f"  Expected: {f['expected']}")
            print(f"  Got: {f['got']}")


if __name__ == "__main__":
    main()
