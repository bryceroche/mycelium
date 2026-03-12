"""
Build Notation MLP training data.

The canonicalizer outputs rough math notation (2x+3, x^2, sin30).
SymPy's strict parser rejects these. SymPy's relaxed parser accepts them.
The Notation MLP learns the mechanical transformation between the two.

Method:
  1. Run all telegrams through strict SymPy parser
  2. Run failures through relaxed parser (transformations="all")
  3. For each (strict_fail, relaxed_pass): extract the transformation
  4. Build (rough_notation → precise_notation) training pairs

This is NOT regex. It's a learned model. But it learns a simple, mechanical
function — SymPy's parser expectations — so it can be tiny.

Usage:
    python build_notation_data.py \
        --telegrams telegrams_validated.jsonl \
        --output notation_training.jsonl
"""

import argparse
import json
import re
import sys
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# SymPy parsing utilities
# ─────────────────────────────────────────────────────────────

def try_strict_parse(expr_str: str) -> Tuple[bool, Optional[str]]:
    """Try parsing with SymPy's strict parser. Returns (success, parsed_str)."""
    try:
        from sympy import sympify
        result = sympify(expr_str, evaluate=False)
        return True, str(result)
    except Exception:
        return False, None


def try_relaxed_parse(expr_str: str) -> Tuple[bool, Optional[str]]:
    """Try parsing with SymPy's relaxed parser (implicit multiplication etc.)."""
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            implicit_application,
            convert_xor,
        )
        transformations = (
            standard_transformations +
            (implicit_multiplication_application,) +
            (implicit_application,) +
            (convert_xor,)
        )
        result = parse_expr(expr_str, transformations=transformations, evaluate=False)
        return True, str(result)
    except Exception:
        return False, None


def try_parse_latex(expr_str: str) -> Tuple[bool, Optional[str]]:
    """Try SymPy's LaTeX parser as a fallback."""
    try:
        from sympy.parsing.latex import parse_latex
        result = parse_latex(expr_str)
        return True, str(result)
    except Exception:
        return False, None


# ─────────────────────────────────────────────────────────────
# Extract math expressions from telegrams
# ─────────────────────────────────────────────────────────────

VERBS = {"GIVEN", "EVAL", "SOLVE", "EXPAND", "SIMPLIFY", "SUBS", "APPLY", "ANSWER"}


def extract_expressions(telegram: str) -> List[Dict]:
    """
    Extract parseable math expressions from a telegram line.
    Returns list of {expr, context, verb, position}.
    """
    parts = telegram.strip().split(None, 1)
    if not parts:
        return []

    verb = parts[0].upper()
    if verb not in VERBS:
        return []

    args = parts[1] if len(parts) > 1 else ""
    if not args:
        return []

    expressions = []

    if verb == "GIVEN":
        # Could be: equation (x^2+y^2=90) or assignment (a=5)
        # Split on space to handle multiple assignments
        for sub_expr in args.split():
            if sub_expr and sub_expr != "_prev":
                # Handle equations: split on = and parse each side
                if "=" in sub_expr:
                    sides = sub_expr.split("=", 1)
                    for side in sides:
                        if side.strip():
                            expressions.append({
                                "expr": side.strip(),
                                "context": "given_side",
                                "verb": verb,
                                "full_telegram": telegram.strip(),
                            })
                else:
                    expressions.append({
                        "expr": sub_expr,
                        "context": "given_value",
                        "verb": verb,
                        "full_telegram": telegram.strip(),
                    })

    elif verb in ("EVAL", "EXPAND", "SIMPLIFY"):
        # The whole argument is one expression
        expr = args.strip()
        if expr and expr != "_prev":
            expressions.append({
                "expr": expr,
                "context": verb.lower(),
                "verb": verb,
                "full_telegram": telegram.strip(),
            })

    elif verb == "SOLVE":
        # SOLVE expr var → parse expr
        solve_parts = args.split()
        if solve_parts:
            # Everything except the last token (which is the variable)
            if len(solve_parts) > 1:
                expr = " ".join(solve_parts[:-1])
            else:
                expr = solve_parts[0]
            if expr != "_prev":
                expressions.append({
                    "expr": expr,
                    "context": "solve_equation",
                    "verb": verb,
                    "full_telegram": telegram.strip(),
                })

    elif verb == "SUBS":
        # SUBS target old new → parse each
        subs_parts = args.split()
        for sp in subs_parts:
            if sp and sp != "_prev":
                expressions.append({
                    "expr": sp,
                    "context": "subs_arg",
                    "verb": verb,
                    "full_telegram": telegram.strip(),
                })

    elif verb == "ANSWER":
        if args.strip() and args.strip() != "_prev":
            expressions.append({
                "expr": args.strip(),
                "context": "answer",
                "verb": verb,
                "full_telegram": telegram.strip(),
            })

    return expressions


# ─────────────────────────────────────────────────────────────
# Build training pairs
# ─────────────────────────────────────────────────────────────

def build_notation_pairs(telegrams_path: str, output_path: str):
    """
    For each expression in each telegram:
      - Try strict parse
      - If fails, try relaxed parse
      - If relaxed succeeds: (rough_expr, precise_expr) is a training pair
      - If both succeed: (expr, expr) is an identity pair (already correct)
    """

    # Load telegrams
    telegrams = []
    with open(telegrams_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("valid", True):
                telegrams.append(item)

    print(f"Loaded {len(telegrams)} telegrams")

    # Extract all expressions
    all_expressions = []
    for t in telegrams:
        telegram_text = t.get("telegram", "")
        exprs = extract_expressions(telegram_text)
        for expr_info in exprs:
            expr_info["problem_id"] = t.get("problem_id")
            all_expressions.append(expr_info)

    print(f"Extracted {len(all_expressions)} expressions")

    # Parse each expression with both parsers
    training_pairs = []
    stats = {
        "strict_pass": 0,
        "strict_fail_relaxed_pass": 0,
        "both_fail": 0,
        "empty_or_prev": 0,
    }

    # Track unique transformations
    transformation_counts = {}

    for expr_info in all_expressions:
        raw = expr_info["expr"]

        if not raw or raw == "_prev":
            stats["empty_or_prev"] += 1
            continue

        strict_ok, strict_result = try_strict_parse(raw)

        if strict_ok:
            # Already valid — identity pair (the model should learn to pass through)
            stats["strict_pass"] += 1
            training_pairs.append({
                "input": raw,
                "output": raw,  # identity — already correct
                "strict_parsed": strict_result,
                "transformation": "identity",
                "context": expr_info["context"],
                "verb": expr_info["verb"],
                "problem_id": expr_info.get("problem_id"),
            })
        else:
            # Try relaxed
            relaxed_ok, relaxed_result = try_relaxed_parse(raw)

            if relaxed_ok:
                # This is the gold: rough notation that relaxed parser fixes
                stats["strict_fail_relaxed_pass"] += 1

                # The "precise" version is what SymPy's relaxed parser produces
                # We need to convert back to string form that strict parser accepts
                precise = relaxed_result

                # Identify the transformation type
                transform_type = identify_transformation(raw, precise)
                transformation_counts[transform_type] = \
                    transformation_counts.get(transform_type, 0) + 1

                training_pairs.append({
                    "input": raw,
                    "output": precise,
                    "strict_parsed": None,
                    "relaxed_parsed": relaxed_result,
                    "transformation": transform_type,
                    "context": expr_info["context"],
                    "verb": expr_info["verb"],
                    "problem_id": expr_info.get("problem_id"),
                })
            else:
                # Both fail — try LaTeX parser as last resort
                latex_ok, latex_result = try_parse_latex(raw)
                if latex_ok:
                    stats["strict_fail_relaxed_pass"] += 1  # count with relaxed
                    training_pairs.append({
                        "input": raw,
                        "output": latex_result,
                        "strict_parsed": None,
                        "relaxed_parsed": None,
                        "latex_parsed": latex_result,
                        "transformation": "latex_parse",
                        "context": expr_info["context"],
                        "verb": expr_info["verb"],
                        "problem_id": expr_info.get("problem_id"),
                    })
                else:
                    stats["both_fail"] += 1
                    training_pairs.append({
                        "input": raw,
                        "output": None,
                        "transformation": "unparseable",
                        "context": expr_info["context"],
                        "verb": expr_info["verb"],
                        "problem_id": expr_info.get("problem_id"),
                    })

    # Save
    with open(output_path, "w") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + "\n")

    # Stats
    total = sum(stats.values())
    print(f"\nParsing results:")
    print(f"  Strict pass (identity):        {stats['strict_pass']} "
          f"({100*stats['strict_pass']/max(total,1):.1f}%)")
    print(f"  Strict fail, relaxed pass:     {stats['strict_fail_relaxed_pass']} "
          f"({100*stats['strict_fail_relaxed_pass']/max(total,1):.1f}%)")
    print(f"  Both fail (unparseable):       {stats['both_fail']} "
          f"({100*stats['both_fail']/max(total,1):.1f}%)")
    print(f"  Empty/_prev (skipped):         {stats['empty_or_prev']}")

    # Trainable pairs (identity + fixable)
    trainable = [p for p in training_pairs if p["output"] is not None]
    fixable = [p for p in training_pairs
               if p["transformation"] not in ("identity", "unparseable")]
    print(f"\nTraining data:")
    print(f"  Total pairs:     {len(trainable)} (identity + fixable)")
    print(f"  Identity pairs:  {stats['strict_pass']} (already correct)")
    print(f"  Fixable pairs:   {len(fixable)} (need notation fix)")
    print(f"  Unparseable:     {stats['both_fail']} (excluded)")

    # Transformation distribution
    if transformation_counts:
        print(f"\nTransformation types:")
        for ttype, count in sorted(transformation_counts.items(), key=lambda x: -x[1]):
            print(f"  {ttype}: {count}")

    # Show examples of each transformation type
    print(f"\nExample transformations:")
    shown_types = set()
    for pair in training_pairs:
        ttype = pair["transformation"]
        if ttype not in shown_types and ttype not in ("identity", "unparseable"):
            print(f"  [{ttype}]")
            print(f"    {pair['input']:30s} → {pair['output']}")
            shown_types.add(ttype)
            if len(shown_types) >= 10:
                break

    return training_pairs


def identify_transformation(raw: str, precise: str) -> str:
    """Identify what transformation the relaxed parser applied."""
    if raw == precise:
        return "identity"

    # Implicit multiplication: 2x → 2*x
    if re.search(r'\d[a-zA-Z]', raw):
        return "implicit_multiplication"

    # Caret to power: x^2 → x**2
    if '^' in raw and '**' in precise:
        return "caret_to_power"

    # Trig without parens: sin30 → sin(30)
    if re.search(r'(sin|cos|tan|log|sqrt)\d', raw):
        return "implicit_function_application"

    # Fraction notation
    if '/' in raw and 'Rational' in precise:
        return "fraction_to_rational"

    # General catch-all
    return "other"


# ─────────────────────────────────────────────────────────────
# Also: generate synthetic notation pairs for augmentation
# ─────────────────────────────────────────────────────────────

def generate_synthetic_pairs(output_path: str, n: int = 5000):
    """
    Generate synthetic rough → precise pairs for common patterns.
    Augments the empirical pairs extracted above.
    """
    import random

    pairs = []
    variables = list("xyzabcnkmrt")
    digits = list("123456789")

    for _ in range(n):
        pattern = random.choice([
            "implicit_mult", "caret", "trig", "trig_pi",
            "implicit_mult_multi", "sqrt_no_parens",
        ])

        if pattern == "implicit_mult":
            # 2x → 2*x, 3y → 3*y
            coeff = random.choice(digits)
            var = random.choice(variables)
            pairs.append({
                "input": f"{coeff}{var}",
                "output": f"{coeff}*{var}",
                "transformation": "implicit_multiplication",
                "synthetic": True,
            })

        elif pattern == "implicit_mult_multi":
            # 2xy → 2*x*y, 3ab → 3*a*b
            coeff = random.choice(digits)
            v1 = random.choice(variables)
            v2 = random.choice([v for v in variables if v != v1])
            pairs.append({
                "input": f"{coeff}{v1}{v2}",
                "output": f"{coeff}*{v1}*{v2}",
                "transformation": "implicit_multiplication",
                "synthetic": True,
            })

        elif pattern == "caret":
            # x^2 → x**2, y^3 → y**3
            var = random.choice(variables)
            exp = random.choice(["2", "3", "4", "-1", "1/2", "1/3"])
            pairs.append({
                "input": f"{var}^{exp}",
                "output": f"{var}**({exp})" if "/" in exp or "-" in exp else f"{var}**{exp}",
                "transformation": "caret_to_power",
                "synthetic": True,
            })

        elif pattern == "trig":
            # sin30 → sin(30), cos45 → cos(45)
            func = random.choice(["sin", "cos", "tan"])
            angle = random.choice(["30", "45", "60", "90", "120", "180"])
            pairs.append({
                "input": f"{func}{angle}",
                "output": f"{func}({angle})",
                "transformation": "implicit_function_application",
                "synthetic": True,
            })

        elif pattern == "trig_pi":
            # sin30 → sin(pi/6) (degree to radian)
            func = random.choice(["sin", "cos", "tan"])
            angle_map = {"30": "pi/6", "45": "pi/4", "60": "pi/3",
                         "90": "pi/2", "120": "2*pi/3", "180": "pi"}
            deg = random.choice(list(angle_map.keys()))
            pairs.append({
                "input": f"{func}{deg}",
                "output": f"{func}({angle_map[deg]})",
                "transformation": "trig_degree_to_radian",
                "synthetic": True,
            })

        elif pattern == "sqrt_no_parens":
            # sqrt2 → sqrt(2), sqrt3 → sqrt(3)
            num = random.choice(digits + ["10", "12", "15", "27"])
            pairs.append({
                "input": f"sqrt{num}",
                "output": f"sqrt({num})",
                "transformation": "implicit_function_application",
                "synthetic": True,
            })

    synth_path = output_path.replace(".jsonl", "_synthetic.jsonl")
    with open(synth_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nGenerated {len(pairs)} synthetic notation pairs → {synth_path}")

    # Distribution
    type_counts = {}
    for p in pairs:
        t = p["transformation"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    return pairs


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build notation MLP training data")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("extract", help="Extract pairs from real telegrams")
    p1.add_argument("--telegrams", required=True)
    p1.add_argument("--output", default="notation_training.jsonl")

    p2 = sub.add_parser("synthetic", help="Generate synthetic augmentation pairs")
    p2.add_argument("--output", default="notation_training.jsonl")
    p2.add_argument("--n", type=int, default=5000)

    p3 = sub.add_parser("both", help="Extract real + generate synthetic")
    p3.add_argument("--telegrams", required=True)
    p3.add_argument("--output", default="notation_training.jsonl")
    p3.add_argument("--n-synthetic", type=int, default=5000)

    args = parser.parse_args()

    if args.command == "extract":
        build_notation_pairs(args.telegrams, args.output)
    elif args.command == "synthetic":
        generate_synthetic_pairs(args.output, args.n)
    elif args.command == "both":
        print("═══ Phase 1: Extract from real telegrams ═══")
        build_notation_pairs(args.telegrams, args.output)
        print("\n═══ Phase 2: Generate synthetic pairs ═══")
        generate_synthetic_pairs(args.output, args.n_synthetic)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
