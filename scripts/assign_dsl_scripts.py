#!/usr/bin/env python3
"""Assign DSL scripts to all signatures based on step type.

This script populates DSL scripts for signatures that don't have them,
using predefined templates based on step_type.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures.db import StepSignatureDB

# Step types that should use guidance-only (no computational DSL)
GUIDANCE_ONLY_TYPES = {
    "synthesize_results",
    "setup_equation",
    "define_variables",
    "define_formula",
    "express_relation",
    "apply_amgm",
    "general_step",  # Too generic, relies on routing to children
}

# DSL templates by step type
DSL_TEMPLATES = {
    # Arithmetic operations
    "compute_sum": {"type": "math", "script": "a + b", "params": ["a", "b"], "fallback": "guidance"},
    "compute_difference": {"type": "math", "script": "a - b", "params": ["a", "b"], "fallback": "guidance"},
    "compute_product": {"type": "math", "script": "a * b", "params": ["a", "b"], "fallback": "guidance"},
    "compute_quotient": {"type": "math", "script": "a / b", "params": ["a", "b"], "fallback": "guidance"},
    "compute_power": {"type": "math", "script": "base ** exponent", "params": ["base", "exponent"], "fallback": "guidance"},
    "compute_remainder": {"type": "math", "script": "a % b", "params": ["a", "b"], "fallback": "guidance"},
    "compute_gcd": {"type": "math", "script": "gcd(a, b)", "params": ["a", "b"], "fallback": "guidance"},
    "compute_lcm": {"type": "math", "script": "(a * b) // gcd(a, b)", "params": ["a", "b"], "fallback": "guidance"},
    "compute_factorial": {"type": "math", "script": "factorial(n)", "params": ["n"], "fallback": "guidance"},
    "compute_sqrt": {"type": "math", "script": "sqrt(n)", "params": ["n"], "fallback": "guidance"},
    "compute_absolute": {"type": "math", "script": "abs(x)", "params": ["x"], "fallback": "guidance"},

    # Geometry - lengths and distances
    "compute_length": {"type": "math", "script": "sqrt((x2-x1)**2 + (y2-y1)**2)", "params": ["x1", "y1", "x2", "y2"], "fallback": "guidance"},
    "compute_distance": {"type": "math", "script": "sqrt((x2-x1)**2 + (y2-y1)**2)", "params": ["x1", "y1", "x2", "y2"], "fallback": "guidance"},
    "compute_radius": {"type": "math", "script": "diameter / 2", "params": ["diameter"], "fallback": "guidance"},
    "compute_diameter": {"type": "math", "script": "2 * radius", "params": ["radius"], "fallback": "guidance"},
    "compute_perimeter": {"type": "math", "script": "2 * (length + width)", "params": ["length", "width"], "fallback": "guidance"},
    "compute_circumference": {"type": "math", "script": "2 * pi * radius", "params": ["radius"], "fallback": "guidance"},

    # Geometry - areas
    "compute_area": {"type": "math", "script": "length * width", "params": ["length", "width"], "fallback": "guidance"},
    "area_rectangle": {"type": "math", "script": "length * width", "params": ["length", "width"], "fallback": "guidance"},
    "area_triangle": {"type": "math", "script": "0.5 * base * height", "params": ["base", "height"], "fallback": "guidance"},
    "area_circle": {"type": "math", "script": "pi * radius**2", "params": ["radius"], "fallback": "guidance"},
    "area_trapezoid": {"type": "math", "script": "0.5 * (a + b) * height", "params": ["a", "b", "height"], "fallback": "guidance"},
    "area_parallelogram": {"type": "math", "script": "base * height", "params": ["base", "height"], "fallback": "guidance"},

    # Geometry - volumes
    "compute_volume": {"type": "math", "script": "length * width * height", "params": ["length", "width", "height"], "fallback": "guidance"},
    "volume_sphere": {"type": "math", "script": "(4/3) * pi * radius**3", "params": ["radius"], "fallback": "guidance"},
    "volume_cylinder": {"type": "math", "script": "pi * radius**2 * height", "params": ["radius", "height"], "fallback": "guidance"},
    "volume_cone": {"type": "math", "script": "(1/3) * pi * radius**2 * height", "params": ["radius", "height"], "fallback": "guidance"},

    # Geometry - angles
    "compute_angle": {"type": "math", "script": "180 - angle1 - angle2", "params": ["angle1", "angle2"], "fallback": "guidance"},
    "angle_sum": {"type": "math", "script": "angle1 + angle2", "params": ["angle1", "angle2"], "fallback": "guidance"},
    "complementary_angle": {"type": "math", "script": "90 - angle", "params": ["angle"], "fallback": "guidance"},
    "supplementary_angle": {"type": "math", "script": "180 - angle", "params": ["angle"], "fallback": "guidance"},

    # Trigonometry
    "trig_function": {"type": "math", "script": "sin(radians(angle))", "params": ["angle"], "fallback": "guidance"},
    "compute_sin": {"type": "math", "script": "sin(radians(angle))", "params": ["angle"], "fallback": "guidance"},
    "compute_cos": {"type": "math", "script": "cos(radians(angle))", "params": ["angle"], "fallback": "guidance"},
    "compute_tan": {"type": "math", "script": "tan(radians(angle))", "params": ["angle"], "fallback": "guidance"},

    # Algebra - equations
    "solve_equation": {"type": "sympy", "script": "solve(equation, x)", "params": ["equation"], "fallback": "guidance"},
    "solve_linear": {"type": "math", "script": "(c - b) / a", "params": ["a", "b", "c"], "fallback": "guidance"},
    "solve_quadratic": {"type": "sympy", "script": "solve(a*x**2 + b*x + c, x)", "params": ["a", "b", "c"], "fallback": "guidance"},
    "solve_system": {"type": "sympy", "script": "solve([eq1, eq2], [x, y])", "params": ["eq1", "eq2"], "fallback": "guidance"},

    # Algebra - expressions
    "simplify_expression": {"type": "sympy", "script": "simplify(expr)", "params": ["expr"], "fallback": "guidance"},
    "expand_expression": {"type": "sympy", "script": "expand(expr)", "params": ["expr"], "fallback": "guidance"},
    "factor_expression": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "fallback": "guidance"},
    "factor_polynomial": {"type": "sympy", "script": "factor(poly)", "params": ["poly"], "fallback": "guidance"},
    "substitute_value": {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "fallback": "guidance"},
    "evaluate_function": {"type": "sympy", "script": "f.subs(x, value)", "params": ["f", "value"], "fallback": "guidance"},

    # Logarithms and exponentials
    "compute_logarithm": {"type": "math", "script": "log(x) / log(base)", "params": ["x", "base"], "fallback": "guidance"},
    "compute_log10": {"type": "math", "script": "log10(x)", "params": ["x"], "fallback": "guidance"},
    "compute_ln": {"type": "math", "script": "log(x)", "params": ["x"], "fallback": "guidance"},
    "compute_exp": {"type": "math", "script": "exp(x)", "params": ["x"], "fallback": "guidance"},

    # Counting and combinatorics
    "count_items": {"type": "math", "script": "n", "params": ["n"], "fallback": "guidance"},
    "count_total_items": {"type": "math", "script": "n1 + n2", "params": ["n1", "n2"], "fallback": "guidance"},
    "count_combinations": {"type": "math", "script": "factorial(n) // (factorial(r) * factorial(n-r))", "params": ["n", "r"], "fallback": "guidance"},
    "count_permutations": {"type": "math", "script": "factorial(n) // factorial(n-r)", "params": ["n", "r"], "fallback": "guidance"},
    "count_permutations_with_replacement": {"type": "math", "script": "n ** r", "params": ["n", "r"], "fallback": "guidance"},
    "count_ways": {"type": "math", "script": "n1 * n2", "params": ["n1", "n2"], "fallback": "guidance"},

    # Probability
    "compute_probability": {"type": "math", "script": "favorable / total", "params": ["favorable", "total"], "fallback": "guidance"},
    "probability_complement": {"type": "math", "script": "1 - p", "params": ["p"], "fallback": "guidance"},
    "probability_union": {"type": "math", "script": "p_a + p_b - p_ab", "params": ["p_a", "p_b", "p_ab"], "fallback": "guidance"},
    "probability_intersection": {"type": "math", "script": "p_a * p_b", "params": ["p_a", "p_b"], "fallback": "guidance"},

    # Sequences and series
    "nth_term": {"type": "math", "script": "a1 + (n-1) * d", "params": ["a1", "n", "d"], "fallback": "guidance"},
    "arithmetic_sum": {"type": "math", "script": "n * (a1 + an) / 2", "params": ["n", "a1", "an"], "fallback": "guidance"},
    "geometric_sum": {"type": "math", "script": "a * (1 - r**n) / (1 - r)", "params": ["a", "r", "n"], "fallback": "guidance"},
    "common_difference": {"type": "math", "script": "a2 - a1", "params": ["a1", "a2"], "fallback": "guidance"},
    "common_ratio": {"type": "math", "script": "a2 / a1", "params": ["a1", "a2"], "fallback": "guidance"},

    # Fractions
    "simplify_fraction": {"type": "math", "script": "numerator // gcd(numerator, denominator)", "params": ["numerator", "denominator"], "fallback": "guidance"},
    "fraction_parts": {"type": "guidance", "script": "Identify numerator and denominator", "params": [], "fallback": "guidance"},
    "add_fractions": {"type": "math", "script": "(a*d + b*c) / (b*d)", "params": ["a", "b", "c", "d"], "fallback": "guidance"},
    "multiply_fractions": {"type": "math", "script": "(a*c) / (b*d)", "params": ["a", "b", "c", "d"], "fallback": "guidance"},

    # Inequalities
    "apply_inequality": {"type": "sympy", "script": "solve_univariate_inequality(ineq, x)", "params": ["ineq"], "fallback": "guidance"},
    "solve_inequality": {"type": "sympy", "script": "solve_univariate_inequality(ineq, x)", "params": ["ineq"], "fallback": "guidance"},

    # Min/Max
    "find_minimum": {"type": "math", "script": "min(values)", "params": ["values"], "fallback": "guidance"},
    "find_maximum": {"type": "math", "script": "max(values)", "params": ["values"], "fallback": "guidance"},

    # Vectors
    "vector_operation": {"type": "math", "script": "sum(a[i] * b[i] for i in range(len(a)))", "params": ["a", "b"], "fallback": "guidance"},
    "vector_magnitude": {"type": "math", "script": "sqrt(sum(x**2 for x in v))", "params": ["v"], "fallback": "guidance"},
    "dot_product": {"type": "math", "script": "sum(a[i] * b[i] for i in range(len(a)))", "params": ["a", "b"], "fallback": "guidance"},

    # Unit conversion
    "convert_units": {"type": "math", "script": "value * conversion_factor", "params": ["value", "conversion_factor"], "fallback": "guidance"},

    # Divisibility
    "check_divisibility": {"type": "math", "script": "n % d == 0", "params": ["n", "d"], "fallback": "guidance"},

    # Ratios and proportions
    "compute_ratio": {"type": "math", "script": "a / b", "params": ["a", "b"], "fallback": "guidance"},
    "solve_proportion": {"type": "math", "script": "(a * d) / b", "params": ["a", "b", "d"], "fallback": "guidance"},

    # Percentages
    "compute_percentage": {"type": "math", "script": "(part / whole) * 100", "params": ["part", "whole"], "fallback": "guidance"},
    "percentage_change": {"type": "math", "script": "((new - old) / old) * 100", "params": ["old", "new"], "fallback": "guidance"},

    # Averages
    "compute_mean": {"type": "math", "script": "sum(values) / len(values)", "params": ["values"], "fallback": "guidance"},
    "compute_median": {"type": "math", "script": "sorted(values)[len(values)//2]", "params": ["values"], "fallback": "guidance"},

    # Modular arithmetic
    "modular_arithmetic": {"type": "math", "script": "a % m", "params": ["a", "m"], "fallback": "guidance"},
    "modular_inverse": {"type": "sympy", "script": "mod_inverse(a, m)", "params": ["a", "m"], "fallback": "guidance"},
}

# Guidance template for guidance-only types
GUIDANCE_TEMPLATE = {"type": "guidance", "script": "Use LLM reasoning", "params": [], "fallback": "guidance"}


def assign_dsl_scripts(db: StepSignatureDB, dry_run: bool = False):
    """Assign DSL scripts to signatures that don't have them."""

    with db._connection() as conn:
        # Get signatures without DSL that aren't semantic umbrellas
        rows = conn.execute('''
            SELECT id, step_type, description
            FROM step_signatures
            WHERE dsl_script IS NULL AND is_semantic_umbrella = 0
        ''').fetchall()

        print(f"Found {len(rows)} signatures without DSL scripts")

        assigned = 0
        guidance_only = 0
        no_template = 0

        step_type_counts = {}

        for row in rows:
            sig_id, step_type, description = row

            # Track counts by step type
            step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1

            # Determine DSL script
            if step_type in GUIDANCE_ONLY_TYPES:
                dsl = GUIDANCE_TEMPLATE
                guidance_only += 1
            elif step_type in DSL_TEMPLATES:
                dsl = DSL_TEMPLATES[step_type]
                assigned += 1
            else:
                # Try to find a partial match
                matched = False
                for template_type, template_dsl in DSL_TEMPLATES.items():
                    if template_type in step_type or step_type in template_type:
                        dsl = template_dsl
                        assigned += 1
                        matched = True
                        break

                if not matched:
                    # No template found - use guidance
                    dsl = GUIDANCE_TEMPLATE
                    no_template += 1
                    continue

            if not dry_run:
                conn.execute(
                    'UPDATE step_signatures SET dsl_script = ? WHERE id = ?',
                    (json.dumps(dsl), sig_id)
                )

        if not dry_run:
            conn.commit()

        print(f"\nResults:")
        print(f"  Assigned DSL scripts: {assigned}")
        print(f"  Marked guidance-only: {guidance_only}")
        print(f"  No template (guidance fallback): {no_template}")

        if no_template > 0:
            print(f"\nStep types without templates:")
            for st, cnt in sorted(step_type_counts.items(), key=lambda x: -x[1]):
                if st not in DSL_TEMPLATES and st not in GUIDANCE_ONLY_TYPES:
                    print(f"  {st}: {cnt}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Assign DSL scripts to signatures")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    db = StepSignatureDB()
    assign_dsl_scripts(db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
