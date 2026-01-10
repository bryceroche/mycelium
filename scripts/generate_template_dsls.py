#!/usr/bin/env python3
"""Generate template-based DSLs for all signatures based on step_type patterns."""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "mycelium.db"

# DSL templates by step_type
DSL_TEMPLATES = {
    "solve_equation": {"type": "sympy", "script": "solve(equation, x)", "params": ["equation"], "fallback": "Solve the equation step by step"},
    "solve_system": {"type": "sympy", "script": "solve([eq1, eq2], [x, y])", "params": ["eq1", "eq2"], "fallback": "Solve the system of equations"},
    "solve_quadratic": {"type": "custom", "script": "apply_quadratic_formula(a, b, c)", "params": ["a", "b", "c"], "fallback": "x = (-b +/- sqrt(b^2-4ac))/(2a)"},
    "compute_sum": {"type": "math", "script": "sum(values)", "params": ["values"], "fallback": "Add all values together"},
    "compute_product": {"type": "math", "script": "product(values)", "params": ["values"], "fallback": "Multiply all values together"},
    "compute_difference": {"type": "math", "script": "a - b", "params": ["a", "b"], "fallback": "Subtract b from a"},
    "compute_sqrt": {"type": "math", "script": "sqrt(value)", "params": ["value"], "fallback": "Compute square root"},
    "compute_power": {"type": "math", "script": "base ** exponent", "params": ["base", "exponent"], "fallback": "Compute base^exponent"},
    "compute_factorial": {"type": "math", "script": "factorial(n)", "params": ["n"], "fallback": "Compute n!"},
    "compute_gcd": {"type": "math", "script": "gcd(a, b)", "params": ["a", "b"], "fallback": "Find greatest common divisor"},
    "compute_lcm": {"type": "math", "script": "lcm(a, b)", "params": ["a", "b"], "fallback": "Find least common multiple"},
    "compute_probability": {"type": "math", "script": "favorable / total", "params": ["favorable", "total"], "fallback": "P = favorable/total"},
    "compute_area": {"type": "math", "script": "length * width", "params": ["length", "width"], "fallback": "Compute area from dimensions"},
    "compute_volume": {"type": "math", "script": "length * width * height", "params": ["length", "width", "height"], "fallback": "Compute volume"},
    "compute_radius": {"type": "math", "script": "diameter / 2", "params": ["diameter"], "fallback": "radius = diameter/2"},
    "compute_angle": {"type": "math", "script": "atan2(opposite, adjacent)", "params": ["opposite", "adjacent"], "fallback": "Compute angle from sides"},
    "compute_length": {"type": "math", "script": "sqrt(dx**2 + dy**2)", "params": ["dx", "dy"], "fallback": "Compute length/distance"},
    "compute_magnitude": {"type": "math", "script": "sqrt(sum(x**2 for x in vector))", "params": ["vector"], "fallback": "Compute vector magnitude"},
    "area_triangle": {"type": "math", "script": "0.5 * base * height", "params": ["base", "height"], "fallback": "A = (1/2)*base*height"},
    "area_circle": {"type": "math", "script": "pi * r**2", "params": ["r"], "fallback": "A = pi*r^2"},
    "area_rectangle": {"type": "math", "script": "length * width", "params": ["length", "width"], "fallback": "A = length * width"},
    "count_items": {"type": "math", "script": "len(items)", "params": ["items"], "fallback": "Count the items"},
    "count_combinations": {"type": "math", "script": "factorial(n) / (factorial(r) * factorial(n-r))", "params": ["n", "r"], "fallback": "C(n,r) = n!/(r!(n-r)!)"},
    "count_permutations": {"type": "math", "script": "factorial(n) / factorial(n-r)", "params": ["n", "r"], "fallback": "P(n,r) = n!/(n-r)!"},
    "simplify_expression": {"type": "sympy", "script": "simplify(expression)", "params": ["expression"], "fallback": "Simplify the expression"},
    "simplify_fraction": {"type": "math", "script": "gcd_reduce(num, den)", "params": ["num", "den"], "fallback": "Divide by GCD to simplify"},
    "expand_expression": {"type": "sympy", "script": "expand(expression)", "params": ["expression"], "fallback": "Expand the expression"},
    "factor_expression": {"type": "sympy", "script": "factor(expression)", "params": ["expression"], "fallback": "Factor the expression"},
    "substitute_value": {"type": "sympy", "script": "expression.subs(var, value)", "params": ["expression", "var", "value"], "fallback": "Substitute value into expression"},
    "evaluate_function": {"type": "math", "script": "f(x)", "params": ["f", "x"], "fallback": "Evaluate function at x"},
    "define_formula": {"type": "none", "fallback": "Define the formula from the problem"},
    "define_variables": {"type": "none", "fallback": "Define variables from problem context"},
    "setup_equation": {"type": "none", "fallback": "Set up equation from given conditions"},
    "express_relation": {"type": "none", "fallback": "Express the relationship mathematically"},
    "apply_inequality": {"type": "sympy", "script": "solve(inequality, x)", "params": ["inequality"], "fallback": "Solve the inequality"},
    "apply_amgm": {"type": "math", "script": "(a + b) / 2 >= sqrt(a * b)", "params": ["a", "b"], "fallback": "AM-GM: (a+b)/2 >= sqrt(ab)"},
    "convert_units": {"type": "math", "script": "value * conversion_factor", "params": ["value", "conversion_factor"], "fallback": "Apply unit conversion"},
    "convert_base": {"type": "custom", "script": "convert_base(number, from_base, to_base)", "params": ["number", "from_base", "to_base"], "fallback": "Convert between number bases"},
    "check_divisibility": {"type": "math", "script": "n % d == 0", "params": ["n", "d"], "fallback": "Check if d divides n"},
    "trig_function": {"type": "math", "script": "sin(theta)", "params": ["theta"], "fallback": "Apply trigonometric function"},
    "vector_operation": {"type": "math", "script": "vector_op(v1, v2)", "params": ["v1", "v2"], "fallback": "Perform vector operation"},
    "matrix_operation": {"type": "custom", "script": "matrix_op(M1, M2)", "params": ["M1", "M2"], "fallback": "Perform matrix operation"},
    "synthesize_results": {"type": "none", "fallback": "Combine previous step results for final answer"},
    "fraction_parts": {"type": "none", "fallback": "Identify numerator and denominator"},
    "common_difference": {"type": "math", "script": "a2 - a1", "params": ["a1", "a2"], "fallback": "d = a_{n+1} - a_n"},
    "general_step": {"type": "none", "fallback": "Follow the step description"},
}

def generate_dsls():
    """Generate DSLs for all signatures without one."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all signatures
    cursor = conn.execute("""
        SELECT id, step_type, description
        FROM step_signatures
        WHERE dsl_script IS NULL OR dsl_script = ''
    """)

    dsls = {}
    for row in cursor:
        sig_id = row['id']
        step_type = row['step_type']
        description = row['description']

        # Get template for this step_type
        if step_type in DSL_TEMPLATES:
            template = DSL_TEMPLATES[step_type].copy()
        else:
            # Default template for unknown types
            template = {"type": "none", "fallback": description or "Follow the step description"}

        # Customize fallback with description if available
        if description and template.get("fallback") == "Follow the step description":
            template["fallback"] = description

        dsls[str(sig_id)] = template

    conn.close()
    return dsls

def main():
    dsls = generate_dsls()
    print(f"Generated DSLs for {len(dsls)} signatures")

    # Save to file
    output_file = Path(__file__).parent / "claude_dsls_generated.json"
    with open(output_file, "w") as f:
        json.dump(dsls, f, indent=2)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
