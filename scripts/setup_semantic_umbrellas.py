#!/usr/bin/env python3
"""Setup semantic umbrella signatures with atomic children.

This script converts problematic high-level signatures into semantic umbrellas
that route to specific atomic children with proper DSLs.

This implements the signature refinement loop from paper.md:
1. IDENTIFY: Query signatures with negative lift (DSL hurts accuracy)
2. ANALYZE: Examine failure cases to understand why DSL fails
3. DECOMPOSE: Split into finer-grained sub-signatures
4. REDIRECT: Mark parent as semantic umbrella, add routing to children
5. GENERATE DSL: Build new DSL for each atomic sub-signature
6. VALIDATE: Test on held-out examples (lift tracking handles this)

Problematic signatures identified:
- compute_probability (id=124): -76% lift, was hardcoded to "12/52"
- solve_equation (id=303): -63% lift, too generic
- simplify_expression (id=268): -20% lift, too generic
- general_step (id=100): -7% lift, catch-all

Each umbrella gets decomposed into atomic children with specific DSLs.

**Practical note for LLM-assisted refinement:**
The LLM may initially resist this task ("I can help you think through approaches...")
or produce overly cautious responses. Insist on concrete outputs: specific sub-signature
names, actual DSL code, explicit routing conditions. The model is capable; it just needs
clear direction that you want executable artifacts, not suggestions.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures.db import StepSignatureDB


def setup_compute_probability_children(db: StepSignatureDB, parent_id: int):
    """Decompose compute_probability into atomic probability types."""

    children = [
        {
            "step_type": "prob_single_draw",
            "description": "Probability of drawing a specific item from a set (single draw, no replacement)",
            "condition": "single draw from set",
            "dsl": {
                "type": "math",
                "script": "favorable / total",
                "params": ["favorable", "total"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "prob_multiple_draws",
            "description": "Probability of multiple draws without replacement",
            "condition": "multiple draws without replacement",
            "dsl": {
                "type": "math",
                "script": "(factorial(favorable) / factorial(favorable - k)) / (factorial(total) / factorial(total - k))",
                "params": ["favorable", "total", "k"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "prob_conditional",
            "description": "Conditional probability P(A|B) = P(A and B) / P(B)",
            "condition": "conditional on another event",
            "dsl": {
                "type": "math",
                "script": "p_a_and_b / p_b",
                "params": ["p_a_and_b", "p_b"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "prob_complement",
            "description": "Probability of complement: P(not A) = 1 - P(A)",
            "condition": "probability of NOT happening",
            "dsl": {
                "type": "math",
                "script": "1 - p",
                "params": ["p"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "prob_independent",
            "description": "Probability of independent events: P(A and B) = P(A) * P(B)",
            "condition": "independent events occurring together",
            "dsl": {
                "type": "math",
                "script": "p_a * p_b",
                "params": ["p_a", "p_b"],
                "fallback": "guidance"
            }
        },
    ]

    print(f"\nSetting up compute_probability (id={parent_id}) as semantic umbrella...")

    for child in children:
        result = db.create_child_signature(
            parent_id=parent_id,
            step_type=child["step_type"],
            description=child["description"],
            dsl_script=json.dumps(child["dsl"]),
            condition=child["condition"],
        )
        if result:
            print(f"  Created child: {child['step_type']} (id={result.id})")
        else:
            print(f"  FAILED: {child['step_type']}")


def setup_solve_equation_children(db: StepSignatureDB, parent_id: int):
    """Decompose solve_equation into atomic equation types."""

    children = [
        {
            "step_type": "solve_linear_single",
            "description": "Solve ax + b = c for x",
            "condition": "linear equation with one variable",
            "dsl": {
                "type": "math",
                "script": "(c - b) / a",
                "params": ["a", "b", "c"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "solve_quadratic_formula",
            "description": "Solve ax^2 + bx + c = 0 using quadratic formula",
            "condition": "quadratic equation",
            "dsl": {
                "type": "sympy",
                "script": "solve(a*x**2 + b*x + c, x)",
                "params": ["a", "b", "c"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "solve_proportion",
            "description": "Solve a/b = c/d for one variable",
            "condition": "proportion or ratio equation",
            "dsl": {
                "type": "math",
                "script": "(a * d) / b",  # Solving for c
                "params": ["a", "b", "d"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "isolate_variable",
            "description": "Isolate a variable from a formula",
            "condition": "rearranging formula for a variable",
            "dsl": {
                "type": "sympy",
                "script": "solve(equation, target_var)",
                "params": ["equation", "target_var"],
                "fallback": "guidance"
            }
        },
    ]

    print(f"\nSetting up solve_equation (id={parent_id}) as semantic umbrella...")

    for child in children:
        result = db.create_child_signature(
            parent_id=parent_id,
            step_type=child["step_type"],
            description=child["description"],
            dsl_script=json.dumps(child["dsl"]),
            condition=child["condition"],
        )
        if result:
            print(f"  Created child: {child['step_type']} (id={result.id})")
        else:
            print(f"  FAILED: {child['step_type']}")


def setup_simplify_expression_children(db: StepSignatureDB, parent_id: int):
    """Decompose simplify_expression into atomic simplification types."""

    children = [
        {
            "step_type": "simplify_fraction",
            "description": "Reduce a fraction to lowest terms using GCD",
            "condition": "fraction to reduce",
            "dsl": {
                "type": "math",
                "script": "(numerator / gcd(numerator, denominator), denominator / gcd(numerator, denominator))",
                "params": ["numerator", "denominator"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "combine_like_terms",
            "description": "Combine like terms in an algebraic expression",
            "condition": "polynomial with like terms",
            "dsl": {
                "type": "sympy",
                "script": "simplify(expand(expr))",
                "params": ["expr"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "simplify_radical",
            "description": "Simplify radical expressions (square roots, etc.)",
            "condition": "expression with radicals",
            "dsl": {
                "type": "sympy",
                "script": "simplify(sqrt(n))",
                "params": ["n"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "factor_common",
            "description": "Factor out common terms from expression",
            "condition": "expression with common factors",
            "dsl": {
                "type": "sympy",
                "script": "factor(expr)",
                "params": ["expr"],
                "fallback": "guidance"
            }
        },
    ]

    print(f"\nSetting up simplify_expression (id={parent_id}) as semantic umbrella...")

    for child in children:
        result = db.create_child_signature(
            parent_id=parent_id,
            step_type=child["step_type"],
            description=child["description"],
            dsl_script=json.dumps(child["dsl"]),
            condition=child["condition"],
        )
        if result:
            print(f"  Created child: {child['step_type']} (id={result.id})")
        else:
            print(f"  FAILED: {child['step_type']}")


def setup_general_step_children(db: StepSignatureDB, parent_id: int):
    """Decompose general_step into common atomic patterns.

    general_step is a catch-all - we route to more specific children.
    """

    children = [
        {
            "step_type": "extract_value",
            "description": "Extract a numeric value from problem text",
            "condition": "extracting given values",
            "dsl": {
                "type": "guidance",
                "script": "Identify and extract the numeric value from: {text}",
                "params": ["text"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "apply_formula",
            "description": "Apply a known formula with given values",
            "condition": "applying a formula",
            "dsl": {
                "type": "sympy",
                "script": "formula.subs(substitutions)",
                "params": ["formula", "substitutions"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "compare_values",
            "description": "Compare two values and determine relationship",
            "condition": "comparing quantities",
            "dsl": {
                "type": "math",
                "script": "a - b",  # Positive if a > b, negative if a < b
                "params": ["a", "b"],
                "fallback": "guidance"
            }
        },
        {
            "step_type": "unit_conversion",
            "description": "Convert between units",
            "condition": "unit conversion needed",
            "dsl": {
                "type": "math",
                "script": "value * conversion_factor",
                "params": ["value", "conversion_factor"],
                "fallback": "guidance"
            }
        },
    ]

    print(f"\nSetting up general_step (id={parent_id}) as semantic umbrella...")

    for child in children:
        result = db.create_child_signature(
            parent_id=parent_id,
            step_type=child["step_type"],
            description=child["description"],
            dsl_script=json.dumps(child["dsl"]),
            condition=child["condition"],
        )
        if result:
            print(f"  Created child: {child['step_type']} (id={result.id})")
        else:
            print(f"  FAILED: {child['step_type']}")


def main():
    """Set up semantic umbrellas for problematic signatures."""

    db = StepSignatureDB()

    # Get stats before
    stats_before = db.get_stats()
    print(f"Signatures before: {stats_before['signatures']}")

    # Map step_type to setup function
    umbrella_setup = {
        "compute_probability": setup_compute_probability_children,
        "solve_equation": setup_solve_equation_children,
        "simplify_expression": setup_simplify_expression_children,
        "general_step": setup_general_step_children,
    }

    # Find the signature IDs for each problematic type
    with db._connection() as conn:
        for step_type, setup_fn in umbrella_setup.items():
            # Find signature with this step_type that has negative lift
            cursor = conn.execute(
                """
                SELECT id, step_type, uses, successes,
                       injected_uses, injected_successes,
                       non_injected_uses, non_injected_successes
                FROM step_signatures
                WHERE step_type = ?
                  AND injected_uses > 0
                  AND non_injected_uses > 0
                ORDER BY uses DESC
                LIMIT 1
                """,
                (step_type,)
            )
            row = cursor.fetchone()

            if row:
                sig_id = row[0]
                uses = row[2]
                inj_rate = row[4] / row[4] if row[4] > 0 else 0
                base_rate = row[6] / row[6] if row[6] > 0 else 0

                # Check if it's already a semantic umbrella
                sig = db.get_signature(sig_id)
                if sig and sig.is_semantic_umbrella:
                    print(f"\n{step_type} (id={sig_id}) is already a semantic umbrella, skipping...")
                    continue

                print(f"\nFound {step_type}: id={sig_id}, uses={uses}")
                setup_fn(db, sig_id)
            else:
                print(f"\nNo signature found for step_type={step_type}")

    # Get stats after
    stats_after = db.get_stats()
    print(f"\n\nSignatures after: {stats_after['signatures']}")
    print(f"New signatures created: {stats_after['signatures'] - stats_before['signatures']}")


if __name__ == "__main__":
    main()
