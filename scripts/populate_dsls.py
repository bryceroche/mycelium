#!/usr/bin/env python3
"""Populate DSL scripts for signatures based on step_type.

V2 uses lazy NL - signatures start empty and get DSLs added as we learn.
This script adds DSLs for common step types.

For general_step signatures, we analyze the description to infer appropriate DSLs.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures import StepSignatureDB

# DSL templates by step_type
# Format: {"type": "math|sympy", "script": "expression", "params": ["p1", "p2"]}
DSL_TEMPLATES = {
    # Arithmetic
    "compute_sum": {
        "type": "math",
        "script": "a + b",
        "params": ["a", "b"],
        "purpose": "Add two numbers together",
    },
    "compute_product": {
        "type": "math",
        "script": "a * b",
        "params": ["a", "b"],
        "purpose": "Multiply two numbers together",
    },
    "compute_difference": {
        "type": "math",
        "script": "a - b",
        "params": ["a", "b"],
        "purpose": "Subtract b from a",
    },
    "compute_quotient": {
        "type": "math",
        "script": "a / b",
        "params": ["a", "b"],
        "purpose": "Divide a by b",
    },
    "compute_power": {
        "type": "math",
        "script": "base ** exponent",
        "params": ["base", "exponent"],
        "purpose": "Raise base to exponent power",
    },
    "compute_factorial": {
        "type": "math",
        "script": "factorial(n)",
        "params": ["n"],
        "purpose": "Calculate n factorial (n!)",
    },
    "compute_sqrt": {
        "type": "math",
        "script": "sqrt(x)",
        "params": ["x"],
        "purpose": "Calculate square root of x",
    },
    "compute_modulo": {
        "type": "math",
        "script": "a % b",
        "params": ["a", "b"],
        "purpose": "Calculate remainder of a divided by b",
    },

    # Number theory
    "compute_gcd": {
        "type": "math",
        "script": "gcd(a, b)",
        "params": ["a", "b"],
        "purpose": "Find greatest common divisor",
    },
    "compute_lcm": {
        "type": "math",
        "script": "lcm(a, b)",
        "params": ["a", "b"],
        "purpose": "Find least common multiple",
    },

    # Geometry
    "compute_area": {
        "type": "math",
        "script": "length * width",
        "params": ["length", "width"],
        "purpose": "Calculate area (rectangle default)",
    },
    "compute_perimeter": {
        "type": "math",
        "script": "2 * (length + width)",
        "params": ["length", "width"],
        "purpose": "Calculate perimeter (rectangle default)",
    },
    "compute_distance": {
        "type": "math",
        "script": "sqrt((x2 - x1)**2 + (y2 - y1)**2)",
        "params": ["x1", "y1", "x2", "y2"],
        "purpose": "Calculate Euclidean distance between two points",
    },

    # Statistics
    "compute_average": {
        "type": "math",
        "script": "(a + b) / 2",
        "params": ["a", "b"],
        "purpose": "Calculate average of two numbers",
    },
    "compute_percentage": {
        "type": "math",
        "script": "(part / whole) * 100",
        "params": ["part", "whole"],
        "purpose": "Calculate percentage",
    },

    # Algebra (SymPy)
    "simplify_expression": {
        "type": "sympy",
        "script": "simplify(expr)",
        "params": ["expr"],
        "purpose": "Simplify an algebraic expression",
    },
    "solve_equation": {
        "type": "sympy",
        "script": "solve(equation, x)",
        "params": ["equation"],
        "purpose": "Solve equation for variable x",
    },
    "factor_expression": {
        "type": "sympy",
        "script": "factor(expr)",
        "params": ["expr"],
        "purpose": "Factor an algebraic expression",
    },
    "evaluate_expression": {
        "type": "math",
        "script": "eval(expr)",
        "params": ["expr"],
        "purpose": "Evaluate a mathematical expression",
    },

    # Geometry
    "compute_angle": {
        "type": "math",
        "script": "degrees",  # Often just return the angle
        "params": ["degrees"],
        "purpose": "Calculate or return an angle in degrees",
    },

    # Combinatorics
    "count_combinations": {
        "type": "math",
        "script": "factorial(n) / (factorial(r) * factorial(n - r))",
        "params": ["n", "r"],
        "purpose": "Calculate n choose r (combinations)",
    },
    "count_permutations": {
        "type": "math",
        "script": "factorial(n) / factorial(n - r)",
        "params": ["n", "r"],
        "purpose": "Calculate permutations P(n, r)",
    },
}

# Clarifying questions and param descriptions
NL_TEMPLATES = {
    "compute_sum": {
        "clarifying_questions": ["What is the first number?", "What is the second number?"],
        "param_descriptions": {"a": "First number to add", "b": "Second number to add"},
    },
    "compute_product": {
        "clarifying_questions": ["What is the first factor?", "What is the second factor?"],
        "param_descriptions": {"a": "First number to multiply", "b": "Second number to multiply"},
    },
    "compute_difference": {
        "clarifying_questions": ["What number to subtract from?", "What number to subtract?"],
        "param_descriptions": {"a": "Number to subtract from", "b": "Number to subtract"},
    },
    "compute_quotient": {
        "clarifying_questions": ["What is the dividend?", "What is the divisor?"],
        "param_descriptions": {"a": "Number being divided (dividend)", "b": "Number to divide by (divisor)"},
    },
    "compute_power": {
        "clarifying_questions": ["What is the base number?", "What is the exponent?"],
        "param_descriptions": {"base": "The number being raised", "exponent": "The power to raise to"},
    },
    "compute_factorial": {
        "clarifying_questions": ["What number to compute factorial of?"],
        "param_descriptions": {"n": "The number to compute factorial of"},
    },
    "compute_sqrt": {
        "clarifying_questions": ["What number to take the square root of?"],
        "param_descriptions": {"x": "The number under the radical"},
    },
    "compute_modulo": {
        "clarifying_questions": ["What is the dividend?", "What is the divisor?"],
        "param_descriptions": {"a": "Number being divided", "b": "Divisor for remainder"},
    },
    "compute_gcd": {
        "clarifying_questions": ["What is the first number?", "What is the second number?"],
        "param_descriptions": {"a": "First number", "b": "Second number"},
    },
    "compute_lcm": {
        "clarifying_questions": ["What is the first number?", "What is the second number?"],
        "param_descriptions": {"a": "First number", "b": "Second number"},
    },
}


def infer_dsl_from_description(description: str) -> dict | None:
    """Infer a DSL template from the step description.

    Analyzes keywords in the description to determine appropriate DSL.
    Returns None if no suitable DSL can be inferred.
    """
    desc_lower = description.lower()

    # Pattern matching for common operations
    patterns = [
        # Synthesis/combination
        (r"combine.*result|final.*answer|synthesize", {
            "type": "guidance",
            "script": "synthesize_results",
            "params": ["results"],
            "purpose": "Combine intermediate results into final answer",
        }),
        # Coordinate/point operations
        (r"coordinate|point.*\(|define.*point", {
            "type": "sympy",
            "script": "Point(x, y)",
            "params": ["x", "y"],
            "purpose": "Define or work with coordinate points",
        }),
        # Substitution
        (r"substitut|plug.*in|replace.*with", {
            "type": "sympy",
            "script": "expr.subs(var, value)",
            "params": ["expr", "var", "value"],
            "purpose": "Substitute values into expression",
        }),
        # Finding min/max
        (r"find.*minimum|find.*maximum|minimize|maximize|min.*value|max.*value", {
            "type": "sympy",
            "script": "solve(diff(expr, x), x)",
            "params": ["expr"],
            "purpose": "Find minimum or maximum value",
        }),
        # Define constraints
        (r"define.*constraint|constraint|given.*condition", {
            "type": "guidance",
            "script": "extract_constraints",
            "params": ["problem"],
            "purpose": "Extract and define problem constraints",
        }),
        # Express in terms of
        (r"express.*in terms|write.*as|rewrite", {
            "type": "sympy",
            "script": "solve(eq, var)",
            "params": ["eq", "var"],
            "purpose": "Express one variable in terms of another",
        }),
        # Find equation
        (r"find.*equation|equation of|derive.*equation", {
            "type": "sympy",
            "script": "Eq(lhs, rhs)",
            "params": ["lhs", "rhs"],
            "purpose": "Find or construct an equation",
        }),
        # Identify/extract
        (r"identify|extract|determine.*value|find.*value", {
            "type": "guidance",
            "script": "extract_values",
            "params": ["text"],
            "purpose": "Identify and extract values from problem",
        }),
        # Magnitude (complex numbers, vectors)
        (r"magnitude|absolute|modulus|\|.*\|", {
            "type": "sympy",
            "script": "Abs(z)",
            "params": ["z"],
            "purpose": "Calculate magnitude/absolute value",
        }),
        # Argument (complex numbers)
        (r"argument|angle.*of|arg\(", {
            "type": "sympy",
            "script": "arg(z)",
            "params": ["z"],
            "purpose": "Calculate argument/angle",
        }),
        # Range/interval
        (r"range|interval|bounds|between", {
            "type": "guidance",
            "script": "find_range",
            "params": ["expr", "var"],
            "purpose": "Determine range or interval",
        }),
        # Solve for variable
        (r"solve for|find.*n\b|find.*x\b|find.*value of", {
            "type": "sympy",
            "script": "solve(eq, var)",
            "params": ["eq", "var"],
            "purpose": "Solve equation for variable",
        }),
        # Critical points
        (r"critical point|derivative.*zero|stationary", {
            "type": "sympy",
            "script": "solve(diff(f, x), x)",
            "params": ["f"],
            "purpose": "Find critical points",
        }),
        # Relationship/connection
        (r"relationship|connection|relate|between.*and", {
            "type": "guidance",
            "script": "find_relationship",
            "params": ["a", "b"],
            "purpose": "Determine relationship between quantities",
        }),
    ]

    for pattern, dsl in patterns:
        if re.search(pattern, desc_lower):
            return dsl

    # Default fallback: guidance DSL for general reasoning
    return {
        "type": "guidance",
        "script": "reason_step",
        "params": ["context"],
        "purpose": f"Execute reasoning step: {description[:50]}",
    }


def main():
    db = StepSignatureDB()

    signatures = db.get_all_signatures()
    print(f"Found {len(signatures)} signatures")

    updated = 0
    skipped = 0

    for sig in signatures:
        step_type = sig.step_type

        # Skip if already has DSL
        if sig.dsl_script:
            print(f"  [{sig.id}] {step_type}: already has DSL, skipping")
            skipped += 1
            continue

        # Try template first, then infer from description
        dsl = None
        inferred = False

        if step_type in DSL_TEMPLATES:
            dsl = DSL_TEMPLATES[step_type]
        else:
            # Infer from description
            dsl = infer_dsl_from_description(sig.description)
            inferred = True

        if not dsl:
            print(f"  [{sig.id}] {step_type}: no template, skipping")
            skipped += 1
            continue

        # Get DSL template
        dsl_script = json.dumps(dsl)
        dsl_type = dsl["type"]

        # Get NL template if available
        nl = NL_TEMPLATES.get(step_type, {})
        clarifying_questions = nl.get("clarifying_questions", [])
        param_descriptions = nl.get("param_descriptions", {})

        # Update signature
        db.update_nl_interface(
            signature_id=sig.id,
            clarifying_questions=clarifying_questions,
            param_descriptions=param_descriptions,
            dsl_script=dsl_script,
            dsl_type=dsl_type,
        )

        source = "inferred" if inferred else "template"
        print(f"  [{sig.id}] {step_type}: added DSL ({dsl_type}, {source})")
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped")


if __name__ == "__main__":
    main()
